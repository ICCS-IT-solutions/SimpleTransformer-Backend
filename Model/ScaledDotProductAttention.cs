using System;
using System.Collections.Generic;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class ScaledDotProductAttention
    {
        private readonly List<TensorBase> _lastQ = new();
        private readonly List<TensorBase> _lastK = new();
        private readonly List<TensorBase> _lastV = new();
        // Entry is null when AttentionMemorySettings.RecomputeSoftmaxWeights is
        // on for this call: Backward then rebuilds the matrix instead of reading
        // it back (see RecomputeSoftmaxWeights).
        private readonly List<TensorBase?> _lastWeights = new();
        // The mask each item was scored under, kept for the same reason: a
        // recompute has to replay masking as well as the Q*K^T and softmax.
        private readonly List<TensorBase?> _lastMasks = new();

        // Attention-weight dropout (Vaswani §5.4). One DropoutSite per
        // (layer, head); the site hands out one mask-carrying DropoutLayer per
        // batch item, keyed from (salt, step, item), because the batch path
        // replays ForwardSequence per slice and Backward must replay each item's
        // exact mask. _lastDroppedWeights caches the post-dropout weights
        // actually fed to the output matmul (needed for dV in Backward), while
        // _lastWeights above keeps the PRE-dropout softmax output for the
        // softmax Jacobian. Parallel to _lastWeights; entries are null when no
        // dropout was applied for that item.
        private readonly DropoutSite _attentionDropout;
        private readonly List<TensorBase?> _lastDroppedWeights = new();
        private readonly List<bool> _dropoutApplied = new();

        private readonly int _headSize;

        public ScaledDotProductAttention(int headSize, string siteKeyPrefix = "attention")
        {
            _headSize = headSize;
            _attentionDropout = new DropoutSite($"{siteKeyPrefix}.attn_w", 0.0f);
        }

        /// <summary>
        /// The attention-weight dropout site. Exposed so the owning head can
        /// include it in checkpoint save/load and per-step re-keying.
        /// </summary>
        public DropoutSite AttentionDropout => _attentionDropout;

        /// <summary>Dropout rate applied to the softmax weights (attention dropout).</summary>
        public float DropoutRate
        {
            get => _attentionDropout.Rate;
            set => _attentionDropout.SetRate(value);
        }

        /// <summary>Train/eval switch. False by default so inference is untouched.</summary>
        public bool DropoutEnabled
        {
            get => _attentionDropout.Enabled;
            set => _attentionDropout.SetEnabled(value);
        }

        public TensorBase Forward(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask, TensorWorkspace workspace)
        {
            return (q.Rank, k.Rank, v.Rank) switch
            {
                (2, 2, 2) => ForwardSequence(q, k, v, mask, batchIndex: 0, isBatch: false, workspace),
                (3, 3, 3) => ForwardBatch(q, k, v, mask, workspace),
                _ => throw new ArgumentException("Q, K and V must all be matrices of Rank 2 or 3.")
            };
        }

        private TensorBase ForwardSequence(
            TensorBase q, 
            TensorBase k, 
            TensorBase v, 
            TensorBase? mask, 
            int batchIndex, 
            bool isBatch, 
            TensorWorkspace workspace)
        {
            if (!isBatch)
            {
                _lastQ.Clear();
                _lastK.Clear();
                _lastV.Clear();
                _lastWeights.Clear();
                _lastMasks.Clear();
                _lastDroppedWeights.Clear();
                _dropoutApplied.Clear();
            }

            _lastQ.Add(q);
            _lastK.Add(k);
            _lastV.Add(v);
            _lastMasks.Add(mask);

            // Borrow temporary buffers for matmul and transposes
            TensorBase kTransposed = workspace.Borrow2D(k.Cols, k.Rows);
            TensorBase scores = workspace.Borrow2D(q.Rows, k.Rows);

            // Sequential chain, one fence wait: the transpose/matmul/scale/mask/
            // softmax/copy/matmul chain round-trips per op today (each op packs,
            // dispatches, waits, then unpacks). The scope keeps the dispatches in
            // one submission; the unpacks still wait per op. CPU backends no-op.
            using (workspace.Backend.BeginBatchScope())
            {
                // 1. Q * K^T
                workspace.Backend.TransposeInto(k, kTransposed);
                workspace.Backend.MatMul(q, kTransposed, scores);
                workspace.Release(kTransposed);

                // 2. Scale scores: 1 / sqrt(d_k)
                workspace.Backend.ScaleInPlace(scores, 1.0f / MathF.Sqrt(_headSize));

                // 3. Apply mask if provided (-1e9f before softmax)
                if (mask != null)
                {
                    workspace.Backend.ApplyMaskInPlace(scores, mask);
                }

                // 4. Softmax computation
                workspace.Backend.SoftmaxInPlace(scores);

                // 5. Cache softmax weights PER BATCH ITEM for backprop (persist until Backward
                // completes).
                //
                // Pooled, NOT `new Tensor(...)`. At seq 2048 this is a 16 MiB
                // array; over 16 layers x 16 heads x 8 batch items that is 32
                // GiB per step. Allocated off-pool, every copy lands on the
                // Large Object Heap, which only a gen2 GC reclaims - so the heap
                // grew without bound between the memory valve's rate-limited
                // compactions and a long run died of exhaustion rather than of
                // its own activation working set. Borrowed, it is bounded by
                // MemoryPressureSettings.WorkspaceCapFractionOfQuota like every
                // other activation and reclaimed by the step's workspace Reset.
                //
                // Under AttentionMemorySettings.RecomputeSoftmaxWeights the cache
                // is skipped entirely and Backward recomputes it (see
                // RecomputeSoftmaxWeights), which removes the seq^2 term.
                bool applyDropout = DropoutEnabled && DropoutRate > 0f;
                bool recompute = AttentionMemorySettings.RecomputeSoftmaxWeights && !applyDropout;

                if (recompute)
                {
                    _lastWeights.Add(null);
                }
                else
                {
                    TensorBase currentWeights = workspace.Borrow2D(scores.Rows, scores.Cols);
                    workspace.Backend.CopyInto(scores, currentWeights);
                    _lastWeights.Add(currentWeights);
                }

                // 6. Attention-weight dropout: the PRE-dropout weights cached
                // above stay intact for the softmax Jacobian; only the tensor
                // fed to the output matmul is dropped.
                TensorBase? dropped = null;

                if (applyDropout)
                {
                    DropoutLayer dropoutLayer = _attentionDropout.ForItem(batchIndex);
                    dropoutLayer.SetRate(DropoutRate);   // pick up config changes
                    dropoutLayer.Enabled = true;         // guarded by applyDropout above

                    dropped = dropoutLayer.Forward(scores, workspace);

                    // Cache the dropped weights for the dV matmul in Backward;
                    // the mask itself lives inside the per-item DropoutLayer.
                    TensorBase droppedCopy = workspace.Borrow2D(dropped.Rows, dropped.Cols);
                    workspace.Backend.CopyInto(dropped, droppedCopy);
                    _lastDroppedWeights.Add(droppedCopy);
                }
                else
                {
                    _lastDroppedWeights.Add(null);
                }

                _dropoutApplied.Add(applyDropout);

                // 7. Output = DroppedWeights * V
                TensorBase output = workspace.Borrow2D(scores.Rows, v.Cols);
                workspace.Backend.MatMul(applyDropout ? dropped! : scores, v, output);

                if (dropped != null)
                    workspace.Release(dropped);
                workspace.Release(scores);

                return output;
            }
        }

        private TensorBase ForwardBatch(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask, TensorWorkspace workspace)
        {
            _lastQ.Clear();
            _lastK.Clear();
            _lastV.Clear();
            _lastWeights.Clear();
            _lastMasks.Clear();
            _lastDroppedWeights.Clear();
            _dropoutApplied.Clear();

            TensorBase batchOutput = workspace.Borrow3D(q.Layers, q.Rows, v.Cols);

            for (int b = 0; b < q.Layers; b++)
            {
                TensorBase qSlice = TensorUtilitiesSimd.GetLayer(q, b);
                TensorBase kSlice = TensorUtilitiesSimd.GetLayer(k, b);
                TensorBase vSlice = TensorUtilitiesSimd.GetLayer(v, b);
                // A causal mask is the same for every batch item (it depends only
                // on position, not on content), so a rank-2 mask is reused as-is
                // rather than sliced. Only a genuinely per-item mask - a padding
                // mask expanded to rank 3 - needs GetLayer here.
                TensorBase? maskSlice = mask switch
                {
                    null => null,
                    var m when m.Rank == 2 => m,
                    var m => TensorUtilitiesSimd.GetLayer(m, b)
                };

                TensorBase result = ForwardSequence(qSlice, kSlice, vSlice, maskSlice, batchIndex: b, isBatch: true, workspace);

                TensorUtilitiesSimd.SetLayer(batchOutput, b, result);
                
                // Release temporary 2D slice output once packed into 3D batch tensor
                workspace.Release(result);
            }

            return batchOutput;
        }

        public (TensorBase dQ, TensorBase dK, TensorBase dV) Backward(TensorBase outputGradient, TensorWorkspace workspace)
        {
            return outputGradient.Rank switch
            {
                2 => BackwardSequence(outputGradient, _lastQ[0], _lastK[0], _lastV[0], _lastWeights[0], batchIndex: 0, workspace),
                3 => BackwardBatch(outputGradient, workspace),
                _ => throw new ArgumentException("Gradient must be Rank 2 or 3.")
            };
        }

        /// <summary>
        /// Rebuilds the post-softmax weight matrix for Backward, replaying exactly
        /// the forward sequence (Q*K^T, scale, mask, softmax) so the result is the
        /// same matrix the cache would have held. Used only when
        /// <see cref="AttentionMemorySettings.RecomputeSoftmaxWeights"/> is on.
        /// <para>
        /// The caller owns the returned buffer and must Release it.
        /// </para>
        /// </summary>
        private TensorBase RecomputeSoftmaxWeights(
            TensorBase q,
            TensorBase k,
            TensorBase? mask,
            TensorWorkspace workspace)
        {
            TensorBase kTransposed = workspace.Borrow2D(k.Cols, k.Rows);
            TensorBase scores = workspace.Borrow2D(q.Rows, k.Rows);

            workspace.Backend.TransposeInto(k, kTransposed);
            workspace.Backend.MatMul(q, kTransposed, scores);
            workspace.Release(kTransposed);

            workspace.Backend.ScaleInPlace(scores, 1.0f / MathF.Sqrt(_headSize));
            if (mask != null)
                workspace.Backend.ApplyMaskInPlace(scores, mask);

            workspace.Backend.SoftmaxInPlace(scores);
            return scores;
        }

        private (TensorBase dQ, TensorBase dK, TensorBase dV) BackwardSequence(
            TensorBase outputGradient, 
            TensorBase q, 
            TensorBase k, 
            TensorBase v,
            TensorBase? savedWeights,
            int batchIndex,
            TensorWorkspace workspace)
        {
            if (savedWeights == null && _dropoutApplied[batchIndex])
                throw new InvalidOperationException(
                    "Attention dropout was applied but no softmax weight cache was kept. " +
                    "AttentionMemorySettings.RecomputeSoftmaxWeights must not be combined " +
                    "with attention dropout - the dropped matrix cannot be rebuilt without " +
                    "consuming fresh dropout RNG.");

            // dV = WeightsActuallyUsedInForward^T * outputGradient.
            // Without attention dropout that is the pre-dropout softmax output
            // (savedWeights); with dropout the forward matmul consumed the
            // DROPPED weights, so dV must use those (cached per batch item).
            // With the cache skipped, savedWeights is null and the matrix is
            // rebuilt here and released before returning.
            TensorBase? recomputed = null;
            TensorBase weightsUsedInForward;
            if (savedWeights != null)
            {
                weightsUsedInForward = _dropoutApplied[batchIndex]
                    ? _lastDroppedWeights[batchIndex]!
                    : savedWeights;
            }
            else
            {
                weightsUsedInForward = recomputed =
                    RecomputeSoftmaxWeights(q, k, _lastMasks[batchIndex], workspace);
            }

            TensorBase weightsTransposed = workspace.Borrow2D(weightsUsedInForward.Cols, weightsUsedInForward.Rows);
            workspace.Backend.TransposeInto(weightsUsedInForward, weightsTransposed);

            TensorBase dV = workspace.Borrow2D(v.Rows, v.Cols);
            workspace.Backend.MatMul(weightsTransposed, outputGradient, dV);
            workspace.Release(weightsTransposed);

            // dWeights = outputGradient * V^T
            TensorBase vTransposed = workspace.Borrow2D(v.Cols, v.Rows);
            workspace.Backend.TransposeInto(v, vTransposed);

            TensorBase dWeights = workspace.Borrow2D(outputGradient.Rows, vTransposed.Cols);
            workspace.Backend.MatMul(outputGradient, vTransposed, dWeights);
            workspace.Release(vTransposed);

            // dScores = SoftmaxBackward(dWeights, SoftmaxWeights)
            // When attention dropout was applied, dWeights is the gradient wrt
            // the DROPPED tensor: push it back through the dropout mask first so
            // the softmax Jacobian receives the gradient wrt the pre-dropout
            // weights cached in _lastWeights.
            TensorBase dScores = workspace.Borrow2D(dWeights.Rows, dWeights.Cols);

            if (_dropoutApplied[batchIndex])
            {
                TensorBase dWeightsDropped = _attentionDropout.ForItem(batchIndex).Backward(dWeights, workspace);
                // savedWeights is the PRE-dropout softmax and is always present
                // here: recompute is suppressed when dropout runs.
                workspace.Backend.SoftmaxBackwardInto(savedWeights!, dWeightsDropped, dScores);
                workspace.Release(dWeightsDropped);
                workspace.Release(dWeights);
            }
            else
            {
                // weightsUsedInForward, not savedWeights: when the cache was
                // skipped the matrix lives in the rebuilt buffer instead.
                workspace.Backend.SoftmaxBackwardInto(weightsUsedInForward, dWeights, dScores);
                workspace.Release(dWeights);
            }

            // Scale dScores back by 1 / Sqrt(headSize)
            workspace.Backend.ScaleInPlace(dScores, 1.0f / MathF.Sqrt(_headSize));

            // dQ = dScores * K
            TensorBase dQ = workspace.Borrow2D(q.Rows, q.Cols);
            workspace.Backend.MatMul(dScores, k, dQ);

            // dK = dScores^T * Q
            TensorBase dScoresTransposed = workspace.Borrow2D(dScores.Cols, dScores.Rows);
            workspace.Backend.TransposeInto(dScores, dScoresTransposed);

            TensorBase dK = workspace.Borrow2D(k.Rows, k.Cols);
            workspace.Backend.MatMul(dScoresTransposed, q, dK);

            // Clean up temporary workspace buffers
            workspace.Release(dScores);
            workspace.Release(dScoresTransposed);
            if (recomputed != null)
                workspace.Release(recomputed);

            return (dQ, dK, dV);
        }

        private (TensorBase dQ, TensorBase dK, TensorBase dV) BackwardBatch(TensorBase outputGradient, TensorWorkspace workspace)
        {
            if (_lastQ.Count == 0)
                throw new InvalidOperationException("Forward pass must be called before Backward.");

            int layers = outputGradient.Layers;

            TensorBase batchDQ = workspace.Borrow3D(layers, _lastQ[0].Rows, _lastQ[0].Cols);
            TensorBase batchDK = workspace.Borrow3D(layers, _lastK[0].Rows, _lastK[0].Cols);
            TensorBase batchDV = workspace.Borrow3D(layers, _lastV[0].Rows, _lastV[0].Cols);

            for (int b = 0; b < layers; b++)
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(outputGradient, b);

                var (dq, dk, dv) = BackwardSequence(
                    gradSlice,
                    _lastQ[b],
                    _lastK[b],
                    _lastV[b],
                    _lastWeights[b],
                    b,
                    workspace);

                TensorUtilitiesSimd.SetLayer(batchDQ, b, dq);
                TensorUtilitiesSimd.SetLayer(batchDK, b, dk);
                TensorUtilitiesSimd.SetLayer(batchDV, b, dv);

                // Release 2D slices borrowed during BackwardSequence
                workspace.Release(dq);
                workspace.Release(dk);
                workspace.Release(dv);
            }

            return (batchDQ, batchDK, batchDV);
        }
    }
}