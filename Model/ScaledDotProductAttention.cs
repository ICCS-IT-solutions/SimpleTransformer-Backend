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
        private readonly List<TensorBase> _lastWeights = new();

        private readonly int _headSize;

        public ScaledDotProductAttention(int headSize)
        {
            _headSize = headSize;
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
            }

            _lastQ.Add(q);
            _lastK.Add(k);
            _lastV.Add(v);

            // Borrow temporary buffers for matmul and transposes
            TensorBase kTransposed = workspace.Borrow2D(k.Cols, k.Rows);
            TensorBase scores = workspace.Borrow2D(q.Rows, k.Rows);

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

            // 5. Cache softmax weights PER BATCH ITEM for backprop (persist until Backward completes)
            Tensor currentWeights = new Tensor(scores.Rows, scores.Cols);
            workspace.Backend.CopyInto(scores, currentWeights);
            _lastWeights.Add(currentWeights);

            // 6. Output = SoftmaxWeights * V
            TensorBase output = workspace.Borrow2D(scores.Rows, v.Cols);
            workspace.Backend.MatMul(scores, v, output);

            workspace.Release(scores);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask, TensorWorkspace workspace)
        {
            _lastQ.Clear();
            _lastK.Clear();
            _lastV.Clear();
            _lastWeights.Clear();

            TensorBase batchOutput = workspace.Borrow3D(q.Layers, q.Rows, v.Cols);

            for (int b = 0; b < q.Layers; b++)
            {
                TensorBase qSlice = TensorUtilitiesSimd.GetLayer(q, b);
                TensorBase kSlice = TensorUtilitiesSimd.GetLayer(k, b);
                TensorBase vSlice = TensorUtilitiesSimd.GetLayer(v, b);
                TensorBase? maskSlice = mask != null ? TensorUtilitiesSimd.GetLayer(mask, b) : null;

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
                2 => BackwardSequence(outputGradient, _lastQ[0], _lastK[0], _lastV[0], _lastWeights[0], workspace),
                3 => BackwardBatch(outputGradient, workspace),
                _ => throw new ArgumentException("Gradient must be Rank 2 or 3.")
            };
        }

        private (TensorBase dQ, TensorBase dK, TensorBase dV) BackwardSequence(
            TensorBase outputGradient, 
            TensorBase q, 
            TensorBase k, 
            TensorBase v,
            TensorBase savedWeights,
            TensorWorkspace workspace)
        {
            // dV = SoftmaxWeights^T * outputGradient
            TensorBase weightsTransposed = workspace.Borrow2D(savedWeights.Cols, savedWeights.Rows);
            workspace.Backend.TransposeInto(savedWeights, weightsTransposed);

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
            TensorBase dScores = workspace.Borrow2D(dWeights.Rows, dWeights.Cols);
            workspace.Backend.SoftmaxBackwardInto(savedWeights, dWeights, dScores);
            workspace.Release(dWeights);

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