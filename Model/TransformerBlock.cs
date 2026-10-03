using System;
using System.Collections.Generic;
using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class TransformerBlock : ITrainableLayer, IDropoutControl
    {
        public string Name { get; }

        private readonly ITrainableLayer _multiHeadAttention;

        /// <summary>Checkpoint/determinism tests: reach the attention module.</summary>
        internal MultiHeadAttention MultiHeadAttentionForTest => (MultiHeadAttention)_multiHeadAttention;
        private readonly ITrainableLayer _feedForward;
        private readonly ITrainableLayer _layerNorm1;
        private readonly ITrainableLayer _layerNorm2;

        // Residual dropout (Vaswani §5.4): applied to each sub-layer's output
        // BEFORE it is added to the skip connection. Disabled by default;
        // toggled by the model's BeginTraining/EndTraining via IDropoutControl.
        // Sites (not bare layers) so the mask stream is keyed from a persisted
        // salt + optimizer step and therefore replays exactly on resume.
        private readonly DropoutSite _dropoutAttention;
        private readonly DropoutSite _dropoutFeedForward;

        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var p in _multiHeadAttention.Parameters)
                    yield return p;

                foreach (var p in _feedForward.Parameters)
                    yield return p;

                foreach (var p in _layerNorm1.Parameters)
                    yield return p;

                foreach (var p in _layerNorm2.Parameters)
                    yield return p;
            }
        }

        public TransformerBlock(
            ITrainableLayer multiHeadAttention, 
            ITrainableLayer feedForward, 
            ITrainableLayer layerNorm1, 
            ITrainableLayer layerNorm2, 
            bool useQLora = false,
            float dropoutRate = 0.0f,
            string name = "transformer_block")
        {
            Name = name;
            _multiHeadAttention = multiHeadAttention ?? throw new ArgumentNullException(nameof(multiHeadAttention));
            _feedForward = feedForward ?? throw new ArgumentNullException(nameof(feedForward));
            _layerNorm1 = layerNorm1 ?? throw new ArgumentNullException(nameof(layerNorm1));
            _layerNorm2 = layerNorm2 ?? throw new ArgumentNullException(nameof(layerNorm2));

            _dropoutAttention = new DropoutSite($"{name}.attn_dropout", dropoutRate);
            _dropoutFeedForward = new DropoutSite($"{name}.ffn_dropout", dropoutRate);
        }

        public void SetDropoutEnabled(bool enabled)
        {
            _dropoutAttention.SetEnabled(enabled);
            _dropoutFeedForward.SetEnabled(enabled);

            if (_multiHeadAttention is IDropoutControl attentionControl)
                attentionControl.SetDropoutEnabled(enabled);
        }

        public void SetDropoutRate(float rate)
        {
            _dropoutAttention.SetRate(rate);
            _dropoutFeedForward.SetRate(rate);

            if (_multiHeadAttention is IDropoutControl attentionControl)
                attentionControl.SetDropoutRate(rate);
        }

        public void PrepareDropoutForStep(long step)
        {
            _dropoutAttention.PrepareForStep(step);
            _dropoutFeedForward.PrepareForStep(step);

            if (_multiHeadAttention is IDropoutControl attentionControl)
                attentionControl.PrepareDropoutForStep(step);
        }

        public void CollectDropoutSites(List<DropoutSite> sites)
        {
            sites.Add(_dropoutAttention);
            sites.Add(_dropoutFeedForward);

            if (_multiHeadAttention is IDropoutControl attentionControl)
                attentionControl.CollectDropoutSites(sites);
        }

        // ILayer forward entry point
        public TensorBase Forward(TensorBase input, TensorWorkspace workspace) => Forward(input, workspace, null);

        /// <summary>
        /// Forward with an optional attention mask, threaded down to
        /// ScaledDotProductAttention. A rank-2 mask is position-only (causal) and
        /// shared by every batch item; a rank-3 mask carries one per item.
        /// </summary>
        public TensorBase Forward(TensorBase input, TensorWorkspace workspace, TensorBase? mask)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, workspace, mask),
                3 => ForwardBatch(input, workspace, mask),
                _ => throw new ArgumentException($"Input must be rank 2 or rank 3. Got Rank {input.Rank}.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorWorkspace workspace, TensorBase? mask)
        {
            // Sub-layer 1: Attention + Dropout + Residual 1 + Norm 1
            TensorBase attention = _multiHeadAttention.Forward(input, workspace, mask);
            TensorBase droppedAttention = _dropoutAttention.ForItem(0).Forward(attention, workspace);
            workspace.Release(attention);

            TensorBase residual1 = workspace.BorrowLike(input);
            workspace.Backend.ElementWiseAddInto(droppedAttention, input, residual1);
            workspace.Release(droppedAttention);

            TensorBase norm1 = _layerNorm1.Forward(residual1, workspace);
            workspace.Release(residual1);

            // Sub-layer 2: FeedForward + Dropout + Residual 2 + Norm 2
            TensorBase ff = _feedForward.Forward(norm1, workspace);
            TensorBase droppedFf = _dropoutFeedForward.ForItem(0).Forward(ff, workspace);
            workspace.Release(ff);

            TensorBase residual2 = workspace.BorrowLike(norm1);
            workspace.Backend.ElementWiseAddInto(droppedFf, norm1, residual2);
            workspace.Release(droppedFf);
            // norm1 stays borrowed: it is _feedForward._expand's _lastInput,
            // which Backward reads for dW. Releasing it here let later
            // same-shape borrows overwrite it behind the cache's back
            // (use-after-release = nondeterministic gradients). Reclaimed by
            // the step's workspace Reset.
            // (The residual1/residual2 releases are kept: LayerNorm's
            // _lastInput cache is shape-only and never reads those values.)

            TensorBase output = _layerNorm2.Forward(residual2, workspace);
            workspace.Release(residual2);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorWorkspace workspace, TensorBase? mask)
        {
            bool verbose = TrainingStepProfile.Enabled;
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(input, "Block Input");

            // 1. Attention Pass
            TensorBase attention = _multiHeadAttention.Forward(input, workspace, mask);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(attention, "Attention Pre-Residual");

            // 1b. Attention-path dropout (inverted dropout; pass-through at inference)
            TensorBase droppedAttention = ApplyResidualDropout(_dropoutAttention, attention, workspace);
            workspace.Release(attention);

            // 2. Residual Addition 1
            TensorBase attentionResidual = workspace.BorrowLike(droppedAttention);
            TensorMathSimd.ElementWiseAddInto(droppedAttention, input, attentionResidual);
            workspace.Release(droppedAttention);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(attentionResidual, "Attention Post-Residual");

            // 3. LayerNorm 1
            TensorBase norm1 = _layerNorm1.Forward(attentionResidual, workspace);
            workspace.Release(attentionResidual);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(norm1, "Norm1");

            // 4. FeedForward Pass
            TensorBase ff = _feedForward.Forward(norm1, workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(ff, "FeedForward Pre-Residual");

            // 4b. FFN-path dropout (inverted dropout; pass-through at inference)
            TensorBase droppedFf = ApplyResidualDropout(_dropoutFeedForward, ff, workspace);
            workspace.Release(ff);

            // 5. Residual Addition 2
            TensorBase ffResidual = workspace.BorrowLike(droppedFf);
            TensorMathSimd.ElementWiseAddInto(droppedFf, norm1, ffResidual);
            workspace.Release(droppedFf);
            // norm1 stays borrowed: _feedForward._expand's _lastInput cache
            // reads it in Backward (see ForwardSequence). Reclaimed by Reset.
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(ffResidual, "FeedForward Post-Residual");

            // 6. LayerNorm 2
            TensorBase output = _layerNorm2.Forward(ffResidual, workspace);
            workspace.Release(ffResidual);

            return output;
        }

        /// <summary>
        /// Applies per-item residual dropout to a Rank-2 or Rank-3 activation.
        /// Rank-2 uses item 0; Rank-3 applies ForItem(b) per batch slice so each
        /// mask is a pure function of (salt, step, item), matching the attention
        /// dropout path. Pass-through (disabled or rate 0) takes a single fast path.
        /// </summary>
        private static TensorBase ApplyResidualDropout(DropoutSite site, TensorBase input, TensorWorkspace workspace)
        {
            if (input.Rank == 2)
                return site.ForItem(0).Forward(input, workspace);

            if (input.Rank != 3)
                throw new ArgumentException($"Input must be rank 2 or rank 3. Got Rank {input.Rank}.");

            TensorBase output = workspace.BorrowLike(input);
            int layers = input.Layers;
            for (int b = 0; b < layers; b++)
            {
                TensorBase inSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase droppedSlice = site.ForItem(b).Forward(inSlice, workspace);
                TensorUtilitiesSimd.SetLayer(output, b, droppedSlice);
                workspace.Release(droppedSlice);
            }

            return output;
        }

        /// <summary>
        /// Pushes a Rank-2 or Rank-3 gradient back through the matching per-item
        /// dropout masks applied by <see cref="ApplyResidualDropout"/>.
        /// </summary>
        private static TensorBase BackwardResidualDropout(DropoutSite site, TensorBase gradient, TensorWorkspace workspace)
        {
            if (gradient.Rank == 2)
                return site.ForItem(0).Backward(gradient, workspace);

            if (gradient.Rank != 3)
                throw new ArgumentException($"Gradient must be rank 2 or rank 3. Got Rank {gradient.Rank}.");

            TensorBase output = workspace.BorrowLike(gradient);
            int layers = gradient.Layers;
            for (int b = 0; b < layers; b++)
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase droppedSlice = site.ForItem(b).Backward(gradSlice, workspace);
                TensorUtilitiesSimd.SetLayer(output, b, droppedSlice);
                workspace.Release(droppedSlice);
            }

            return output;
        }

        // ILayer backward entry point
        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient, workspace),
                3 => BackwardBatch(gradient, workspace),
                _ => throw new ArgumentException($"Gradient must be rank 2 or rank 3. Got Rank {gradient.Rank}.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient, TensorWorkspace workspace)
        {
            if (TrainingStepProfile.Enabled)
                DiagonisticUtilities.AssertNoNaN(gradient, "Block Gradient");

            // 1. Backprop LayerNorm2
            TensorBase dResidual2 = _layerNorm2.Backward(gradient, workspace);

            // 2. Backprop FFN-path dropout, then FeedForward
            TensorBase dDroppedFf = _dropoutFeedForward.ForItem(0).Backward(dResidual2, workspace);
            TensorBase dFf = _feedForward.Backward(dDroppedFf, workspace);
            workspace.Release(dDroppedFf);

            // 3. Split gradient at Residual 2 (FFN path + skip connection)
            TensorBase dNorm1 = workspace.BorrowLike(dFf);
            workspace.Backend.ElementWiseAddInto(dFf, dResidual2, dNorm1);
            workspace.Release(dFf);
            workspace.Release(dResidual2);

            // 4. Backprop LayerNorm1
            TensorBase dResidual1 = _layerNorm1.Backward(dNorm1, workspace);
            workspace.Release(dNorm1);

            // 5. Backprop attention-path dropout, then Attention
            TensorBase dDroppedAttention = _dropoutAttention.ForItem(0).Backward(dResidual1, workspace);
            TensorBase dAttention = _multiHeadAttention.Backward(dDroppedAttention, workspace);
            workspace.Release(dDroppedAttention);

            // 6. Split gradient at Residual 1 (Attention path + skip connection)
            TensorBase dInput = workspace.BorrowLike(dAttention);
            workspace.Backend.ElementWiseAddInto(dAttention, dResidual1, dInput);
            workspace.Release(dAttention);
            workspace.Release(dResidual1);

            return dInput;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            bool verbose = TrainingStepProfile.Enabled;
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(gradient, "Block Input Gradient");

            // 1. Backprop LayerNorm2
            TensorBase dFfResidual = _layerNorm2.Backward(gradient, workspace);

            // 2. Backprop FFN-path dropout, then FeedForward
            TensorBase dDroppedFf = BackwardResidualDropout(_dropoutFeedForward, dFfResidual, workspace);
            TensorBase dFf = _feedForward.Backward(dDroppedFf, workspace);
            workspace.Release(dDroppedFf);

            // 3. Split gradient at Residual 2
            TensorBase dNorm1 = workspace.BorrowLike(dFf);
            workspace.Backend.ElementWiseAddInto(dFf, dFfResidual, dNorm1);
            workspace.Release(dFf);
            workspace.Release(dFfResidual);

            // 4. Backprop LayerNorm1
            TensorBase dAttnResidual = _layerNorm1.Backward(dNorm1, workspace);
            workspace.Release(dNorm1);

            // 5. Backprop attention-path dropout, then MultiHeadAttention
            TensorBase dDroppedAttention = BackwardResidualDropout(_dropoutAttention, dAttnResidual, workspace);
            TensorBase dAttention = _multiHeadAttention.Backward(dDroppedAttention, workspace);
            workspace.Release(dDroppedAttention);

            // 6. Split gradient at Residual 1
            TensorBase dInput = workspace.BorrowLike(dAttention);
            workspace.Backend.ElementWiseAddInto(dAttention, dAttnResidual, dInput);
            workspace.Release(dAttention);
            workspace.Release(dAttnResidual);

            return dInput;
        }

        public void ZeroGradients()
        {
            _multiHeadAttention.ZeroGradients();
            _feedForward.ZeroGradients();
            _layerNorm1.ZeroGradients();
            _layerNorm2.ZeroGradients();
        }

        public void Dispose()
        {
            _multiHeadAttention.Dispose();
            _feedForward.Dispose();
            _layerNorm1.Dispose();
            _layerNorm2.Dispose();
        }
    }
}