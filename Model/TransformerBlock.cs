using System;
using System.Collections.Generic;
using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class TransformerBlock : ITrainableLayer
    {
        public string Name { get; }

        private readonly ITrainableLayer _multiHeadAttention;
        private readonly ITrainableLayer _feedForward;
        private readonly ITrainableLayer _layerNorm1;
        private readonly ITrainableLayer _layerNorm2;

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
            string name = "transformer_block")
        {
            Name = name;
            _multiHeadAttention = multiHeadAttention ?? throw new ArgumentNullException(nameof(multiHeadAttention));
            _feedForward = feedForward ?? throw new ArgumentNullException(nameof(feedForward));
            _layerNorm1 = layerNorm1 ?? throw new ArgumentNullException(nameof(layerNorm1));
            _layerNorm2 = layerNorm2 ?? throw new ArgumentNullException(nameof(layerNorm2));
        }

        // ILayer forward entry point
        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, workspace),
                3 => ForwardBatch(input, workspace),
                _ => throw new ArgumentException($"Input must be rank 2 or rank 3. Got Rank {input.Rank}.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorWorkspace workspace)
        {
            // Sub-layer 1: Attention + Residual 1 + Norm 1
            TensorBase attention = _multiHeadAttention.Forward(input, workspace);

            TensorBase residual1 = workspace.BorrowLike(input);
            workspace.Backend.ElementWiseAddInto(attention, input, residual1);
            workspace.Release(attention);

            TensorBase norm1 = _layerNorm1.Forward(residual1, workspace);
            workspace.Release(residual1);

            // Sub-layer 2: FeedForward + Residual 2 + Norm 2
            TensorBase ff = _feedForward.Forward(norm1, workspace);

            TensorBase residual2 = workspace.BorrowLike(norm1);
            workspace.Backend.ElementWiseAddInto(ff, norm1, residual2);
            workspace.Release(ff);
            workspace.Release(norm1);

            TensorBase output = _layerNorm2.Forward(residual2, workspace);
            workspace.Release(residual2);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorWorkspace workspace)
        {
            DiagonisticUtilities.AssertNoNaN(input, "Block Input");

            // 1. Attention Pass
            TensorBase attention = _multiHeadAttention.Forward(input, workspace);
            DiagonisticUtilities.AssertNoNaN(attention, "Attention Pre-Residual");

            // 2. Residual Addition 1
            TensorBase attentionResidual = workspace.BorrowLike(attention);
            TensorMathSimd.ElementWiseAddInto(attention, input, attentionResidual);
            workspace.Release(attention);
            DiagonisticUtilities.AssertNoNaN(attentionResidual, "Attention Post-Residual");

            // 3. LayerNorm 1
            TensorBase norm1 = _layerNorm1.Forward(attentionResidual, workspace);
            workspace.Release(attentionResidual);
            DiagonisticUtilities.AssertNoNaN(norm1, "Norm1");

            // 4. FeedForward Pass
            TensorBase ff = _feedForward.Forward(norm1, workspace);
            DiagonisticUtilities.AssertNoNaN(ff, "FeedForward Pre-Residual");

            // 5. Residual Addition 2
            TensorBase ffResidual = workspace.BorrowLike(ff);
            TensorMathSimd.ElementWiseAddInto(ff, norm1, ffResidual);
            workspace.Release(ff);
            workspace.Release(norm1);
            DiagonisticUtilities.AssertNoNaN(ffResidual, "FeedForward Post-Residual");

            // 6. LayerNorm 2
            TensorBase output = _layerNorm2.Forward(ffResidual, workspace);
            workspace.Release(ffResidual);

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
            DiagonisticUtilities.AssertNoNaN(gradient, "Block Gradient");

            // 1. Backprop LayerNorm2
            TensorBase dResidual2 = _layerNorm2.Backward(gradient, workspace);

            // 2. Backprop FeedForward
            TensorBase dFf = _feedForward.Backward(dResidual2, workspace);

            // 3. Split gradient at Residual 2 (FFN path + skip connection)
            TensorBase dNorm1 = workspace.BorrowLike(dFf);
            workspace.Backend.ElementWiseAddInto(dFf, dResidual2, dNorm1);
            workspace.Release(dFf);
            workspace.Release(dResidual2);

            // 4. Backprop LayerNorm1
            TensorBase dResidual1 = _layerNorm1.Backward(dNorm1, workspace);
            workspace.Release(dNorm1);

            // 5. Backprop Attention
            TensorBase dAttention = _multiHeadAttention.Backward(dResidual1, workspace);

            // 6. Split gradient at Residual 1 (Attention path + skip connection)
            TensorBase dInput = workspace.BorrowLike(dAttention);
            workspace.Backend.ElementWiseAddInto(dAttention, dResidual1, dInput);
            workspace.Release(dAttention);
            workspace.Release(dResidual1);

            return dInput;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            DiagonisticUtilities.AssertNoNaN(gradient, "Block Input Gradient");

            // 1. Backprop LayerNorm2
            TensorBase dFfResidual = _layerNorm2.Backward(gradient, workspace);

            // 2. Backprop FeedForward
            TensorBase dFf = _feedForward.Backward(dFfResidual, workspace);

            // 3. Split gradient at Residual 2
            TensorBase dNorm1 = workspace.BorrowLike(dFf);
            workspace.Backend.ElementWiseAddInto(dFf, dFfResidual, dNorm1);
            workspace.Release(dFf);
            workspace.Release(dFfResidual);

            // 4. Backprop LayerNorm1
            TensorBase dAttnResidual = _layerNorm1.Backward(dNorm1, workspace);
            workspace.Release(dNorm1);

            // 5. Backprop MultiHeadAttention
            TensorBase dAttention = _multiHeadAttention.Backward(dAttnResidual, workspace);

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