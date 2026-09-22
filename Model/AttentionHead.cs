using System;
using System.Collections.Generic;
using System.Linq;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class AttentionHead : ITrainableLayer
    {
        public string Name { get; }
        private readonly ILinearLayer _queryProjection;
        private readonly ILinearLayer _keyProjection;
        private readonly ILinearLayer _valueProjection;
        private readonly ScaledDotProductAttention _attention;

        public ILinearLayer QueryProjection => _queryProjection;
        public ILinearLayer KeyProjection => _keyProjection;
        public ILinearLayer ValueProjection => _valueProjection;

        public IEnumerable<TrainableParameter> Parameters =>
            _queryProjection.Parameters
                .Concat(_keyProjection.Parameters)
                .Concat(_valueProjection.Parameters);

        public AttentionHead(int embeddingSize, int headSize, string name = "attention_head", bool useQLora = false)
        {
            Name = name;
            if(useQLora)
            {
                _queryProjection = new QLoraLinearLayer(embeddingSize, headSize, name: $"{name}.query");
                _keyProjection   = new QLoraLinearLayer(embeddingSize, headSize, name: $"{name}.key");
                _valueProjection = new QLoraLinearLayer(embeddingSize, headSize, name: $"{name}.value");
            }
            else
            {
                _queryProjection = new LinearLayer(embeddingSize, headSize, name: $"{name}.query");
                _keyProjection   = new LinearLayer(embeddingSize, headSize, name: $"{name}.key");
                _valueProjection = new LinearLayer(embeddingSize, headSize, name: $"{name}.value");
            }

            _attention = new ScaledDotProductAttention(headSize);
        }

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace) => Forward(input, workspace, null);

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace, TensorBase? mask)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, mask, workspace),
                3 => ForwardBatch(input, mask, workspace),
                _ => throw new ArgumentException($"Input must be Rank 2 or 3. Got Rank {input.Rank}.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorBase? mask, TensorWorkspace workspace)
        {
            if (input.Rank != 2) 
                throw new ArgumentException("Input must be a matrix (Rank 2).");

            // 1. Project input into Q, K, V workspace tensors
            TensorBase q = _queryProjection.Forward(input, workspace);
            TensorBase k = _keyProjection.Forward(input, workspace);
            TensorBase v = _valueProjection.Forward(input, workspace);

            // 2. Compute attention output
            TensorBase output = _attention.Forward(q, k, v, mask, workspace);

            // 3. Release intermediate Q, K, V buffers back to workspace pool
            workspace.Release(q);
            workspace.Release(k);
            workspace.Release(v);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorBase? mask, TensorWorkspace workspace)
        {
            if (input.Rank != 3)
                throw new ArgumentException("Input must be a stacked matrix (Rank 3).");

            TensorBase q = _queryProjection.Forward(input, workspace);
            TensorBase k = _keyProjection.Forward(input, workspace);
            TensorBase v = _valueProjection.Forward(input, workspace);

            TensorBase output = _attention.Forward(q, k, v, mask, workspace);

            workspace.Release(q);
            workspace.Release(k);
            workspace.Release(v);

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient, workspace),
                3 => BackwardBatch(gradient, workspace),
                _ => throw new ArgumentException($"Gradient must be Rank 2 or 3. Got Rank {gradient.Rank}.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient, TensorWorkspace workspace)
        {
            // 1. Backward pass through scaled dot-product attention
            var (dQ, dK, dV) = _attention.Backward(gradient, workspace);

            // 2. Backward pass through projection linear layers
            TensorBase dInputQ = _queryProjection.Backward(dQ, workspace);
            TensorBase dInputK = _keyProjection.Backward(dK, workspace);
            TensorBase dInputV = _valueProjection.Backward(dV, workspace);

            // 3. Borrow destination tensor for accumulated input gradient
            TensorBase dInput = workspace.Borrow2D(dInputQ.Rows, dInputQ.Cols);

            // 3-way elementwise addition through the backend: dInput = dInputQ + dInputK + dInputV
            workspace.Backend.ElementWiseAddInto(dInputQ, dInputK, dInput);
            workspace.Backend.ElementWiseAddInPlace(dInput, dInputV);

            // 4. Release all intermediate gradient buffers back to workspace pool
            workspace.Release(dQ);
            workspace.Release(dK);
            workspace.Release(dV);
            workspace.Release(dInputQ);
            workspace.Release(dInputK);
            workspace.Release(dInputV);

            return dInput;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            if (gradient.Rank != 3)
                throw new ArgumentException("Gradient must be a stacked matrix (Rank 3).");

            var (dQ, dK, dV) = _attention.Backward(gradient, workspace);

            TensorBase dInputQ = _queryProjection.Backward(dQ, workspace);
            TensorBase dInputK = _keyProjection.Backward(dK, workspace);
            TensorBase dInputV = _valueProjection.Backward(dV, workspace);

            TensorBase dInput = workspace.Borrow3D(dInputQ.Layers, dInputQ.Rows, dInputQ.Cols);

            workspace.Backend.ElementWiseAddInto(dInputQ, dInputK, dInput);
            workspace.Backend.ElementWiseAddInPlace(dInput, dInputV);

            workspace.Release(dQ);
            workspace.Release(dK);
            workspace.Release(dV);
            workspace.Release(dInputQ);
            workspace.Release(dInputK);
            workspace.Release(dInputV);

            return dInput;
        }

        public void ZeroGradients()
        {
            _queryProjection.ZeroGradients();
            _keyProjection.ZeroGradients();
            _valueProjection.ZeroGradients();
        }

        public void Dispose()
        {
            _queryProjection.Dispose();
            _keyProjection.Dispose();
            _valueProjection.Dispose();
        }
    }
}