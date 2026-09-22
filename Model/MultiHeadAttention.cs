using System;
using System.Collections.Generic;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class MultiHeadAttention : ITrainableLayer
    {
        public string Name { get; }
        private readonly AttentionHead[] _heads;
        private readonly ILinearLayer _outputProjection;
        private readonly int _embeddingSize;
        private readonly int _headSize;

        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var head in _heads)
                {
                    foreach (var p in head.Parameters)
                        yield return p;
                }

                foreach (var p in _outputProjection.Parameters)
                    yield return p;
            }
        }

        public MultiHeadAttention(int embeddingSize, int numHeads, string name = "attention", bool useQLora = false)
        {
            if (embeddingSize % numHeads != 0) 
                throw new ArgumentException("Embedding size must be divisible by number of heads.");

            Name = name;
            _embeddingSize = embeddingSize;
            _headSize = embeddingSize / numHeads;
            _heads = new AttentionHead[numHeads];

            for (int i = 0; i < numHeads; i++)
            {
                _heads[i] = new AttentionHead(embeddingSize, _headSize, name: $"{Name}.heads.{i}", useQLora);
            }

            _outputProjection = useQLora
                ? new QLoraLinearLayer(embeddingSize, embeddingSize, useBias: false, name: $"{Name}.out_proj")
                : new LinearLayer(embeddingSize, embeddingSize, useBias: false, name: $"{Name}.out_proj");
        }

        // Standard ILayer entry points
        public TensorBase Forward(TensorBase input, TensorWorkspace workspace) => Forward(input, workspace, null);

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
            int rows = input.Rows;
            
            // Borrow buffer from workspace instead of 'new Tensor(...)'
            TensorBase concatenated = workspace.Borrow2D(rows, _embeddingSize);

            ComputeForwardSequenceInternal(input, mask, concatenated, workspace);

            TensorBase output = _outputProjection.Forward(concatenated, workspace);

            // Release intermediate buffer after projection completes
            workspace.Release(concatenated);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorWorkspace workspace, TensorBase? mask)
        {
            int layers = input.Layers;
            int rows = input.Rows;

            // Borrow 3D buffer from workspace
            TensorBase concatenatedBatch = workspace.Borrow3D(layers, rows, _embeddingSize);

            for (int b = 0; b < layers; b++)
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase? maskSlice = mask != null ? TensorUtilitiesSimd.GetLayer(mask, b) : null;
                TensorBase concatSlice = TensorUtilitiesSimd.GetLayer(concatenatedBatch, b);

                ComputeForwardSequenceInternal(inputSlice, maskSlice, concatSlice, workspace);
            }

            TensorBase output = _outputProjection.Forward(concatenatedBatch, workspace);

            // Release 3D intermediate buffer
            workspace.Release(concatenatedBatch);

            return output;
        }

        private void ComputeForwardSequenceInternal(TensorBase input, TensorBase? mask, TensorBase targetConcat, TensorWorkspace workspace)
        {
            int numHeads = _heads.Length;
            int rows = input.Rows;

            for (int i = 0; i < numHeads; i++)
            {
                TensorBase headOutput = _heads[i].Forward(input, workspace, mask);
                int startCol = i * _headSize;
                
                // Copy slice into target concatenation buffer
                TensorUtilitiesSimd.CopyBlock(headOutput, 0, targetConcat, startCol, rows, _headSize);

                // Release individual head output immediately after copying
                workspace.Release(headOutput);
            }
        }

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
            TensorBase dConcat = _outputProjection.Backward(gradient, workspace);
            
            // Borrow gradient target tensor from workspace
            TensorBase inputGradient = workspace.Borrow2D(dConcat.Rows, _embeddingSize);

            ComputeBackwardSequenceInternal(dConcat, inputGradient, workspace);

            // Clean up output projection backward output
            workspace.Release(dConcat);

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            TensorBase dConcatBatch = _outputProjection.Backward(gradient, workspace);
            
            int layers = gradient.Layers;
            TensorBase inputGradientBatch = workspace.Borrow3D(layers, gradient.Rows, _embeddingSize);

            for (int b = 0; b < layers; b++)
            {
                TensorBase dConcatSlice = TensorUtilitiesSimd.GetLayer(dConcatBatch, b);
                TensorBase inputGradSlice = TensorUtilitiesSimd.GetLayer(inputGradientBatch, b);

                ComputeBackwardSequenceInternal(dConcatSlice, inputGradSlice, workspace);
            }

            workspace.Release(dConcatBatch);

            return inputGradientBatch;
        }

        private void ComputeBackwardSequenceInternal(TensorBase dConcat, TensorBase targetInputGradient, TensorWorkspace workspace)
        {
            int numHeads = _heads.Length;
            int rows = dConcat.Rows;

            for (int i = 0; i < numHeads; i++)
            {
                int startColumn = i * _headSize;

                // Borrow slice buffer from workspace
                TensorBase headGradient = workspace.Borrow2D(rows, _headSize);
                TensorUtilitiesSimd.CopyBlock(dConcat, startColumn, headGradient, 0, rows, _headSize);

                TensorBase dInputHead = _heads[i].Backward(headGradient, workspace);

                // Accumulate gradients into target Input Gradient through the backend
                workspace.Backend.ElementWiseAddInPlace(targetInputGradient, dInputHead);

                // Release local workspace buffers
                workspace.Release(headGradient);
                workspace.Release(dInputHead);
            }
        }

        public void ZeroGradients()
        {
            for (int i = 0; i < _heads.Length; i++)
            {
                _heads[i].ZeroGradients();
            }

            _outputProjection.ZeroGradients();
        }

        public void Dispose()
        {
            for (int i = 0; i < _heads.Length; i++)
            {
                _heads[i].Dispose();
            }
        }
    }
}