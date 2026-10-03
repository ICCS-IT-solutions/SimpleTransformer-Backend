using System;
using System.Collections.Generic;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class MultiHeadAttention : ITrainableLayer, IDropoutControl
    {
        public string Name { get; }
        private readonly AttentionHead[] _heads;
        private readonly ILinearLayer _outputProjection;

        /// <summary>Checkpoint/determinism tests: reach the out-projection.</summary>
        internal ILinearLayer OutputProjectionForTest => _outputProjection;

        /// <summary>
        /// Determinism diagnostics: the concatenation buffers created by the
        /// forwards on this thread, so the workspace can report if anything
        /// releases one (a use-after-release would let a later borrow overwrite
        /// Backward's dW operand).
        /// </summary>
        internal static readonly System.Collections.Generic.List<TensorBase> WatchedConcatsForTest = new();

        /// <summary>Determinism diagnostics: enable concat use-after-release watching.</summary>
        internal static bool WatchEnabledForTest;
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
            WatchedConcatsForTest.Add(concatenated);

            ComputeForwardSequenceInternal(input, mask, concatenated, workspace);

            TensorBase output = _outputProjection.Forward(concatenated, workspace);

            // concatenated must stay borrowed: it is _outputProjection's
            // _lastInput, which Backward reads for dW. Releasing it here let
            // later same-shape borrows overwrite it behind the cache's back
            // (use-after-release = nondeterministic gradients). Reclaimed by
            // the step's workspace Reset.
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
                // A causal mask is position-only and identical for every batch
                // item, so a rank-2 mask is passed straight through. Only a
                // genuinely per-item mask (e.g. a padding mask expanded to rank 3)
                // needs slicing.
                TensorBase? maskSlice = mask switch
                {
                    null => null,
                    var m when m.Rank == 2 => m,
                    var m => TensorUtilitiesSimd.GetLayer(m, b)
                };
                TensorBase concatSlice = TensorUtilitiesSimd.GetLayer(concatenatedBatch, b);

                ComputeForwardSequenceInternal(inputSlice, maskSlice, concatSlice, workspace);
            }

            TensorBase output = _outputProjection.Forward(concatenatedBatch, workspace);

            // concatenatedBatch stays borrowed for _outputProjection's
            // _lastInput cache (Backward reads it); see ForwardSequence.
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

        // Push dropout control down to every head's attention module.
        public void SetDropoutEnabled(bool enabled)
        {
            for (int i = 0; i < _heads.Length; i++)
            {
                _heads[i].SetDropoutEnabled(enabled);
            }
        }

        public void SetDropoutRate(float rate)
        {
            for (int i = 0; i < _heads.Length; i++)
            {
                _heads[i].SetDropoutRate(rate);
            }
        }

        public void PrepareDropoutForStep(long step)
        {
            for (int i = 0; i < _heads.Length; i++)
            {
                _heads[i].PrepareDropoutForStep(step);
            }
        }

        public void CollectDropoutSites(List<DropoutSite> sites)
        {
            for (int i = 0; i < _heads.Length; i++)
            {
                _heads[i].CollectDropoutSites(sites);
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