using System;
using System.Collections.Generic;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class EmbeddingLayer : ITrainableLayer, IDisposable
    {
        private readonly int _vocabSize;
        private readonly int _embeddingSize;
        private readonly Tensor _embeddings;
        private readonly Tensor _embeddingGradient;
        
        private readonly TrainableParameter[] _parameters;
        public string Name { get; }
        public IEnumerable<TrainableParameter> Parameters => _parameters;

        private TensorBase? _lastInput;
        private readonly Random _random = new();

        public EmbeddingLayer(int vocabSize, int embeddingSize, string name = "token_embeddings")
        {
            Name = name;
            _vocabSize = vocabSize;
            _embeddingSize = embeddingSize;

            _embeddings = new Tensor(vocabSize, embeddingSize);
            _embeddingGradient = new Tensor(vocabSize, embeddingSize);
            _parameters = new[] 
            { 
                new TrainableParameter($"{Name}.weight", _embeddings, _embeddingGradient) 
            };

            InitEmbeddings();
        }

        private void InitEmbeddings()
        {
            float limit = MathF.Sqrt(6f / (_vocabSize + _embeddingSize));
            Span<float> data = _embeddings.Data.AsSpan();
            
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)_random.NextDouble() * 2f * limit - limit;
            }
        }

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            // Ensure embeddings table hasn't been poisoned by the previous optimizer step
            if (float.IsNaN(_embeddings.Data[0]))
            {
                throw new InvalidOperationException("Embedding weights contain NaN prior to forward pass. Check optimizer step or learning rate.");
            }
            if (input.Rank != 1 && input.Rank != 2)
                throw new ArgumentException("Embedding layer expects a rank 1 or rank 2 tensor.");

            _lastInput = input;

            int batchSize = input.Rank == 1 ? 1 : input.Rows;
            int sequenceLength = input.Rank == 1 ? input.Length : input.Cols;

            // Borrow destination tensor from pooled workspace
            TensorBase output = input.Rank == 1 
                ? workspace.Borrow2D(sequenceLength, _embeddingSize)
                : workspace.Borrow3D(batchSize, sequenceLength, _embeddingSize);

            ReadOnlySpan<float> inputData = input.Data.AsSpan();
            ReadOnlySpan<float> embeddingData = _embeddings.Data.AsSpan();
            Span<float> outputData = output.Data.AsSpan();

            int totalTokens = batchSize * sequenceLength;

            for (int i = 0; i < totalTokens; i++)
            {
                int tokenId = (int)inputData[i];

                if ((uint)tokenId >= (uint)_vocabSize)
                    throw new ArgumentOutOfRangeException(nameof(input), $"Token ID {tokenId} out of bounds (0..{_vocabSize - 1}).");

                // Direct memory copy using Spans
                ReadOnlySpan<float> sourceRow = embeddingData.Slice(tokenId * _embeddingSize, _embeddingSize);
                Span<float> targetRow = outputData.Slice(i * _embeddingSize, _embeddingSize);
                
                sourceRow.CopyTo(targetRow);
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            if (float.IsNaN(_embeddings.Data[0]))
            {
                throw new InvalidOperationException("Embedding weights contain NaN prior to backward pass. Check optimizer step or learning rate.");
            }
            if (_lastInput == null)
                throw new InvalidOperationException("Forward pass must be executed prior to Backward pass.");

            ReadOnlySpan<float> inputData = _lastInput.Data.AsSpan();
            ReadOnlySpan<float> gradData = gradient.Data.AsSpan();
            Span<float> embGradData = _embeddingGradient.Data.AsSpan();

            int totalTokens = _lastInput.Rank == 1 ? _lastInput.Length : (_lastInput.Rows * _lastInput.Cols);

            // Accumulate gradients back to the embedding parameters table
            for (int i = 0; i < totalTokens; i++)
            {
                int tokenId = (int)inputData[i];

                ReadOnlySpan<float> incomingGradRow = gradData.Slice(i * _embeddingSize, _embeddingSize);
                Span<float> targetGradRow = embGradData.Slice(tokenId * _embeddingSize, _embeddingSize);

                // Accumulate gradient row into the embedding parameter gradient table
                for (int e = 0; e < _embeddingSize; e++)
                {
                    targetGradRow[e] += incomingGradRow[e];
                }
            }

            // Borrow dummy input gradient from workspace to match ITrainableLayer interface contract
            TensorBase dummyInputGradient = _lastInput.Rank == 1
                ? workspace.Borrow1D(_lastInput.Length)
                : workspace.Borrow2D(_lastInput.Rows, _lastInput.Cols);

            // Clear buffer in case workspace handed us recycled memory
            workspace.Backend.Fill(dummyInputGradient, 0f);

            return dummyInputGradient;
        }

        public void ClipGradients(float maxNorm = 1.0f)
        {
            Span<float> gradData = _embeddingGradient.Data.AsSpan();
            
            float sumSq = 0f;
            for (int i = 0; i < gradData.Length; i++)
            {
                sumSq += gradData[i] * gradData[i];
            }

            float norm = MathF.Sqrt(sumSq);
            if (norm > maxNorm)
            {
                float scale = maxNorm / (norm + 1e-6f);
                for (int i = 0; i < gradData.Length; i++)
                {
                    gradData[i] *= scale;
                }
            }
        }        

        public void ZeroGradients()
        {
            Array.Fill(_embeddingGradient.Data, 0f);
        }

        public void Dispose()
        {
            //Clean up anything used here
            _embeddings.Dispose();
            _embeddingGradient.Dispose();
        }
    }
}