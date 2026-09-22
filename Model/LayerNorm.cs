using System;
using System.Buffers;
using System.Collections.Generic;
using System.Threading.Tasks;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class LayerNorm : ITrainableLayer
    {
        public string Name { get; }
        private readonly int _embeddingSize;
        private readonly float _epsilon;

        private float[]? _lastInvStdBuffer;
        private int _lastInvStdCount;
        private TensorBase? _lastNormalized;

        private readonly Tensor _gamma;
        private readonly Tensor _beta;
        private readonly Tensor _gradientGamma;
        private readonly Tensor _gradientBeta;

        public Tensor Gamma => _gamma;
        public Tensor Beta => _beta;
        public Tensor GradientGamma => _gradientGamma;
        public Tensor GradientBeta => _gradientBeta;

        public IEnumerable<TrainableParameter> Parameters => _parameters;
        private readonly TrainableParameter[] _parameters;

        private TensorBase? _lastInput;

        public LayerNorm(int embeddingSize, float epsilon = 1e-5f, string name = "layer_norm")
        {
            Name = name;
            _embeddingSize = embeddingSize;
            _epsilon = epsilon;

            _gamma = new Tensor(embeddingSize);
            _beta = new Tensor(embeddingSize);
            _gradientGamma = new Tensor(embeddingSize);
            _gradientBeta = new Tensor(embeddingSize);

            // Standard PyTorch/HuggingFace naming: .weight for scale (gamma), .bias for shift (beta)
            _parameters = new[]
            {
                new TrainableParameter($"{Name}.weight", _gamma, _gradientGamma),
                new TrainableParameter($"{Name}.bias", _beta, _gradientBeta)
            };

            InitParameters();
        }

        private void InitParameters()
        {
            // Pure managed init - these are owned, contiguous parameter buffers
            Array.Fill(_gamma.Data, 1.0f);
            Array.Fill(_beta.Data, 0.0f);
            Array.Fill(_gradientGamma.Data, 0.0f);
            Array.Fill(_gradientBeta.Data, 0.0f);
        }

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, workspace),
                3 => ForwardBatch(input, workspace),
                _ => throw new ArgumentException("LayerNorm expects Rank 2 or Rank 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException($"Expected {_embeddingSize} columns, got {input.Cols}.");

            _lastInput = input;
            EnsureCacheCapacity(totalRows: input.Rows, cols: input.Cols, workspace);

            TensorBase output = workspace.BorrowLike(input);
            ComputeLayerNormForward(input, output, startCacheRow: 0);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException($"Expected {_embeddingSize} columns, got {input.Cols}.");

            _lastInput = input;
            int totalRows = input.Layers * input.Rows;
            EnsureCacheCapacity(totalRows: totalRows, cols: input.Cols, workspace);

            TensorBase output = workspace.BorrowLike(input);
            int layers = input.Layers;

            Parallel.For(0, layers, l =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, l);
                TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, l);
                int cacheRowOffset = l * input.Rows;

                ComputeLayerNormForward(inputSlice, outputSlice, cacheRowOffset);
            });

            return output;
        }

        private void ComputeLayerNormForward(TensorBase input, TensorBase output, int startCacheRow)
        {
            int rows = input.Rows;
            int cols = input.Cols;

            ReadOnlySpan<float> inData = input.Data.AsSpan();
            Span<float> outData = output.Data.AsSpan();
            Span<float> normData = _lastNormalized!.Data.AsSpan();
            ReadOnlySpan<float> gammaData = _gamma.Data.AsSpan();
            ReadOnlySpan<float> betaData = _beta.Data.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int inRowOffset = input.Offset + r * input.Stride;
                int outRowOffset = output.Offset + r * output.Stride;
                int normRowOffset = (startCacheRow + r) * cols;

                var (mean, variance) = StatsUtilities.MeanAndVarianceRow(input, r);
                
                float safeVariance = MathF.Max(0f, variance);
                float invStd = 1f / MathF.Sqrt(safeVariance + _epsilon);

                if (float.IsNaN(invStd) || float.IsInfinity(invStd))
                {
                    invStd = 1f; // Fallback sanity cap
                }

                _lastInvStdBuffer![startCacheRow + r] = invStd;

                for (int c = 0; c < cols; c++)
                {
                    float x = inData[inRowOffset + c];
                    float xHat = (x - mean) * invStd;

                    normData[normRowOffset + c] = xHat;
                    outData[outRowOffset + c] = xHat * gammaData[c] + betaData[c];
                }
            }
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient, workspace),
                3 => BackwardBatch(gradient, workspace),
                _ => throw new ArgumentException("LayerNorm expects Rank 2 or Rank 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient, TensorWorkspace workspace)
        {
            ValidateBackwardState(gradient);

            TensorBase inputGradient = workspace.BorrowLike(_lastInput!);
            ComputeLayerNormBackward(gradient, inputGradient, startCacheRow: 0, accumulationLock: null);

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            ValidateBackwardState(gradient);

            int layers = gradient.Layers;
            TensorBase inputGradient = workspace.BorrowLike(_lastInput!);
            object accumulationLock = new object();

            Parallel.For(0, layers, l =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, l);
                TensorBase dInputSlice = TensorUtilitiesSimd.GetLayer(inputGradient, l);
                int cacheRowOffset = l * gradient.Rows;

                ComputeLayerNormBackward(gradSlice, dInputSlice, cacheRowOffset, accumulationLock);
            });

            return inputGradient;
        }

        private void ComputeLayerNormBackward(
            TensorBase gradient, 
            TensorBase inputGradient, 
            int startCacheRow, 
            object? accumulationLock)
        {
            int rows = gradient.Rows;
            int cols = _embeddingSize;

            ReadOnlySpan<float> gradData = gradient.Data.AsSpan();
            Span<float> dInputData = inputGradient.Data.AsSpan();
            ReadOnlySpan<float> normData = _lastNormalized!.Data.AsSpan();
            ReadOnlySpan<float> gammaData = _gamma.Data.AsSpan();

            // Borrow thread-local accumulation buffers from ArrayPool instead of heap allocations
            float[] localDGamma = ArrayPool<float>.Shared.Rent(cols);
            float[] localDBeta = ArrayPool<float>.Shared.Rent(cols);
            Array.Clear(localDGamma, 0, cols);
            Array.Clear(localDBeta, 0, cols);

            try
            {
                for (int r = 0; r < rows; r++)
                {
                    int gradRowOffset = gradient.Offset + r * gradient.Stride;
                    int dInputRowOffset = inputGradient.Offset + r * inputGradient.Stride;
                    int normRowOffset = (startCacheRow + r) * cols;
                    float invStd = _lastInvStdBuffer![startCacheRow + r];

                    float sumDxHat = 0f;
                    float sumDxHatXHat = 0f;

                    // Pass 1: Local reduction sums & parameter accumulation
                    for (int c = 0; c < cols; c++)
                    {
                        float g = gradData[gradRowOffset + c];
                        float xHat = normData[normRowOffset + c];
                        float dxHat = g * gammaData[c];

                        localDBeta[c] += g;
                        localDGamma[c] += g * xHat;

                        sumDxHat += dxHat;
                        sumDxHatXHat += dxHat * xHat;
                    }

                    // Pass 2: Calculate input gradients
                    float invCols = 1.0f / cols;
                    for (int c = 0; c < cols; c++)
                    {
                        float g = gradData[gradRowOffset + c];
                        float xHat = normData[normRowOffset + c];
                        float dxHat = g * gammaData[c];

                        dInputData[dInputRowOffset + c] = invStd * (dxHat - (sumDxHat + xHat * sumDxHatXHat) * invCols);
                    }
                }

                // Thread-safe update to global parameter gradients
                if (accumulationLock != null)
                {
                    lock (accumulationLock)
                    {
                        AccumulateGradients(localDGamma, localDBeta);
                    }
                }
                else
                {
                    AccumulateGradients(localDGamma, localDBeta);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(localDGamma);
                ArrayPool<float>.Shared.Return(localDBeta);
            }
        }

        private void AccumulateGradients(ReadOnlySpan<float> dGamma, ReadOnlySpan<float> dBeta)
        {
            Span<float> gGrad = _gradientGamma.Data.AsSpan();
            Span<float> bGrad = _gradientBeta.Data.AsSpan();

            for (int c = 0; c < _embeddingSize; c++)
            {
                gGrad[c] += dGamma[c];
                bGrad[c] += dBeta[c];
            }
        }

        private void EnsureCacheCapacity(int totalRows, int cols, TensorWorkspace workspace)
        {
            if (_lastInvStdBuffer == null || _lastInvStdCount < totalRows)
            {
                if (_lastInvStdBuffer != null)
                {
                    ArrayPool<float>.Shared.Return(_lastInvStdBuffer);
                }
                _lastInvStdBuffer = ArrayPool<float>.Shared.Rent(totalRows);
                _lastInvStdCount = totalRows;
            }

            // Borrow lastNormalized from workspace pool using ReadOnlySpan<int> shape
            if (_lastNormalized == null || _lastNormalized.Rows != totalRows || _lastNormalized.Cols != cols)
            {
                _lastNormalized = workspace.Borrow2D(totalRows, cols);
            }
            else
            {
                workspace.Backend.Fill(_lastNormalized!, 0f);
            }
        }

        private void ValidateBackwardState(TensorBase gradient)
        {
            if (_lastInput == null || _lastInvStdBuffer == null || _lastNormalized == null)
            {
                throw new InvalidOperationException("Forward must be called before Backward.");
            }

            TensorUtilitiesSimd.ValidateSameShape(gradient, _lastInput);
        }

        public void ZeroGradients()
        {
            Array.Fill(_gradientGamma.Data, 0f);
            Array.Fill(_gradientBeta.Data, 0f);
        }

        /// <summary>
        /// Clears references to cached input/activations and returns rented pooled arrays.
        /// </summary>
        public void ClearState()
        {
            _lastInput = null;
            _lastNormalized = null;

            if (_lastInvStdBuffer != null)
            {
                ArrayPool<float>.Shared.Return(_lastInvStdBuffer);
                _lastInvStdBuffer = null;
                _lastInvStdCount = 0;
            }
        }

        public void Dispose() => ClearState();
    }
}