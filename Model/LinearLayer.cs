using Serilog;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class LinearLayer : ILinearLayer
    {
        public string Name { get; }
        private readonly int _inputSize;
        public int InputSize => _inputSize;
        private readonly int _outputSize;
        public int OutputSize => _outputSize;
        private readonly bool _useBias;
        public bool UseBias => _useBias;
        private readonly TrainableParameter[] _parameters;
        public IEnumerable<TrainableParameter> Parameters => _parameters;

        private readonly Tensor _weights;
        private readonly Tensor _weightGradient;
        private readonly Tensor? _bias;
        private readonly Tensor? _biasGradient;

        public Tensor Weights => _weights;
        public Tensor? Bias => _bias;

        private TensorBase? _lastInput;

        // VRAM residency (raw training): the weight matrix is a MatMul operand on
        // every forward/backward call, so the Vulkan backend can keep one device
        // copy instead of re-uploading it per call. Registration is attempted once
        // on first use (CPU backends return false and change nothing); the device
        // copy is refreshed after each optimizer step and released on dispose.
        private bool _residentRegistrationAttempted;
        private IAccelerationBackend? _residentBackend;

        // ThreadLocal storage to reuse dW and dB buffers across Parallel.For execution without allocations
        private readonly ThreadLocal<Tensor> _threadLocalDW;
        private readonly ThreadLocal<Tensor?> _threadLocalDB;

        public LinearLayer(int inputSize, int outputSize, bool useBias = true, string name = "linear")
        {
            Name = name;
            _inputSize = inputSize;
            _outputSize = outputSize;
            _useBias = useBias;

            // Conventional shapes: Weights are [outputSize, inputSize]
            _weights = new Tensor(outputSize, inputSize);
            _weightGradient = new Tensor(outputSize, inputSize);

            if (_useBias)
            {
                _bias = new Tensor(1, outputSize);
                _biasGradient = new Tensor(1, outputSize);
            }

            // Standard naming convention: .weight for matrix weights, .bias for bias vector
            _parameters = _useBias
                ? new[]
                {
                    new TrainableParameter($"{Name}.weight", _weights, _weightGradient),
                    new TrainableParameter($"{Name}.bias", _bias!, _biasGradient!)
                }
                : new[]
                {
                    new TrainableParameter($"{Name}.weight", _weights, _weightGradient)
                };

            // Instantiate ThreadLocal scratch buffers for thread safety without dynamic heap allocations
            _threadLocalDW = new ThreadLocal<Tensor>(() => new Tensor(_outputSize, _inputSize), trackAllValues: true);
            _threadLocalDB = new ThreadLocal<Tensor?>(() => _useBias ? new Tensor(1, _outputSize) : null, trackAllValues: true);

            InitWeights();
        }

        private readonly Random _random = new();

        private void InitWeights()
        {
            float limit = MathF.Sqrt(6.0f / (_inputSize + _outputSize));

            TensorUtilitiesSimd.FillRandom(
                _weights,
                _random,
                -limit,
                limit);

            if (_useBias)
            {
                Array.Clear(_bias!.Data, 0, _bias.Data.Length);
            }
        }

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            EnsureResidentWeights(workspace.Backend);

            return input.Rank switch
            {
                2 => ForwardSequence(input, workspace),
                3 => ForwardBatch(input, workspace),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }

        /// <summary>
        /// One-time promotion of the weight matrix into the backend's device
        /// memory (Vulkan only). Subsequent MatMul calls bind the VRAM copy
        /// directly; if the backend declines (CPU, over budget) the ordinary
        /// per-call path keeps working unchanged.
        /// </summary>
        private void EnsureResidentWeights(IAccelerationBackend backend)
        {
            if (_residentRegistrationAttempted)
                return;

            _residentRegistrationAttempted = true;
            _residentBackend = backend;
            backend.TryRegisterResidentWeights(_weights, $"{Name}.weight");
        }

        private TensorBase ForwardSequence(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Cols != _inputSize)
                throw new ArgumentException($"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;

            // Borrow output buffer from workspace instead of 'new Tensor(...)'
            TensorBase output = workspace.Borrow(
                input.Rows, _outputSize, 
                shape => new Tensor(shape[0], shape[1])
            );

            workspace.Backend.MatMul(input, _weights, output, transposeB: true);

            if (_useBias)
            {
                AddBiasInPlace(output);
            }

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Cols != _inputSize)
                throw new ArgumentException($"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;

            int layers = input.Layers;

            // Borrow 3D tensor buffer [layers, rows, outputSize]
            TensorBase output = workspace.Borrow(
                layers, input.Rows, _outputSize, 
                shape => new Tensor(shape[0], shape[1], shape[2])
            );

            Parallel.For(0, layers, b =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, b);

                workspace.Backend.MatMul(inputSlice, _weights, outputSlice, transposeB: true);

                if (_useBias)
                {
                    AddBiasInPlace(outputSlice);
                }
            });

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient, workspace),
                3 => BackwardBatch(gradient, workspace),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient, TensorWorkspace workspace)
        {
            if (_lastInput == null) 
                throw new InvalidOperationException("Last input is null.");

            TensorBase input = _lastInput;

            // 1. dW = G^T * X
            workspace.Backend.MatMul(gradient, input, _weightGradient, transposeA: true);

            // 2. dBias = sum(G, axis=0)
            if (_useBias)
            {
                AccumulateBiasGradient(gradient, _biasGradient!);
            }

            // 3. dX = G * W (Borrowed from workspace instead of 'new Tensor(...)')
            TensorBase inputGradient = workspace.Borrow(
                input.Rows, input.Cols, 
                shape => new Tensor(shape[0], shape[1])
            );

            workspace.Backend.MatMul(gradient, _weights, inputGradient);

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            if (_lastInput == null)
                throw new InvalidOperationException("Last input is null.");

            TensorBase input = _lastInput;
            int layers = gradient.Layers;

            // Borrow dX buffer [layers, rows, inputSize]
            TensorBase inputGradient = workspace.Borrow(
                layers, gradient.Rows, _inputSize, 
                shape => new Tensor(shape[0], shape[1], shape[2])
            );

            // 1. Clear thread-local gradient buffers across participating threads
            foreach (var localBuffer in _threadLocalDW.Values)
            {
                workspace.Backend.Fill(localBuffer, 0f);
            }
            if (_useBias)
            {
                foreach (var localBuffer in _threadLocalDB.Values)
                {
                    if (localBuffer != null)
                        workspace.Backend.Fill(localBuffer, 0f);
                }
            }

            // 2. Compute slices in parallel
            Parallel.For(0, layers, b =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase dInputSlice = TensorUtilitiesSimd.GetLayer(inputGradient, b);

                Tensor localDW = _threadLocalDW.Value!;

                workspace.Backend.MatMulAccumulate(gradSlice, inputSlice, localDW, transposeA: true);
                workspace.Backend.MatMul(gradSlice, _weights, dInputSlice);

                if (_useBias)
                {
                    Tensor localDB = _threadLocalDB.Value!;
                    AccumulateBiasGradient(gradSlice, localDB);
                }
            });

            // 3. Reduce thread-local gradients into main weight gradient
            foreach (var localDW in _threadLocalDW.Values)
            {
                workspace.Backend.ElementWiseAddInPlace(_weightGradient, localDW);
            }

            if (_useBias)
            {
                foreach (var localDB in _threadLocalDB.Values)
                {
                    if (localDB != null)
                        workspace.Backend.ElementWiseAddInPlace(_biasGradient!, localDB);
                }
            }

            return inputGradient;
        }

        public void ZeroGradients()
        {
            // Zero weight gradient (pure managed - gradients are plain owned buffers)
            Array.Fill(_weightGradient.Data, 0f);
            if (_useBias)
            {
                Array.Fill(_biasGradient!.Data, 0f);
            }
        }

        #region Helper Methods (Optimized Memory Spans)

        private void AddBiasInPlace(TensorBase target)
        {
            int rows = target.Rows;
            int cols = target.Cols;
            ReadOnlySpan<float> biasSpan = _bias!.Data.AsSpan(0, _outputSize);
            Span<float> targetSpan = target.Data.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = target.Offset + (r * target.Stride);
                Span<float> rowSpan = targetSpan.Slice(rowOffset, cols);

                // Pure managed broadcast add (bias-sized rows are memory-bound, SIMD adds little here)
                for (int j = 0; j < cols; j++)
                {
                    rowSpan[j] += biasSpan[j];
                }
            }
        }

        private void AccumulateBiasGradient(TensorBase gradient, Tensor targetBiasGrad)
        {
            int rows = gradient.Rows;
            int cols = gradient.Cols;
            ReadOnlySpan<float> gradData = gradient.Data.AsSpan();
            Span<float> biasGradSpan = targetBiasGrad.Data.AsSpan(targetBiasGrad.Offset, cols);

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = gradient.Offset + (r * gradient.Stride);
                ReadOnlySpan<float> rowSpan = gradData.Slice(rowOffset, cols);

                // Pure managed row-sum accumulation
                for (int j = 0; j < cols; j++)
                {
                    biasGradSpan[j] += rowSpan[j];
                }
            }
        }
        public void Dispose()
        {
            //Give the device-local weight copy back to the driver before the
            //managed tensors go away (no-op for CPU backends / unregistered).
            _residentBackend?.ReleaseResidentWeights(_weights);

            _weightGradient.Dispose();
            _biasGradient?.Dispose();
        }

        #endregion
    }
}