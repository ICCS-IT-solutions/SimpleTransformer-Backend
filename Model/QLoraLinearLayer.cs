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
    public class QLoraLinearLayer : ILinearLayer
    {
        //Likely not used in QLoRA or LoRA, but here due to the interface requirement.
        public Tensor Weights
        {
            get
            {
                Tensor dequantized = new Tensor(_outputSize, _inputSize);
                DequantizeBaseWeights(dequantized);
                return dequantized;
            }
        }
        public string Name { get; }
        private readonly int _inputSize;
        public int InputSize => _inputSize;
        private readonly int _outputSize;
        public int OutputSize => _outputSize;
        private readonly int _rank;
        private readonly float _loraScale; // alpha / rank
        private readonly bool _useBias;
        public bool UseBias => _useBias;

        // Frozen Quantized Base Weight Data
        private readonly byte[] _quantizedWeights4Bit; // Packed 4-bit indices (2 values per byte)
        private readonly float[] _quantizationScales;  // Block-wise scale factors
        private readonly int _blockSize;

        // Trainable LoRA Matrices
        private readonly Tensor _loraA;         // Shape: [rank, inputSize]
        private readonly Tensor _loraAGradient; // Shape: [rank, inputSize]
        private readonly Tensor _loraB;         // Shape: [outputSize, rank]
        private readonly Tensor _loraBGradient; // Shape: [outputSize, rank]

        private readonly Tensor? _bias;
        public Tensor? Bias => _bias;
        private readonly Tensor? _biasGradient;

        public IEnumerable<TrainableParameter> Parameters => _parameters;
        private readonly TrainableParameter[] _parameters;

        private TensorBase? _lastInput;
        private TensorBase? _lastLoraAOutput; // Cache X * A^T to eliminate recomputation in backward

        // Frozen base weights, dequantized once and kept for the model's lifetime
        // (they never change). Previously every forward pass unpacked the 4-bit
        // weights into workspace scratch and re-uploaded the result per matmul.
        private Tensor? _dequantizedBaseWeights;
        private IAccelerationBackend? _baseWeightBackend;
        private bool _residentRegistrationAttempted;
        private readonly object _baseWeightGate = new();

        // ThreadLocal storage for multi-threaded parallel reductions across 3D batch slices
        private readonly ThreadLocal<Tensor> _threadLocalDA;
        private readonly ThreadLocal<Tensor> _threadLocalDB;
        private readonly ThreadLocal<Tensor?> _threadLocalDBias;

        public QLoraLinearLayer(
            int inputSize, 
            int outputSize, 
            int rank = 8, 
            float loraAlpha = 16.0f, 
            bool useBias = true, 
            int blockSize = 64,
            string name = "qlora_linear")
        {
            Name = name;
            _inputSize = inputSize;
            _outputSize = outputSize;
            _rank = rank;
            _loraScale = loraAlpha / rank;
            _useBias = useBias;
            _blockSize = blockSize;

            // 1. Storage for Frozen Quantized Base Weights W0 [outputSize, inputSize]
            int totalElements = outputSize * inputSize;
            _quantizedWeights4Bit = new byte[(totalElements + 1) / 2];
            _quantizationScales = new float[(totalElements + blockSize - 1) / blockSize];

            // 2. Allocate Trainable LoRA Matrices
            _loraA = new Tensor(_rank, _inputSize);
            _loraAGradient = new Tensor(_rank, _inputSize);

            _loraB = new Tensor(_outputSize, _rank);
            _loraBGradient = new Tensor(_outputSize, _rank);

            if (_useBias)
            {
                _bias = new Tensor(1, _outputSize);
                _biasGradient = new Tensor(1, _outputSize);
            }

            // Register ONLY LoRA parameters (& optional bias) for optimization
            var paramList = new List<TrainableParameter>
            {
                new TrainableParameter($"{Name}.lora_A", _loraA, _loraAGradient),
                new TrainableParameter($"{Name}.lora_B", _loraB, _loraBGradient)
            };

            if (_useBias)
            {
                paramList.Add(new TrainableParameter($"{Name}.bias", _bias!, _biasGradient!));
            }
            _parameters = paramList.ToArray();

            // 3. Thread-local scratchpad buffers for multi-threaded batch backprop
            _threadLocalDA = new ThreadLocal<Tensor>(() => new Tensor(_rank, _inputSize), trackAllValues: true);
            _threadLocalDB = new ThreadLocal<Tensor>(() => new Tensor(_outputSize, _rank), trackAllValues: true);
            _threadLocalDBias = new ThreadLocal<Tensor?>(() => _useBias ? new Tensor(1, _outputSize) : null, trackAllValues: true);

            InitLoraWeights();
        }

        private void InitLoraWeights()
        {
            // Standard LoRA Init: A is Gaussian/Uniform random, B is initialized to exact zero
            float limit = MathF.Sqrt(6.0f / (_inputSize + _rank));
            Random random = new Random();
            TensorUtilitiesSimd.FillRandom(_loraA, random, -limit, limit);
            Array.Fill(_loraB.Data, 0.0f);

            if (_useBias)
            {
                Array.Clear(_bias!.Data, 0, _bias.Data.Length);
            }
        }

        #region Forward Pass

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, workspace),
                3 => ForwardBatch(input, workspace),
                _ => throw new ArgumentException("QLoraLinearLayer expects rank 2 or rank 3 inputs.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Cols != _inputSize)
                throw new ArgumentException($"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;
            int rows = input.Rows;

            // Borrow buffers from TensorWorkspace
            TensorBase output = workspace.Borrow(rows, _outputSize, shape => new Tensor(shape[0], shape[1]));
            TensorBase loraAOut = workspace.Borrow(rows, _rank, shape => new Tensor(shape[0], shape[1]));
            _lastLoraAOutput = loraAOut;

            // Step 1: frozen base weights, dequantized once and VRAM-resident
            TensorBase dequantW = GetDequantizedBaseWeights(workspace);

            // Step 2: Frozen Base Pass -> Output = Input * W0^T
            workspace.Backend.MatMul(input, dequantW, output, transposeB: true);

            // Step 3: LoRA Branch -> loraAOut = Input * A^T
            workspace.Backend.MatMul(input, _loraA, loraAOut, transposeB: true);

            // Temp buffer for B^T multiplication: loraBOut = loraAOut * B^T
            TensorBase loraBOut = workspace.Borrow(rows, _outputSize, shape => new Tensor(shape[0], shape[1]));
            workspace.Backend.MatMul(loraAOut, _loraB, loraBOut, transposeB: true);

            // Step 4: Scale and accumulate LoRA branch output into base output
            ScaleAndAccumulate(loraBOut, output, _loraScale);

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
            int rows = input.Rows;

            TensorBase output = workspace.Borrow(layers, rows, _outputSize, shape => new Tensor(shape[0], shape[1], shape[2]));
            TensorBase loraAOut = workspace.Borrow(layers, rows, _rank, shape => new Tensor(shape[0], shape[1], shape[2]));
            _lastLoraAOutput = loraAOut;

            // Frozen base weights: dequantized once, shared by all threads
            TensorBase dequantW = GetDequantizedBaseWeights(workspace);

            Parallel.For(0, layers, b =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, b);
                TensorBase loraAOutSlice = TensorUtilitiesSimd.GetLayer(loraAOut, b);

                // Base pass
                workspace.Backend.MatMul(inputSlice, dequantW, outputSlice, transposeB: true);

                // LoRA pass
                workspace.Backend.MatMul(inputSlice, _loraA, loraAOutSlice, transposeB: true);

                // Slice scratch buffer for LoRA B product
                Tensor loraBOutSlice = new Tensor(rows, _outputSize);
                workspace.Backend.MatMul(loraAOutSlice, _loraB, loraBOutSlice, transposeB: true);

                ScaleAndAccumulate(loraBOutSlice, outputSlice, _loraScale);

                if (_useBias)
                {
                    AddBiasInPlace(outputSlice);
                }
            });

            return output;
        }

        #endregion

        #region Backward Pass

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient, workspace),
                3 => BackwardBatch(gradient, workspace),
                _ => throw new ArgumentException("QLoraLinearLayer expects rank 2 or rank 3 gradients.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient, TensorWorkspace workspace)
        {
            if (_lastInput == null || _lastLoraAOutput == null)
                throw new InvalidOperationException("Backward pass executed before Forward pass.");

            TensorBase input = _lastInput;
            TensorBase loraAOut = _lastLoraAOutput;
            int rows = gradient.Rows;

            // Borrow temporary workspace buffers
            TensorBase inputGradient = workspace.Borrow(rows, _inputSize, shape => new Tensor(shape[0], shape[1]));
            TensorBase dequantW = GetDequantizedBaseWeights(workspace);
            TensorBase dLoraAOut = workspace.Borrow(rows, _rank, shape => new Tensor(shape[0], shape[1]));
            TensorBase dInputAdapter = workspace.Borrow(rows, _inputSize, shape => new Tensor(shape[0], shape[1]));

            // 1. BASE PATH: dX_base = Gradient * W0
            workspace.Backend.MatMul(gradient, dequantW, inputGradient);
            // 2. LORA ADAPTER GRADIENTS:
            // dB += scale * (Gradient^T * loraAOut)
            TensorBase dBAccumulator = workspace.Borrow(_outputSize, _rank, shape => new Tensor(shape[0], shape[1]));
            workspace.Backend.MatMul(gradient, loraAOut, dBAccumulator, transposeA: true);
            ScaleAndAccumulate(dBAccumulator, _loraBGradient, _loraScale);

            // dLoraAOut = Gradient * B
            workspace.Backend.MatMul(gradient, _loraB, dLoraAOut);

            // dA += scale * (dLoraAOut^T * Input)
            TensorBase dAAccumulator = workspace.Borrow(_rank, _inputSize, shape => new Tensor(shape[0], shape[1]));
            workspace.Backend.MatMul(dLoraAOut, input, dAAccumulator, transposeA: true);
            ScaleAndAccumulate(dAAccumulator, _loraAGradient, _loraScale);

            // dX_adapter = scale * (dLoraAOut * A)
            workspace.Backend.MatMul(dLoraAOut, _loraA, dInputAdapter);

            // Total Input Gradient: dX = dX_base + scale * dX_adapter
            ScaleAndAccumulate(dInputAdapter, inputGradient, _loraScale);

            // 3. BIAS GRADIENT
            if (_useBias)
            {
                AccumulateBiasGradient(gradient, _biasGradient!);
            }

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient, TensorWorkspace workspace)
        {
            if (_lastInput == null || _lastLoraAOutput == null)
                throw new InvalidOperationException("Backward pass executed before Forward pass.");

            TensorBase input = _lastInput;
            TensorBase loraAOut = _lastLoraAOutput;
            int layers = gradient.Layers;
            int rows = gradient.Rows;

            TensorBase inputGradient = workspace.Borrow(layers, rows, _inputSize, shape => new Tensor(shape[0], shape[1], shape[2]));
            TensorBase dequantW = GetDequantizedBaseWeights(workspace);

            // Reset thread-local gradient buffers
            foreach (var localDA in _threadLocalDA.Values) workspace.Backend.Fill(localDA, 0f);
            foreach (var localDB in _threadLocalDB.Values) workspace.Backend.Fill(localDB, 0f);
            if (_useBias)
            {
                foreach (var localDBias in _threadLocalDBias.Values)
                {
                    if (localDBias != null) workspace.Backend.Fill(localDBias, 0f);
                }
            }

            Parallel.For(0, layers, b =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase loraAOutSlice = TensorUtilitiesSimd.GetLayer(loraAOut, b);
                TensorBase dInputSlice = TensorUtilitiesSimd.GetLayer(inputGradient, b);

                Tensor localDA = _threadLocalDA.Value!;
                Tensor localDB = _threadLocalDB.Value!;

                // 1. Base Gradient: dX_base = Gradient * W0
                workspace.Backend.MatMul(gradSlice, dequantW, dInputSlice);

                // 2. LoRA B Gradient: localDB += Grad^T * loraAOut
                workspace.Backend.MatMulAccumulate(gradSlice, loraAOutSlice, localDB, transposeA: true);

                // 3. dLoraAOut = Grad * B
                Tensor dLoraAOutSlice = new Tensor(rows, _rank);
                workspace.Backend.MatMul(gradSlice, _loraB, dLoraAOutSlice);

                // 4. LoRA A Gradient: localDA += dLoraAOut^T * Input
                workspace.Backend.MatMulAccumulate(dLoraAOutSlice, inputSlice, localDA, transposeA: true);

                // 5. Adapter Input Gradient: dX_adapter = dLoraAOut * A
                Tensor dInputAdapterSlice = new Tensor(rows, _inputSize);
                workspace.Backend.MatMul(dLoraAOutSlice, _loraA, dInputAdapterSlice);

                // Accumulate adapter contribution into dInput
                ScaleAndAccumulate(dInputAdapterSlice, dInputSlice, _loraScale);

                if (_useBias)
                {
                    AccumulateBiasGradient(gradSlice, _threadLocalDBias.Value!);
                }
            });

            // Reduce thread-local gradients into global gradients with LoRA scale factor
            foreach (var localDA in _threadLocalDA.Values)
            {
                ScaleAndAccumulate(localDA, _loraAGradient, _loraScale);
            }
            foreach (var localDB in _threadLocalDB.Values)
            {
                ScaleAndAccumulate(localDB, _loraBGradient, _loraScale);
            }
            if (_useBias)
            {
                foreach (var localDBias in _threadLocalDBias.Values)
                {
                    if (localDBias != null)
                        workspace.Backend.ElementWiseAddInPlace(_biasGradient!, localDBias);
                }
            }

            return inputGradient;
        }

        #endregion

        public void ZeroGradients()
        {
            Array.Fill(_loraAGradient.Data, 0f);
            Array.Fill(_loraBGradient.Data, 0f);
            if (_useBias)
            {
                Array.Fill(_biasGradient!.Data, 0f);
            }
        }

        #region SIMD & Quantization Helpers

        /// <summary>
        /// The frozen base weights W0, dequantized exactly once and reused by every
        /// forward and backward pass. Unpacking the 4-bit weights into workspace
        /// scratch on each call (four times per layer per step: forward sequence,
        /// forward batch, backward sequence, backward batch) plus re-uploading the
        /// result for every matmul was the dominant cost of a training step.
        /// The tensor is also registered with the backend as device-resident, so
        /// the GPU keeps one VRAM copy and never receives it again.
        /// </summary>
        private Tensor GetDequantizedBaseWeights(TensorWorkspace workspace)
        {
            lock (_baseWeightGate)
            {
                if (_dequantizedBaseWeights == null)
                {
                    var weights = new Tensor(_outputSize, _inputSize);
                    DequantizeBaseWeights(weights);
                    _dequantizedBaseWeights = weights;
                }

                //Backends that cannot cache (CPU) return false and the tensor is
                //simply used as-is.
                if (!_residentRegistrationAttempted)
                {
                    _residentRegistrationAttempted = true;
                    _baseWeightBackend = workspace.Backend;
                    workspace.Backend.TryRegisterResidentWeights(
                        _dequantizedBaseWeights, $"{Name}.base");
                }

                return _dequantizedBaseWeights;
            }
        }

        private void DequantizeBaseWeights(TensorBase targetDequantized)
        {
            // Simple block-wise 4-bit dequantization mapping 0..15 nibbles back to float
            Span<float> outputSpan = targetDequantized.Data.AsSpan(targetDequantized.Offset, _outputSize * _inputSize);

            for (int i = 0; i < outputSpan.Length; i++)
            {
                int byteIdx = i >> 1;
                byte packed = _quantizedWeights4Bit[byteIdx];
                int nibble = (i % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

                float scale = _quantizationScales[i / _blockSize];
                // Map [0..15] -> centered [-8..7] range scaled by block scale factor
                outputSpan[i] = (nibble - 8) * scale;
            }
        }

        private static void ScaleAndAccumulate(TensorBase source, TensorBase destination, float scale)
        {
            Span<float> src = source.Data.AsSpan(source.Offset, source.Rows * source.Cols);
            Span<float> dst = destination.Data.AsSpan(destination.Offset, destination.Rows * destination.Cols);

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] += src[i] * scale;
            }
        }

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

                // Pure managed broadcast add (bias-sized rows are memory-bound)
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
            _loraAGradient.Dispose();
            _loraBGradient.Dispose();
            _biasGradient?.Dispose();

            //Hand the frozen base weights' VRAM copy back before dropping the host
            //tensor it was uploaded from, so unloading a model frees VRAM now
            //rather than at backend shutdown.
            if (_dequantizedBaseWeights != null)
                _baseWeightBackend?.ReleaseResidentWeights(_dequantizedBaseWeights);

            _dequantizedBaseWeights?.Dispose();
        }

        #endregion
    }
}