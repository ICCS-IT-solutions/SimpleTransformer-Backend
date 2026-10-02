using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimpleTransformer.Model
{
    public class SgdOptimizer : IOptimizer
    {
        public float LearningRate { get; set; }
        private readonly float _momentum;
        private readonly float _weightDecay;
        private readonly bool _useNesterov;

        // Tracks the velocity buffer per parameter NAME (not object reference),
        // so the state survives a resume that rebuilds the parameter objects and
        // can be keyed into a checkpoint.
        private readonly Dictionary<string, Tensor> _velocityState = new();

        public SgdOptimizer(
            float learningRate, 
            float momentum = 0.0f, 
            float weightDecay = 0.0f, 
            bool useNesterov = false)
        {
            LearningRate = learningRate;
            _momentum = momentum;
            _weightDecay = weightDecay;
            _useNesterov = useNesterov;
        }

        public void Step(IEnumerable<TrainableParameter> parameters)
        {
            float lr = LearningRate;
            foreach (var param in parameters)
            {
                if (param?.Value?.Data == null || param?.Gradient?.Data == null)
                    continue;

                float[] values = param.Value.Data;
                float[] gradients = param.Gradient.Data;

                // Vanilla SGD path (no momentum buffers needed)
                if (_momentum == 0.0f && _weightDecay == 0.0f)
                {
                    UpdateParametersVanillaSimd(values, gradients, lr);
                    continue;
                }

                // Ensure a velocity buffer exists for this parameter
                string paramKey = param.Name;
                if (!_velocityState.TryGetValue(paramKey, out var velocityTensor))
                {
                    velocityTensor = new Tensor(param.Value.Shape);
                    _velocityState[paramKey] = velocityTensor;
                }

                float[] velocity = velocityTensor.Data;

                UpdateParametersMomentumSimd(
                    values, 
                    gradients, 
                    velocity, 
                    lr, 
                    _momentum, 
                    _weightDecay, 
                    _useNesterov);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UpdateParametersMomentumSimd(
            float[] values, 
            float[] gradients, 
            float[] velocity, 
            float lr, 
            float momentum, 
            float weightDecay, 
            bool nesterov)
        {
            int length = values.Length;
            int i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                int vectorSize = Vector<float>.Count;
                var lrVec = new Vector<float>(lr);
                var momentumVec = new Vector<float>(momentum);
                var decayVec = new Vector<float>(weightDecay);

                int simdBoundary = length - (length % vectorSize);

                for (; i < simdBoundary; i += vectorSize)
                {
                    var w = new Vector<float>(values, i);
                    var g = new Vector<float>(gradients, i);
                    var v = new Vector<float>(velocity, i);

                    // 1. Ingest Weight Decay: g' = g + weightDecay * w
                    if (weightDecay != 0.0f)
                    {
                        g += decayVec * w;
                    }

                    // 2. Velocity Update: v = momentum * v + g'
                    v = (momentumVec * v) + g;
                    v.CopyTo(velocity, i);

                    // 3. Weight Update
                    Vector<float> step;
                    if (nesterov)
                    {
                        // Nesterov step: momentum * v + g'
                        step = (momentumVec * v) + g;
                    }
                    else
                    {
                        // Standard momentum step: v
                        step = v;
                    }

                    var updated = w - (lrVec * step);
                    updated.CopyTo(values, i);
                }
            }

            // Fallback tail loop for non-vectorized elements
            for (; i < length; i++)
            {
                float w = values[i];
                float g = gradients[i];

                if (weightDecay != 0.0f)
                {
                    g += weightDecay * w;
                }

                float v = momentum * velocity[i] + g;
                velocity[i] = v;

                float step = nesterov ? (momentum * v + g) : v;
                values[i] = w - lr * step;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UpdateParametersVanillaSimd(float[] values, float[] gradients, float learningRate)
        {
            int length = values.Length;
            int i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                int vectorSize = Vector<float>.Count;
                var lrVector = new Vector<float>(learningRate);
                int simdBoundary = length - (length % vectorSize);

                for (; i < simdBoundary; i += vectorSize)
                {
                    var v = new Vector<float>(values, i);
                    var g = new Vector<float>(gradients, i);
                    var updated = v - (lrVector * g);
                    updated.CopyTo(values, i);
                }
            }

            for (; i < length; i++)
            {
                values[i] -= learningRate * gradients[i];
            }
        }

        /// <summary>
        /// Snapshot of the momentum velocity buffers keyed by parameter name.
        /// Vanilla SGD (momentum and weight decay both zero) keeps no state, so
        /// the snapshot is empty and the run is already reproducible.
        /// </summary>
        public OptimizerState ExportState()
        {
            var entries = new List<OptimizerParamState>(_velocityState.Count);

            foreach (var kvp in _velocityState)
            {
                entries.Add(new OptimizerParamState
                {
                    Name = kvp.Key,
                    Velocity = new TensorData
                    {
                        Shape = kvp.Value.Shape,
                        Data = kvp.Value.Data
                    }
                });
            }

            return new OptimizerState
            {
                Kind = OptimizerStateKinds.Sgd,
                StepCount = 0,   // SGD has no bias correction
                Parameters = entries
            };
        }

        public int ImportState(OptimizerState state, IReadOnlyList<TrainableParameter> parameters)
        {
            ArgumentNullException.ThrowIfNull(state);
            ArgumentNullException.ThrowIfNull(parameters);

            if (state.Kind != OptimizerStateKinds.Sgd)
            {
                throw new InvalidDataException(
                    $"Optimizer state was produced by optimizer kind {state.Kind}, " +
                    $"but this is an SGD optimizer (kind {OptimizerStateKinds.Sgd}).");
            }

            var expectedShapes = new Dictionary<string, int[]>(parameters.Count);
            foreach (var p in parameters)
                expectedShapes[p.Name] = p.Value.Shape;

            ResetState();

            int skipped = 0;
            foreach (var entry in state.Parameters)
            {
                if (entry.Velocity == null ||
                    !expectedShapes.TryGetValue(entry.Name, out int[]? expectedShape) ||
                    !ShapeMatches(expectedShape, entry.Velocity.Shape) ||
                    !entry.Velocity.IsValid)
                {
                    skipped++;
                    continue;
                }

                var velocity = new Tensor(entry.Velocity.Shape);
                Array.Copy(entry.Velocity.Data, velocity.Data, velocity.Data.Length);
                _velocityState[entry.Name] = velocity;
            }

            return skipped;
        }

        private static bool ShapeMatches(int[] expected, int[] actual)
        {
            if (expected.Length != actual.Length) return false;
            for (int i = 0; i < expected.Length; i++)
            {
                if (expected[i] != actual[i]) return false;
            }
            return true;
        }

        public void ResetState()
        {
            foreach (var kvp in _velocityState)
            {
                kvp.Value.Dispose();
            }
            _velocityState.Clear();
        }
    }
}