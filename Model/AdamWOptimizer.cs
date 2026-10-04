using System;
using System.Collections.Generic;
using System.IO;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class AdamWOptimizer : IOptimizer
    {
        public float LearningRate { get; set; }
        public float Beta1 { get; set; }
        public float Beta2 { get; set; }
        public float Epsilon { get; set; }
        public float WeightDecay { get; set; }

        public int StepCount => _stepCount;

        private int _stepCount;

        // Keyed by Parameter Name (or ID) to maintain state across checkpoint reloads
        private readonly Dictionary<string, (Tensor FirstMoment, Tensor SecondMoment)> _state = new();

        public AdamWOptimizer(
            float learningRate = 0.001f, 
            float beta1 = 0.9f, 
            float beta2 = 0.999f, 
            float epsilon = 1e-8f, 
            float weightDecay = 0.01f)
        {
            LearningRate = learningRate;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
            WeightDecay = weightDecay;
            _stepCount = 0;
        }

        public void Step(IEnumerable<TrainableParameter> parameters)
        {
            _stepCount++;

            // Compute standard Adam bias correction factors
            float biasCorrection1 = 1.0f - MathF.Pow(Beta1, _stepCount);
            float biasCorrection2 = 1.0f - MathF.Pow(Beta2, _stepCount);
            
            // Effective step size alpha = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
            float alpha = LearningRate * (MathF.Sqrt(biasCorrection2) / biasCorrection1);

            foreach (var p in parameters)
            {
                if (p.Gradient == null || p.Value == null)
                    continue;

                // Use parameter name (or fallback to unique ID) so state survives object reinstantiation
                string paramKey = p.Name ?? p.GetHashCode().ToString();

                if (!_state.TryGetValue(paramKey, out var moments))
                {
                    moments = (
                        new Tensor(p.Value.Shape),
                        new Tensor(p.Value.Shape)
                    );
                    _state[paramKey] = moments;
                }

                UpdateParameterAdamW(
                    p.Value,
                    p.Gradient,
                    moments.FirstMoment,
                    moments.SecondMoment,
                    alpha,
                    ShouldDecayWeight(p));
            }
        }

/// <summary>
        /// Whether a parameter takes decoupled weight decay.
        /// <para>
        /// AdamW as specified decays the MATRIX weights only. Bias vectors and the
        /// LayerNorm gain/bias are excluded: decaying a LayerNorm gain pulls the
        /// normalisation scale toward zero as training proceeds, degrading the model
        /// in a way that is easy to misdiagnose as a learning-rate problem. This is
        /// the usual "do not decay 1-D parameters" convention, matched to the
        /// checkpoint's own naming - biases are always "*.bias", and LayerNorm
        /// registers its gain and shift as rank-1 ".weight"/".beta".
        /// </summary>
        public static bool ShouldDecayWeight(TrainableParameter parameter)
        {
            ArgumentNullException.ThrowIfNull(parameter);

            // LayerNorm gain and shift are rank-1 tensors.
            if (parameter.Value.Rank == 1)
                return false;

            // Bias vectors are held as [1, N] (rank 2), so rank alone will not catch
            // them; the checkpoint name is the reliable signal.
            if (parameter.Name.EndsWith(".bias", StringComparison.Ordinal))
                return false;

            return true;
        }
        private void UpdateParameterAdamW(
            TensorBase weights, 
            TensorBase gradients, 
            TensorBase m, 
            TensorBase v, 
            float alpha,
            bool decay)
        {
            Span<float> wSpan = weights.Data;
            ReadOnlySpan<float> gSpan = gradients.Data;
            Span<float> mSpan = m.Data;
            Span<float> vSpan = v.Data;

            int count = wSpan.Length;
            float lrDecay = LearningRate * WeightDecay;

            for (int i = 0; i < count; i++)
            {
                float g = gSpan[i];
                float w = wSpan[i];

                // 1. Decoupled Weight Decay (w = w - lr * decay * w), applied ONLY to decayable
                // parameters. The canonical AdamW rule (Loshchilov & Hutter, and
                // what GPT-2 / LLaMA / every standard implementation do) is: decay
                // the matrix weights, never the biases or the LayerNorm gain. A
                // decaying LayerNorm gain is not merely untuned, it is harmful - it
                // shrinks the normalisation scale toward zero over training. See
                // ShouldDecayWeight.
                if (decay)
                {
                    w -= lrDecay * w;
                }

                // 2. Update first moment: m = beta1 * m + (1 - beta1) * g
                float mVal = Beta1 * mSpan[i] + (1.0f - Beta1) * g;
                mSpan[i] = mVal;

                // 3. Update second moment: v = beta2 * v + (1 - beta2) * g^2
                float vVal = Beta2 * vSpan[i] + (1.0f - Beta2) * (g * g);
                vSpan[i] = vVal;

                // 4. Parameter update step
                w -= alpha * (mVal / (MathF.Sqrt(vVal) + Epsilon));

                wSpan[i] = w;
            }
        }

        public void SetStepCount(int stepCount)
        {
            _stepCount = stepCount;
        }

        /// <summary>
        /// Snapshot of the first/second moments keyed by parameter name (the same
        /// key the checkpoint's parameter block uses) plus the bias-correction step.
        /// </summary>
        public OptimizerState ExportState()
        {
            var entries = new List<OptimizerParamState>(_state.Count);

            foreach (var kvp in _state)
            {
                entries.Add(new OptimizerParamState
                {
                    Name = kvp.Key,
                    FirstMoment = new TensorData
                    {
                        Shape = kvp.Value.FirstMoment.Shape,
                        Data = kvp.Value.FirstMoment.Data
                    },
                    SecondMoment = new TensorData
                    {
                        Shape = kvp.Value.SecondMoment.Shape,
                        Data = kvp.Value.SecondMoment.Data
                    }
                });
            }

            return new OptimizerState
            {
                Kind = OptimizerStateKinds.AdamW,
                StepCount = _stepCount,
                Parameters = entries
            };
        }

        public int ImportState(OptimizerState state, IReadOnlyList<TrainableParameter> parameters)
        {
            ArgumentNullException.ThrowIfNull(state);
            ArgumentNullException.ThrowIfNull(parameters);

            if (state.Kind != OptimizerStateKinds.AdamW)
            {
                throw new InvalidDataException(
                    $"Optimizer state was produced by optimizer kind {state.Kind}, " +
                    $"but this is an AdamW optimizer (kind {OptimizerStateKinds.AdamW}).");
            }

            // Expected shape per parameter name, so a corrupt or stale snapshot
            // cannot install a moment buffer that the update loop would index past.
            var expectedShapes = new Dictionary<string, int[]>(parameters.Count);
            foreach (var p in parameters)
                expectedShapes[p.Name] = p.Value.Shape;

            // Replace any live state rather than merging into it.
            ResetState();

            // Bias correction resumes at the saved step: zeroing it would mis-scale
            // the first post-resume steps even though the schedule continues.
            _stepCount = state.StepCount;

            int skipped = 0;
            foreach (var entry in state.Parameters)
            {
                if (entry.FirstMoment == null || entry.SecondMoment == null ||
                    !expectedShapes.TryGetValue(entry.Name, out int[]? expectedShape) ||
                    !ShapeMatches(expectedShape, entry.FirstMoment.Shape) ||
                    !ShapeMatches(expectedShape, entry.SecondMoment.Shape) ||
                    !entry.FirstMoment.IsValid || !entry.SecondMoment.IsValid)
                {
                    skipped++;
                    continue;
                }

                var firstMoment = new Tensor(entry.FirstMoment.Shape);
                Array.Copy(entry.FirstMoment.Data, firstMoment.Data, firstMoment.Data.Length);

                var secondMoment = new Tensor(entry.SecondMoment.Shape);
                Array.Copy(entry.SecondMoment.Data, secondMoment.Data, secondMoment.Data.Length);

                _state[entry.Name] = (firstMoment, secondMoment);
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
            foreach (var kvp in _state)
            {
                kvp.Value.FirstMoment.Dispose();
                kvp.Value.SecondMoment.Dispose();
            }
            _state.Clear();
            _stepCount = 0;
        }
    }
}