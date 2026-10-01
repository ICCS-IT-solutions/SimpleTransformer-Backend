using System;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Per-step learning rate policy: linear warmup from 0 to the peak rate,
    /// then cosine decay to a floor. Pure function of the global step index,
    /// so pause/resume reproduces the identical curve as long as the step
    /// counter is checkpointed alongside the weights.
    /// </summary>
    public sealed class LRScheduler
    {
        public float PeakLearningRate { get; }
        public float MinLearningRate { get; }
        public int WarmupSteps { get; }
        public int TotalSteps { get; }

        public LRScheduler(
            float peakLearningRate,
            float minLearningRate,
            int warmupSteps,
            int totalSteps)
        {
            if (peakLearningRate <= 0f)
                throw new ArgumentOutOfRangeException(nameof(peakLearningRate), "Peak learning rate must be greater than 0.");
            if (minLearningRate < 0f)
                throw new ArgumentOutOfRangeException(nameof(minLearningRate), "Min learning rate must be non-negative.");
            if (minLearningRate > peakLearningRate)
                throw new ArgumentOutOfRangeException(nameof(minLearningRate), "Min learning rate must not exceed the peak learning rate.");
            if (warmupSteps < 0)
                throw new ArgumentOutOfRangeException(nameof(warmupSteps), "Warmup steps must be non-negative.");
            if (totalSteps <= 0)
                throw new ArgumentOutOfRangeException(nameof(totalSteps), "Total steps must be positive.");
            if (warmupSteps >= totalSteps)
                throw new ArgumentOutOfRangeException(nameof(warmupSteps), "Warmup steps must be fewer than total steps.");

            PeakLearningRate = peakLearningRate;
            MinLearningRate = minLearningRate;
            WarmupSteps = warmupSteps;
            TotalSteps = totalSteps;
        }

        /// <summary>
        /// Learning rate for a 0-based global step. Past the horizon the
        /// floor is returned (never extrapolates upward).
        /// </summary>
        public float GetLR(int step)
        {
            if (step < 0)
                throw new ArgumentOutOfRangeException(nameof(step), "Step must be non-negative.");

            // Phase 1: linear warmup 0 -> peak. Step 0 yields 0 so the very
            // first update on fresh (or freshly resumed) weights is a no-op.
            if (step < WarmupSteps)
            {
                if (WarmupSteps == 0)
                    return PeakLearningRate;
                return PeakLearningRate * ((float)(step + 1) / WarmupSteps);
            }

            // Phase 2: cosine decay peak -> floor over the remaining horizon.
            int decaySteps = TotalSteps - WarmupSteps;
            int decayStep = Math.Min(step - WarmupSteps, decaySteps);
            double progress = (double)decayStep / decaySteps; // 0..1
            double cosine = 0.5 * (1.0 + Math.Cos(Math.PI * progress));
            return (float)(MinLearningRate + (PeakLearningRate - MinLearningRate) * cosine);
        }
    }
}
