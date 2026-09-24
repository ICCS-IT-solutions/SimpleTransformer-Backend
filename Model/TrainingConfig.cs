namespace SimpleTransformer.Model
{
    public enum OptimizerType
    {
        AdamW,
        Sgd
    }

    public class TrainingConfig
    {
        // --- Core Hyperparameters ---
        public float LearningRate { get; set; } = 0.001f;
        public int BatchSize { get; set; } = 8;
        public int Epochs { get; set; } = 10;
        public float DropoutRate { get; set; } = 0.0f;

        /// <summary>
        /// When true, a trailing mini-batch that is smaller than
        /// <see cref="BatchSize"/> is discarded instead of being trained on.
        /// <para>
        /// A short final batch is trained, but it is a poor gradient estimate
        /// measured from far fewer samples than the rest of the epoch, so it adds
        /// noise for no benefit. Dropping it keeps every optimizer step based on a
        /// full batch. The trade-off is that up to BatchSize-1 samples are not
        /// used in an epoch, which only matters when the dataset barely fills one
        /// batch; in that case the final batch is kept so training still runs.
        /// </para>
        /// </summary>
        public bool DropLast { get; set; } = true;

        // --- Optimizer Selection ---
        public OptimizerType Optimizer { get; set; } = OptimizerType.AdamW;

        // --- Weight Decay (Common to AdamW & SGD) ---
        public float WeightDecay { get; set; } = 0.01f;

        // --- AdamW Specifics ---
        public float Beta1 { get; set; } = 0.9f;
        public float Beta2 { get; set; } = 0.999f;
        public float Epsilon { get; set; } = 1e-8f;

        // --- SGD Specifics ---
        public float SgdMomentum { get; set; } = 0.9f;
        public bool UseNesterov { get; set; } = false;

        // --- Gradient Stability & Regularization ---
        /// <summary>
        /// Maximum global norm for gradient clipping. Set to <= 0 to disable clipping.
        /// </summary>
        public float MaxGradientNorm { get; set; } = 1.0f;

        // --- Learning Rate Scheduler ---
        public int WarmupSteps { get; set; } = 0;
        public float MinLearningRate { get; set; } = 1e-5f;

        // --- Factory Presets ---
        
        /// <summary>
        /// Optimal defaults for standard Transformer pre-training with AdamW.
        /// Uses lr = 3e-4, weight decay = 0.01, and AdamW momentum.
        /// </summary>
        public static TrainingConfig DefaultAdamWConfig => new TrainingConfig
        {
            Optimizer = OptimizerType.AdamW,
            LearningRate = 0.0003f, // 3e-4 is the standard baseline for Transformers
            WeightDecay = 0.01f,
            Beta1 = 0.9f,
            Beta2 = 0.999f,
            Epsilon = 1e-8f,
            MaxGradientNorm = 1.0f
        };

        /// <summary>
        /// Tuned defaults for SGD with Momentum and Nesterov acceleration.
        /// Uses a higher learning rate (0.01) necessary for SGD gradient scaling.
        /// </summary>
        public static TrainingConfig DefaultSgdConfig => new TrainingConfig
        {
            Optimizer = OptimizerType.Sgd,
            LearningRate = 0.01f,   // SGD requires ~30x larger LR than AdamW
            SgdMomentum = 0.9f,     // Essential for velocity tracking
            UseNesterov = true,
            WeightDecay = 0.0001f,
            MaxGradientNorm = 1.0f
        };

        //Update existing config using a new set of core hyperparameters
        public void UpdateCoreConfig(
            float? learningRate = null, 
            int? batchSize = null, 
            int? epochs = null, 
            float? dropoutRate = null, 
            float? weightDecay = null, 
            float? maxGradientNorm = null,
            bool? dropLast = null)
        {
            // Validate arguments before assigning if values exist
            if (learningRate.HasValue && learningRate.Value <= 0f)
                throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be greater than 0.");
            
            if (batchSize.HasValue && batchSize.Value <= 0)
                throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than 0.");
            
            if (epochs.HasValue && epochs.Value <= 0)
                throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be greater than 0.");

            if (dropoutRate.HasValue && (dropoutRate.Value < 0f || dropoutRate.Value >= 1f))
                throw new ArgumentOutOfRangeException(nameof(dropoutRate), "Dropout rate must be between [0.0, 1.0).");

            // Modern C# Null-Coalescing Assignment
            LearningRate = learningRate ?? LearningRate;
            BatchSize = batchSize ?? BatchSize;
            Epochs = epochs ?? Epochs;
            DropoutRate = dropoutRate ?? DropoutRate;
            WeightDecay = weightDecay ?? WeightDecay;
            MaxGradientNorm = maxGradientNorm ?? MaxGradientNorm;
            DropLast = dropLast ?? DropLast;
        }
    }
}

