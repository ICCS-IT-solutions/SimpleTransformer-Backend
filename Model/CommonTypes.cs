namespace SimpleTransformer.Model
{
    /// <summary>
    /// Compute precision mode for training.
    /// </summary>
    public enum PrecisionMode
    {
        /// <summary>Full FP32 compute and storage.</summary>
        FP32 = 0,

        /// <summary>FP16 storage with FP32 compute (automatic loss scaling).</summary>
        FP16 = 1,

        /// <summary>BF16 storage with FP32 compute (no loss scaling needed).</summary>
        BF16 = 2,
    }

    /// <summary>
    /// Policy for activation checkpointing to reduce activation memory.
    /// </summary>
    public enum CheckpointPolicy
    {
        /// <summary>No checkpointing - store all activations.</summary>
        None = 0,

        /// <summary>Checkpoint every layer - maximum memory savings, ~30% compute overhead.</summary>
        EveryLayer = 1,

        /// <summary>Checkpoint every other layer - balanced memory/compute tradeoff.</summary>
        EveryOtherLayer = 2,

        /// <summary>Checkpoint every N layers - configurable tradeoff.</summary>
        EveryNLayers = 3,
    }

    public enum OptimizerType
    {
        AdamW,
        Sgd
    }
}