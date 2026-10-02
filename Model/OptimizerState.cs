using System.Collections.Generic;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Encodes which optimizer produced a serialized <see cref="OptimizerState"/>,
    /// so a checkpoint can never be loaded into the wrong optimizer (their state
    /// layouts are not interchangeable).
    /// </summary>
    public static class OptimizerStateKinds
    {
        public const int AdamW = 0;
        public const int Sgd = 1;
    }

    /// <summary>
    /// A serializable snapshot of an optimizer's per-parameter training state.
    /// <para>
    /// AdamW's first/second moments and bias-correction step, and SGD's momentum
    /// velocity, are what make a resumed run continue rather than restart. They
    /// were previously never persisted, so a process restart reset every moment
    /// to zero and restarted bias correction at t=1 while the LR schedule carried
    /// on - a much larger deviation than the dropout masks.
    /// </para>
    /// <para>
    /// Deliberately a plain data holder: the checkpoint byte format lives in
    /// <see cref="CheckpointStateExtensions"/>, so optimizers stay format-agnostic.
    /// </para>
    /// </summary>
    public sealed class OptimizerState
    {
        /// <summary>0 = AdamW, 1 = SGD. Validated against the configured optimizer on load.</summary>
        public int Kind { get; init; }

        /// <summary>
        /// AdamW's bias-correction step (t). Zero for SGD, which has no bias
        /// correction.
        /// </summary>
        public int StepCount { get; init; }

        /// <summary>
        /// Per-parameter state, keyed by parameter name (the same key the
        /// checkpoint's parameter block uses). Empty when the checkpoint was
        /// written before any optimizer step, or for vanilla SGD.
        /// </summary>
        public List<OptimizerParamState> Parameters { get; init; } = new();
    }

    /// <summary>
    /// One parameter's optimizer state. Only the fields relevant to the
    /// optimizer kind are populated.
    /// </summary>
    public sealed class OptimizerParamState
    {
        public required string Name { get; init; }

        /// <summary>AdamW first moment (m).</summary>
        public TensorData? FirstMoment { get; init; }

        /// <summary>AdamW second moment (v).</summary>
        public TensorData? SecondMoment { get; init; }

        /// <summary>SGD momentum velocity buffer.</summary>
        public TensorData? Velocity { get; init; }
    }
}