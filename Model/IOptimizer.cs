namespace SimpleTransformer.Model
{
    public interface IOptimizer
    {
        /// <summary>
        /// Current learning rate. Owned per-step by the LR scheduler
        /// (<see cref="LRScheduler"/>); the model sets it before each step.
        /// </summary>
        float LearningRate { get; set; }

        void Step(IEnumerable<TrainableParameter> layers);

        /// <summary>
        /// Snapshots the optimizer's training state (AdamW first/second moments
        /// and bias-correction step, SGD momentum velocity) for the checkpoint.
        /// </summary>
        OptimizerState ExportState();

        /// <summary>
        /// Restores a snapshot produced by <see cref="ExportState"/>.
        /// <para>
        /// <paramref name="parameters"/> supplies the expected name and shape for
        /// validation: entries that name an unknown parameter, or whose shape does
        /// not match it, are skipped rather than allowed to corrupt the next step.
        /// </para>
        /// </summary>
        /// <returns>Count of snapshot entries that were skipped (0 on a clean load).</returns>
        int ImportState(OptimizerState state, IReadOnlyList<TrainableParameter> parameters);
    }
}