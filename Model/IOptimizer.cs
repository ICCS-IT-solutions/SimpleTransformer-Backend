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
    }
}