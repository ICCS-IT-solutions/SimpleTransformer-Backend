namespace SimpleTransformer.Model
{
    public interface ILossFunction
    {
        float Forward(TensorBase prediction, TensorBase target);

        TensorBase Backward(TensorBase prediction, TensorBase target);

        /// <summary>
        /// Workspace-pooled variant: the returned gradient is borrowed from
        /// <paramref name="workspace"/> and must be released (or swept by
        /// Reset) by the caller. Default implementation ignores the workspace
        /// and delegates to the off-pool Backward (one step of GC pressure);
        /// overrides should borrow so a training step allocates nothing here.
        /// </summary>
        TensorBase Backward(TensorBase prediction, TensorBase target, TensorWorkspace workspace)
            => Backward(prediction, target);
    }
}