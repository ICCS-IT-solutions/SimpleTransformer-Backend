namespace SimpleTransformer.Model
{
    public interface ILayer
    {
        TensorBase Forward(TensorBase input, TensorWorkspace workspace);

        /// <summary>
        /// Forward with an optional attention mask. Only attention consumes it:
        /// a rank-2 mask is position-only (e.g. causal) and shared by every batch
        /// item, a rank-3 mask carries one per item.
        /// <para>
        /// The default implementation ignores the mask, so every non-attention
        /// layer (feed-forward, norm, activation) keeps its 2-argument entry point
        /// and nothing else has to change. TransformerBlock overrides this to
        /// thread the mask down to ScaledDotProductAttention.
        /// </para>
        /// </summary>
        TensorBase Forward(TensorBase input, TensorWorkspace workspace, TensorBase? mask)
            => Forward(input, workspace);

        TensorBase Backward(TensorBase gradient, TensorWorkspace workspace);

        /// <summary>
        /// Backward with an optional NaN tripwire. Only layers that can host a
        /// meaningful sub-stage probe consume it - today TransformerBlock, to
        /// attribute a non-finite gradient to one of its six internal steps.
        /// <para>
        /// The default implementation ignores the probe and delegates, so every
        /// other layer keeps its 2-argument entry point and nothing else changes.
        /// This mirrors the mask overload above.
        /// </para>
        /// </summary>
        TensorBase Backward(TensorBase gradient, TensorWorkspace workspace, BackwardProbe? probe)
            => Backward(gradient, workspace);
        //Leaving this disabled for now until I need to bring it in.
        // IEnumerable<Tensor> Parameters { get; }
    }
}