namespace SimpleTransformer.Model
{
    /// <summary>
    /// Memory policy for the attention softmax cache, pushed in by the host
    /// process (Server / CLI) from config.ini before any model is constructed.
    /// <para>
    /// Scaled dot-product attention caches the post-softmax weight matrix so
    /// Backward can apply the softmax Jacobian. That tensor is
    /// heads x batch x seq^2 floats - at seq 2048, 16 heads, batch 8, 16 layers
    /// it is 32 GiB, and it dwarfs every other activation in the model.
    /// <see cref="RecomputeSoftmaxWeights"/> trades the cache for a second
    /// Q*K^T + softmax in Backward: memory drops to O(seq) per head, at roughly
    /// a third more attention FLOPs.
    /// <para>
    /// Recompute is refused while attention-weight dropout is enabled: the
    /// dropped matrix the forward matmul actually consumed is a function of the
    /// per-item dropout mask, and re-running DropoutLayer.Forward to rebuild it
    /// would consume fresh RNG. With dropout on, attention keeps caching.
    /// </para>
    /// </summary>
    public static class AttentionMemorySettings
    {
        /// <summary>
        /// Default false: caching is the fast path and the only one that keeps
        /// Backward's cost identical to a non-recomputing implementation. Turn
        /// on when the seq^2 cache will not fit.
        /// </summary>
        public static bool RecomputeSoftmaxWeights { get; set; }
    }
}