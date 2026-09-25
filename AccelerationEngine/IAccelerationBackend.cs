using SimpleTransformer.Model;
namespace SimpleTransformer.AccelerationEngine
{
    public interface IAccelerationBackend : IDisposable
    {
        string Name { get; }

        bool IsGpuAccelerated { get; }

        /// <summary>True when the backend initialized successfully and can execute ops.</summary>
        bool IsAvailable { get; }

        // Element-wise operations
        void ScaleInPlace(
            TensorBase tensor,
            float scalar);

        void ElementWiseAddInPlace(
            TensorBase target,
            TensorBase source);

        void ElementWiseAddInto(
            TensorBase a,
            TensorBase b,
            TensorBase result);

        void ElementWiseMultiplyInPlace(
            TensorBase target,
            TensorBase source);

        void ElementWiseMultiplyInto(
            TensorBase a,
            TensorBase b,
            TensorBase result);

        // Activation / normalization
        void GeluInPlace(
            TensorBase tensor);

        void GeluInto(
            TensorBase input,
            TensorBase result);

        void GeluBackwardInto(
            TensorBase input,
            TensorBase outputGradient,
            TensorBase inputGradient);

        void LayerNormInPlace(
            TensorBase tensor,
            TensorBase gamma,
            TensorBase beta,
            float epsilon = 1e-5f);

        void LayerNormInto(
            TensorBase input,
            TensorBase gamma,
            TensorBase beta,
            TensorBase result,
            float epsilon = 1e-5f);

        // Softmax
        void SoftmaxInPlace(
            TensorBase tensor);

        void SoftmaxBackwardInto(
            TensorBase softmaxOutput,
            TensorBase outputGradient,
            TensorBase inputGradient);

        // Matrix operations
        void MatMul(
            TensorBase a,
            TensorBase b,
            TensorBase result,
            bool transposeA = false,
            bool transposeB = false);

        void MatMulAccumulate(
            TensorBase a,
            TensorBase b,
            TensorBase result,
            bool transposeA = false,
            bool transposeB = false);

        // Attention
        void ApplyMaskInPlace(
            TensorBase scores,
            TensorBase mask);

        // Tensor layout / memory movement
        /// <summary>
        /// Writes the transpose of a rank-2 <paramref name="source"/> into <paramref name="destination"/>.
        /// </summary>
        void TransposeInto(
            TensorBase source,
            TensorBase destination);

        /// <summary>
        /// Copies the logical contents of <paramref name="source"/> into <paramref name="destination"/> (stride-aware).
        /// </summary>
        void CopyInto(
            TensorBase source,
            TensorBase destination);

        /// <summary>
        /// Fills the whole logical tensor with <paramref name="value"/>.
        /// </summary>
        void Fill(
            TensorBase tensor,
            float value);

        /// <summary>
        /// Optional hint: <paramref name="tensor"/> is immutable for the model's
        /// lifetime (e.g. frozen QLoRA base weights) and may therefore be kept in
        /// device memory and reused by later ops instead of being uploaded per
        /// call. Backends that can cache return true; the default implementation
        /// (CPU backends, where the arrays are already where they are needed) is
        /// a no-op returning false.
        /// </summary>
        bool TryRegisterResidentWeights(TensorBase tensor, string? name = null) => false;

        /// <summary>
        /// Optional counterpart to <see cref="TryRegisterResidentWeights"/>: releases
        /// any device memory the backend cached for <paramref name="tensor"/>. A
        /// layer owning an immutable weight calls this when it is disposed so
        /// unloading a model gives the memory back instead of holding it until the
        /// backend shuts down. The default implementation does nothing.
        /// </summary>
        void ReleaseResidentWeights(TensorBase tensor) { }

        /// <summary>
        /// Optional refresh: re-uploads a tensor's current contents into its existing
        /// VRAM-resident buffer after the host copy was mutated in place (an optimizer
        /// step or checkpoint hydration). Returns false when the tensor is not resident
        /// or the backend cannot cache (CPU backends) - callers may safely ignore the
        /// result, the ordinary per-call upload path stays correct either way.
        /// </summary>
        bool TryRefreshResidentWeights(TensorBase tensor) => false;

        /// <summary>
        /// Optional one-line memory telemetry for logs, e.g. VRAM-resident bytes,
        /// this process's VRAM usage and dispatch count. Empty string when the
        /// backend has nothing to report (CPU backends).
        /// </summary>
        string DescribeMemoryUsage() => string.Empty;

        // Synchronization
        void Synchronize();
    }
}