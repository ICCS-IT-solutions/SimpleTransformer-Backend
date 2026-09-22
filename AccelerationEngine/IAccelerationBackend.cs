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

        // Synchronization
        void Synchronize();
    }
}