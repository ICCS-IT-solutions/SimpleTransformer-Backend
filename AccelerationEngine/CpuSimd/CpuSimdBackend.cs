using System;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.AccelerationEngine.CpuSimd
{
    /// <summary>
    /// CPU backend that routes every operation to the existing System.Numerics.Vector
    /// based SIMD kernel library (TensorMathSimd / TensorUtilitiesSimd / MaskUtilitiesSimd).
    ///
    /// This is the loosely coupled adapter: the model code only ever sees
    /// <see cref="IAccelerationBackend"/>, while the SIMD kernels remain encapsulated here.
    /// Swapping this class for a GPU backend therefore requires no changes to model code.
    /// </summary>
    public class CpuSimdBackend : IAccelerationBackend
    {
        public string Name => "CpuSimd (System.Numerics.Vector)";
        public bool IsGpuAccelerated => false;
        public bool IsAvailable => true;

        #region Element-wise operations

        public void ScaleInPlace(TensorBase tensor, float scalar)
            => TensorMathSimd.ScaleInPlace(tensor, scalar);

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
            => TensorMathSimd.ElementWiseAddInPlace(target, source);

        public void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
            => TensorMathSimd.ElementWiseAddInto(a, b, result);

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
            => TensorMathSimd.ElementWiseMultiplyInPlace(target, source);

        public void ElementWiseMultiplyInto(TensorBase a, TensorBase b, TensorBase result)
            => TensorMathSimd.ElementWiseMultiplyInto(a, b, result);

        #endregion

        #region Activations

        public void GeluInPlace(TensorBase tensor)
            => TensorMathSimd.GeluInPlace(tensor);

        public void GeluInto(TensorBase input, TensorBase result)
            => TensorMathSimd.GeluInto(input, result);

        public void GeluBackwardInto(TensorBase input, TensorBase outputGradient, TensorBase inputGradient)
            => TensorMathSimd.GeluBackwardInto(input, outputGradient, inputGradient);

        #endregion

        #region Normalization

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f)
            => TensorMathSimd.LayerNormInPlace(tensor, gamma, beta, epsilon);

        public void LayerNormInto(TensorBase input, TensorBase gamma, TensorBase beta, TensorBase result, float epsilon = 1e-5f)
            => TensorMathSimd.LayerNormInto(input, gamma, beta, result, epsilon);

        #endregion

        #region Softmax

        public void SoftmaxInPlace(TensorBase tensor)
        {
            // The SIMD kernel library exposes row-wise softmax on rank-2 matrices
            // (and per-row spans); route rank-2 straight through.
            if (tensor.Rank == 2 && tensor is Tensor matrix)
            {
                TensorUtilitiesSimd.SoftmaxRowsInPlace(matrix);
                return;
            }

            FallbackToScalarRowwise(tensor, static row => TensorUtilitiesSimd.SoftmaxInPlace(row));
        }

        public void SoftmaxBackwardInto(TensorBase softmaxOutput, TensorBase outputGradient, TensorBase inputGradient)
        {
            // Note: TensorUtilitiesSimd.SoftmaxBackwardInto takes (outputGradient, softmaxOutput, inputGradient)
            // as concrete Tensors; the backend contract uses the (softmaxOutput, outputGradient, inputGradient) order.
            if (softmaxOutput is Tensor soft && outputGradient is Tensor dy && inputGradient is Tensor dx)
            {
                TensorUtilitiesSimd.SoftmaxBackwardInto(dy, soft, dx);
                return;
            }

            FallbackToScalarRowwise3(
                softmaxOutput,
                outputGradient,
                inputGradient,
                static (soft, dy, dx) =>
                {
                    float dot = 0f;
                    for (int i = 0; i < soft.Length; i++)
                    {
                        dot += dy[i] * soft[i];
                    }
                    for (int i = 0; i < soft.Length; i++)
                    {
                        dx[i] = soft[i] * (dy[i] - dot);
                    }
                });
        }

        #endregion

        #region Matrix operations

        public void MatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => TensorMathSimd.MatMul(a, b, result, transposeA, transposeB);

        public void MatMulAccumulate(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
        {
            // The SIMD library exposes one native accumulate kernel (left-transposed),
            // used by the Linear/QLoRA gradient path. Any other transpose combination
            // falls back to product-into-scratch + add.
            if (transposeA && !transposeB)
            {
                TensorMathSimd.MatrixMultiplyLeftTransposedAccumulateInto(a, b, result);
                return;
            }

            var product = new Tensor(result.Shape);
            try
            {
                TensorMathSimd.MatMul(a, b, product, transposeA, transposeB);
                TensorMathSimd.ElementWiseAddInPlace(result, product);
            }
            finally
            {
                product.Dispose();
            }
        }

        #endregion

        #region Attention masking

        public void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
            => MaskUtilitiesSimd.ApplyMaskInPlace(scores, mask);

        #endregion

        #region Layout / memory movement

        public void TransposeInto(TensorBase source, TensorBase destination)
            => TensorUtilitiesSimd.TransposeInto(source, destination);

        public void CopyInto(TensorBase source, TensorBase destination)
            => TensorUtilitiesSimd.CopyInto(source, destination);

        public void Fill(TensorBase tensor, float value)
            => TensorUtilitiesSimd.Fill(tensor, value);

        #endregion

        #region Fallback helpers

        private static void FallbackToScalarRowwise(TensorBase tensor, Action<Span<float>> rowOp)
        {
            int rows = tensor.Rank switch
            {
                1 => 1,
                2 => tensor.Rows,
                3 => tensor.Layers * tensor.Rows,
                _ => throw new ArgumentException($"Tensor rank {tensor.Rank} is not supported.")
            };

            for (int r = 0; r < rows; r++)
            {
                rowOp(GetLinearRowSpan(tensor, r));
            }
        }

        private static void FallbackToScalarRowwise3(
            TensorBase a,
            TensorBase b,
            TensorBase c,
            Action<Span<float>, ReadOnlySpan<float>, Span<float>> rowOp)
        {
            int rows = a.Rank switch
            {
                1 => 1,
                2 => a.Rows,
                3 => a.Layers * a.Rows,
                _ => throw new ArgumentException($"Tensor rank {a.Rank} is not supported.")
            };

            for (int r = 0; r < rows; r++)
            {
                rowOp(
                    GetLinearRowSpan(a, r),
                    GetLinearRowSpan(b, r),
                    GetLinearRowSpan(c, r));
            }
        }

        private static Span<float> GetLinearRowSpan(TensorBase tensor, int linearRow)
        {
            switch (tensor.Rank)
            {
                case 1:
                    return tensor.Buffer.AsSpan(tensor.Offset, tensor.Cols);
                case 2:
                    return tensor.Buffer.AsSpan(tensor.Offset + linearRow * tensor.Stride, tensor.Cols);
                case 3:
                    int layer = linearRow / tensor.Rows;
                    int row = linearRow % tensor.Rows;
                    return tensor.Buffer.AsSpan(
                        tensor.Offset + layer * tensor.LayerStride + row * tensor.Stride,
                        tensor.Cols);
                default:
                    throw new ArgumentException($"Tensor rank {tensor.Rank} is not supported.");
            }
        }

        #endregion

        #region Lifecycle

        /// <summary>
        /// The SIMD backend is stateless - nothing to release.
        /// </summary>
        public void Dispose()
        {
        }

        /// <summary>
        /// All operations complete synchronously on the calling thread - nothing to wait for.
        /// </summary>
        public void Synchronize()
        {
        }

        #endregion
    }
}