using System;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.CpuReference
{
    /// <summary>
    /// Pure managed reference implementation of <see cref="IAccelerationBackend"/>.
    ///
    /// Design goals:
    ///  - ZERO low-level code: no System.Numerics.Vector, no unsafe pointers, no
    ///    hardware intrinsics and no parallelisation. Just plain, sequential C# loops.
    ///  - Deterministic and single-threaded, so results are reproducible run to run.
    ///  - Stride-aware: works on owning <see cref="Tensor"/>s as well as strided
    ///    <see cref="TensorView"/>s, because rows are addressed through
    ///    Buffer/Offset/Stride instead of assuming one flat block.
    ///
    /// This backend is the correctness baseline used to validate the SIMD and GPU
    /// backends (see AccelerationEngine.BackendSelfTest).
    /// </summary>
    public class CpuReferenceBackend : IAccelerationBackend
    {
        private const float Sqrt2OverPi = 0.7978845608f; // sqrt(2 / pi)
        private const float GeluC = 0.044715f;
        private const float MaskValue = -1e9f;

        public string Name => "CpuReference (pure managed scalar)";
        public bool IsGpuAccelerated => false;
        public bool IsAvailable => true;

        #region Shared helpers

        /// <summary>
        /// Number of logical rows when the last dimension (Cols) is the normalized axis.
        /// Rank 1 = single row, rank 2 = matrix rows, rank 3 = layer-major flattened rows.
        /// </summary>
        private static int GetRowCount(TensorBase tensor) => tensor.Rank switch
        {
            1 => 1,
            2 => tensor.Rows,
            3 => tensor.Layers * tensor.Rows,
            _ => throw new ArgumentException($"Tensor rank {tensor.Rank} is not supported (expected 1, 2 or 3).")
        };

        /// <summary>
        /// Returns a contiguous span over one logical row of the tensor.
        /// Works for owning tensors and strided views alike.
        /// </summary>
        private static Span<float> GetRow(TensorBase tensor, int linearRow)
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
                    throw new ArgumentException($"Tensor rank {tensor.Rank} is not supported (expected 1, 2 or 3).");
            }
        }

        private static void ValidateSameShape(TensorBase a, TensorBase b, string operation)
        {
            if (a.Rank != b.Rank)
                throw new ArgumentException($"{operation}: tensor ranks do not match ({a.Rank} vs {b.Rank}).");

            if (a.Layers != b.Layers || a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException(
                    $"{operation}: tensor shapes do not match " +
                    $"({a.Layers}x{a.Rows}x{a.Cols}) vs ({b.Layers}x{b.Rows}x{b.Cols}).");
        }

        private static void ValidateSameShape(TensorBase a, TensorBase b, TensorBase c, string operation)
        {
            ValidateSameShape(a, b, operation);
            ValidateSameShape(a, c, operation);
        }

        private static void ValidateLayerNormArguments(TensorBase gamma, TensorBase beta, int normalizedSize, string operation)
        {
            if (gamma.Length != normalizedSize || beta.Length != normalizedSize)
                throw new ArgumentException(
                    $"{operation}: gamma and beta dimensions must match the normalized dimension ({normalizedSize}).");
        }

        #endregion

        #region Element-wise operations

        public void ScaleInPlace(TensorBase tensor, float scalar)
        {
            int rows = GetRowCount(tensor);
            for (int r = 0; r < rows; r++)
            {
                Span<float> row = GetRow(tensor, r);
                for (int c = 0; c < row.Length; c++)
                {
                    row[c] *= scalar;
                }
            }
        }

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
        {
            ValidateSameShape(target, source, nameof(ElementWiseAddInPlace));

            int rows = GetRowCount(target);
            for (int r = 0; r < rows; r++)
            {
                Span<float> targetRow = GetRow(target, r);
                ReadOnlySpan<float> sourceRow = GetRow(source, r);
                for (int c = 0; c < targetRow.Length; c++)
                {
                    targetRow[c] += sourceRow[c];
                }
            }
        }

        public void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
        {
            ValidateSameShape(a, b, result, nameof(ElementWiseAddInto));

            int rows = GetRowCount(a);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> rowA = GetRow(a, r);
                ReadOnlySpan<float> rowB = GetRow(b, r);
                Span<float> rowResult = GetRow(result, r);
                for (int c = 0; c < rowResult.Length; c++)
                {
                    rowResult[c] = rowA[c] + rowB[c];
                }
            }
        }

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
        {
            ValidateSameShape(target, source, nameof(ElementWiseMultiplyInPlace));

            int rows = GetRowCount(target);
            for (int r = 0; r < rows; r++)
            {
                Span<float> targetRow = GetRow(target, r);
                ReadOnlySpan<float> sourceRow = GetRow(source, r);
                for (int c = 0; c < targetRow.Length; c++)
                {
                    targetRow[c] *= sourceRow[c];
                }
            }
        }

        public void ElementWiseMultiplyInto(TensorBase a, TensorBase b, TensorBase result)
        {
            ValidateSameShape(a, b, result, nameof(ElementWiseMultiplyInto));

            int rows = GetRowCount(a);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> rowA = GetRow(a, r);
                ReadOnlySpan<float> rowB = GetRow(b, r);
                Span<float> rowResult = GetRow(result, r);
                for (int c = 0; c < rowResult.Length; c++)
                {
                    rowResult[c] = rowA[c] * rowB[c];
                }
            }
        }

        #endregion

        #region Activations

        private static float Gelu(float value)
        {
            float x2 = value * value;
            float x3 = x2 * value;
            float u = Sqrt2OverPi * (value + GeluC * x3);
            return 0.5f * value * (1f + MathF.Tanh(u));
        }

        public void GeluInPlace(TensorBase tensor)
        {
            int rows = GetRowCount(tensor);
            for (int r = 0; r < rows; r++)
            {
                Span<float> row = GetRow(tensor, r);
                for (int c = 0; c < row.Length; c++)
                {
                    row[c] = Gelu(row[c]);
                }
            }
        }

        public void GeluInto(TensorBase input, TensorBase result)
        {
            ValidateSameShape(input, result, nameof(GeluInto));

            int rows = GetRowCount(input);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> inputRow = GetRow(input, r);
                Span<float> resultRow = GetRow(result, r);
                for (int c = 0; c < resultRow.Length; c++)
                {
                    resultRow[c] = Gelu(inputRow[c]);
                }
            }
        }

        public void GeluBackwardInto(TensorBase input, TensorBase outputGradient, TensorBase inputGradient)
        {
            ValidateSameShape(input, outputGradient, inputGradient, nameof(GeluBackwardInto));

            int rows = GetRowCount(input);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> xRow = GetRow(input, r);
                ReadOnlySpan<float> dyRow = GetRow(outputGradient, r);
                Span<float> dxRow = GetRow(inputGradient, r);

                for (int c = 0; c < dxRow.Length; c++)
                {
                    float value = xRow[c];
                    float x2 = value * value;
                    float x3 = x2 * value;

                    float u = Sqrt2OverPi * (value + GeluC * x3);
                    float t = MathF.Tanh(u);

                    float derivative =
                        0.5f * (1f + t)
                        + 0.5f * value * (1f - t * t) * Sqrt2OverPi * (1f + 3f * GeluC * x2);

                    dxRow[c] = dyRow[c] * derivative;
                }
            }
        }

        #endregion

        #region Normalization

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f)
        {
            ValidateSameShape(tensor, tensor, nameof(LayerNormInPlace));
            ValidateLayerNormArguments(gamma, beta, tensor.Cols, nameof(LayerNormInPlace));

            int rows = GetRowCount(tensor);
            ReadOnlySpan<float> gammaSpan = gamma.Buffer.AsSpan(gamma.Offset, gamma.Length);
            ReadOnlySpan<float> betaSpan = beta.Buffer.AsSpan(beta.Offset, beta.Length);

            for (int r = 0; r < rows; r++)
            {
                Span<float> row = GetRow(tensor, r);
                LayerNormRow(row, row, gammaSpan, betaSpan, epsilon);
            }
        }

        public void LayerNormInto(TensorBase input, TensorBase gamma, TensorBase beta, TensorBase result, float epsilon = 1e-5f)
        {
            ValidateSameShape(input, result, nameof(LayerNormInto));
            ValidateLayerNormArguments(gamma, beta, input.Cols, nameof(LayerNormInto));

            int rows = GetRowCount(input);
            ReadOnlySpan<float> gammaSpan = gamma.Buffer.AsSpan(gamma.Offset, gamma.Length);
            ReadOnlySpan<float> betaSpan = beta.Buffer.AsSpan(beta.Offset, beta.Length);

            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> inputRow = GetRow(input, r);
                Span<float> resultRow = GetRow(result, r);
                LayerNormRow(inputRow, resultRow, gammaSpan, betaSpan, epsilon);
            }
        }

        /// <summary>
        /// Normalizes one row: y = (x - mean) * invStd * gamma + beta,
        /// with invStd = 1 / sqrt(biasedVariance + epsilon).
        /// </summary>
        private static void LayerNormRow(
            ReadOnlySpan<float> input,
            Span<float> output,
            ReadOnlySpan<float> gamma,
            ReadOnlySpan<float> beta,
            float epsilon)
        {
            int len = input.Length;
            if (len == 0)
                return;

            // 1. Mean
            float sum = 0f;
            for (int i = 0; i < len; i++)
            {
                sum += input[i];
            }
            float mean = sum / len;

            // 2. Biased variance
            float variance = 0f;
            for (int i = 0; i < len; i++)
            {
                float diff = input[i] - mean;
                variance += diff * diff;
            }
            variance /= len;

            // 3. Normalize + scale + shift
            float invStd = 1.0f / MathF.Sqrt(variance + epsilon);
            for (int i = 0; i < len; i++)
            {
                float normalized = (input[i] - mean) * invStd;
                output[i] = normalized * gamma[i] + beta[i];
            }
        }

        #endregion

        #region Softmax

        public void SoftmaxInPlace(TensorBase tensor)
        {
            int rows = GetRowCount(tensor);
            for (int r = 0; r < rows; r++)
            {
                Span<float> row = GetRow(tensor, r);
                SoftmaxRow(row);
            }
        }

        private static void SoftmaxRow(Span<float> row)
        {
            int len = row.Length;
            if (len == 0)
                throw new ArgumentException("Cannot compute softmax on an empty row.");

            // 1. Row maximum (for numerical stability)
            float max = row[0];
            for (int i = 1; i < len; i++)
            {
                if (row[i] > max)
                    max = row[i];
            }

            // 2. Exponentiate and accumulate the sum
            float sum = 0f;
            for (int i = 0; i < len; i++)
            {
                float e = MathF.Exp(row[i] - max);
                row[i] = e;
                sum += e;
            }

            // 3. Normalize
            float invSum = 1f / sum;
            for (int i = 0; i < len; i++)
            {
                row[i] *= invSum;
            }
        }

        public void SoftmaxBackwardInto(TensorBase softmaxOutput, TensorBase outputGradient, TensorBase inputGradient)
        {
            ValidateSameShape(softmaxOutput, outputGradient, inputGradient, nameof(SoftmaxBackwardInto));

            int rows = GetRowCount(softmaxOutput);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> softRow = GetRow(softmaxOutput, r);
                ReadOnlySpan<float> dyRow = GetRow(outputGradient, r);
                Span<float> dxRow = GetRow(inputGradient, r);

                // dot = sum(dy * soft)
                float dot = 0f;
                for (int i = 0; i < softRow.Length; i++)
                {
                    dot += dyRow[i] * softRow[i];
                }

                // dx = soft * (dy - dot)
                for (int i = 0; i < softRow.Length; i++)
                {
                    dxRow[i] = softRow[i] * (dyRow[i] - dot);
                }
            }
        }

        #endregion

        #region Matrix operations

        public void MatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
        {
            MatMulCore(a, b, result, transposeA, transposeB, accumulate: false);
        }

        public void MatMulAccumulate(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
        {
            MatMulCore(a, b, result, transposeA, transposeB, accumulate: true);
        }

        /// <summary>
        /// Sequential GEMM supporting rank 2 and rank 3 (batched) tensors.
        /// Logical dims: A is (m x k), B is (k x n), result is (m x n), where the
        /// physical layout is adjusted by the transpose flags.
        /// </summary>
        private static void MatMulCore(
            TensorBase a,
            TensorBase b,
            TensorBase result,
            bool transposeA,
            bool transposeB,
            bool accumulate)
        {
            if (a.Rank != b.Rank || a.Rank != result.Rank)
                throw new ArgumentException(
                    $"MatMul requires matching ranks (got {a.Rank}, {b.Rank}, {result.Rank}).");

            if (a.Rank != 2 && a.Rank != 3)
                throw new ArgumentException($"MatMul expects rank 2 or rank 3 tensors (got rank {a.Rank}).");

            if (a.Rank == 3 && (a.Layers != b.Layers || a.Layers != result.Layers))
                throw new ArgumentException("MatMul batch sizes do not match across a/b/result.");

            // Logical dimensions of the operation.
            int m = transposeA ? a.Cols : a.Rows;
            int k = transposeA ? a.Rows : a.Cols;
            int kb = transposeB ? b.Cols : b.Rows;
            int n = transposeB ? b.Rows : b.Cols;

            if (k != kb)
                throw new ArgumentException($"MatMul inner dimensions do not match ({k} vs {kb}).");

            if (result.Rows != m || result.Cols != n)
                throw new ArgumentException(
                    $"MatMul result buffer is ({result.Rows}x{result.Cols}) but expected ({m}x{n}).");

            int layers = a.Rank == 3 ? a.Layers : 1;
            int aLayerStride = a.Rank == 3 ? a.LayerStride : 0;
            int bLayerStride = b.Rank == 3 ? b.LayerStride : 0;
            int rLayerStride = result.Rank == 3 ? result.LayerStride : 0;

            float[] aBuf = a.Buffer;
            float[] bBuf = b.Buffer;
            float[] rBuf = result.Buffer;

            int aOffset = a.Offset;
            int bOffset = b.Offset;
            int rOffset = result.Offset;

            int aStride = a.Stride;
            int bStride = b.Stride;
            int rStride = result.Stride;

            for (int layer = 0; layer < layers; layer++)
            {
                int aBase = aOffset + layer * aLayerStride;
                int bBase = bOffset + layer * bLayerStride;
                int rBase = rOffset + layer * rLayerStride;

                for (int i = 0; i < m; i++)
                {
                    int rRowBase = rBase + i * rStride;

                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0f;

                        for (int e = 0; e < k; e++)
                        {
                            float av = transposeA
                                ? aBuf[aBase + e * aStride + i]     // logical A[i, e] = physical A[e, i]
                                : aBuf[aBase + i * aStride + e];    // logical A[i, e] = physical A[i, e]

                            float bv = transposeB
                                ? bBuf[bBase + j * bStride + e]     // logical B[e, j] = physical B[j, e]
                                : bBuf[bBase + e * bStride + j];    // logical B[e, j] = physical B[e, j]

                            sum += av * bv;
                        }

                        int rIdx = rRowBase + j;
                        rBuf[rIdx] = accumulate ? rBuf[rIdx] + sum : sum;
                    }
                }
            }
        }

        #endregion

        #region Attention masking

        public void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
        {
            if (scores.Rank != 2)
                throw new ArgumentException("Attention scores must be a matrix (Rank 2).");

            if (scores.Rows != scores.Cols)
                throw new ArgumentException("Attention scores must be a square matrix.");

            if (mask.Rank != 2 || mask.Rows != scores.Rows || mask.Cols != scores.Cols)
                throw new ArgumentException(
                    $"Mask must be a ({scores.Rows}x{scores.Cols}) matrix.");

            int rows = scores.Rows;
            int cols = scores.Cols;

            float[] scoresBuf = scores.Buffer;
            int scoresOffset = scores.Offset;
            int scoresStride = scores.Stride;

            float[] maskBuf = mask.Buffer;
            int maskOffset = mask.Offset;
            int maskStride = mask.Stride;

            for (int r = 0; r < rows; r++)
            {
                int scoresRowBase = scoresOffset + r * scoresStride;
                int maskRowBase = maskOffset + r * maskStride;

                for (int c = 0; c < cols; c++)
                {
                    if (maskBuf[maskRowBase + c] == 0f)
                    {
                        scoresBuf[scoresRowBase + c] = MaskValue;
                    }
                }
            }
        }

        #endregion

        #region Layout / memory movement

        public void TransposeInto(TensorBase source, TensorBase destination)
        {
            if (source.Rank != 2 || destination.Rank != 2)
                throw new ArgumentException("TransposeInto requires rank 2 tensors.");

            if (destination.Rows != source.Cols || destination.Cols != source.Rows)
                throw new ArgumentException(
                    $"TransposeInto destination shape mismatch. Expected {source.Cols}x{source.Rows}.");

            int rows = source.Rows;
            int cols = source.Cols;

            float[] srcBuf = source.Buffer;
            int srcOffset = source.Offset;
            int srcStride = source.Stride;

            float[] dstBuf = destination.Buffer;
            int dstOffset = destination.Offset;
            int dstStride = destination.Stride;

            for (int r = 0; r < rows; r++)
            {
                int srcRowBase = srcOffset + r * srcStride;
                for (int c = 0; c < cols; c++)
                {
                    dstBuf[dstOffset + c * dstStride + r] = srcBuf[srcRowBase + c];
                }
            }
        }

        public void CopyInto(TensorBase source, TensorBase destination)
        {
            ValidateSameShape(source, destination, nameof(CopyInto));

            int rows = GetRowCount(source);
            for (int r = 0; r < rows; r++)
            {
                GetRow(source, r).CopyTo(GetRow(destination, r));
            }
        }

        public void Fill(TensorBase tensor, float value)
        {
            int rows = GetRowCount(tensor);
            for (int r = 0; r < rows; r++)
            {
                GetRow(tensor, r).Fill(value);
            }
        }

        #endregion

        #region Lifecycle

        /// <summary>
        /// The reference backend is fully managed and stateless - nothing to release.
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