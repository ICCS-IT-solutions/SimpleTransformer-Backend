using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.CpuReference.Math;

public sealed class MatrixOperations
{
    #region Matrix Multiplication

    public TensorBase MatrixMultiply(TensorBase a, TensorBase b)
    {
        ValidateMatrixMultiply(a, b);

        var result = new Tensor(a.Rows, b.Cols);
        MatrixMultiplyInto(a, b, result);

        return result;
    }

    public void MatrixMultiplyInto(
        TensorBase a,
        TensorBase b,
        TensorBase result)
    {
        ValidateMatrixMultiply(a, b);
        ValidateResult(result, a.Rows, b.Cols);

        int m = a.Rows;
        int n = b.Cols;
        int k = a.Cols;

        float[] aData = a.Data;
        float[] bData = b.Data;
        float[] resultData = result.Data;

        int aOffset = a.Offset;
        int bOffset = b.Offset;
        int resultOffset = result.Offset;

        int aStride = a.Stride;
        int bStride = b.Stride;
        int resultStride = result.Stride;

        for (int i = 0; i < m; i++)
        {
            int aRowOffset = aOffset + (i * aStride);
            int resultRowOffset = resultOffset + (i * resultStride);

            for (int j = 0; j < n; j++)
            {
                float sum = 0f;

                for (int p = 0; p < k; p++)
                {
                    sum +=
                        aData[aRowOffset + p] *
                        bData[bOffset + (p * bStride) + j];
                }

                resultData[resultRowOffset + j] = sum;
            }
        }
    }

    #endregion

    #region Transposed Multiplication

    public TensorBase MultiplyTransposeRight(
        TensorBase a,
        TensorBase b)
    {
        ValidateMatrixMultiplyTransposeRight(a, b);

        var transposeBuffer = new Tensor(b.Cols, b.Rows);
        var result = new Tensor(a.Rows, b.Rows);

        MultiplyTransposeRightInto(
            a,
            b,
            transposeBuffer,
            result);

        return result;
    }

    public void MultiplyTransposeRightInto(
        TensorBase a,
        TensorBase b,
        TensorBase transposeBuffer,
        TensorBase result)
    {
        ValidateMatrixMultiplyTransposeRight(a, b);

        ValidateResult(
            result,
            a.Rows,
            b.Rows);

        ValidateTransposeBuffer(
            transposeBuffer,
            b);

        TransposeInto(b, transposeBuffer);
        MatrixMultiplyInto(a, transposeBuffer, result);
    }

    public TensorBase MultiplyTransposeLeft(
        TensorBase a,
        TensorBase b)
    {
        ValidateMatrixMultiplyTransposeLeft(a, b);

        var transposeBuffer = new Tensor(a.Cols, a.Rows);
        var result = new Tensor(a.Cols, b.Cols);

        MultiplyTransposeLeftInto(
            a,
            b,
            transposeBuffer,
            result);

        return result;
    }
    public void MultiplyTransposeLeftInto(
        TensorBase a,
        TensorBase b,
        TensorBase transposeBuffer,
        TensorBase result)
    {
        ValidateMatrixMultiplyTransposeLeft(a, b);

        ValidateResult(
            result,
            a.Cols,
            b.Cols);

        ValidateTransposeBuffer(
            transposeBuffer,
            a);

        TransposeInto(a, transposeBuffer);
        MatrixMultiplyInto(transposeBuffer, b, result);
    }

    #endregion

    #region Cached Transposed Multiplication

    /// <summary>
    /// Multiplies A by a matrix that has already been transposed.
    ///
    /// The supplied transposed matrix is treated as an ordinary
    /// matrix whose dimensions already represent B^T.
    /// No additional transpose is performed.
    /// </summary>
    public void MatrixMultiplyWithRightTransposed(
        TensorBase a,
        TensorBase transposedB,
        TensorBase result)
    {
        MatrixMultiplyInto(
            a,
            transposedB,
            result);
    }

    /// <summary>
    /// Multiplies a matrix that has already been transposed by B.
    ///
    /// The supplied transposed matrix is treated as an ordinary
    /// matrix whose dimensions already represent A^T.
    /// No additional transpose is performed.
    /// </summary>
    public void MatrixMultiplyWithLeftTransposed(
        TensorBase transposedA,
        TensorBase b,
        TensorBase result)
    {
        MatrixMultiplyInto(
            transposedA,
            b,
            result);
    }

    #endregion

    #region Transpose

    public TensorBase Transpose(TensorBase matrix)
    {
        ValidateMatrix(matrix);

        var result = new Tensor(
            matrix.Cols,
            matrix.Rows);

        TransposeInto(matrix, result);

        return result;
    }

    public void TransposeInto(
        TensorBase source,
        TensorBase destination)
    {
        ValidateMatrix(source);
        ValidateMatrix(destination);

        if (destination.Rows != source.Cols ||
            destination.Cols != source.Rows)
        {
            throw new ArgumentException(
                $"Destination must have shape " +
                $"({source.Cols}, {source.Rows}).");
        }

        float[] src = source.Data;
        float[] dst = destination.Data;

        int srcOffset = source.Offset;
        int dstOffset = destination.Offset;

        int srcStride = source.Stride;
        int dstStride = destination.Stride;

        int rows = source.Rows;
        int cols = source.Cols;

        for (int r = 0; r < rows; r++)
        {
            int srcRowOffset =
                srcOffset + (r * srcStride);

            for (int c = 0; c < cols; c++)
            {
                dst[
                    dstOffset +
                    (c * dstStride) +
                    r
                ] = src[srcRowOffset + c];
            }
        }
    }

    #endregion

    #region Validation

    private static void ValidateMatrix(TensorBase tensor)
    {
        if (tensor.Rank != 2)
        {
            throw new ArgumentException(
                "Tensor must be a matrix (Rank 2).");
        }
    }

    private static void ValidateMatrixMultiply(
        TensorBase a,
        TensorBase b)
    {
        ValidateMatrix(a);
        ValidateMatrix(b);

        if (a.Cols != b.Rows)
        {
            throw new ArgumentException(
                $"Cannot multiply " +
                $"({a.Rows}x{a.Cols}) by " +
                $"({b.Rows}x{b.Cols}).");
        }
    }

    private static void ValidateMatrixMultiplyTransposeRight(
        TensorBase a,
        TensorBase b)
    {
        ValidateMatrix(a);
        ValidateMatrix(b);

        // A (m x k) * B^T (k x n)
        // B therefore has physical shape (n x k).
        if (a.Cols != b.Cols)
        {
            throw new ArgumentException(
                $"Cannot multiply " +
                $"({a.Rows}x{a.Cols}) by " +
                $"({b.Rows}x{b.Cols})^T.");
        }
    }

    private static void ValidateMatrixMultiplyTransposeLeft(
        TensorBase a,
        TensorBase b)
    {
        ValidateMatrix(a);
        ValidateMatrix(b);

        // A^T (m x k) * B (k x n)
        // A therefore has physical shape (k x m).
        if (a.Rows != b.Rows)
        {
            throw new ArgumentException(
                $"Cannot multiply " +
                $"({a.Rows}x{a.Cols})^T by " +
                $"({b.Rows}x{b.Cols}).");
        }
    }

    private static void ValidateResult(
        TensorBase result,
        int rows,
        int cols)
    {
        if (result.Rank != 2 ||
            result.Rows != rows ||
            result.Cols != cols)
        {
            throw new ArgumentException(
                $"Destination tensor has incorrect dimensions. " +
                $"Expected ({rows}x{cols}), " +
                $"got ({result.Rows}x{result.Cols}).");
        }
    }

    private static void ValidateTransposeBuffer(
        TensorBase buffer,
        TensorBase source)
    {
        if (buffer.Rank != 2 ||
            buffer.Rows != source.Cols ||
            buffer.Cols != source.Rows)
        {
            throw new ArgumentException(
                $"Transpose buffer has incorrect dimensions. " +
                $"Expected ({source.Cols}x{source.Rows}).");
        }
    }

    #endregion
}