namespace SimpleTransformer.Model.Extensions
{
    public static class TensorMath
    {
        #region Math


        #region Element-wise matrix-scalar math operations
        public static Tensor ElementWiseAddScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] + scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseSubtractScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] - scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseMultiplyScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] * scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseDivideScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if (scalar == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    result[i, j] = a[i, j] / scalar;
                }
            }
            return result;
        }


        #endregion

        #region Element-wise matrix-matrix math operations
        //Tensor math ops: Multiply, Add, Subtract

        public static Tensor ElementWiseAdd (Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);
            int rows = a.Rows;
            int cols = a.Cols;
            //Perform an element-wise addition on the two matrices
            var result = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return result;
        }

        //Element-wise subtraction of two matrices
        public static Tensor ElementWiseSubtract(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);
            int rows = a.Rows;
            int cols = a.Cols;
            //Perform an element-wise subtraction on the two matrices
            var result = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] - b[i, j];
                }
            }
            return result;
        }

        public static Tensor ElementWiseMultiply(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);
            int rows = a.Rows;
            int cols = a.Cols;
            //Perform an element-wise multiplication on the two matrices
            var result = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }
            return result;
        }
        //Element-wise divide of two matrices
        public static Tensor ElementWiseDivide(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);

            int rows = a.Rows;
            int cols = a.Cols;
            //Perform an element-wise division on the two matrices
            var result = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if(b[i, j] == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    result[i, j] = a[i, j] / b[i, j]; 
                }
            }
            return result;
        }

        //Performs element-wise addition on two inputs and stores the result in the third input
        public static void ElementWiseAddInto(
            Tensor a,
            Tensor b,
            Tensor result)
        {
            TensorUtilities.ValidateSameShape(a, b);
            TensorUtilities.ValidateSameShape(a, result);

            float[] A = a.Data;
            float[] B = b.Data;
            float[] C = result.Data;

            int length = C.Length;

            for (int i = 0; i < length; i++)
            {
                C[i] = A[i] + B[i];
            }
        }

        public static void ElementWiseSubtractInto(Tensor a, Tensor b, Tensor result)
        {
            TensorUtilities.ValidateSameShape(a, b);
            TensorUtilities.ValidateSameShape(a, result);

            float[] A = a.Data;
            float[] B = b.Data;
            float[] C = result.Data;

            int length = C.Length;

            for (int i = 0; i < length; i++)
            {
                C[i] = A[i] - B[i];
            }
        }

        public static void ElementWiseMultiplyInto(Tensor a, Tensor b, Tensor result)
        {
            TensorUtilities.ValidateSameShape(a, b);
            TensorUtilities.ValidateSameShape(a, result);

            float[] A = a.Data;
            float[] B = b.Data;
            float[] C = result.Data;

            int length = C.Length;

            for (int i = 0; i < length; i++)
            {
                C[i] = A[i] * B[i];
            }
        }

        public static void ElementWiseDivideInto(Tensor a, Tensor b, Tensor result)
        {
            TensorUtilities.ValidateSameShape(a, b);
            TensorUtilities.ValidateSameShape(a, result);

            float[] A = a.Data;
            float[] B = b.Data;
            float[] C = result.Data;

            int length = C.Length;

            for (int i = 0; i < length; i++)
            {
                C[i] = A[i] / B[i];
            }
        }
                

        //Enhancement: Added in-place versions of the above methods. 
        //These will become profoundly useful when I need to perform in-place mutations.
        public static void ElementWiseAddInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            int rows = a.Rows;
            int cols = a.Cols;
            int layers = a.Layers;
            switch(a.Rank)
            {
                case 1:
                    for (int i = 0; i < rows; i++)
                    {
                        a[i] += b[i];
                    }
                    break;
                case 2:
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            a[i, j] += b[i, j];
                        }
                    }
                    break;
                case 3:
                    for (int layer = 0; layer < layers; layer++)
                    {
                        for (int row = 0; row < rows; row++)
                        {
                            for (int col = 0; col < cols; col++)
                            {
                                a[layer, row, col] += b[layer, row, col];
                            }
                        }
                    }
                    break;
                default:
                    throw new ArgumentException("Tensor rank not supported.");
            }
        }

        public static void ElementWiseSubtractInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            int rows = a.Rows;
            int cols = a.Cols;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    a[i, j] -= b[i, j];
                }
            }
        }

        public static void ElementWiseMultiplyInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            int rows = a.Rows;
            int cols = a.Cols;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    a[i, j] *= b[i, j];
                }
            }
        }

        public static void ElementWiseDivideInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);
            int rows = a.Rows;
            int cols = a.Cols;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (b[i, j] == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    a[i, j] /= b[i, j];
                }
            }
        }     

        //Helpers for matrix multiplication, softmax and transpose
        #endregion

        #region Matrix Multiplication
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            var result = new Tensor(a.Rows, b.Cols);
            MatrixMultiplyInto(a, b, result);
            return result;
        }

        public static void MatrixMultiplyInto(Tensor a, Tensor b, Tensor result)
        {
            if (a.Rank != 2)
                throw new ArgumentException("Left operand must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Right operand must be a matrix.");

            if (a.Cols != b.Rows)
                throw new ArgumentException(
                    $"Cannot multiply ({a.Rows}x{a.Cols}) by ({b.Rows}x{b.Cols}).");

            if (result.Rank != 2 ||
                result.Rows != a.Rows ||
                result.Cols != b.Cols)
            {
                throw new ArgumentException(
                    "Destination tensor has incorrect dimensions.");
            }    

            float[] A = a.Data;
            float[] B = b.Data;
            float[] C = result.Data;

            //Cache the input dimensions locally.
            int m = a.Rows;
            int n = b.Cols;
            int k = a.Cols;    

            for (int i = 0; i < m; i++)
            {
                int aRow = i * k;
                int cRow = i * n;

                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    int bIndex = j;

                    for (int p = 0; p < k; p++)
                    {
                        sum += A[aRow + p] * B[bIndex];
                        bIndex += n;
                    }

                    C[cRow + j] = sum;
                }
            }
        }

        //Transpose then multiply against an input matrix. 
        public static Tensor MultiplyTransposeRight(Tensor a, Tensor b)
        {
            return MatrixMultiply(a, TensorUtilities.Transpose(b));
        }

        public static void MultiplyTransposeRightInto(
            Tensor a,
            Tensor b,
            Tensor transposeBuffer,
            Tensor result)
        {
            TensorUtilities.TransposeInto(b, transposeBuffer);
            MatrixMultiplyInto(a, transposeBuffer, result);
        }
                
        public static Tensor MultiplyTransposeLeft(Tensor a, Tensor b)
        {
            return MatrixMultiply(TensorUtilities.Transpose(a), b);
        }

        public static void MultiplyTransposeLeftInto(
            Tensor a,
            Tensor b,
            Tensor transposeBuffer,
            Tensor result)
        {
            TensorUtilities.TransposeInto(a, transposeBuffer);
            MatrixMultiplyInto(transposeBuffer, b, result);
        }

        //Works against a cached right transposed matrix - this does not perform the transpose.
        public static void MatrixMultiplyWithRightTransposed(
            Tensor a,
            Tensor transposedB,
            Tensor result)
        {
            MatrixMultiplyInto(a, transposedB, result);
        }
        //Works against a cached left transposed matrix - this does not perform the transpose.
        public static void MatrixMultiplyWithLeftTransposed(
            Tensor transposedA,
            Tensor b,
            Tensor result)
        {
            MatrixMultiplyInto(transposedA, b, result);
        }

        #endregion

        #region Special functions
        public static Tensor Gelu(Tensor src)
        {
            var result = src.Clone();
            GeluInPlace(result);
            return result;
        }

        public static void GeluInPlace(Tensor tensor)
        {
            const float sqrt2OverPi = 0.7978845608f;

            float[] data = tensor.Data;
            int length = data.Length;

            for (int i = 0; i < length; i++)
            {
                float x = data[i];

                // Saturate extreme inputs (see TensorMathSimd.GeluInto).
                if (float.IsNaN(x)) { data[i] = float.NaN; continue; }
                if (x >= 12f) { data[i] = x; continue; }
                if (x <= -12f) { data[i] = 0f; continue; }

                float x2 = x * x;
                float x3 = x2 * x;

                float u =
                    sqrt2OverPi *
                    (x + 0.044715f * x3);

                float t = MathF.Tanh(u);

                data[i] =
                    0.5f * x * (1f + t);
            }
        }

        public static void GeluBackwardInto(
            Tensor input,
            Tensor outputGradient,
            Tensor inputGradient)
        {
            TensorUtilities.ValidateSameShape(input, outputGradient);
            TensorUtilities.ValidateSameShape(input, inputGradient);

            const float sqrt2OverPi = 0.7978845608f;

            float[] x = input.Data;
            float[] dy = outputGradient.Data;
            float[] dx = inputGradient.Data;

            int length = x.Length;

            for (int i = 0; i < length; i++)
            {
                float value = x[i];

                float x2 = value * value;
                float x3 = x2 * value;

                float u =
                    sqrt2OverPi *
                    (value + 0.044715f * x3);

                float t = MathF.Tanh(u);

                float derivative =
                    0.5f * (1f + t)
                    +
                    0.5f * value
                    * (1f - t * t)
                    * sqrt2OverPi
                    * (1f + 3f * 0.044715f * x2);

                dx[i] = dy[i] * derivative;
            }
        }

        #endregion

        #region Dot product and cosine similarity

        public static float Dot(Tensor a, Tensor b)
        {
            if (a.Rank != 1)
                throw new ArgumentException("First tensor must be a vector.");

            if (b.Rank != 1)
                throw new ArgumentException("Second tensor must be a vector."); 
            
            return Dot(a.Data, b.Data);
        }

        public static float Dot(Tensor matrixA, int rowA, Tensor matrixB, int rowB)
        {
            if (matrixA.Rank != 2)
                throw new ArgumentException("First tensor must be a matrix.");

            if (matrixB.Rank != 2)
                throw new ArgumentException("Second tensor must be a matrix.");

            if (matrixA.Cols != matrixB.Cols)
                throw new ArgumentException("Row lengths do not match.");

            if (rowA < 0 || rowA >= matrixA.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowA));

            if (rowB < 0 || rowB >= matrixB.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowB));            
            return Dot(RowUtilities.GetRow(matrixA, rowA), RowUtilities.GetRow(matrixB, rowB));
        }

        public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vector lengths do not match.");
            float sum = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }
        #endregion
        //Scale or multiply a tensor by a scalar value and return a new tensor
        public static Tensor Scale(Tensor src, float scalar)
        {       
            var t = src.Clone();     
            ScaleInPlace(t, scalar);
            return t;   
        }

        //Scale or multiply a tensor by a scalar value in place
        public static void ScaleInPlace(Tensor tensor, float scalar)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.Data[i] *= scalar;
            }
        }        
    }
    #endregion
}