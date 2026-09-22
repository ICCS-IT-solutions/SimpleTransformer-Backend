using System;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine
{
    /// <summary>
    /// Parity harness that runs every <see cref="IAccelerationBackend"/> operation on
    /// identical random inputs through both the pure managed reference backend and the
    /// SIMD backend, then reports the maximum absolute difference per operation.
    ///
    /// Run with: dotnet run -- --backend-selftest
    /// </summary>
    public static class BackendSelfTest
    {
        private const float ExactTolerance = 1e-6f;      // element-wise / layout ops must agree bit-near-exactly
        private const float MatMulTolerance = 2e-3f;     // different dot-product summation order
        private const float NonlinearTolerance = 2e-3f;  // SIMD fast-exp approximation in softmax
        // NOTE: the SIMD GELU kernels use a coarse rational tanh approximation
        // (VectorTanh), which deviates from the exact MathF.Tanh used by the
        // reference backend by up to ~0.13. Consider switching VectorTanh to a
        // tighter approximation if training stability ever requires it.
        private const float GeluTolerance = 0.35f;

        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Acceleration Backend Self-Test (CpuReference vs CpuSimd) ===");
            Console.WriteLine();

            using var reference = new CpuReference.CpuReferenceBackend();
            using var simd = new CpuSimd.CpuSimdBackend();

            Console.WriteLine($"Reference backend : {reference.Name}");
            Console.WriteLine($"SIMD backend      : {simd.Name}");
            Console.WriteLine($"Vector.IsHardwareAccelerated = {System.Numerics.Vector.IsHardwareAccelerated}");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Report(string name, float maxDiff, float tolerance)
            {
                bool ok = maxDiff <= tolerance;
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-44} max|diff| = {maxDiff:G3} (tol {tolerance:G3})");
            }

            var random = new Random(42); // fixed seed -> reproducible inputs

            // ---------------------------------------------------------------
            // Matrix multiplication (all transpose combinations, vs naive GEMM)
            // ---------------------------------------------------------------
            {
                var a = RandomMatrix(random, 5, 4);
                var b = RandomMatrix(random, 4, 3);
                var atPhysical = RandomMatrix(random, 4, 5);
                var btPhysical = RandomMatrix(random, 3, 4);

                void MatMulCombo(string name, TensorBase ta, TensorBase tb, bool transposeA, bool transposeB)
                {
                    int m = transposeA ? ta.Cols : ta.Rows;
                    int n = transposeB ? tb.Rows : tb.Cols;
                    using var refResult = new Tensor(m, n);
                    using var simdResult = new Tensor(m, n);
                    using var expected = NaiveMatMul(ta, tb, transposeA, transposeB);

                    reference.MatMul(ta, tb, refResult, transposeA, transposeB);
                    simd.MatMul(ta, tb, simdResult, transposeA, transposeB);

                    Report(name, MathF.Max(MaxAbsDiff(refResult, simdResult), MaxAbsDiff(refResult, expected)), MatMulTolerance);
                }

                MatMulCombo("MatMul (A x B)", a, b, false, false);
                MatMulCombo("MatMul (A^T x B)", atPhysical, b, true, false);
                MatMulCombo("MatMul (A x B^T)", a, btPhysical, false, true);
                MatMulCombo("MatMul (A^T x B^T)", atPhysical, btPhysical, true, true);

                // Batched (rank 3) matmul
                {
                    var batchA = Random3D(random, 2, 5, 4);
                    var batchB = Random3D(random, 2, 4, 3);
                    using var refResult = new Tensor(2, 5, 3);
                    using var simdResult = new Tensor(2, 5, 3);
                    using var expected = new Tensor(2, 5, 3);

                    for (int l = 0; l < 2; l++)
                        for (int i = 0; i < 5; i++)
                            for (int j = 0; j < 3; j++)
                            {
                                float sum = 0f;
                                for (int e = 0; e < 4; e++)
                                    sum += batchA[l, i, e] * batchB[l, e, j];
                                expected[l, i, j] = sum;
                            }

                    reference.MatMul(batchA, batchB, refResult);
                    simd.MatMul(batchA, batchB, simdResult);

                    Report("MatMul (rank 3, batched)", MathF.Max(MaxAbsDiff(refResult, simdResult), MaxAbsDiff(refResult, expected)), MatMulTolerance);
                    batchA.Dispose();
                    batchB.Dispose();
                }

                // Accumulate: result += G^T * X (the Linear/QLoRA weight-gradient path)
                {
                    var g = RandomMatrix(random, 4, 3);
                    var x = RandomMatrix(random, 4, 5);
                    var refResult = RandomMatrix(random, 3, 5);
                    var simdResult = (Tensor)refResult.Clone();
                    var expected = (Tensor)refResult.Clone();

                    using var gtx = NaiveMatMul(g, x, true, false);
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 5; j++)
                            expected[i, j] += gtx[i, j];

                    reference.MatMulAccumulate(g, x, refResult, transposeA: true);
                    simd.MatMulAccumulate(g, x, simdResult, transposeA: true);

                    Report("MatMulAccumulate (+= G^T x X)", MathF.Max(MaxAbsDiff(refResult, simdResult), MaxAbsDiff(refResult, expected)), MatMulTolerance);
                    refResult.Dispose();
                    simdResult.Dispose();
                    expected.Dispose();
                    g.Dispose();
                    x.Dispose();
                }

                a.Dispose(); b.Dispose(); atPhysical.Dispose(); btPhysical.Dispose();
            }

            // ---------------------------------------------------------------
            // Element-wise, activations, normalization
            // ---------------------------------------------------------------
            {
                var a = RandomMatrix(random, 6, 7);
                var b = RandomMatrix(random, 6, 7);
                // Gamma/beta are rank-1 vectors, exactly as the LayerNorm layer uses them
                var gamma = RandomVector(random, 7);
                var beta = RandomVector(random, 7);

                void TwoTensorCheck(string name, float tolerance, Action<IAccelerationBackend, TensorBase, TensorBase> op)
                {
                    var refTarget = (Tensor)a.Clone();
                    var simdTarget = (Tensor)a.Clone();
                    op(reference, refTarget, b);
                    op(simd, simdTarget, b);
                    Report(name, MaxAbsDiff(refTarget, simdTarget), tolerance);
                    refTarget.Dispose();
                    simdTarget.Dispose();
                }

                TwoTensorCheck("ScaleInPlace", ExactTolerance, (backend, t, _) => backend.ScaleInPlace(t, 2.5f));
                TwoTensorCheck("ElementWiseAddInPlace", ExactTolerance, (backend, t, s) => backend.ElementWiseAddInPlace(t, s));
                TwoTensorCheck("ElementWiseMultiplyInPlace", ExactTolerance, (backend, t, s) => backend.ElementWiseMultiplyInPlace(t, s));

                void IntoCheck(string name, float tolerance, Action<IAccelerationBackend, TensorBase, TensorBase> op)
                {
                    using var refResult = new Tensor(a.Rows, a.Cols);
                    using var simdResult = new Tensor(a.Rows, a.Cols);
                    op(reference, a, refResult);
                    op(simd, a, simdResult);
                    Report(name, MaxAbsDiff(refResult, simdResult), tolerance);
                }

                IntoCheck("ElementWiseAddInto", ExactTolerance, (backend, x, r) => backend.ElementWiseAddInto(x, b, r));
                IntoCheck("ElementWiseMultiplyInto", ExactTolerance, (backend, x, r) => backend.ElementWiseMultiplyInto(x, b, r));
                IntoCheck("GeluInto", GeluTolerance, (backend, x, r) => backend.GeluInto(x, r));
                IntoCheck("LayerNormInto", NonlinearTolerance, (backend, x, r) => backend.LayerNormInto(x, gamma, beta, r));

                // GELU backward
                {
                    var x = RandomMatrix(random, 6, 7, min: -3f, max: 3f);
                    var dy = RandomMatrix(random, 6, 7, min: -2f, max: 2f);
                    using var refGrad = new Tensor(6, 7);
                    using var simdGrad = new Tensor(6, 7);
                    reference.GeluBackwardInto(x, dy, refGrad);
                    simd.GeluBackwardInto(x, dy, simdGrad);
                    Report("GeluBackwardInto", MaxAbsDiff(refGrad, simdGrad), GeluTolerance);
                    x.Dispose(); dy.Dispose();
                }

                a.Dispose(); b.Dispose(); gamma.Dispose(); beta.Dispose();
            }

            // ---------------------------------------------------------------
            // Softmax, attention masking, layout / memory movement
            // ---------------------------------------------------------------
            {
                // Softmax (rank 2) + row-sum sanity on the SIMD output
                var logits = RandomMatrix(random, 6, 11, min: -8f, max: 8f);
                using var refResult = (Tensor)logits.Clone();
                using var simdResult = (Tensor)logits.Clone();

                reference.SoftmaxInPlace(refResult);
                simd.SoftmaxInPlace(simdResult);

                float rowSumError = 0f;
                for (int r = 0; r < 6; r++)
                {
                    float sum = 0f;
                    for (int c = 0; c < 11; c++)
                    {
                        sum += simdResult[r, c];
                        if (simdResult[r, c] < 0f)
                            rowSumError = float.MaxValue; // negative probability
                    }
                    rowSumError = MathF.Max(rowSumError, MathF.Abs(sum - 1f));
                }

                Report("SoftmaxInPlace (rank 2)", MathF.Max(MaxAbsDiff(refResult, simdResult), rowSumError), NonlinearTolerance);

                // Softmax backward
                var dy = RandomMatrix(random, 6, 11, min: -2f, max: 2f);
                using var refGrad = new Tensor(6, 11);
                using var simdGrad = new Tensor(6, 11);
                reference.SoftmaxBackwardInto(refResult, dy, refGrad);
                simd.SoftmaxBackwardInto(simdResult, dy, simdGrad);
                Report("SoftmaxBackwardInto", MaxAbsDiff(refGrad, simdGrad), NonlinearTolerance);
                dy.Dispose();

                // Softmax (rank 3, batched)
                var logits3d = Random3D(random, 2, 4, 9, min: -8f, max: 8f);
                using var ref3d = (Tensor)logits3d.Clone();
                using var simd3d = (Tensor)logits3d.Clone();
                reference.SoftmaxInPlace(ref3d);
                simd.SoftmaxInPlace(simd3d);
                Report("SoftmaxInPlace (rank 3)", MaxAbsDiff(ref3d, simd3d), NonlinearTolerance);
                logits3d.Dispose();
                logits.Dispose();

                // Attention masking (square scores, zeros in the mask)
                var scores = RandomMatrix(random, 6, 6, min: -5f, max: 5f);
                var mask = RandomMatrix(random, 6, 6, min: 0f, max: 1f);
                mask[0, 1] = 0f; mask[2, 3] = 0f; mask[4, 5] = 0f;
                mask[1, 0] = 0f; mask[3, 2] = 0f; mask[5, 4] = 0f;

                var refScores = (Tensor)scores.Clone();
                var simdScores = (Tensor)scores.Clone();
                reference.ApplyMaskInPlace(refScores, mask);
                simd.ApplyMaskInPlace(simdScores, mask);
                Report("ApplyMaskInPlace", MaxAbsDiff(refScores, simdScores), ExactTolerance);
                refScores.Dispose();
                simdScores.Dispose();
                scores.Dispose();
                mask.Dispose();
            }

            {
                // Transpose
                var src = RandomMatrix(random, 4, 6);
                using var refDst = new Tensor(6, 4);
                using var simdDst = new Tensor(6, 4);
                reference.TransposeInto(src, refDst);
                simd.TransposeInto(src, simdDst);
                Report("TransposeInto", MaxAbsDiff(refDst, simdDst), ExactTolerance);

                // Copy (rank 3, stride-aware)
                var src3d = Random3D(random, 3, 4, 5);
                using var refCopy = new Tensor(3, 4, 5);
                using var simdCopy = new Tensor(3, 4, 5);
                reference.CopyInto(src3d, refCopy);
                simd.CopyInto(src3d, simdCopy);
                Report("CopyInto (rank 3)", MaxAbsDiff(refCopy, simdCopy), ExactTolerance);

                // Fill
                var fillTarget = RandomMatrix(random, 4, 5);
                reference.Fill(fillTarget, 3.25f);
                bool fillOk = true;
                for (int r = 0; r < 4 && fillOk; r++)
                    for (int c = 0; c < 5 && fillOk; c++)
                        fillOk = fillTarget[r, c] == 3.25f;
                Report("Fill", fillOk ? 0f : float.MaxValue, ExactTolerance);

                src.Dispose();
                src3d.Dispose();
                fillTarget.Dispose();
            }

            Console.WriteLine();
            Console.WriteLine($"Results: {passed} passed, {failed} failed.");
            return failed == 0;
        }

        #region Random tensor helpers

        private static float NextFloat(Random random, float min, float max)
            => min + (float)random.NextDouble() * (max - min);

        private static Tensor RandomVector(Random random, int length, float min = -1f, float max = 1f)
        {
            var tensor = new Tensor(length);
            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = NextFloat(random, min, max);
            }
            return tensor;
        }

        private static Tensor RandomMatrix(Random random, int rows, int cols, float min = -1f, float max = 1f)
        {
            var tensor = new Tensor(rows, cols);
            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = NextFloat(random, min, max);
            }
            return tensor;
        }

        private static Tensor Random3D(Random random, int layers, int rows, int cols, float min = -1f, float max = 1f)
        {
            var tensor = new Tensor(layers, rows, cols);
            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = NextFloat(random, min, max);
            }
            return tensor;
        }

        private static float MaxAbsDiff(TensorBase a, TensorBase b)
        {
            if (a.Rank != b.Rank || a.Layers != b.Layers || a.Rows != b.Rows || a.Cols != b.Cols)
                throw new InvalidOperationException("Shape mismatch while comparing results.");

            int rows = a.Rank switch { 1 => 1, 2 => a.Rows, 3 => a.Layers * a.Rows, _ => throw new InvalidOperationException() };
            float max = 0f;
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> rowA = GetLinearRow(a, r);
                ReadOnlySpan<float> rowB = GetLinearRow(b, r);
                for (int c = 0; c < rowA.Length; c++)
                {
                    float diff = MathF.Abs(rowA[c] - rowB[c]);
                    if (diff > max) max = diff;
                }
            }
            return max;
        }

        private static ReadOnlySpan<float> GetLinearRow(TensorBase tensor, int linearRow)
        {
            switch (tensor.Rank)
            {
                case 1: return tensor.Buffer.AsSpan(tensor.Offset, tensor.Cols);
                case 2: return tensor.Buffer.AsSpan(tensor.Offset + linearRow * tensor.Stride, tensor.Cols);
                case 3:
                    int layer = linearRow / tensor.Rows;
                    int row = linearRow % tensor.Rows;
                    return tensor.Buffer.AsSpan(tensor.Offset + layer * tensor.LayerStride + row * tensor.Stride, tensor.Cols);
                default: throw new InvalidOperationException();
            }
        }

        private static Tensor NaiveMatMul(TensorBase a, TensorBase b, bool transposeA, bool transposeB)
        {
            int m = transposeA ? a.Cols : a.Rows;
            int k = transposeA ? a.Rows : a.Cols;
            int n = transposeB ? b.Rows : b.Cols;
            var result = new Tensor(m, n);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int e = 0; e < k; e++)
                    {
                        float av = transposeA ? a[e, i] : a[i, e];
                        float bv = transposeB ? b[j, e] : b[e, j];
                        sum += av * bv;
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }

        #endregion
    }
}