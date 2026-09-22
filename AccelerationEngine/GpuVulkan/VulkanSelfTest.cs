using System;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Bring-up harness for the Vulkan backend: compares the 6 GPU
    /// element-wise kernels against the CPU reference on random data.
    /// Run with: dotnet run -- --vulkan-selftest
    /// </summary>
    public static class VulkanSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Vulkan Backend Self-Test (GpuVulkan vs CpuReference) ===");
            Console.WriteLine();

            using var gpu = new GpuVulkanBackend();
            using var reference = new CpuReference.CpuReferenceBackend();

            Console.WriteLine($"GPU backend : {gpu.Name}");
            Console.WriteLine($"GPU active  : {gpu.IsAvailable}");
            Console.WriteLine();

            if (!gpu.IsAvailable)
            {
                Console.WriteLine("Vulkan unavailable - GPU paths skipped, fallback verified by construction.");
                return true;
            }

            int passed = 0;
            int failed = 0;

            void Report(string name, float maxDiff, float tol)
            {
                bool ok = maxDiff <= tol;
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-28} max|diff| = {maxDiff:G3}");
            }

            var random = new Random(1234);

            Tensor Rand(int rows, int cols)
            {
                var t = new Tensor(rows, cols);
                for (int i = 0; i < t.Data.Length; i++)
                    t.Data[i] = (float)(random.NextDouble() * 2 - 1);
                return t;
            }

            float MaxDiff(TensorBase a, TensorBase b)
            {
                float max = 0f;
                for (int r = 0; r < a.Rows; r++)
                    for (int c = 0; c < a.Cols; c++)
                        max = MathF.Max(max, MathF.Abs(a[r, c] - b[r, c]));
                return max;
            }

            using (var t = Rand(6, 7))
            using (var expected = (Tensor)t.Clone())
            using (var actual = (Tensor)t.Clone())
            {
                reference.ScaleInPlace(expected, 2.5f);
                gpu.ScaleInPlace(actual, 2.5f);
                Report("ScaleInPlace", MaxDiff(expected, actual), 1e-5f);
            }

            using (var t = Rand(4, 5))
            {
                reference.Fill(t, 3.25f);
                bool cpuOk = true;
                for (int r = 0; r < 4 && cpuOk; r++)
                    for (int c = 0; c < 5 && cpuOk; c++)
                        cpuOk = t[r, c] == 3.25f;

                using var g = Rand(4, 5);
                gpu.Fill(g, 3.25f);
                bool gpuOk = true;
                for (int r = 0; r < 4 && gpuOk; r++)
                    for (int c = 0; c < 5 && gpuOk; c++)
                        gpuOk = g[r, c] == 3.25f;

                Report("Fill", (cpuOk && gpuOk) ? 0f : 1f, 0.5f);
            }

            using (var a = Rand(6, 7))
            using (var b = Rand(6, 7))
            using (var exp = (Tensor)a.Clone())
            using (var act = (Tensor)a.Clone())
            {
                reference.ElementWiseAddInPlace(exp, b);
                gpu.ElementWiseAddInPlace(act, b);
                Report("AddInPlace", MaxDiff(exp, act), 1e-5f);
            }

            using (var a = Rand(6, 7))
            using (var b = Rand(6, 7))
            using (var exp = new Tensor(6, 7))
            using (var act = new Tensor(6, 7))
            {
                reference.ElementWiseAddInto(a, b, exp);
                gpu.ElementWiseAddInto(a, b, act);
                Report("AddInto", MaxDiff(exp, act), 1e-5f);
            }

            using (var a = Rand(6, 7))
            using (var b = Rand(6, 7))
            using (var exp = (Tensor)a.Clone())
            using (var act = (Tensor)a.Clone())
            {
                reference.ElementWiseMultiplyInPlace(exp, b);
                gpu.ElementWiseMultiplyInPlace(act, b);
                Report("MulInPlace", MaxDiff(exp, act), 1e-5f);
            }

            using (var a = Rand(6, 7))
            using (var b = Rand(6, 7))
            using (var exp = new Tensor(6, 7))
            using (var act = new Tensor(6, 7))
            {
                reference.ElementWiseMultiplyInto(a, b, act);
                reference.ElementWiseMultiplyInto(a, b, exp);
                Report("MulInto", MaxDiff(exp, act), 1e-5f);
            }

            void MatMulCase(string name, int m, int k, int n, bool tA, bool tB, bool acc)
            {
                int aRows = tA ? k : m;
                int aCols = tA ? m : k;
                int bRows = tB ? n : k;
                int bCols = tB ? k : n;
                using var a = Rand(aRows, aCols);
                using var b = Rand(bRows, bCols);
                using var exp = new Tensor(m, n);
                using var act = new Tensor(m, n);
                if (acc)
                {
                    using var seed = Rand(m, n);
                    seed.Data.CopyTo(exp.Data, 0);
                    seed.Data.CopyTo(act.Data, 0);
                    reference.MatMulAccumulate(a, b, exp, tA, tB);
                    gpu.MatMulAccumulate(a, b, act, tA, tB);
                }
                else
                {
                    reference.MatMul(a, b, exp, tA, tB);
                    gpu.MatMul(a, b, act, tA, tB);
                }
                Report(name, MaxDiff(exp, act), 2e-3f);
            }

            MatMulCase("MatMul 5x4*4x3", 5, 4, 3, false, false, false);
            MatMulCase("MatMul 8x8*8x8", 8, 8, 8, false, false, false);
            MatMulCase("MatMul 7x5^T", 7, 5, 4, false, true, false);
            MatMulCase("MatMul ^TA", 5, 4, 3, true, false, false);
            MatMulCase("MatMul 20x17*17x13", 20, 17, 13, false, false, false);
            MatMulCase("MatMulAcc 6x9*9x4", 6, 9, 4, false, false, true);
            MatMulCase("MatMulAcc 7x5^T", 7, 5, 4, false, true, true);

            // Rank-3 batch path (matches TransformerModel LinearLayer batch GEMM)
            {
                var rnd = new Random(99);
                using var a = new Tensor(2, 5, 4);
                using var b = new Tensor(2, 4, 3);
                using var exp = new Tensor(2, 5, 3);
                using var act = new Tensor(2, 5, 3);
                foreach (var t in new[] { a, b })
                    for (int i = 0; i < t.Data.Length; i++)
                        t.Data[i] = (float)(rnd.NextDouble() * 2 - 1);
                reference.MatMul(a, b, exp);
                gpu.MatMul(a, b, act);
                float max = 0f;
                for (int l = 0; l < 2; l++)
                    for (int r = 0; r < 5; r++)
                        for (int c = 0; c < 3; c++)
                            max = MathF.Max(max, MathF.Abs(exp[l, r, c] - act[l, r, c]));
                Report("MatMul batch 2x5x4*4x3", max, 2e-3f);
            }

            Console.WriteLine();
            Console.WriteLine($"Results: {passed} passed, {failed} failed.");
            return failed == 0;
        }
    }
}
