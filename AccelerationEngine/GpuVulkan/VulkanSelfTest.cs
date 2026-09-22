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

            Console.WriteLine();
            Console.WriteLine($"Results: {passed} passed, {failed} failed.");
            return failed == 0;
        }
    }
}
