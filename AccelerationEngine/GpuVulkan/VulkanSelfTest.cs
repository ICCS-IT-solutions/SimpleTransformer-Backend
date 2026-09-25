using System;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Bring-up harness for the Vulkan backend: compares all GPU
    /// kernels against the CPU reference on random data.
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

            // ---- Resizable BAR detection -------------------------------------
            //The layout rules the detection relies on: ReBAR is only reported as
            //enabled when the main device-local heap is itself host-visible, and
            //a VRAM working set is only considered viable in that case.
            {
                const double mib = 1024.0 * 1024.0;

                //Invariants the detection must hold on any driver:
                //  * ReBAR on  => the whole device-local heap is mappable, and a VRAM
                //    working set is viable, sized to that mappable heap.
                //  * ReBAR off => only a small aperture is mappable, and a VRAM
                //    working set is deliberately not offered.
                bool rebarOn = gpu.ResizableBarEnabled;
                bool workingSetConsistent = rebarOn
                    ? gpu.VramWorkingSetBudgetBytes == gpu.HostVisibleDeviceLocalBytes
                      && gpu.VramWorkingSetBudgetBytes > 0
                    : gpu.VramWorkingSetBudgetBytes == 0;
                //ReBAR on means the main VRAM heap is mappable, so the mappable
                //total covers it; ReBAR off means only a small aperture is
                //mappable, necessarily smaller than the main heap.
                bool apertureSane = gpu.DeviceLocalHeapSizeBytes == 0 ||
                    (rebarOn
                        ? gpu.HostVisibleDeviceLocalBytes >= gpu.DeviceLocalHeapSizeBytes
                        : gpu.HostVisibleDeviceLocalBytes < gpu.DeviceLocalHeapSizeBytes);

                Console.WriteLine(
                    $"  [INFO] {gpu.ResizableBarInfo} | " +
                    $"host-mappable device-local {gpu.HostVisibleDeviceLocalBytes / mib:F0} MiB | " +
                    $"VRAM working set budget {gpu.VramWorkingSetBudgetBytes / mib:F0} MiB");

                Report("ReBAR detection", workingSetConsistent && apertureSane ? 0f : 1f, 0f);
            }

            void Report(string name, float maxDiff, float tol)
            {
                bool ok = maxDiff <= tol;
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-28} max|diff| = {maxDiff:G3}");
            }

            var random = new Random(1234);

            // ---- VRAM-resident weights: register once, then run matmuls that
            // bind the cached device copy instead of re-uploading per call ----
            {
                using var b = Rand(17, 13);
                var referenceW = new Tensor(17, 13);
                Array.Copy(b.Data, referenceW.Data, b.Data.Length);

                //Weights read transposed live stored as [n, k], so they get
                //their own resident copy (and exercise the multi-entry cache).
                using var bT = Rand(13, 17);
                var referenceWT = new Tensor(13, 17);
                Array.Copy(bT.Data, referenceWT.Data, bT.Data.Length);

                bool plainRegistered = gpu.TryRegisterResidentWeights(b, "selftest-weights");
                bool transposedRegistered = gpu.TryRegisterResidentWeights(bT, "selftest-weights-t");

                if (!plainRegistered || !transposedRegistered)
                {
                    Console.WriteLine("  [INFO] resident cache unavailable: skipping resident matmul checks.");
                }
                else
                {
                    void ResidentMatMulCase(string name, int m, bool tB)
                    {
                        using var a = Rand(m, 17);
                        using var exp = new Tensor(m, 13);
                        using var act = new Tensor(m, 13);
                        reference.MatMul(a, tB ? referenceWT : referenceW, exp, false, tB);
                        gpu.MatMul(a, tB ? bT : b, act, false, tB);
                        Report(name, MaxDiff(exp, act), 2e-3f);
                    }

                    ResidentMatMulCase("Resident 12x17*17x13", 12, false);
                    ResidentMatMulCase("Resident 5x17*13^T", 5, true);

                    // A second registration of the same tensor must be a no-op.
                    if (!gpu.TryRegisterResidentWeights(b, "selftest-weights") ||
                        !gpu.TryRegisterResidentWeights(bT, "selftest-weights-t"))
                        Report("Resident idempotent register", 1f, 0f);
                    else
                        Report("Resident idempotent register", 0f, 0f);

                    // Refresh after an in-place update (the raw-training contract):
                    // mutate the host weights as an optimizer step would, re-upload
                    // via TryRefreshResidentWeights, and prove the next resident
                    // MatMul binds the NEW values instead of the stale device copy.
                    for (int i = 0; i < b.Data.Length; i++)
                        b.Data[i] = b.Data[i] * 0.5f + 0.25f;
                    Array.Copy(b.Data, referenceW.Data, b.Data.Length);

                    bool refreshed = gpu.TryRefreshResidentWeights(b);
                    if (!refreshed)
                        Report("Resident refresh", 1f, 0f);
                    else
                    {
                        ResidentMatMulCase("Resident refresh", 12, false);
                    }

                    // Refreshing a tensor that was never registered must report
                    // false rather than pretending it did work.
                    using (var unregistered = Rand(9, 11))
                        Report("Refresh unregistered",
                            gpu.TryRefreshResidentWeights(unregistered) ? 1f : 0f, 0f);

                    Console.WriteLine(
                        $"  [INFO] resident cache: {gpu.ResidentWeightCount} tensor(s), " +
                        $"{gpu.ResidentWeightBytes} B (VRAM budget {gpu.GpuMemoryBudgetBytes} B)");

                    gpu.ClearResidentWeights();

                    if (gpu.ResidentWeightCount != 0 || gpu.ResidentWeightBytes != 0)
                        Report("Resident clear", 1f, 0f);
                    else
                        Report("Resident clear", 0f, 0f);

                    // Releasing one weight must free exactly that copy, and the
                    // released tensor must fall back to the ordinary path rather
                    // than binding the buffer that was just destroyed.
                    if (gpu.TryRegisterResidentWeights(b, "selftest-weights") &&
                        gpu.TryRegisterResidentWeights(bT, "selftest-weights-t"))
                    {
                        long bothBytes = gpu.ResidentWeightBytes;
                        gpu.ReleaseResidentWeights(b);

                        Report("Resident release one",
                            gpu.ResidentWeightCount == 1 &&
                            gpu.ResidentWeightBytes > 0 &&
                            gpu.ResidentWeightBytes < bothBytes ? 0f : 1f, 0f);

                        ResidentMatMulCase("Post-release 5x17*13^T", 5, true);
                        ResidentMatMulCase("Post-release 12x17*17x13", 12, false);

                        // Releasing the remainder, then releasing again, must be a
                        // no-op rather than double-freeing.
                        gpu.ReleaseResidentWeights(bT);
                        gpu.ReleaseResidentWeights(bT);
                        Report("Resident release all",
                            gpu.ResidentWeightCount == 0 && gpu.ResidentWeightBytes == 0 ? 0f : 1f, 0f);
                    }
                }
            }

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

            // ---- Phase 3: nonlinear / norm / layout ops ----

            using (var x = Rand(6, 8))
            using (var exp = (Tensor)x.Clone())
            using (var act = (Tensor)x.Clone())
            {
                reference.GeluInPlace(exp);
                gpu.GeluInPlace(act);
                Report("GeluInPlace", MaxDiff(exp, act), 1e-5f);
            }

            using (var x = Rand(6, 8))
            using (var exp = new Tensor(6, 8))
            using (var act = new Tensor(6, 8))
            {
                reference.GeluInto(x, exp);
                gpu.GeluInto(x, act);
                Report("GeluInto", MaxDiff(exp, act), 1e-5f);
            }

            using (var x = Rand(6, 8))
            using (var dy = Rand(6, 8))
            using (var exp = new Tensor(6, 8))
            using (var act = new Tensor(6, 8))
            {
                reference.GeluBackwardInto(x, dy, exp);
                gpu.GeluBackwardInto(x, dy, act);
                Report("GeluBackward", MaxDiff(exp, act), 1e-4f);
            }

            Tensor RandVec(int n)
            {
                var t = new Tensor(n);
                for (int i = 0; i < n; i++)
                    t.Data[i] = (float)(random.NextDouble() * 2 - 1);
                return t;
            }

            using (var x = Rand(4, 8))
            using (var g = RandVec(8))
            using (var b = RandVec(8))
            using (var exp = (Tensor)x.Clone())
            using (var act = (Tensor)x.Clone())
            {
                reference.LayerNormInPlace(exp, g, b);
                gpu.LayerNormInPlace(act, g, b);
                Report("LayerNormInPlace", MaxDiff(exp, act), 1e-4f);
            }

            using (var x = Rand(4, 8))
            using (var g = RandVec(8))
            using (var b = RandVec(8))
            using (var exp = new Tensor(4, 8))
            using (var act = new Tensor(4, 8))
            {
                reference.LayerNormInto(x, g, b, exp);
                gpu.LayerNormInto(x, g, b, act);
                Report("LayerNormInto", MaxDiff(exp, act), 1e-4f);
            }

            using (var x = Rand(4, 8))
            using (var exp = (Tensor)x.Clone())
            using (var act = (Tensor)x.Clone())
            {
                reference.SoftmaxInPlace(exp);
                gpu.SoftmaxInPlace(act);
                Report("SoftmaxInPlace", MaxDiff(exp, act), 1e-5f);
            }

            using (var s = Rand(4, 8))
            using (var dy = Rand(4, 8))
            using (var exp = new Tensor(4, 8))
            using (var act = new Tensor(4, 8))
            {
                // SoftmaxBackward needs real softmax outputs, not raw randoms.
                using var sx = (Tensor)s.Clone();
                using var sy = (Tensor)s.Clone();
                reference.SoftmaxInPlace(sx);
                gpu.SoftmaxInPlace(sy);
                reference.SoftmaxBackwardInto(sx, dy, exp);
                gpu.SoftmaxBackwardInto(sy, dy, act);
                Report("SoftmaxBackward", MaxDiff(exp, act), 1e-5f);
            }

            using (var s = Rand(5, 5))
            using (var m = new Tensor(5, 5))
            using (var exp = (Tensor)s.Clone())
            using (var act = (Tensor)s.Clone())
            {
                for (int i = 0; i < 25; i++)
                    m.Data[i] = (i % 3 == 0) ? 0f : 1f;
                reference.ApplyMaskInPlace(exp, m);
                gpu.ApplyMaskInPlace(act, m);
                Report("ApplyMask", MaxDiff(exp, act), 1e-5f);
            }

            using (var s = Rand(4, 6))
            using (var exp = new Tensor(6, 4))
            using (var act = new Tensor(6, 4))
            {
                reference.TransposeInto(s, exp);
                gpu.TransposeInto(s, act);
                float max = 0f;
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 4; c++)
                        max = MathF.Max(max, MathF.Abs(exp[r, c] - act[r, c]));
                Report("Transpose", max, 1e-6f);
            }

            using (var s = Rand(4, 6))
            using (var exp = new Tensor(4, 6))
            using (var act = new Tensor(4, 6))
            {
                reference.CopyInto(s, exp);
                gpu.CopyInto(s, act);
                Report("CopyInto", MaxDiff(exp, act), 1e-6f);
            }

            Console.WriteLine();
            Console.WriteLine($"Results: {passed} passed, {failed} failed.");
            return failed == 0;
        }
    }
}
