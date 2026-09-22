using System;
using System.Diagnostics;
using SimpleTransformer.AccelerationEngine.CpuSimd;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Phase 4 harness. Two jobs:
    /// <list type="number">
    /// <item>latency of each <see cref="IAccelerationBackend"/> op on transformer-shaped
    /// workloads, GPU against the SIMD CPU backend (the speed baseline),</item>
    /// <item>proof of the Phase 4 reuse work: after warm-up the same op mix creates
    /// zero new Vulkan buffers and zero new descriptor sets.</item>
    /// </list>
    /// Numerical parity is checked against the exact CPU reference backend, not the
    /// SIMD backend, because the SIMD GELU/softmax kernels deliberately trade
    /// accuracy for speed (approximate tanh/exp) and would otherwise be reported as
    /// GPU mismatches.
    ///
    /// Run with: dotnet run -c Release -- --vulkan-bench
    /// </summary>
    public static class VulkanBenchmark
    {
        private static readonly Random _random = new(7);
        private static int _passed;
        private static int _failed;

        /// <summary>Element-wise / layout ops: GPU must match the reference to float precision.</summary>
        private const float ExactTolerance = 1e-6f;

        /// <summary>K-dimension summation order differs between the tiled kernels.</summary>
        private const float MatMulTolerance = 2e-3f;

        /// <summary>GPU <c>exp</c>/<c>inversesqrt</c>/<c>tanh</c> vs CPU <c>MathF</c> differ by a few ULP.</summary>
        private const float NonlinearTolerance = 1e-4f;

        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Vulkan Backend Benchmark (GpuVulkan vs CpuSimd) ===");
            Console.WriteLine();

            using var gpu = new GpuVulkanBackend();
            using var cpu = new CpuSimdBackend();
            using var reference = new SimpleTransformer.AccelerationEngine.CpuReference.CpuReferenceBackend();

            Console.WriteLine($"GPU backend : {gpu.Name}");
            Console.WriteLine($"CPU backend : {cpu.Name}   (timing baseline)");
            Console.WriteLine($"Parity vs   : {reference.Name}   (exact spec)");
            Console.WriteLine($"Working memory tier : {gpu.WorkingMemoryTier}");
            Console.WriteLine();

            if (!gpu.IsAvailable)
            {
                Console.WriteLine("Vulkan unavailable - nothing to benchmark.");
                return true;
            }

            Console.WriteLine("  op                                  cpu ms      gpu ms   speedup   throughput");
            Console.WriteLine("  --------------------------------------------------------------------------------");

            BenchMatMul(cpu, gpu, reference, "MatMul 128x256 * 256x128", 128, 256, 128);
            BenchMatMul(cpu, gpu, reference, "MatMul 128x512 * 512x256", 128, 512, 256);
            BenchMatMul(cpu, gpu, reference, "MatMul 128x256^T * 256x256", 128, 256, 256, transposeA: true);
            BenchMatMul(cpu, gpu, reference, "MatMul 256x256 * 256x256^T", 256, 256, 256, transposeB: true);
            BenchMatMul(cpu, gpu, reference, "MatMul batch4 256x256 * 256x256", 512, 256, 256, batch: 4);
            BenchMatMul(cpu, gpu, reference, "MatMul 512x1024 * 1024x1024", 512, 1024, 1024);

            BenchRows(cpu, gpu, reference, "LayerNormInPlace 512x512", 512, 512, RowOp.LayerNorm, NonlinearTolerance);
            BenchRows(cpu, gpu, reference, "GeluInPlace 512x512", 512, 512, RowOp.Gelu, NonlinearTolerance);
            BenchRows(cpu, gpu, reference, "SoftmaxInPlace 512x512", 512, 512, RowOp.Softmax, NonlinearTolerance);
            BenchRows(cpu, gpu, reference, "ScaleInPlace 512x512", 512, 512, RowOp.Scale, ExactTolerance);
            BenchElementWise(cpu, gpu, reference, "AddInto 512x512", 512, 512);
            BenchTranspose(cpu, gpu, reference, "TransposeInto 512x512", 512, 512);
            BenchMask(cpu, gpu, reference, "ApplyMaskInPlace 128x128", 128, 128);

            Console.WriteLine();

            // Warm-up must cover the exact op mix being measured, including the
            // small gamma/beta vectors (256 floats) LayerNorm needs. Otherwise
            // the first measured batch pays first-touch bucket creation and the
            // steady-state proof fails on pool misses, not on reuse bugs.
            // Two batches: the pool's LIFO stacks alternate physical buffer
            // handles between batches, so the pool state (and therefore the
            // descriptor-set cache keys) only converges after the second one.
            RunMixedBatch(cpu, gpu);
            RunMixedBatch(cpu, gpu);

            long buffersBefore = GpuVulkanBackend.BuffersCreated;
            long dispatchesBefore = gpu.DispatchCount;
            int setsBefore = gpu.CachedDescriptorSetCount;

            for (int i = 0; i < 3; i++)
                RunMixedBatch(cpu, gpu);

            long buffersDelta = GpuVulkanBackend.BuffersCreated - buffersBefore;
            long dispatchDelta = gpu.DispatchCount - dispatchesBefore;
            int setsDelta = gpu.CachedDescriptorSetCount - setsBefore;

            Console.WriteLine("Steady-state allocation (3 mixed batches after warm-up):");
            Console.WriteLine($"  dispatches         : {dispatchDelta}");
            Console.WriteLine($"  buffers created    : {buffersDelta}    (0 = bucket pool fully reused)");
            Console.WriteLine($"  new descriptor sets: {setsDelta}    (0 = descriptor cache working)");
            Console.WriteLine($"  pool recycles      : {gpu.DescriptorPoolRecycles}");
            Console.WriteLine();
            Console.WriteLine("Lifetime counters:");
            Console.WriteLine($"  dispatches         : {gpu.DispatchCount}");
            Console.WriteLine($"  pooled buffers     : {gpu.PooledBufferCount}");
            Console.WriteLine($"  cached sets        : {gpu.CachedDescriptorSetCount}");
            Console.WriteLine($"  host -> device     : {GpuVulkanBackend.UploadedBytes / 1024.0 / 1024.0:F2} MiB");
            Console.WriteLine($"  device -> host     : {GpuVulkanBackend.DownloadedBytes / 1024.0 / 1024.0:F2} MiB");
            Console.WriteLine();

            bool allocOk = buffersDelta == 0 && setsDelta == 0;
            _failed += allocOk ? 0 : 1;
            _passed += allocOk ? 1 : 0;
            Console.WriteLine(allocOk
                ? "  [PASS] steady-state state reuse"
                : "  [FAIL] steady-state state reuse (per-op Vulkan objects created)");
            Console.WriteLine();
            Console.WriteLine($"Results: {_passed} passed, {_failed} failed.");
            return _failed == 0;
        }

        // -------------------------------------------------------------------
        // Workloads
        // -------------------------------------------------------------------

        /// <summary>Representative mix of the ops a transformer step issues.</summary>
        private static void RunMixedBatch(IAccelerationBackend cpu, IAccelerationBackend gpu)
        {
            foreach (var backend in new[] { cpu, gpu })
            {
                using var a = Rand(128, 256);
                using var b = Rand(256, 128);
                using var r = new Tensor(128, 128);
                backend.MatMul(a, b, r);

                using var x = Rand(256, 256);
                using var gamma = RandVec(256);
                using var beta = RandVec(256);
                backend.LayerNormInPlace(x, gamma, beta);
                backend.GeluInPlace(x);
                backend.ScaleInPlace(x, 1.0001f);

                using var scores = Rand(128, 128);
                using var mask = new Tensor(128, 128);
                for (int i = 0; i < mask.Data.Length; i++)
                    mask.Data[i] = (i % 3 == 0) ? 0f : 1f;
                backend.ApplyMaskInPlace(scores, mask);
            }
        }

        private static void BenchMatMul(
            IAccelerationBackend cpu,
            IAccelerationBackend gpu,
            IAccelerationBackend reference,
            string name,
            int m,
            int k,
            int n,
            bool transposeA = false,
            bool transposeB = false,
            int batch = 1)
        {
            double flops = 2.0 * m * n * k * batch;

            using var a = Rand(transposeA ? k : m, transposeA ? m : k);
            using var b = Rand(transposeB ? n : k, transposeB ? k : n);
            using var refR = new Tensor(m, n);
            using var cpuR = new Tensor(m, n);
            using var gpuR = new Tensor(m, n);

            reference.MatMul(a, b, refR, transposeA, transposeB);
            cpu.MatMul(a, b, cpuR, transposeA, transposeB);
            gpu.MatMul(a, b, gpuR, transposeA, transposeB);

            // Parity is GPU-vs-reference; the SIMD figure is reported for context
            // so a divergence in either CPU path stays visible.
            float diff = MaxDiff(refR, gpuR);
            float simdDiff = MaxDiff(refR, cpuR);

            int iterations = Calibrate(() => cpu.MatMul(a, b, cpuR, transposeA, transposeB));
            double cpuMs = Median(iterations, () => cpu.MatMul(a, b, cpuR, transposeA, transposeB));
            double gpuMs = Median(iterations, () => gpu.MatMul(a, b, gpuR, transposeA, transposeB));

            Report(name, cpuMs, gpuMs, diff, MatMulTolerance, flops, simdDiff);
        }

        private enum RowOp { LayerNorm, Gelu, Softmax, Scale }

        private static void BenchRows(
            IAccelerationBackend cpu,
            IAccelerationBackend gpu,
            IAccelerationBackend reference,
            string name,
            int rows,
            int cols,
            RowOp op,
            float tolerance)
        {
            using var cpuT = Rand(rows, cols);
            // Same values on every side: clone, never re-draw from the RNG.
            using var gpuT = (Tensor)cpuT.Clone();
            using var refT = (Tensor)cpuT.Clone();
            using var gamma = RandVec(cols);
            using var beta = RandVec(cols);
            using var cpuSeed = (Tensor)cpuT.Clone();
            using var gpuSeed = (Tensor)gpuT.Clone();

            void Apply(IAccelerationBackend backend, TensorBase t)
            {
                switch (op)
                {
                    case RowOp.LayerNorm: backend.LayerNormInPlace(t, gamma, beta); break;
                    case RowOp.Gelu: backend.GeluInPlace(t); break;
                    case RowOp.Softmax: backend.SoftmaxInPlace(t); break;
                    default: backend.ScaleInPlace(t, 1.0001f); break;
                }
            }

            Apply(reference, refT);
            Apply(cpu, cpuT);
            Apply(gpu, gpuT);
            float diff = MaxDiff(refT, gpuT);
            float simdDiff = MaxDiff(refT, cpuT);

            // All four ops are in-place and non-idempotent (repeated scaling
            // overflows, repeated softmax converges), so each timed iteration
            // is preceded by a raw array restore. The restore-only cost is
            // measured separately and subtracted to leave pure op latency.
            Action cpuReset = () => Array.Copy(cpuSeed.Data, cpuT.Data, cpuT.Data.Length);
            Action gpuReset = () => Array.Copy(gpuSeed.Data, gpuT.Data, gpuT.Data.Length);
            Action cpuOp = () => Apply(cpu, cpuT);
            Action gpuOp = () => Apply(gpu, gpuT);

            int iterations = Calibrate(() => { cpuReset(); cpuOp(); });
            double cpuMs = TimeOpWithReset(iterations, cpuOp, cpuReset);
            double gpuMs = TimeOpWithReset(iterations, gpuOp, gpuReset);

            Report(name, cpuMs, gpuMs, diff, tolerance, 0, simdDiff);
        }

        private static void BenchElementWise(
            IAccelerationBackend cpu,
            IAccelerationBackend gpu,
            IAccelerationBackend reference,
            string name,
            int rows,
            int cols)
        {
            using var src = Rand(rows, cols);
            using var refR = new Tensor(rows, cols);
            using var cpuR = new Tensor(rows, cols);
            using var gpuR = new Tensor(rows, cols);

            reference.ElementWiseAddInto(src, src, refR);
            cpu.ElementWiseAddInto(src, src, cpuR);
            gpu.ElementWiseAddInto(src, src, gpuR);
            float diff = MaxDiff(refR, gpuR);
            float simdDiff = MaxDiff(refR, cpuR);

            int iterations = Calibrate(() => cpu.ElementWiseAddInto(src, src, cpuR));
            double cpuMs = Median(iterations, () => cpu.ElementWiseAddInto(src, src, cpuR));
            double gpuMs = Median(iterations, () => gpu.ElementWiseAddInto(src, src, gpuR));

            Report(name, cpuMs, gpuMs, diff, ExactTolerance, 0, simdDiff);
        }

        private static void BenchTranspose(
            IAccelerationBackend cpu,
            IAccelerationBackend gpu,
            IAccelerationBackend reference,
            string name,
            int rows,
            int cols)
        {
            using var src = Rand(rows, cols);
            using var refD = new Tensor(cols, rows);
            using var cpuD = new Tensor(cols, rows);
            using var gpuD = new Tensor(cols, rows);

            reference.TransposeInto(src, refD);
            cpu.TransposeInto(src, cpuD);
            gpu.TransposeInto(src, gpuD);
            float diff = MaxDiff(refD, gpuD);
            float simdDiff = MaxDiff(refD, cpuD);

            int iterations = Calibrate(() => cpu.TransposeInto(src, cpuD));
            double cpuMs = Median(iterations, () => cpu.TransposeInto(src, cpuD));
            double gpuMs = Median(iterations, () => gpu.TransposeInto(src, gpuD));

            Report(name, cpuMs, gpuMs, diff, ExactTolerance, 0, simdDiff);
        }

        private static void BenchMask(
            IAccelerationBackend cpu,
            IAccelerationBackend gpu,
            IAccelerationBackend reference,
            string name,
            int rows,
            int cols)
        {
            using var src = Rand(rows, cols);
            using var cpuS = (Tensor)src.Clone();
            using var gpuS = (Tensor)src.Clone();
            using var refS = (Tensor)src.Clone();
            using var mask = new Tensor(rows, cols);
            for (int i = 0; i < mask.Data.Length; i++)
                mask.Data[i] = (i % 3 == 0) ? 0f : 1f;

            reference.ApplyMaskInPlace(refS, mask);
            cpu.ApplyMaskInPlace(cpuS, mask);
            gpu.ApplyMaskInPlace(gpuS, mask);
            float diff = MaxDiff(refS, gpuS);
            float simdDiff = MaxDiff(refS, cpuS);

            // Masking is in-place; re-seed both sides each iteration.
            Action cpuReset = () => { for (int i = 0; i < cpuS.Data.Length; i++) cpuS.Data[i] = 0.5f; };
            Action gpuReset = () => { for (int i = 0; i < gpuS.Data.Length; i++) gpuS.Data[i] = 0.5f; };

            int iterations = Calibrate(() => { cpuReset(); cpu.ApplyMaskInPlace(cpuS, mask); });
            double cpuMs = TimeOpWithReset(iterations, () => cpu.ApplyMaskInPlace(cpuS, mask), cpuReset);
            double gpuMs = TimeOpWithReset(iterations, () => gpu.ApplyMaskInPlace(gpuS, mask), gpuReset);

            Report(name, cpuMs, gpuMs, diff, ExactTolerance, 0, simdDiff);
        }

        // -------------------------------------------------------------------
        // Timing helpers
        // -------------------------------------------------------------------

        /// <summary>
        /// Runs <paramref name="action"/> once and picks a repetition count that
        /// keeps the slowest (CPU) measurement brief but still resolvable.
        /// </summary>
        private static int Calibrate(Action action)
        {
            var sw = Stopwatch.StartNew();
            action();
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;

            if (ms > 50) return 3;
            if (ms > 10) return 10;
            if (ms > 1) return 30;
            return 100;
        }

        /// <summary>
        /// Median wall-clock time of <paramref name="iterations"/> runs, in
        /// milliseconds. Median (not mean) so a stray OS scheduling hiccup or a
        /// driver-side flush does not skew the result.
        /// </summary>
        private static double Median(int iterations, Action action)
        {
            if (iterations < 1)
                iterations = 1;

            var samples = new double[iterations];
            var sw = new Stopwatch();
            for (int i = 0; i < iterations; i++)
            {
                sw.Restart();
                action();
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(samples);
            return samples[samples.Length / 2];
        }

        /// <summary>
        /// Latency of a destructive in-place op, with the cost of re-seeding the
        /// input measured separately and subtracted. Reporting the raw
        /// "reset + op" figure would overstate every backend, and would bias the
        /// comparison because the reset is a constant added to both sides.
        /// </summary>
        private static double TimeOpWithReset(int iterations, Action op, Action reset)
        {
            double withReset = Median(iterations, () => { reset(); op(); });
            double resetOnly = Median(iterations, () => reset());
            return Math.Max(0, withReset - resetOnly);
        }

        /// <summary>
        /// Prints one benchmark row and gates GPU correctness.
        /// <para>
        /// <paramref name="maxDiff"/> is GPU-vs-exact-reference and is gated against
        /// <paramref name="tolerance"/> - that is the correctness claim for the
        /// Vulkan kernels. <paramref name="simdDiff"/> (SIMD-vs-reference) is reported
        /// for context only: the SIMD GELU/softmax kernels intentionally use
        /// approximate tanh/exp, so their deviation from the exact reference is a
        /// known, accepted trade-off rather than a GPU regression.
        /// </para>
        /// </summary>
        private static void Report(
            string name,
            double cpuMs,
            double gpuMs,
            float maxDiff,
            float tolerance,
            double flops,
            float simdDiff)
        {
            bool parityOk = float.IsFinite(maxDiff) && maxDiff <= tolerance;
            _passed += parityOk ? 1 : 0;
            _failed += parityOk ? 0 : 1;

            double speedup = gpuMs > 0 ? cpuMs / gpuMs : double.PositiveInfinity;
            string throughput = flops > 0
                ? $"{flops / (gpuMs * 1e6),7:F2} GFLOP/s"
                : $"{cpuMs / (gpuMs > 0 ? gpuMs : 1),7:F2}x";

            Console.WriteLine(
                $"  {name,-34} {cpuMs,8:F3} {gpuMs,10:F3}   {speedup,6:F2}x   {throughput}   " +
                $"gpu|diff| {maxDiff:G3} (tol {tolerance:G3})  simd|diff| {simdDiff:G3}" +
                (parityOk ? "" : "  <-- MISMATCH"));
        }

        // -------------------------------------------------------------------
        // Reference data + comparison
        // -------------------------------------------------------------------

        private static Tensor Rand(int rows, int cols)
        {
            var t = new Tensor(rows, cols);
            for (int i = 0; i < t.Data.Length; i++)
                t.Data[i] = (float)(_random.NextDouble() * 2 - 1);
            return t;
        }

        private static Tensor RandVec(int cols)
        {
            var t = new Tensor(cols);
            for (int i = 0; i < t.Data.Length; i++)
                t.Data[i] = (float)(_random.NextDouble() * 2 - 1);
            return t;
        }

        /// <summary>Row-major element-wise difference between two 2D tensors.</summary>
        private static float MaxDiff(TensorBase a, TensorBase b)
        {
            float max = 0f;
            for (int r = 0; r < a.Rows; r++)
                for (int c = 0; c < a.Cols; c++)
                    max = MathF.Max(max, MathF.Abs(a[r, c] - b[r, c]));
            return max;
        }
    }
}
