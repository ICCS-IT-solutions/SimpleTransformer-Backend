using System;
using SimpleTransformer.AccelerationEngine.Common;
using SimpleTransformer.AccelerationEngine.CpuReference;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Vulkan compute backend. Phase 4: buffer pooling, persistent fence,
    /// descriptor-set reuse - zero Vulkan object create/destroy per op.
    /// Tensors are densified on upload (stride-aware pack) so
    /// TensorView inputs work correctly.
    /// </summary>
    public sealed class GpuVulkanBackend : IAccelerationBackend
    {
        private readonly CpuReferenceBackend _fallback = new();
        private readonly VulkanContext? _ctx;
        private readonly VulkanShaderCompiler? _compiler;
        private readonly VulkanKernelLauncher? _launcher;
        private readonly VulkanBufferPool? _pool;

        private bool _disposed;

        private static int _memoryInfoLogged;

        /// <summary>
        /// Detected VRAM facts plus the cap actually in force, e.g.
        /// "VRAM 8192 MiB, driver budget 7900 MiB, used by us 140 MiB, effective cap 7900 MiB".
        /// "n/a" when no Vulkan device was initialised.
        /// </summary>
        public string GpuMemoryInfo { get; private set; } = "n/a";

        /// <summary>
        /// Effective cap in bytes for DEVICE_LOCAL (VRAM) allocations: the config
        /// override when set, else the driver's live budget, else the heap size.
        /// The buffer pool keeps retained VRAM under this.
        /// </summary>
        public ulong GpuMemoryBudgetBytes { get; private set; }

        /// <summary>
        /// Effective cap in bytes for host-visible staging buffers (system RAM):
        /// the config override when set, else a quarter of physical RAM clamped
        /// to [1 GiB, 8 GiB]. The pool keeps retained host buffers under this,
        /// so Vulkan staging can never exhaust system memory.
        /// </summary>
        public ulong HostStagingBudgetBytes { get; private set; }

        /// <summary>Human-readable host-staging budget, e.g. "host staging cap 8192 MiB of 32768 MiB system RAM (auto)".</summary>
        public string HostStagingInfo { get; private set; } = "n/a";

        /// <summary>
        /// Auto policy for the host (system RAM) staging pool: a quarter of
        /// physical RAM clamped to [1 GiB, 8 GiB]. On a 32 GiB machine that is
        /// 8 GiB - a hard ceiling that leaves 24 GiB for the OS, the Vulkan
        /// driver and the GC heap holding the model and optimizer state, so the
        /// unmanaged pool can never exhaust system RAM no matter how long a run
        /// gets or how many distinct tensor sizes it touches.
        /// </summary>
        private static ulong ComputeHostStagingBudget()
        {
            long overrideBytes = VulkanMemorySettings.HostBudgetBytesOverride;
            if (overrideBytes > 0)
                return (ulong)overrideBytes;

            long totalRam = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            if (totalRam <= 0)
                return 1UL << 30;

            return Math.Clamp((ulong)totalRam / 4UL, 1UL << 30, 8UL << 30);
        }

        private static string DescribeHostStaging(ulong budget)
        {
            const double mib = 1024.0 * 1024.0;
            long totalRam = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            bool overridden = VulkanMemorySettings.HostBudgetBytesOverride > 0;
            return $"host staging cap {budget / mib:F0} MiB of {totalRam / mib:F0} MiB system RAM" +
                (overridden ? " (config override)" : " (auto)");
        }

        /// <summary>
        /// Effective VRAM cap for one context: an explicit override is clamped to
        /// what the hardware/driver actually offers; auto uses the live driver
        /// budget when present, otherwise the fixed device-local heap size.
        /// </summary>
        private static ulong ComputeVramBudget(VulkanContext ctx)
        {
            ulong available = ctx.MemoryBudgetSupported && ctx.DeviceLocalBudgetBytes > 0
                ? ctx.DeviceLocalBudgetBytes
                : ctx.DeviceLocalHeapSizeBytes;

            long overrideBytes = VulkanMemorySettings.BudgetBytesOverride;
            if (overrideBytes <= 0)
                return available;

            //The override is a cap, never a licence to exceed the hardware: clamp
            //it to the heap (and to the driver budget when that is tighter).
            ulong cap = Math.Min((ulong)overrideBytes, ctx.DeviceLocalHeapSizeBytes);
            if (ctx.MemoryBudgetSupported && ctx.DeviceLocalBudgetBytes > 0)
                cap = Math.Min(cap, ctx.DeviceLocalBudgetBytes);
            return cap;
        }

        private static string DescribeGpuMemory(VulkanContext ctx, ulong effectiveBudget)
        {
            const double mib = 1024.0 * 1024.0;
            string description = $"VRAM {ctx.DeviceLocalHeapSizeBytes / mib:F0} MiB";

            if (ctx.MemoryBudgetSupported)
            {
                description +=
                    $", driver budget {ctx.DeviceLocalBudgetBytes / mib:F0} MiB" +
                    $", used by us {ctx.DeviceLocalUsageBytes / mib:F0} MiB";
            }
            else
            {
                description += ", live budget unavailable (VK_EXT_memory_budget not enabled)";
            }

            return $"{description}, effective cap {effectiveBudget / mib:F0} MiB" +
                (VulkanMemorySettings.BudgetBytesOverride > 0 ? " (config override)" : " (auto)");
        }

        public GpuVulkanBackend()
        {
            var ctx = new VulkanContext();
            if (!ctx.TryInitialize())
            {
                ctx.Dispose();
                return;
            }

            try
            {
                var compiler = new VulkanShaderCompiler();
                var launcher = new VulkanKernelLauncher(ctx, compiler);
                _ctx = ctx;
                _compiler = compiler;
                _launcher = launcher;

                GpuMemoryBudgetBytes = ComputeVramBudget(ctx);
                GpuMemoryInfo = DescribeGpuMemory(ctx, GpuMemoryBudgetBytes);
                HostStagingBudgetBytes = ComputeHostStagingBudget();
                HostStagingInfo = DescribeHostStaging(HostStagingBudgetBytes);
                _pool = new VulkanBufferPool(ctx, GpuMemoryBudgetBytes, HostStagingBudgetBytes);

                //One line per process so repeated backend probes (the /backends
                //endpoint constructs one) do not spam the log.
                if (Interlocked.Exchange(ref _memoryInfoLogged, 1) == 0)
                    Console.WriteLine(
                        $"[GpuVulkan] {ctx.DeviceName}: {GpuMemoryInfo} | {HostStagingInfo}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GpuVulkan] init failed, CPU fallback: {ex.Message}");
                ctx.Dispose();
            }
        }

        public string Name => _launcher != null
            ? $"GpuVulkan ({_ctx!.DeviceName})"
            : "GpuVulkan (unavailable - CPU reference fallback)";

        public bool IsGpuAccelerated => _launcher != null;

        public bool IsAvailable => _launcher != null;

        /// <summary>Number of pooled Vulkan buffers (perf introspection).</summary>
        public int PooledBufferCount => _pool?.LiveBufferCount ?? 0;

        // ---- Phase 4 telemetry -------------------------------------------------

        /// <summary>Total compute dispatches issued to the GPU.</summary>
        public long DispatchCount => _launcher?.DispatchCount ?? 0;

        /// <summary>Distinct descriptor sets cached (flat growth = reuse working).</summary>
        public int CachedDescriptorSetCount => _launcher?.CachedDescriptorSetCount ?? 0;

        /// <summary>Times the descriptor pool had to be recycled.</summary>
        public long DescriptorPoolRecycles => _launcher?.DescriptorPoolRecycles ?? 0;

        /// <summary>Cumulative host-to-device bytes copied.</summary>
        public static long UploadedBytes => VulkanBuffer.TotalUploadedBytes;

        /// <summary>Cumulative device-to-host bytes copied.</summary>
        public static long DownloadedBytes => VulkanBuffer.TotalDownloadedBytes;

        /// <summary>
        /// Buffers created process-wide. Flat across repeated op batches once
        /// the bucket pool is warm - the Phase 4 allocation-free proof.
        /// </summary>
        public static long BuffersCreated => VulkanBuffer.TotalBuffersCreated;

        /// <summary>
        /// True only if per-op buffers landed in VRAM. Expected to be FALSE:
        /// the measured policy prefers host-cached system RAM because every op
        /// round trips through the host (see VulkanContext.SelectMemoryType).
        /// Kept as a diagnostic.
        /// </summary>
        public bool UsingDeviceLocalMemory => _pool?.AnyDeviceLocal ?? false;

        /// <summary>
        /// Memory tier the per-op staging buffers actually use, e.g.
        /// "type  5: HOST_VISIBLE|HOST_COHERENT|HOST_CACHED". This is the tier
        /// that governs per-op latency, so it is what the benchmarks report.
        /// </summary>
        public string WorkingMemoryTier
        {
            get
            {
                if (_ctx == null || _pool?.WorkingMemoryTypeIndex is not uint index)
                    return "n/a";
                return _ctx.DescribeMemoryType(index);
            }
        }

        /// <summary>
        /// Diagnostic dump of the device heap/memory-type layout plus the memory
        /// type the buffer pool actually gets for transformer-sized buffers, and
        /// a measured host read-back bandwidth for that type.
        /// Run with: dotnet run -c Release -- --vulkan-meminfo
        /// </summary>
        public void LogMemoryInfo()
        {
            if (_ctx == null || _launcher == null)
            {
                Console.WriteLine("Vulkan unavailable - no memory information.");
                return;
            }

            Console.WriteLine($"Device: {_ctx.DeviceName}");
            _ctx.LogMemoryProperties();
            Console.WriteLine($"VRAM budget: {GpuMemoryInfo}");
            Console.WriteLine($"System RAM: {HostStagingInfo}");
            Console.WriteLine(
                $"Effective cap for VRAM allocations: {GpuMemoryBudgetBytes / 1024.0 / 1024.0:F0} MiB " +
                $"(memory_budget_mb={(VulkanMemorySettings.BudgetBytesOverride > 0 ? (VulkanMemorySettings.BudgetBytesOverride / 1024 / 1024).ToString() : "0 = auto")})");
            Console.WriteLine();

            // The per-op cost is dominated by the host<->device round trip, so
            // measure every host-visible memory type the device offers. The
            // winner is what the staging path should allocate from.
            Console.WriteLine("Host-visible memory tiers (1 MiB payload, up+down round trip):");
            uint[] candidates = { 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15 };
            uint typeCount = _ctx.MemoryTypeCount;
            foreach (uint typeIndex in candidates)
            {
                if (typeIndex >= typeCount)
                    continue;
                if (!_ctx.IsHostVisibleMemoryType(typeIndex))
                    continue;

                ProbeTier(typeIndex, _ctx.DescribeMemoryType(typeIndex));
            }

            Console.WriteLine();
            Console.WriteLine("Selected policy tiers (what the backend actually allocates):");
            foreach (int floats in new[] { 16_384, 262_144, 1_048_576 })
            {
                var buffer = RentBuffer(floats);
                try
                {
                    ReportProbe(floats, buffer, "policy");
                }
                finally
                {
                    ReturnBuffer(buffer);
                }
            }
        }

        /// <summary>Measures one forced memory type with a 1 MiB payload.</summary>
        private void ProbeTier(uint typeIndex, string description)
        {
            const int floats = 262_144;
            VulkanBuffer buffer;
            try
            {
                buffer = new VulkanBuffer(_ctx!, (ulong)(floats * 4), false, (int)typeIndex);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  type {typeIndex,2} {description,-40} unavailable ({ex.Message})");
                return;
            }

            try
            {
                ReportProbe(floats, buffer, $"type {typeIndex,2}");
                Console.WriteLine($"                                        ({description})");
            }
            finally
            {
                buffer.Dispose();
            }
        }

        /// <summary>Times repeated upload+download round trips and prints the bandwidth.</summary>
        private static void ReportProbe(int floats, VulkanBuffer buffer, string label)
        {
            int bytes = floats * 4;
            var payload = new float[floats];
            for (int i = 0; i < floats; i++)
                payload[i] = i * 0.5f;
            var readBack = new float[floats];

            buffer.Upload(payload);

            // Best of a few round trips so first-touch page faults and clock
            // ramping do not dominate the reported bandwidth.
            double best = double.MaxValue;
            for (int i = 0; i < 5; i++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                buffer.Upload(payload);
                buffer.Download(readBack);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }

            double mib = bytes / 1024.0 / 1024.0;
            double mbPerSec = best > 0 ? (mib * 2) / (best / 1000.0) : 0;
            Console.WriteLine(
                $"  {label} {bytes / 1024,6} KiB: deviceLocal={buffer.IsDeviceLocal,-5} " +
                $"roundTrip={best,7:F3} ms  {mbPerSec,7:F1} MiB/s (up+down)");
        }

        private static int RowCount(TensorBase t) => t.Rank switch
        {
            1 => 1,
            2 => t.Rows,
            3 => t.Layers * t.Rows,
            _ => throw new ArgumentException($"Rank {t.Rank} unsupported.")
        };

        private static Span<float> Row(TensorBase t, int r)
        {
            switch (t.Rank)
            {
                case 1: return t.Buffer.AsSpan(t.Offset, t.Cols);
                case 2: return t.Buffer.AsSpan(t.Offset + r * t.Stride, t.Cols);
                case 3:
                    int layer = r / t.Rows;
                    int row = r % t.Rows;
                    return t.Buffer.AsSpan(
                        t.Offset + layer * t.LayerStride + row * t.Stride, t.Cols);
                default: throw new ArgumentException($"Rank {t.Rank} unsupported.");
            }
        }

        private static float[] Pack(TensorBase t)
        {
            int rows = RowCount(t);
            var flat = new float[rows * t.Cols];
            for (int r = 0; r < rows; r++)
                Row(t, r).CopyTo(flat.AsSpan(r * t.Cols, t.Cols));
            return flat;
        }

        private static void PackInto(TensorBase t, Span<float> flat)
        {
            int rows = RowCount(t);
            for (int r = 0; r < rows; r++)
                Row(t, r).CopyTo(flat.Slice(r * t.Cols, t.Cols));
        }

        private float[] RentScratch(int count)
        {
            // Small-count scratch comes from a lock-free per-size cache;
            // rented arrays are returned explicitly by callers below.
            return ScratchCache.Rent(count);
        }

        private static void ReturnScratch(float[] array)
        {
            ScratchCache.Return(array);
        }

        private VulkanBuffer RentBuffer(int floatCount)
        {
            return _pool!.Rent(floatCount);
        }

        private void ReturnBuffer(VulkanBuffer buffer)
        {
            _pool!.Return(buffer);
        }

        private static void Unpack(ReadOnlySpan<float> flat, TensorBase t)
        {
            int rows = RowCount(t);
            for (int r = 0; r < rows; r++)
                flat.Slice(r * t.Cols, t.Cols).CopyTo(Row(t, r));
        }

        private void RunUnary(VulkanKernel kernel, TensorBase tensor, float alpha)
        {
            if (_launcher == null)
            {
                if (kernel == VulkanKernel.Scale) _fallback.ScaleInPlace(tensor, alpha);
                else _fallback.Fill(tensor, alpha);
                return;
            }

            int count = RowCount(tensor) * tensor.Cols;
            float[] flat = RentScratch(count);
            var buf = RentBuffer(count);
            try
            {
                PackInto(tensor, flat.AsSpan(0, count));
                buf.Upload(flat.AsSpan(0, count));
                _launcher.Dispatch(kernel, new[] { buf }, (uint)count, alpha);
                buf.Download(flat.AsSpan(0, count));
                Unpack(flat.AsSpan(0, count), tensor);
            }
            finally
            {
                ReturnBuffer(buf);
                ReturnScratch(flat);
            }
        }

        private void RunBinaryInPlace(VulkanKernel kernel, TensorBase target, TensorBase source)
        {
            if (target.Rank != source.Rank || target.Layers != source.Layers ||
                target.Rows != source.Rows || target.Cols != source.Cols)
                throw new ArgumentException(
                    $"RunBinaryInPlace shape mismatch: target {TensorValidation.DescribeShape(target)}" +
                    $" vs source {TensorValidation.DescribeShape(source)}.");

            if (_launcher == null)
            {
                if (kernel == VulkanKernel.AddInPlace) _fallback.ElementWiseAddInPlace(target, source);
                else _fallback.ElementWiseMultiplyInPlace(target, source);
                return;
            }

            int count = RowCount(target) * target.Cols;
            float[] a = RentScratch(count);
            float[] b = RentScratch(count);
            var ba = RentBuffer(count);
            var bb = RentBuffer(count);
            try
            {
                PackInto(target, a.AsSpan(0, count));
                PackInto(source, b.AsSpan(0, count));
                ba.Upload(a.AsSpan(0, count));
                bb.Upload(b.AsSpan(0, count));
                _launcher.Dispatch(kernel, new[] { ba, bb }, (uint)count, 0f);
                ba.Download(a.AsSpan(0, count));
                Unpack(a.AsSpan(0, count), target);
            }
            finally
            {
                ReturnBuffer(ba);
                ReturnBuffer(bb);
                ReturnScratch(a);
                ReturnScratch(b);
            }
        }

        private void RunBinaryInto(VulkanKernel kernel, TensorBase a, TensorBase b, TensorBase result)
        {
            if (a.Rank != b.Rank || a.Layers != b.Layers || a.Rows != b.Rows || a.Cols != b.Cols ||
                a.Rank != result.Rank || a.Layers != result.Layers ||
                a.Rows != result.Rows || a.Cols != result.Cols)
                throw new ArgumentException(
                    $"RunBinaryInto shape mismatch: a {TensorValidation.DescribeShape(a)}," +
                    $" b {TensorValidation.DescribeShape(b)}," +
                    $" result {TensorValidation.DescribeShape(result)}.");

            if (_launcher == null)
            {
                if (kernel == VulkanKernel.AddInto) _fallback.ElementWiseAddInto(a, b, result);
                else _fallback.ElementWiseMultiplyInto(a, b, result);
                return;
            }

            int count = RowCount(a) * a.Cols;
            float[] fa = RentScratch(count);
            float[] fb = RentScratch(count);
            float[] fr = RentScratch(count);
            var ba = RentBuffer(count);
            var bb = RentBuffer(count);
            var br = RentBuffer(count);
            try
            {
                PackInto(a, fa.AsSpan(0, count));
                PackInto(b, fb.AsSpan(0, count));
                ba.Upload(fa.AsSpan(0, count));
                bb.Upload(fb.AsSpan(0, count));
                _launcher.Dispatch(kernel, new[] { ba, bb, br }, (uint)count, 0f);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), result);
            }
            finally
            {
                ReturnBuffer(ba);
                ReturnBuffer(bb);
                ReturnBuffer(br);
                ReturnScratch(fa);
                ReturnScratch(fb);
                ReturnScratch(fr);
            }
        }

        private void RunMatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA, bool transposeB, bool accumulate)
        {
            int batch = a.Rank == 3 ? a.Layers : 1;
            int m = transposeA ? a.Cols : a.Rows;
            int kA = transposeA ? a.Rows : a.Cols;
            int kB = transposeB ? b.Cols : b.Rows;
            int n = transposeB ? b.Rows : b.Cols;
            if (kA != kB)
                throw new ArgumentException($"MatMul inner dims mismatch ({kA} vs {kB}).");
            if (result.Rank == 3)
            {
                if (result.Layers != batch || result.Rows != m || result.Cols != n)
                    throw new ArgumentException("MatMul result shape mismatch.");
            }
            else if (result.Rows != m || result.Cols != n)
            {
                throw new ArgumentException("MatMul result shape mismatch.");
            }

            if (_launcher == null)
            {
                if (accumulate) _fallback.MatMulAccumulate(a, b, result, transposeA, transposeB);
                else _fallback.MatMul(a, b, result, transposeA, transposeB);
                return;
            }

            float[] fa = Pack(a);
            float[] fb = Pack(b);
            int k = kA;
            int countA = RowCount(a) * a.Cols;
            int countB = RowCount(b) * b.Cols;
            int countR = RowCount(result) * result.Cols;
            if (countA < fa.Length || countB < fb.Length)
                throw new ArgumentException("MatMul operand pack size mismatch.");
            if (countR != batch * m * n)
                throw new ArgumentException("MatMul result pack size mismatch.");

            // Phase 4: pooled scratch + pooled buffers, no per-op allocation.
            float[] fr = RentScratch(countR);
            var ba = RentBuffer(countA);
            var bb = RentBuffer(countB);
            var br = RentBuffer(countR);
            try
            {
                // PackInto handles strided/views; accumulate needs the current
                // result contents on the device (the kernel reads r[ri]).
                PackInto(a, fa.AsSpan(0, countA));
                PackInto(b, fb.AsSpan(0, countB));
                ba.Upload(fa.AsSpan(0, countA));
                bb.Upload(fb.AsSpan(0, countB));
                if (accumulate)
                {
                    PackInto(result, fr.AsSpan(0, countR));
                    br.Upload(fr.AsSpan(0, countR));
                }

                _launcher.DispatchMatMul(
                    accumulate ? VulkanKernel.MatMulAccumulate : VulkanKernel.MatMul,
                    ba, bb, br, (uint)m, (uint)n, (uint)k, (uint)batch,
                    transposeA, transposeB, accumulate);

                br.Download(fr.AsSpan(0, countR));
                Unpack(fr.AsSpan(0, countR), result);
            }
            finally
            {
                ReturnBuffer(ba);
                ReturnBuffer(bb);
                ReturnBuffer(br);
                ReturnScratch(fa);
                ReturnScratch(fb);
                ReturnScratch(fr);
            }
        }

        public void ScaleInPlace(TensorBase tensor, float scalar)
            => RunUnary(VulkanKernel.Scale, tensor, scalar);

        public void Fill(TensorBase tensor, float value)
            => RunUnary(VulkanKernel.Fill, tensor, value);

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
            => RunBinaryInPlace(VulkanKernel.AddInPlace, target, source);

        public void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
            => RunBinaryInto(VulkanKernel.AddInto, a, b, result);

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
            => RunBinaryInPlace(VulkanKernel.MulInPlace, target, source);

        public void ElementWiseMultiplyInto(TensorBase a, TensorBase b, TensorBase result)
            => RunBinaryInto(VulkanKernel.MulInto, a, b, result);

        public void GeluInPlace(TensorBase tensor)
        {
            if (_launcher == null) { _fallback.GeluInPlace(tensor); return; }
            int count = RowCount(tensor) * tensor.Cols;
            float[] f = RentScratch(count);
            var b = RentBuffer(count);
            try
            {
                PackInto(tensor, f.AsSpan(0, count));
                b.Upload(f.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.GeluInPlace, new[] { b }, (uint)RowCount(tensor), (uint)tensor.Cols);
                b.Download(f.AsSpan(0, count));
                Unpack(f.AsSpan(0, count), tensor);
            }
            finally
            {
                ReturnBuffer(b);
                ReturnScratch(f);
            }
        }

        public void GeluInto(TensorBase input, TensorBase result)
        {
            ValidateSame(input, result, "GeluInto");
            if (_launcher == null) { _fallback.GeluInto(input, result); return; }
            int count = RowCount(input) * input.Cols;
            float[] fa = RentScratch(count);
            float[] fr = RentScratch(count);
            var ba = RentBuffer(count);
            var br = RentBuffer(count);
            try
            {
                PackInto(input, fa.AsSpan(0, count));
                ba.Upload(fa.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.GeluInto, new[] { ba, br }, (uint)RowCount(input), (uint)input.Cols);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), result);
            }
            finally
            {
                ReturnBuffer(ba);
                ReturnBuffer(br);
                ReturnScratch(fa);
                ReturnScratch(fr);
            }
        }

        public void GeluBackwardInto(TensorBase input, TensorBase outputGradient, TensorBase inputGradient)
        {
            ValidateSame(input, outputGradient, "GeluBackwardInto");
            ValidateSame(input, inputGradient, "GeluBackwardInto");
            if (_launcher == null) { _fallback.GeluBackwardInto(input, outputGradient, inputGradient); return; }
            int count = RowCount(input) * input.Cols;
            float[] fx = RentScratch(count);
            float[] fy = RentScratch(count);
            float[] fr = RentScratch(count);
            var bx = RentBuffer(count);
            var by = RentBuffer(count);
            var br = RentBuffer(count);
            try
            {
                PackInto(input, fx.AsSpan(0, count));
                PackInto(outputGradient, fy.AsSpan(0, count));
                bx.Upload(fx.AsSpan(0, count));
                by.Upload(fy.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.GeluBackward, new[] { bx, by, br }, (uint)RowCount(input), (uint)input.Cols);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), inputGradient);
            }
            finally
            {
                ReturnBuffer(bx);
                ReturnBuffer(by);
                ReturnBuffer(br);
                ReturnScratch(fx);
                ReturnScratch(fy);
                ReturnScratch(fr);
            }
        }

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f)
        {
            ValidateNorm(tensor, gamma, beta, "LayerNormInPlace");
            if (_launcher == null) { _fallback.LayerNormInPlace(tensor, gamma, beta, epsilon); return; }
            int count = RowCount(tensor) * tensor.Cols;
            float[] f = RentScratch(count);
            float[] g = Pack1D(gamma);
            float[] bb = Pack1D(beta);
            var bx = RentBuffer(count);
            var bg = RentBuffer(g.Length);
            var bbb = RentBuffer(bb.Length);
            try
            {
                PackInto(tensor, f.AsSpan(0, count));
                bx.Upload(f.AsSpan(0, count));
                bg.Upload(g);
                bbb.Upload(bb);
                _launcher.DispatchRowwise(VulkanKernel.LayerNormInPlace, new[] { bx, bg, bbb }, (uint)RowCount(tensor), (uint)tensor.Cols, epsilon);
                bx.Download(f.AsSpan(0, count));
                Unpack(f.AsSpan(0, count), tensor);
            }
            finally
            {
                ReturnBuffer(bx);
                ReturnBuffer(bg);
                ReturnBuffer(bbb);
                ReturnScratch(f);
            }
        }

        public void LayerNormInto(TensorBase input, TensorBase gamma, TensorBase beta, TensorBase result, float epsilon = 1e-5f)
        {
            ValidateSame(input, result, "LayerNormInto");
            ValidateNorm(input, gamma, beta, "LayerNormInto");
            if (_launcher == null) { _fallback.LayerNormInto(input, gamma, beta, result, epsilon); return; }
            int count = RowCount(input) * input.Cols;
            if (RowCount(result) * result.Cols != count)
                throw new ArgumentException("LayerNormInto: shape mismatch.");
            float[] fa = RentScratch(count);
            float[] fr = RentScratch(count);
            float[] g = Pack1D(gamma);
            float[] bb = Pack1D(beta);
            var ba = RentBuffer(count);
            var bg = RentBuffer(g.Length);
            var bbb = RentBuffer(bb.Length);
            var br = RentBuffer(count);
            try
            {
                PackInto(input, fa.AsSpan(0, count));
                ba.Upload(fa.AsSpan(0, count));
                bg.Upload(g);
                bbb.Upload(bb);
                _launcher.DispatchRowwise(VulkanKernel.LayerNormInto, new[] { ba, bg, bbb, br }, (uint)RowCount(input), (uint)input.Cols, epsilon);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), result);
            }
            finally
            {
                ReturnBuffer(ba);
                ReturnBuffer(bg);
                ReturnBuffer(bbb);
                ReturnBuffer(br);
                ReturnScratch(fa);
                ReturnScratch(fr);
            }
        }

        public void SoftmaxInPlace(TensorBase tensor)
        {
            if (_launcher == null) { _fallback.SoftmaxInPlace(tensor); return; }
            int count = RowCount(tensor) * tensor.Cols;
            float[] f = RentScratch(count);
            var b = RentBuffer(count);
            try
            {
                PackInto(tensor, f.AsSpan(0, count));
                b.Upload(f.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.SoftmaxInPlace, new[] { b }, (uint)RowCount(tensor), (uint)tensor.Cols);
                b.Download(f.AsSpan(0, count));
                Unpack(f.AsSpan(0, count), tensor);
            }
            finally
            {
                ReturnBuffer(b);
                ReturnScratch(f);
            }
        }

        public void SoftmaxBackwardInto(TensorBase softmaxOutput, TensorBase outputGradient, TensorBase inputGradient)
        {
            ValidateSame(softmaxOutput, outputGradient, "SoftmaxBackwardInto");
            ValidateSame(softmaxOutput, inputGradient, "SoftmaxBackwardInto");
            if (_launcher == null) { _fallback.SoftmaxBackwardInto(softmaxOutput, outputGradient, inputGradient); return; }
            int count = RowCount(softmaxOutput) * softmaxOutput.Cols;
            float[] fs = RentScratch(count);
            float[] fy = RentScratch(count);
            float[] fr = RentScratch(count);
            var bs = RentBuffer(count);
            var by = RentBuffer(count);
            var br = RentBuffer(count);
            try
            {
                PackInto(softmaxOutput, fs.AsSpan(0, count));
                PackInto(outputGradient, fy.AsSpan(0, count));
                bs.Upload(fs.AsSpan(0, count));
                by.Upload(fy.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.SoftmaxBackward, new[] { bs, by, br }, (uint)RowCount(softmaxOutput), (uint)softmaxOutput.Cols);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), inputGradient);
            }
            finally
            {
                ReturnBuffer(bs);
                ReturnBuffer(by);
                ReturnBuffer(br);
                ReturnScratch(fs);
                ReturnScratch(fy);
                ReturnScratch(fr);
            }
        }

        public void MatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => RunMatMul(a, b, result, transposeA, transposeB, false);

        public void MatMulAccumulate(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => RunMatMul(a, b, result, transposeA, transposeB, true);

        public void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
        {
            if (scores.Rank != 2 || mask.Rank != 2 || scores.Rows != mask.Rows || scores.Cols != mask.Cols)
                throw new ArgumentException("Mask shape mismatch.");
            if (_launcher == null) { _fallback.ApplyMaskInPlace(scores, mask); return; }
            int count = scores.Rows * scores.Cols;
            float[] fs = RentScratch(count);
            float[] fm = RentScratch(count);
            var bs = RentBuffer(count);
            var bm = RentBuffer(count);
            try
            {
                PackInto(scores, fs.AsSpan(0, count));
                PackInto(mask, fm.AsSpan(0, count));
                bs.Upload(fs.AsSpan(0, count));
                bm.Upload(fm.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.ApplyMask, new[] { bs, bm }, (uint)count, 1);
                bs.Download(fs.AsSpan(0, count));
                Unpack(fs.AsSpan(0, count), scores);
            }
            finally
            {
                ReturnBuffer(bs);
                ReturnBuffer(bm);
                ReturnScratch(fs);
                ReturnScratch(fm);
            }
        }

        public void TransposeInto(TensorBase source, TensorBase destination)
        {
            if (source.Rank != 2 || destination.Rank != 2 ||
                destination.Rows != source.Cols || destination.Cols != source.Rows)
                throw new ArgumentException("Transpose shape mismatch.");
            if (_launcher == null) { _fallback.TransposeInto(source, destination); return; }
            int count = source.Rows * source.Cols;
            float[] fs = RentScratch(count);
            float[] fr = RentScratch(count);
            var bs = RentBuffer(count);
            var br = RentBuffer(count);
            try
            {
                PackInto(source, fs.AsSpan(0, count));
                bs.Upload(fs.AsSpan(0, count));
                _launcher.DispatchTranspose(bs, br, (uint)source.Rows, (uint)source.Cols);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), destination);
            }
            finally
            {
                ReturnBuffer(bs);
                ReturnBuffer(br);
                ReturnScratch(fs);
                ReturnScratch(fr);
            }
        }

        public void CopyInto(TensorBase source, TensorBase destination)
        {
            ValidateSame(source, destination, "CopyInto");
            if (_launcher == null) { _fallback.CopyInto(source, destination); return; }
            int count = RowCount(source) * source.Cols;
            float[] fs = RentScratch(count);
            float[] fr = RentScratch(count);
            var bs = RentBuffer(count);
            var br = RentBuffer(count);
            try
            {
                PackInto(source, fs.AsSpan(0, count));
                bs.Upload(fs.AsSpan(0, count));
                _launcher.DispatchRowwise(VulkanKernel.Copy, new[] { bs, br }, (uint)count, 1);
                br.Download(fr.AsSpan(0, count));
                Unpack(fr.AsSpan(0, count), destination);
            }
            finally
            {
                ReturnBuffer(bs);
                ReturnBuffer(br);
                ReturnScratch(fs);
                ReturnScratch(fr);
            }
        }

        private static void ValidateSame(TensorBase a, TensorBase b, string op)
        {
            if (a.Rank != b.Rank || a.Layers != b.Layers || a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException(
                    $"{op}: shape mismatch ({TensorValidation.DescribeShape(a)} vs {TensorValidation.DescribeShape(b)}).");
        }

        private static void ValidateNorm(TensorBase t, TensorBase gamma, TensorBase beta, string op)
        {
            if (gamma.Rank != 1 || beta.Rank != 1 || gamma.Cols != t.Cols || beta.Cols != t.Cols)
                throw new ArgumentException($"{op}: gamma/beta must be rank-1 length Cols.");
        }

        private static float[] Pack1D(TensorBase t)
        {
            var flat = new float[t.Cols];
            t.Buffer.AsSpan(t.Offset, t.Cols).CopyTo(flat);
            return flat;
        }

        public void Synchronize()
        {
            _ctx?.Synchronize();
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _launcher?.Dispose();
            _compiler?.Dispose();
            _fallback.Dispose();
            _ctx?.Dispose();
        }
    }
}
