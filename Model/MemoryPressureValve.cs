using System;
using System.Diagnostics;

namespace SimpleTransformer.Model
{
    /// <summary>How hard the valve pushed on its most recent evaluation.</summary>
    public enum MemoryReliefLevel
    {
        /// <summary>Below the relief band. Nothing to do.</summary>
        None = 0,

        /// <summary>
        /// Drop idle pooled device buffers (Vulkan staging, descriptor caches,
        /// scratch arrays). Cheap and bounded; the backend already exposes it
        /// as <c>TrimIdleMemory</c>.
        /// </summary>
        TrimDevicePools = 1,

        /// <summary>
        /// Hand most of the pooled activation memory back to the GC, then a
        /// non-blocking gen-0 collection. Costs one re-allocation round of
        /// activations on the next step.
        /// </summary>
        TrimWorkspace = 2,

        /// <summary>
        /// Emergency rung: drop the workspace pool to a floor and force a
        /// blocking, compacting gen-2 collection. The only rung that reclaims
        /// a fragmented multi-gigabyte heap, and the only one that can stall
        /// the training thread - hence rate-limited and gated behind
        /// <see cref="MemoryPressureSettings.AllowBlockingCompact"/>.
        /// </summary>
        Compact = 3
    }

    /// <summary>One observation of process memory pressure.</summary>
    public readonly struct MemoryPressureSample
    {
        /// <summary>
        /// Committed private bytes for this process. This is the figure that
        /// tracks "memory usage reaches 95%" as the OS reports it, and unlike
        /// the managed heap reading it includes unmanaged allocations - the
        /// Vulkan staging pool and pinned buffers sit entirely outside the
        /// managed heap, which is why a managed-heap-only reading understates
        /// the pressure that actually causes the slowdown.
        /// </summary>
        public readonly long PrivateBytes;

        /// <summary>Managed heap size as last known to the GC. No forced collection.</summary>
        public readonly long ManagedHeapBytes;

        /// <summary>Total memory the runtime considers available (the denominator).</summary>
        public readonly long PhysicalBytes;

        /// <summary>Resolved quota after clamping. See <c>ResolveQuotaBytes</c>.</summary>
        public readonly long QuotaBytes;

        /// <summary>Fraction of physical memory committed, clamped to [0, 1].</summary>
        public readonly double UsedFraction;

        public MemoryPressureSample(
            long privateBytes,
            long managedHeapBytes,
            long physicalBytes,
            long quotaBytes)
        {
            PrivateBytes = privateBytes;
            ManagedHeapBytes = managedHeapBytes;
            PhysicalBytes = physicalBytes;
            QuotaBytes = quotaBytes;
            UsedFraction = physicalBytes > 0
                ? Math.Clamp((double)privateBytes / physicalBytes, 0.0, 1.0)
                : 0.0;
        }

        /// <summary>False when the runtime reported no usable memory container.</summary>
        public bool IsUsable => PhysicalBytes > 0;
    }

    /// <summary>
    /// Decides <em>whether</em> to relieve memory pressure; deliberately does
    /// not perform the relief. Keeping the policy free of any handle on the
    /// workspace or the backend is what makes <c>Evaluate</c> a pure function,
    /// and therefore testable without allocating a real heap - see
    /// <c>MemoryPressureSelfTest</c> (dotnet run -- --memory-valve-selftest).
    /// <para>
    /// The rungs escalate: the cheap ones run routinely, the blocking one only
    /// past the upper bound and never more often than the cooldown. Once the
    /// valve has acted it latches, and stays quiet until usage falls back below
    /// the lower bound. Without that latch a heap that merely plateaus above
    /// the threshold would be collected on every sample, which is a far worse
    /// outcome than the pressure being relieved.
    /// </para>
    /// </summary>
    public sealed class MemoryPressureValve
    {
        /// <summary>Lower clamp for the auto quota, mirroring ComputeHostStagingBudget.</summary>
        public const long AutoMinQuotaBytes = 1L << 30;

        /// <summary>Upper clamp for the auto quota.</summary>
        public const long AutoMaxQuotaBytes = 48L << 30;

        // Process.PrivateMemorySize64 is cached on the instance and only updated
        // by Refresh(), so the handle is held for the valve's lifetime -
        // Process.GetCurrentProcess() per sample would be a syscall plus a
        // fresh allocation on the training hot path.
        private readonly Process? _process;

        private long _ticksSinceRelief;
        private bool _latched;
        private long _tick;
        private int _reliefCount;
        private MemoryReliefLevel _lastRelief = MemoryReliefLevel.None;
        private MemoryPressureSample _lastSample;

        public MemoryPressureValve()
        {
            try
            {
                _process = Process.GetCurrentProcess();
            }
            catch
            {
                // No process handle: the valve still works off the managed heap
                // reading, just without the private-bytes figure that matches
                // what the OS reports.
                _process = null;
            }
        }

        public MemoryPressureSample LastSample => _lastSample;

        /// <summary>Resolves the configured quota into a usable ceiling, in bytes.</summary>
        public long QuotaBytes
        {
            get
            {
                long physical = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                return ResolveQuotaBytes(
                    MemoryPressureSettings.MaxQuotaBytes,
                    MemoryPressureSettings.MinQuotaBytes,
                    physical);
            }
        }

        /// <summary>
        /// Clamps a configured quota into the band the machine can actually
        /// honour. 0/absent quota = auto. An explicit value is a cap, never a
        /// licence to exceed the hardware - the same rule
        /// <c>GpuVulkanBackend.ComputeVramBudget</c> applies to the VRAM
        /// override, and for the same reason: a quota above what the box can
        /// give would just be an OOM with extra steps.
        /// </summary>
        public static long ResolveQuotaBytes(long configuredBytes, long minBytes, long physicalBytes)
        {
            if (physicalBytes <= 0)
                return Math.Max(configuredBytes, minBytes);

            long ceiling = (long)(physicalBytes * MemoryPressureSettings.MaxUsableFractionOfPhysical);
            if (ceiling <= 0)
                return Math.Max(configuredBytes, minBytes);

            // A min_quota larger than the machine is itself clamped, otherwise
            // it would push the result back through the ceiling.
            long floor = Math.Min(Math.Max(0L, minBytes), ceiling);

            long quota = configuredBytes > 0
                ? configuredBytes
                : Math.Clamp(
                    (long)(physicalBytes * MemoryPressureSettings.AutoQuotaFractionOfPhysical),
                    AutoMinQuotaBytes,
                    AutoMaxQuotaBytes);

            return Math.Clamp(quota, floor, ceiling);
        }

        /// <summary>Samples process memory. Safe to call at any cadence.</summary>
        public MemoryPressureSample Sample()
        {
            long physical = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long managed = GC.GetTotalMemory(forceFullCollection: false);
            long privateBytes = 0;

            if (_process != null)
            {
                try
                {
                    _process.Refresh();
                    privateBytes = _process.PrivateMemorySize64;
                }
                catch (Exception)
                {
                    // The process handle can go stale (exit, permissions).
                    // Degrade to a managed-heap-only reading rather than
                    // throwing out of a training step.
                    privateBytes = 0;
                }
            }

            return new MemoryPressureSample(
                privateBytes,
                managed,
                physical,
                ResolveQuotaBytes(
                    MemoryPressureSettings.MaxQuotaBytes,
                    MemoryPressureSettings.MinQuotaBytes,
                    physical));
        }

        /// <summary>
        /// Per-step entry point. Returns the relief the caller should apply,
        /// or <see cref="MemoryReliefLevel.None"/> when there is nothing to do.
        /// Costs one bool read and a modulo when there is nothing to report.
        /// </summary>
        public MemoryReliefLevel Tick()
        {
            if (!MemoryPressureSettings.Enabled)
                return MemoryReliefLevel.None;

            int interval = Math.Max(1, MemoryPressureSettings.CheckEveryNSteps);
            if (++_tick % interval != 0)
                return MemoryReliefLevel.None;

            return Evaluate(Sample());
        }

        /// <summary>
        /// Pure policy step: maps one sample onto a relief level. The only
        /// side effects are the valve's own latch and cooldown bookkeeping, so
        /// the whole escalation ladder can be exercised against synthetic
        /// samples instead of a real heap.
        /// </summary>
        public MemoryReliefLevel Evaluate(MemoryPressureSample sample)
        {
            _lastSample = sample;

            if (!sample.IsUsable)
                return MemoryReliefLevel.None;

            _ticksSinceRelief++;

            double usedPct = sample.UsedFraction * 100.0;
            double lower = Math.Clamp(MemoryPressureSettings.LowerBoundPercent, 1.0, 100.0);
            double upper = Math.Clamp(
                Math.Max(MemoryPressureSettings.UpperBoundPercent, lower),
                lower,
                100.0);

            // Back under the band: clear the latch so a later excursion gets
            // the full ladder again.
            if (usedPct < lower)
            {
                _latched = false;
                return MemoryReliefLevel.None;
            }

            // Latched and still cooling down: stay quiet. This is the
            // difference between relieving pressure and stalling the job.
            int cooldown = Math.Max(1, MemoryPressureSettings.MinCooldownSteps);
            if (_latched && _ticksSinceRelief < cooldown)
                return MemoryReliefLevel.None;

            MemoryReliefLevel level;
            if (usedPct >= upper && MemoryPressureSettings.AllowBlockingCompact)
            {
                level = MemoryReliefLevel.Compact;
            }
            else if (usedPct >= (lower + upper) * 0.5)
            {
                level = MemoryReliefLevel.TrimWorkspace;
            }
            else
            {
                level = MemoryReliefLevel.TrimDevicePools;
            }

            _latched = true;
            _ticksSinceRelief = 0;
            _lastRelief = level;
            _reliefCount++;
            return level;
        }

        /// <summary>
        /// One-line telemetry for logs and the frontend job message, in the
        /// same spirit as <c>IAccelerationBackend.DescribeMemoryUsage</c>.
        /// </summary>
        public string Describe()
        {
            const double mib = 1024.0 * 1024.0;

            if (!MemoryPressureSettings.Enabled)
                return "memory valve off";

            MemoryPressureSample s = _lastSample;
            if (!s.IsUsable)
                return "memory valve armed (no memory container reported)";

            string physical = s.PhysicalBytes > 0
                ? $"{s.PhysicalBytes / mib:F0} MiB"
                : "unknown";

            return $"host {s.PrivateBytes / mib:F0} of {physical} " +
                   $"({s.UsedFraction * 100:F0}%, quota {s.QuotaBytes / mib:F0} MiB), " +
                   $"managed heap {s.ManagedHeapBytes / mib:F0} MiB, " +
                   $"{_reliefCount} reliefs (last {_lastRelief})";
        }
    }
}
