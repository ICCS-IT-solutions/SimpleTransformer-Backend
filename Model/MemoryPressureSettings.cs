namespace SimpleTransformer.Model
{
    /// <summary>
    /// Optional host-memory pressure-relief overrides, pushed in by the host
    /// process (Server / CLI) from config.ini before any model is constructed.
    /// The model itself stays configuration-free: it only reads these statics.
    ///
    /// This is the managed-heap counterpart to
    /// <see cref="SimpleTransformer.AccelerationEngine.GpuVulkan.VulkanMemorySettings"/>,
    /// which only budgets unmanaged Vulkan buffers. The two cover different
    /// halves of system RAM, so a long training run can blow past the Vulkan
    /// budgets and still hit 95% process memory: model weights, AdamW moment
    /// state, gradients and pooled activations all live on the managed heap.
    /// </summary>
    public static class MemoryPressureSettings
    {
        /// <summary>
        /// Master switch. When false the valve costs one static bool read per
        /// training step and never samples, collects or trims anything.
        /// </summary>
        public static bool Enabled { get; set; } = true;

        /// <summary>
        /// Ceiling for this process, in bytes, that the valve treats as
        /// "full". 0 = auto (80% of the memory container the runtime reports,
        /// clamped to [1 GiB, 48 GiB]). Always clamped to
        /// <see cref="MaxUsableFractionOfPhysical"/> of physical memory - the
        /// same rule <c>ComputeVramBudget</c> applies to the VRAM override: a
        /// configured value is a cap, never a licence to exhaust the machine.
        /// </summary>
        public static long MaxQuotaBytes { get; set; }

        /// <summary>
        /// Floor for the resolved quota, in bytes. 0 = no explicit floor. Also
        /// clamped down to the physical ceiling, so a min_quota_mb larger than
        /// the machine cannot push the quota back up past it.
        /// </summary>
        public static long MinQuotaBytes { get; set; }

        /// <summary>
        /// Hardest ceiling the valve will ever resolve, as a fraction of
        /// physical memory. 5% stays reserved for the OS, the Vulkan driver
        /// and native allocations that are outside the managed heap entirely.
        /// </summary>
        public const double MaxUsableFractionOfPhysical = 0.95;

        /// <summary>
        /// Fraction of physical memory the auto quota targets. Sits between
        /// <see cref="LowerBoundPercent"/> and <see cref="UpperBoundPercent"/>
        /// so the auto policy has room for the cheap relief rungs to act
        /// before the expensive one is considered.
        /// </summary>
        public const double AutoQuotaFractionOfPhysical = 0.80;

        /// <summary>
        /// Start of the relief band, as a percent of physical memory. Crossing
        /// it drops idle pooled device buffers (the cheap rung). Usage must
        /// fall back below this before any further relief is considered, which
        /// is what stops the valve thrashing on a heap that stays warm.
        /// </summary>
        public static double LowerBoundPercent { get; set; } = 70.0;

        /// <summary>
        /// End of the relief band, as a percent of physical memory. Crossing
        /// it permits the blocking compacting collection - the rung that
        /// actually reclaims a fragmented multi-gigabyte heap, and the one that
        /// can stall the training thread, so it is rate-limited and gated on
        /// <see cref="AllowBlockingCompact"/>.
        /// </summary>
        public static double UpperBoundPercent { get; set; } = 85.0;

        /// <summary>
        /// Evaluate the valve once every N training steps. Sampling process
        /// memory costs a syscall, so the hot path is a modulo, not a query.
        /// Values &lt; 1 are treated as 1 (check every step).
        /// </summary>
        public static int CheckEveryNSteps { get; set; } = 25;

        /// <summary>
        /// Minimum number of sampled ticks between two relief actions. The
        /// second half of the anti-thrash guarantee: a blocking compacting
        /// collection on a multi-gigabyte heap can cost seconds, and running
        /// one per step would make the run far slower than the memory pressure
        /// it is trying to escape.
        /// </summary>
        public static int MinCooldownSteps { get; set; } = 250;

        /// <summary>
        /// Percent of the workspace pool's retained bytes to keep when the
        /// valve trims it. The remainder is handed back to the GC, at the cost
        /// of re-allocating those activations on the next step.
        /// </summary>
        public static double WorkspaceRetainPercent { get; set; } = 50.0;

        /// <summary>
        /// Fraction of the resolved quota the workspace pool may retain
        /// regardless of pressure. This is the part of the fix that needs no
        /// valve at all: an unconditional byte cap on the tensor pool, which
        /// previously had no bound at all, so a long run accumulated one
        /// pooled tensor per activation shape it ever saw.
        /// 0 = uncapped (not recommended for long runs).
        /// </summary>
        public static double WorkspaceCapFractionOfQuota { get; set; } = 0.35;

        /// <summary>
        /// Whether the valve may run a blocking, compacting gen-2 collection
        /// once usage reaches <see cref="UpperBoundPercent"/>. Default true
        /// because it is the only rung that recovers a fragmented heap, but it
        /// should be unreachable in steady state: if the workspace cap and the
        /// two cheap rungs keep the run below the upper bound, it never fires.
        /// Set false where a multi-second stall is unacceptable - the valve
        /// then tops out at trimming the workspace and a non-blocking collect.
        /// </summary>
        public static bool AllowBlockingCompact { get; set; } = true;
    }
}
