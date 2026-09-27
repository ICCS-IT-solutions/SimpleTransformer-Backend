namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>Where the per-op working buffers of the Vulkan pool live.</summary>
    public enum VulkanPerOpMemoryTier
    {
        /// <summary>
        /// Mapped HOST_VISIBLE|HOST_CACHED buffers in system RAM (the original
        /// policy). The GPU reaches every operand over PCIe, but the host never
        /// maps VRAM - which matters only while every op round-trips through
        /// the host.
        /// </summary>
        HostCached = 0,

        /// <summary>
        /// DEVICE_LOCAL (VRAM) payloads reached through a host-visible staging
        /// buffer, with both copies recorded into the same command batch as the
        /// dispatch. The GPU works from VRAM at VRAM bandwidth and the host
        /// still touches system RAM. Falls back to <see cref="HostCached"/>
        /// automatically when the device has no device-local memory or the
        /// allocation fails.
        /// </summary>
        DeviceLocal = 1
    }

    /// <summary>
    /// Optional GPU-memory overrides, pushed in by the host process (Server /
    /// CLI) from config.ini before any backend is constructed. The accelerator
    /// itself stays configuration-free: it only reads these statics.
    /// </summary>
    public static class VulkanMemorySettings
    {
        /// <summary>
        /// Hard cap in bytes for VRAM (DEVICE_LOCAL) allocations.
        /// 0 = auto: use the driver's live budget (VK_EXT_memory_budget) when the
        /// driver provides it, otherwise the device-local heap size.
        /// </summary>
        public static long BudgetBytesOverride { get; set; }

        /// <summary>
        /// Hard cap in bytes for host-visible (system RAM) staging buffers.
        /// 0 = auto: a quarter of physical RAM clamped to [1 GiB, 8 GiB], so the
        /// pool can never crowd out the OS, the Vulkan driver or the GC heap
        /// (model + optimizer state) - 8 GiB on a 32 GiB machine.
        /// </summary>
        public static long HostBudgetBytesOverride { get; set; }

        /// <summary>
        /// Tier the pooled per-op buffers are allocated from. Defaults to
        /// <see cref="VulkanPerOpMemoryTier.DeviceLocal"/> so the working set
        /// lands in VRAM; the pool falls back to system RAM by itself when the
        /// device cannot provide device-local memory.
        /// </summary>
        public static VulkanPerOpMemoryTier PerOpMemoryTier { get; set; } =
            VulkanPerOpMemoryTier.DeviceLocal;

        /// <summary>
        /// Threshold in bytes below which the buffer pool rents host-visible
        /// memory directly, avoiding staging buffers entirely.
        ///
        /// Staging copies cost 2 full memory passes (host->staging->VRAM and
        /// back) plus PCIe transfer overhead. For small operations (<= 1 MiB)
        /// the CPU-side copy time dominates the PCIe execution time, so host-visible
        /// buffers are faster. Above this threshold, VRAM bandwidth beats host
        /// access and the device tier wins clearly (e.g. 37x on 512x1024x1024 MatMul).
        ///
        /// 0 = disabled (all pooled buffers staged to VRAM when DeviceLocal is selected).
        /// Default: 1 MiB (262,144 floats).
        /// </summary>
        public static ulong DeviceLocalThresholdBytes { get; set; } = 1024UL * 1024UL;
    }
}
