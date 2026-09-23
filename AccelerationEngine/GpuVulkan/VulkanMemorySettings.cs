namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
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
    }
}
