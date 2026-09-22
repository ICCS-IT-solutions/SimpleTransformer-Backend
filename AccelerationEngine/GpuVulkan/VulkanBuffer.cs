using System;
using Silk.NET.Vulkan;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Host-visible storage buffer wrapper.
    /// Phase 4: the memory is mapped once at creation and stays mapped for
    /// the buffer's lifetime, so Upload/Download are pure memcpy with no
    /// vkMapMemory/vkUnmapMemory round trips per dispatch.
    /// Memory type selection is delegated to
    /// <see cref="VulkanContext.SelectMemoryType"/>, which prefers
    /// HOST_VISIBLE|HOST_COHERENT|HOST_CACHED system RAM. That is
    /// counter-intuitive but measured: on this RX 5700 XT the resizable-BAR
    /// (DEVICE_LOCAL|HOST_VISIBLE) heap is ~1000x slower for host round trips
    /// (37-52 MiB/s vs ~43 GiB/s), and every op here round trips through the
    /// host, so device-local VRAM would be the worst possible choice.
    /// </summary>
    internal sealed unsafe class VulkanBuffer : IDisposable
    {
        private readonly VulkanContext _ctx;
        public Silk.NET.Vulkan.Buffer Handle;
        public DeviceMemory Memory;
        public ulong SizeBytes { get; }

        /// <summary>True when the chosen memory type is device-local VRAM.</summary>
        public bool IsDeviceLocal { get; private set; }

        /// <summary>Memory type index actually allocated (diagnostics).</summary>
        public uint ChosenMemoryTypeIndex { get; private set; }

        // Phase 4 telemetry. Dispatch is single-threaded and synchronous,
        // so plain counters are sufficient (no interlocked needed).
        /// <summary>Total host-to-device bytes copied (map memcpy).</summary>
        public static long TotalUploadedBytes;
        /// <summary>Total device-to-host bytes copied (map memcpy).</summary>
        public static long TotalDownloadedBytes;
        /// <summary>Buffers actually created; stays flat once the pool is warm.</summary>
        public static long TotalBuffersCreated;

        private void* _mapped;
        private bool _disposed;

        public VulkanBuffer(VulkanContext ctx, ulong sizeBytes)
            : this(ctx, sizeBytes, preferHostCached: true)
        {
        }

        /// <summary>
        /// Memory type bits a storage buffer of <paramref name="sizeBytes"/> is
        /// compatible with. Queried without allocating: a buffer is created,
        /// interrogated and destroyed, so probes can skip incompatible types.
        /// </summary>
        internal static uint QueryMemoryTypeBits(VulkanContext ctx, ulong sizeBytes)
        {
            var vk = ctx.Vk;
            var info = new BufferCreateInfo
            {
                SType = StructureType.BufferCreateInfo,
                Size = sizeBytes,
                Usage = BufferUsageFlags.StorageBufferBit |
                        BufferUsageFlags.TransferSrcBit |
                        BufferUsageFlags.TransferDstBit,
                SharingMode = SharingMode.Exclusive
            };

            Result r = vk.CreateBuffer(ctx.Device, info, null, out Silk.NET.Vulkan.Buffer probe);
            if (r != Result.Success)
                throw new InvalidOperationException($"vkCreateBuffer (probe) failed: {r}");
            try
            {
                vk.GetBufferMemoryRequirements(ctx.Device, probe, out MemoryRequirements req);
                return req.MemoryTypeBits;
            }
            finally
            {
                vk.DestroyBuffer(ctx.Device, probe, null);
            }
        }

        public VulkanBuffer(VulkanContext ctx, ulong sizeBytes, bool preferHostCached)
            : this(ctx, sizeBytes, preferHostCached, forcedMemoryTypeIndex: -1)
        {
        }

        /// <summary>
        /// Allocates on an explicit memory type. Only the bandwidth probe uses this,
        /// so it can measure each host-visible tier in isolation.
        /// </summary>
        public VulkanBuffer(VulkanContext ctx, ulong sizeBytes, bool preferHostCached, int forcedMemoryTypeIndex)
        {
            _ctx = ctx;
            SizeBytes = sizeBytes;
            TotalBuffersCreated++;

            var vk = ctx.Vk;
            var bufferInfo = new BufferCreateInfo
            {
                SType = StructureType.BufferCreateInfo,
                Size = sizeBytes,
                Usage = BufferUsageFlags.StorageBufferBit |
                        BufferUsageFlags.TransferSrcBit |
                        BufferUsageFlags.TransferDstBit,
                SharingMode = SharingMode.Exclusive
            };

            Result result = vk.CreateBuffer(ctx.Device, bufferInfo, null, out Handle);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkCreateBuffer failed: {result}");

            vk.GetBufferMemoryRequirements(ctx.Device, Handle, out MemoryRequirements req);
            uint memIndex = forcedMemoryTypeIndex >= 0
                ? (uint)forcedMemoryTypeIndex
                : ctx.SelectMemoryType(req.MemoryTypeBits, preferHostCached);

            var allocInfo = new MemoryAllocateInfo
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = req.Size,
                MemoryTypeIndex = memIndex
            };

            result = vk.AllocateMemory(ctx.Device, allocInfo, null, out Memory);
            if (result != Result.Success && forcedMemoryTypeIndex < 0)
            {
                // Device-local host-visible heaps can be small (a 256 MiB BAR
                // window on this RX 5700 XT); retry on system-visible memory
                // before giving up.
                memIndex = ctx.SelectMemoryType(req.MemoryTypeBits, preferHostCached: false);
                allocInfo.MemoryTypeIndex = memIndex;
                result = vk.AllocateMemory(ctx.Device, allocInfo, null, out Memory);
            }
            if (result != Result.Success)
                throw new InvalidOperationException($"vkAllocateMemory failed: {result}");

            result = vk.BindBufferMemory(ctx.Device, Handle, Memory, 0);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkBindBufferMemory failed: {result}");

            ChosenMemoryTypeIndex = memIndex;
            IsDeviceLocal = ctx.IsDeviceLocalMemoryType(memIndex);

            void* mapped = null;
            result = vk.MapMemory(ctx.Device, Memory, 0, sizeBytes, 0, &mapped);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkMapMemory failed: {result}");
            _mapped = mapped;
        }

        // Memory-type selection lives in VulkanContext.SelectMemoryType so the
        // buffer, and anything else that allocates, share one policy
        // (device-local -> host-cached -> coherent -> host-visible).

        public void Upload(ReadOnlySpan<float> data)
        {
            if (_mapped == null)
                throw new InvalidOperationException("Buffer memory is not mapped.");
            int bytes = data.Length * 4;
            if ((ulong)bytes > SizeBytes)
                throw new ArgumentOutOfRangeException(
                    nameof(data), $"Payload {bytes} B exceeds buffer capacity {SizeBytes} B.");

            fixed (float* src = data)
            {
                new ReadOnlySpan<byte>(src, bytes)
                    .CopyTo(new Span<byte>(_mapped, bytes));
            }
            TotalUploadedBytes += bytes;
        }

        public void Download(Span<float> data)
        {
            if (_mapped == null)
                throw new InvalidOperationException("Buffer memory is not mapped.");
            int bytes = data.Length * 4;
            if ((ulong)bytes > SizeBytes)
                throw new ArgumentOutOfRangeException(
                    nameof(data), $"Payload {bytes} B exceeds buffer capacity {SizeBytes} B.");

            fixed (float* dst = data)
            {
                new ReadOnlySpan<byte>(_mapped, bytes)
                    .CopyTo(new Span<byte>(dst, bytes));
            }
            TotalDownloadedBytes += bytes;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_mapped != null)
            {
                _ctx.Vk.UnmapMemory(_ctx.Device, Memory);
                _mapped = null;
            }
            if (Handle.Handle != 0)
            {
                _ctx.Vk.DestroyBuffer(_ctx.Device, Handle, null);
                Handle = default;
            }
            if (Memory.Handle != 0)
            {
                _ctx.Vk.FreeMemory(_ctx.Device, Memory, null);
                Memory = default;
            }
        }
    }
}
