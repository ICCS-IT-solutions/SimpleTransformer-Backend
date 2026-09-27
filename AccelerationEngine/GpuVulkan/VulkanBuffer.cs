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
    /// <summary>
    /// Storage buffer wrapper in one of two tiers.
    /// <list type="bullet">
    /// <item>host tier - a mapped HOST_VISIBLE|HOST_CACHED buffer in system RAM.
    /// Upload/Download are pure memcpy, which is what the old per-op sync model
    /// needed (every op round-tripped through the host).</item>
    /// <item>device tier - a DEVICE_LOCAL (VRAM) buffer with a mapped
    /// host-visible staging companion. The payload lives in VRAM for the GPU,
    /// the host reads and writes system RAM, and the bus is crossed only by
    /// copies recorded into the command batch (see
    /// <see cref="VulkanKernelLauncher"/>). This is what puts the working set
    /// in VRAM instead of in system RAM.</item>
    /// </list>
    /// Memory type selection is delegated to
    /// <see cref="VulkanContext.SelectMemoryType"/>, which prefers
    /// HOST_VISIBLE|HOST_COHERENT|HOST_CACHED system RAM. That is
    /// counter-intuitive but measured: on this RX 5700 XT the resizable-BAR
    /// (DEVICE_LOCAL|HOST_VISIBLE) heap is ~1000x slower for host round trips
    /// (37-52 MiB/s vs ~43 GiB/s), so the device tier must reach VRAM through
    /// copies instead of by mapping it.
    /// </summary>
    public sealed unsafe class VulkanBuffer : IDisposable
    {
        private readonly VulkanContext _ctx;
        public Silk.NET.Vulkan.Buffer Handle;
        public DeviceMemory Memory;
        public ulong SizeBytes { get; }

        /// <summary>Host-visible companion of a device-local buffer, else null.</summary>
        private StagingMemory? _staging;

        /// <summary>True when the payload lives in VRAM behind a staging buffer.</summary>
        public bool UsesStaging => _staging != null;

        /// <summary>Host-visible buffer the copies move bytes through.</summary>
        public Silk.NET.Vulkan.Buffer StagingHandle => _staging!.Handle;

        /// <summary>Bytes this buffer holds on the host side (0 unless staged).</summary>
        public ulong StagingSizeBytes => _staging != null ? SizeBytes : 0;

        /// <summary>Set by <see cref="Upload"/>, consumed when the next dispatch records the copy.</summary>
        public bool PendingUpload { get; set; }

        /// <summary>Set by <see cref="MarkReadback"/>, consumed by the dispatch that produces the result.</summary>
        public bool ReadbackWanted { get; set; }

        /// <summary>True once the buffer and its memory have been destroyed.</summary>
        public bool IsDisposed => _disposed;

        /// <summary>True when the chosen memory type is device-local VRAM.</summary>
        public bool IsDeviceLocal { get; private set; }

        /// <summary>Memory type index actually allocated (diagnostics).</summary>
        public uint ChosenMemoryTypeIndex { get; private set; }

        /// <summary>True when the chosen memory type can be mapped by the host.</summary>
        public bool IsHostVisible { get; private set; }

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

            Result r = vk.CreateBuffer(ctx.Device, in info, null, out Silk.NET.Vulkan.Buffer probe);
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

            Result result = vk.CreateBuffer(ctx.Device, in bufferInfo, null, out Handle);
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

            result = vk.AllocateMemory(ctx.Device, in allocInfo, null, out Memory);
            if (result != Result.Success && forcedMemoryTypeIndex < 0)
            {
                // Device-local host-visible heaps can be small (a 256 MiB BAR
                // window on this RX 5700 XT); retry on system-visible memory
                // before giving up.
                memIndex = ctx.SelectMemoryType(req.MemoryTypeBits, preferHostCached: false);
                allocInfo.MemoryTypeIndex = memIndex;
                result = vk.AllocateMemory(ctx.Device, in allocInfo, null, out Memory);
            }
            if (result != Result.Success)
                throw new InvalidOperationException($"vkAllocateMemory failed: {result}");

            result = vk.BindBufferMemory(ctx.Device, Handle, Memory, 0);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkBindBufferMemory failed: {result}");

            ChosenMemoryTypeIndex = memIndex;
            IsDeviceLocal = ctx.IsDeviceLocalMemoryType(memIndex);
            IsHostVisible = ctx.IsHostVisibleMemoryType(memIndex);

            void* mapped = null;
            result = vk.MapMemory(ctx.Device, Memory, 0, sizeBytes, 0, &mapped);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkMapMemory failed: {result}");
            _mapped = mapped;
        }

        // Memory-type selection lives in VulkanContext.SelectMemoryType so the
        // buffer, and anything else that allocates, share one policy
        // (device-local -> host-cached -> coherent -> host-visible).

        /// <summary>
        /// Allocates a GPU-resident storage buffer: it prefers true VRAM
        /// (DEVICE_LOCAL without HOST_VISIBLE) so compute reads run at VRAM
        /// bandwidth instead of across PCIe. The memory stays unmapped, so the
        /// caller has to fill it with a staged device copy
        /// (see <see cref="VulkanKernelLauncher.CopyBuffer(VulkanBuffer, VulkanBuffer, ulong)"/>).
        /// Returns null when the device offers no device-local type or the
        /// allocation fails, letting callers fall back to host-visible buffers.
        /// </summary>
        public static VulkanBuffer? TryCreateDeviceResident(VulkanContext ctx, ulong sizeBytes)
        {
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

            if (vk.CreateBuffer(ctx.Device, in bufferInfo, null, out Silk.NET.Vulkan.Buffer handle) != Result.Success)
                return null;

            vk.GetBufferMemoryRequirements(ctx.Device, handle, out MemoryRequirements req);

            uint memIndex = ctx.SelectDeviceLocalMemoryType(req.MemoryTypeBits);
            if (memIndex == uint.MaxValue)
            {
                vk.DestroyBuffer(ctx.Device, handle, null);
                return null;
            }

            var allocInfo = new MemoryAllocateInfo
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = req.Size,
                MemoryTypeIndex = memIndex
            };

            if (vk.AllocateMemory(ctx.Device, in allocInfo, null, out DeviceMemory memory) != Result.Success)
            {
                vk.DestroyBuffer(ctx.Device, handle, null);
                return null;
            }

            if (vk.BindBufferMemory(ctx.Device, handle, memory, 0) != Result.Success)
            {
                vk.FreeMemory(ctx.Device, memory, null);
                vk.DestroyBuffer(ctx.Device, handle, null);
                return null;
            }

            TotalBuffersCreated++;
            return new VulkanBuffer(ctx, sizeBytes, handle, memory, memIndex);
        }

        /// <summary>Wraps an already-allocated, unmapped buffer (device-resident).</summary>
        private VulkanBuffer(VulkanContext ctx, ulong sizeBytes, Silk.NET.Vulkan.Buffer handle, DeviceMemory memory, uint memoryTypeIndex)
        {
            _ctx = ctx;
            SizeBytes = sizeBytes;
            Handle = handle;
            Memory = memory;
            ChosenMemoryTypeIndex = memoryTypeIndex;
            IsDeviceLocal = ctx.IsDeviceLocalMemoryType(memoryTypeIndex);
            IsHostVisible = ctx.IsHostVisibleMemoryType(memoryTypeIndex);
            _mapped = null;
        }

        /// <summary>
        /// Allocates a device-tier buffer: a DEVICE_LOCAL (VRAM) payload plus a
        /// mapped host-visible staging companion, so per-op data lives in VRAM
        /// while the host keeps touching fast system RAM. Returns null when the
        /// host process asked for the host tier, when the device offers no
        /// device-local type, or when either allocation fails - the pool then
        /// falls back to a plain host-visible buffer.
        /// </summary>
        public static VulkanBuffer? TryCreateStaged(VulkanContext ctx, ulong sizeBytes)
        {
            if (VulkanMemorySettings.PerOpMemoryTier != VulkanPerOpMemoryTier.DeviceLocal)
                return null;

            VulkanBuffer? device = TryCreateDeviceResident(ctx, sizeBytes);
            if (device == null)
                return null;

            if (!device.AttachStaging(ctx))
            {
                device.Dispose();
                return null;
            }

            return device;
        }

        /// <summary>Adds the host-visible companion a device-tier buffer needs.</summary>
        private bool AttachStaging(VulkanContext ctx)
        {
            var vk = ctx.Vk;
            var info = new BufferCreateInfo
            {
                SType = StructureType.BufferCreateInfo,
                Size = SizeBytes,
                Usage = BufferUsageFlags.TransferSrcBit | BufferUsageFlags.TransferDstBit,
                SharingMode = SharingMode.Exclusive
            };

            if (vk.CreateBuffer(ctx.Device, in info, null, out Silk.NET.Vulkan.Buffer handle)
                != Result.Success)
            {
                return false;
            }

            vk.GetBufferMemoryRequirements(ctx.Device, handle, out MemoryRequirements req);

            uint memIndex;
            try
            {
                // Host-cached system RAM: the staging side of every copy, and
                // the tier the old per-op path measured as by far the fastest
                // for host access.
                memIndex = ctx.SelectMemoryType(req.MemoryTypeBits, preferHostCached: true);
            }
            catch (InvalidOperationException)
            {
                vk.DestroyBuffer(ctx.Device, handle, null);
                return false;
            }

            var allocInfo = new MemoryAllocateInfo
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = req.Size,
                MemoryTypeIndex = memIndex
            };
            if (vk.AllocateMemory(ctx.Device, in allocInfo, null, out DeviceMemory memory)
                != Result.Success)
            {
                vk.DestroyBuffer(ctx.Device, handle, null);
                return false;
            }

            void* mapped = null;
            if (vk.BindBufferMemory(ctx.Device, handle, memory, 0) != Result.Success ||
                vk.MapMemory(ctx.Device, memory, 0, SizeBytes, 0, ref mapped) != Result.Success)
            {
                vk.FreeMemory(ctx.Device, memory, null);
                vk.DestroyBuffer(ctx.Device, handle, null);
                return false;
            }

            TotalBuffersCreated++;
            _staging = new StagingMemory
            {
                Handle = handle,
                Memory = memory,
                Mapped = mapped,
                SizeBytes = SizeBytes
            };
            return true;
        }

        /// <summary>The mapped host-visible companion of a device-tier buffer.</summary>
        private sealed class StagingMemory
        {
            public Silk.NET.Vulkan.Buffer Handle;
            public DeviceMemory Memory;
            public void* Mapped;
            public ulong SizeBytes;
        }

        public void Upload(ReadOnlySpan<float> data)
        {
            void* target = _mapped;
            if (target == null && _staging != null)
                target = _staging.Mapped;
            if (target == null)
                throw new InvalidOperationException("Buffer memory is not mapped.");
            int bytes = data.Length * 4;
            if ((ulong)bytes > SizeBytes)
                throw new ArgumentOutOfRangeException(
                    nameof(data), $"Payload {bytes} B exceeds buffer capacity {SizeBytes} B.");

            fixed (float* src = data)
            {
                new ReadOnlySpan<byte>(src, bytes)
                    .CopyTo(new Span<byte>(target, bytes));
            }
            TotalUploadedBytes += bytes;

            // A device-local buffer only sees these bytes through a copy the
            // next dispatch records; the dispatcher consumes the flag.
            if (_staging != null)
                PendingUpload = true;
        }

        public void Download(Span<float> data)
        {
            void* source = _mapped;
            if (source == null && _staging != null)
                source = _staging.Mapped;
            if (source == null)
                throw new InvalidOperationException("Buffer memory is not mapped.");
            int bytes = data.Length * 4;
            if ((ulong)bytes > SizeBytes)
                throw new ArgumentOutOfRangeException(
                    nameof(data), $"Payload {bytes} B exceeds buffer capacity {SizeBytes} B.");

            // Staging memory only holds what a recorded copy put there; a
            // pending flag means the dispatch that should have copied it back
            // never ran, so the bytes on hand are stale.
            if (_staging != null && ReadbackWanted)
            {
                throw new InvalidOperationException(
                    "Downloaded a staged buffer whose read-back copy was never recorded: " +
                    "call MarkReadback() before the dispatch that produces this result.");
            }

            fixed (float* dst = data)
            {
                new ReadOnlySpan<byte>(source, bytes)
                    .CopyTo(new Span<byte>(dst, bytes));
            }
            TotalDownloadedBytes += bytes;
        }

        /// <summary>
        /// Declares that the caller will <see cref="Download"/> this buffer's
        /// result after the next dispatch, so the dispatcher records the
        /// device->staging copy into that batch instead of costing an extra
        /// round trip. No-op on the host tier, where the read is a plain memcpy
        /// over the mapped memory.
        /// </summary>
        public void MarkReadback()
        {
            if (_staging != null)
                ReadbackWanted = true;
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
            if (_staging != null)
            {
                _ctx.Vk.UnmapMemory(_ctx.Device, _staging.Memory);
                if (_staging.Handle.Handle != 0)
                    _ctx.Vk.DestroyBuffer(_ctx.Device, _staging.Handle, null);
                if (_staging.Memory.Handle != 0)
                    _ctx.Vk.FreeMemory(_ctx.Device, _staging.Memory, null);
                _staging = null;
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
