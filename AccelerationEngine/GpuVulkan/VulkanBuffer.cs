using System;
using Silk.NET.Vulkan;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Host-visible coherent device buffer wrapper.
    /// Simple and correct for Phase 1 (no staging); swap for
    /// device-local + staging in the perf phase if needed.
    /// </summary>
    internal sealed unsafe class VulkanBuffer : IDisposable
    {
        private readonly VulkanContext _ctx;
        public Silk.NET.Vulkan.Buffer Handle;
        public DeviceMemory Memory;
        public ulong SizeBytes { get; }

        private bool _disposed;

        public VulkanBuffer(VulkanContext ctx, ulong sizeBytes)
        {
            _ctx = ctx;
            SizeBytes = sizeBytes;

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
            uint memIndex = FindMemoryType(
                ctx, req.MemoryTypeBits,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);

            var allocInfo = new MemoryAllocateInfo
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = req.Size,
                MemoryTypeIndex = memIndex
            };

            result = vk.AllocateMemory(ctx.Device, allocInfo, null, out Memory);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkAllocateMemory failed: {result}");

            result = vk.BindBufferMemory(ctx.Device, Handle, Memory, 0);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkBindBufferMemory failed: {result}");
        }

        private static uint FindMemoryType(
            VulkanContext ctx, uint typeBits, MemoryPropertyFlags flags)
        {
            ctx.Vk.GetPhysicalDeviceMemoryProperties(ctx.PhysicalDevice, out PhysicalDeviceMemoryProperties props);
            for (uint i = 0; i < props.MemoryTypeCount; i++)
            {
                if ((typeBits & (1u << (int)i)) == 0)
                    continue;
                if ((props.MemoryTypes[(int)i].PropertyFlags & flags) == flags)
                    return i;
            }
            throw new InvalidOperationException("No suitable Vulkan memory type found.");
        }

        public void Upload(ReadOnlySpan<float> data)
        {
            void* mapped = null;
            Result result = _ctx.Vk.MapMemory(
                _ctx.Device, Memory, 0, SizeBytes, 0, &mapped);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkMapMemory failed: {result}");
            try
            {
                fixed (float* src = data)
                {
                    new ReadOnlySpan<byte>(src, data.Length * 4)
                        .CopyTo(new Span<byte>(mapped, data.Length * 4));
                }
            }
            finally
            {
                _ctx.Vk.UnmapMemory(_ctx.Device, Memory);
            }
        }

        public void Download(Span<float> data)
        {
            void* mapped = null;
            Result result = _ctx.Vk.MapMemory(
                _ctx.Device, Memory, 0, SizeBytes, 0, &mapped);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkMapMemory failed: {result}");
            try
            {
                fixed (float* dst = data)
                {
                    new ReadOnlySpan<byte>(mapped, data.Length * 4)
                        .CopyTo(new Span<byte>(dst, data.Length * 4));
                }
            }
            finally
            {
                _ctx.Vk.UnmapMemory(_ctx.Device, Memory);
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
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
