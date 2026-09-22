using System;
using System.Collections.Generic;
using Silk.NET.Vulkan;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Pools host-visible coherent storage buffers by size bucket
    /// (power-of-two float capacity). Buffers are reset (not destroyed)
    /// between dispatches, eliminating vkCreateBuffer/vkAllocateMemory/
    /// vkBindBufferMemory/vkDestroyBuffer/vkFreeMemory per op.
    /// Not thread-safe: one pool per backend instance, matching the
    /// single-threaded synchronous dispatch model.
    /// </summary>
    internal sealed class VulkanBufferPool : IDisposable
    {
        private readonly VulkanContext _ctx;
        private readonly Dictionary<int, Stack<VulkanBuffer>> _buckets = new();
        private readonly List<VulkanBuffer> _all = new();
        private bool _disposed;

        public VulkanBufferPool(VulkanContext ctx)
        {
            _ctx = ctx;
        }

        public VulkanBuffer Rent(int floatCount)
        {
            int bucket = BucketFor(floatCount);
            if (_buckets.TryGetValue(bucket, out var stack) && stack.Count > 0)
                return stack.Pop();

            var buffer = new VulkanBuffer(_ctx, (ulong)(bucket * 4));
            if (buffer.IsDeviceLocal)
                AnyDeviceLocal = true;
            WorkingMemoryTypeIndex ??= buffer.ChosenMemoryTypeIndex;
            _all.Add(buffer);
            return buffer;
        }

        /// <summary>True when at least one pooled buffer got DEVICE_LOCAL memory.</summary>
        public bool AnyDeviceLocal { get; private set; }

        /// <summary>
        /// Memory type index the pool's working buffers actually use. With the
        /// default policy this is the host-cached system-RAM tier, not VRAM.
        /// </summary>
        public uint? WorkingMemoryTypeIndex { get; private set; }

        public void Return(VulkanBuffer buffer)
        {
            int bucket = BucketFor((int)(buffer.SizeBytes / 4));
            if (!_buckets.TryGetValue(bucket, out var stack))
            {
                stack = new Stack<VulkanBuffer>();
                _buckets[bucket] = stack;
            }
            stack.Push(buffer);
        }

        public int LiveBufferCount => _all.Count;

        private static int BucketFor(int floatCount)
        {
            int bucket = 256;
            while (bucket < floatCount)
                bucket <<= 1;
            return bucket;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            foreach (var buffer in _all)
                buffer.Dispose();
            _all.Clear();
            _buckets.Clear();
        }
    }
}
