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
    /// Thread-safe: callers may rent/return from parallel loops (e.g. QLoRA
    /// forward batches), so all pool state is guarded by a gate.
    /// </summary>
    internal sealed class VulkanBufferPool : IDisposable
    {
        private readonly VulkanContext _ctx;
        private readonly Dictionary<int, Stack<VulkanBuffer>> _buckets = new();
        private readonly List<VulkanBuffer> _all = new();
        private readonly object _gate = new();
        private readonly ulong _deviceLocalBudgetBytes;
        private readonly ulong _hostBudgetBytes;
        private ulong _deviceLocalRetainedBytes;
        private ulong _hostRetainedBytes;
        private bool _disposed;

        /// <param name="ctx">Vulkan context the buffers live in.</param>
        /// <param name="deviceLocalBudgetBytes">
        /// Cap on retained DEVICE_LOCAL (VRAM) bytes (0 = unlimited). Idle VRAM
        /// buffers are evicted to stay under it; rented ones are never touched.
        /// </param>
        /// <param name="hostBudgetBytes">
        /// Cap on retained host-visible (system RAM) bytes (0 = unlimited).
        /// Without this the pool - every size bucket ever created, kept until
        /// process exit - is the one unmanaged consumer that could grow into the
        /// rest of the machine's RAM over a long training run.
        /// </param>
        public VulkanBufferPool(VulkanContext ctx, ulong deviceLocalBudgetBytes = 0, ulong hostBudgetBytes = 0)
        {
            _ctx = ctx;
            _deviceLocalBudgetBytes = deviceLocalBudgetBytes;
            _hostBudgetBytes = hostBudgetBytes;
        }

        public VulkanBuffer Rent(int floatCount)
        {
            lock (_gate)
            {
                int bucket = BucketFor(floatCount);
                if (_buckets.TryGetValue(bucket, out var stack) && stack.Count > 0)
                    return stack.Pop()!;

                var buffer = new VulkanBuffer(_ctx, (ulong)(bucket * 4));
                if (buffer.IsDeviceLocal)
                {
                    AnyDeviceLocal = true;
                    _deviceLocalRetainedBytes += buffer.SizeBytes;

                    //Keep retained VRAM under the detected/configured budget by
                    //dropping idle buffers first (this new one is not pooled yet).
                    if (_deviceLocalBudgetBytes > 0 &&
                        _deviceLocalRetainedBytes > _deviceLocalBudgetBytes)
                        EvictIdleOverBudget(deviceLocalTier: true);
                }
                else
                {
                    _hostRetainedBytes += buffer.SizeBytes;

                    //Same guard for system RAM: the host tier is where every
                    //per-op staging buffer lands today, and it used to be unbounded.
                    if (_hostBudgetBytes > 0 &&
                        _hostRetainedBytes > _hostBudgetBytes)
                        EvictIdleOverBudget(deviceLocalTier: false);
                }
                WorkingMemoryTypeIndex ??= buffer.ChosenMemoryTypeIndex;
                _all.Add(buffer);
                return buffer;
            }
        }

        /// <summary>Live retained bytes for one tier. Callers hold _gate.</summary>
        private ulong RetainedBytes(bool deviceLocalTier) =>
            deviceLocalTier ? _deviceLocalRetainedBytes : _hostRetainedBytes;

        private void RemoveLiveLocked(VulkanBuffer buffer)
        {
            _all.Remove(buffer);
            if (buffer.IsDeviceLocal)
                _deviceLocalRetainedBytes -= buffer.SizeBytes;
            else
                _hostRetainedBytes -= buffer.SizeBytes;
        }

        /// <summary>
        /// Frees idle buffers of one tier until its retained bytes fit the budget.
        /// Must be called with _gate held; only buffers sitting in the buckets
        /// (i.e. not currently rented) are candidates, and the other tier's
        /// buckets keep their LIFO order.
        /// </summary>
        private void EvictIdleOverBudget(bool deviceLocalTier)
        {
            ulong budget = deviceLocalTier ? _deviceLocalBudgetBytes : _hostBudgetBytes;
            if (budget == 0)
                return;

            foreach (var stack in _buckets.Values)
            {
                if (RetainedBytes(deviceLocalTier) <= budget)
                    break;

                if (stack.Count == 0)
                    continue;

                //Drain the bucket, disposing only the tier that is over budget,
                //then push the survivors back with their order intact.
                var keep = new List<VulkanBuffer>(stack.Count);
                while (stack.Count > 0 && RetainedBytes(deviceLocalTier) > budget)
                {
                    var candidate = stack.Pop();
                    if (candidate.IsDeviceLocal == deviceLocalTier)
                    {
                        RemoveLiveLocked(candidate);
                        candidate.Dispose();
                    }
                    else
                    {
                        keep.Add(candidate);
                    }
                }

                for (int i = keep.Count - 1; i >= 0; i--)
                    stack.Push(keep[i]);
            }
        }

        /// <summary>True when at least one pooled buffer got DEVICE_LOCAL memory.</summary>
        public bool AnyDeviceLocal { get; private set; }

        /// <summary>
        /// Memory type index the pool's working buffers actually use. With the
        /// default policy this is the host-cached system-RAM tier, not VRAM.
        /// </summary>
        public uint? WorkingMemoryTypeIndex { get; private set; }

        /// <summary>Live retained host-visible (system RAM) bytes across the pool.</summary>
        public ulong HostRetainedBytes
        {
            get { lock (_gate) return _hostRetainedBytes; }
        }

        /// <summary>Live retained DEVICE_LOCAL (VRAM) bytes across the pool.</summary>
        public ulong DeviceLocalRetainedBytes
        {
            get { lock (_gate) return _deviceLocalRetainedBytes; }
        }

        public void Return(VulkanBuffer buffer)
        {
            if (buffer == null)
                return; // defensive: never poison the pool with a null slot

            lock (_gate)
            {
                int bucket = BucketFor((int)(buffer.SizeBytes / 4));
                if (!_buckets.TryGetValue(bucket, out var stack))
                {
                    stack = new Stack<VulkanBuffer>();
                    _buckets[bucket] = stack;
                }
                stack.Push(buffer);
            }
        }

        public int LiveBufferCount
        {
            get { lock (_gate) return _all.Count; }
        }

        private static int BucketFor(int floatCount)
        {
            int bucket = 256;
            while (bucket < floatCount)
                bucket <<= 1;
            return bucket;
        }

        public void Dispose()
        {
            lock (_gate)
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
}
