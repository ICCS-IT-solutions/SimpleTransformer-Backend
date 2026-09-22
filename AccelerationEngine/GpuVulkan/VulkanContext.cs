using System;
using System.Collections.Generic;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Headless Vulkan compute context: instance, physical device, logical
    /// device with one compute queue, one reusable command pool.
    /// Never throws for missing GPUs - check IsAvailable, fall back to CPU.
    /// </summary>
    internal sealed unsafe class VulkanContext : IDisposable
    {
        public Vk Vk { get; } = Vk.GetApi();

        public Instance Instance;
        public PhysicalDevice PhysicalDevice;
        public Device Device;
        public Queue Queue;
        public uint QueueFamilyIndex;
        public CommandPool CommandPool;

        public string DeviceName { get; private set; } = string.Empty;
        public bool IsAvailable { get; private set; }

        private bool _disposed;

        public bool TryInitialize()
        {
            try
            {
                CreateInstance();
                if (!PickPhysicalDevice())
                    return false;
                CreateLogicalDevice();
                CreateCommandPool();
                IsAvailable = true;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GpuVulkan] unavailable, CPU fallback: {ex.Message}");
                Cleanup();
                return false;
            }
        }

        private void CreateInstance()
        {
            using var appName = SilkMarshal.StringToMemory("SimpleTransformer", NativeStringEncoding.UTF8);
            using var engineName = SilkMarshal.StringToMemory("SimpleTransformerBackend", NativeStringEncoding.UTF8);

            var appInfo = new ApplicationInfo
            {
                SType = StructureType.ApplicationInfo,
                PApplicationName = (byte*)appName.AsPtr<byte>(),
                ApplicationVersion = Vk.Version10,
                PEngineName = (byte*)engineName.AsPtr<byte>(),
                EngineVersion = Vk.Version10,
                ApiVersion = Vk.Version10
            };

            var createInfo = new InstanceCreateInfo
            {
                SType = StructureType.InstanceCreateInfo,
                PApplicationInfo = &appInfo
            };

            Result result = Vk.CreateInstance(createInfo, null, out Instance);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkCreateInstance failed: {result}");
            Vk.CurrentInstance = Instance;
        }

        private bool PickPhysicalDevice()
        {
            uint count = 0;
            if (Vk.EnumeratePhysicalDevices(Instance, &count, null) != Result.Success || count == 0)
                return false;

            var devices = new PhysicalDevice[count];
            fixed (PhysicalDevice* ptr = devices)
            {
                if (Vk.EnumeratePhysicalDevices(Instance, &count, ptr) != Result.Success)
                    return false;
            }

            PhysicalDevice firstCompute = default;
            bool haveFirst = false;

            foreach (var device in devices)
            {
                uint family = FindComputeQueueFamily(device);
                if (family == uint.MaxValue)
                    continue;

                Vk.GetPhysicalDeviceProperties(device, out PhysicalDeviceProperties props);
                string name = SilkMarshal.PtrToString((nint)props.DeviceName, NativeStringEncoding.UTF8) ?? "unknown";

                if (!haveFirst)
                {
                    firstCompute = device;
                    haveFirst = true;
                }

                if (props.DeviceType == PhysicalDeviceType.DiscreteGpu)
                {
                    PhysicalDevice = device;
                    QueueFamilyIndex = family;
                    DeviceName = name;
                    return true;
                }
            }

            if (haveFirst)
            {
                PhysicalDevice = firstCompute;
                QueueFamilyIndex = FindComputeQueueFamily(firstCompute);
                Vk.GetPhysicalDeviceProperties(firstCompute, out PhysicalDeviceProperties props);
                DeviceName = SilkMarshal.PtrToString((nint)props.DeviceName, NativeStringEncoding.UTF8) ?? "unknown";
                return true;
            }

            return false;
        }

        private uint FindComputeQueueFamily(PhysicalDevice device)
        {
            uint count = 0;
            Vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, null);
            if (count == 0)
                return uint.MaxValue;

            var families = new QueueFamilyProperties[count];
            fixed (QueueFamilyProperties* ptr = families)
            {
                Vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, ptr);
            }

            for (uint i = 0; i < count; i++)
            {
                if (families[i].QueueFlags.HasFlag(QueueFlags.ComputeBit) &&
                    !families[i].QueueFlags.HasFlag(QueueFlags.GraphicsBit))
                    return i;
            }
            for (uint i = 0; i < count; i++)
            {
                if (families[i].QueueFlags.HasFlag(QueueFlags.ComputeBit))
                    return i;
            }
            return uint.MaxValue;
        }

        private void CreateLogicalDevice()
        {
            float priority = 1.0f;
            var queueInfo = new DeviceQueueCreateInfo
            {
                SType = StructureType.DeviceQueueCreateInfo,
                QueueFamilyIndex = QueueFamilyIndex,
                QueueCount = 1,
                PQueuePriorities = &priority
            };

            var createInfo = new DeviceCreateInfo
            {
                SType = StructureType.DeviceCreateInfo,
                QueueCreateInfoCount = 1,
                PQueueCreateInfos = &queueInfo
            };

            Result result = Vk.CreateDevice(PhysicalDevice, createInfo, null, out Device);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkCreateDevice failed: {result}");

            Vk.CurrentDevice = Device;
            Vk.GetDeviceQueue(Device, QueueFamilyIndex, 0, out Queue);
        }

        private void CreateCommandPool()
        {
            var poolInfo = new CommandPoolCreateInfo
            {
                SType = StructureType.CommandPoolCreateInfo,
                Flags = CommandPoolCreateFlags.CommandPoolCreateResetCommandBufferBit,
                QueueFamilyIndex = QueueFamilyIndex
            };

            Result result = Vk.CreateCommandPool(Device, poolInfo, null, out CommandPool);
            if (result != Result.Success)
                throw new InvalidOperationException($"vkCreateCommandPool failed: {result}");
        }

        public void Synchronize()
        {
            if (IsAvailable && !_disposed)
                Vk.DeviceWaitIdle(Device);
        }

        /// <summary>
        /// Chooses a memory type for a host-visible working buffer. Preference
        /// order matters enormously: AMD's Windows driver lists the
        /// write-combined host heap FIRST, and the resizable-BAR heap looks
        /// attractive by name but is the slowest tier for *host* access.
        ///
        /// Measured on this RX 5700 XT for a 1 MiB upload+download round trip:
        ///
        ///   1. HOST_VISIBLE|HOST_COHERENT|HOST_CACHED   - system RAM the CPU
        ///      reads through cache. Measured ~43 000 MiB/s. This is the only
        ///      tier fast enough when every op does a host round trip.
        ///   2. DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT  - VRAM over PCIe
        ///      (the 256 MiB resizable-BAR window). Measured 37-52 MiB/s: ~1000x
        ///      SLOWER than host-cached, because uncached host reads cross PCIe.
        ///      Only worth it for buffers the GPU touches many times without a
        ///      host round trip, which this pack/dispatch/unpack path never does.
        ///   3. HOST_VISIBLE|HOST_COHERENT               - write-combined,
        ///      uncached. Measured 330-500 MiB/s. Correct, ~10x faster than BAR.
        ///   4. HOST_VISIBLE                              - needs explicit flush.
        /// </summary>
        public uint SelectMemoryType(uint typeBits, bool preferHostCached = true)
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);

            const MemoryPropertyFlags coherent =
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit;

            Span<MemoryPropertyFlags> preferences = stackalloc MemoryPropertyFlags[4];
            int count = 0;
            if (preferHostCached)
            {
                preferences[count++] = coherent | MemoryPropertyFlags.HostCachedBit;
                preferences[count++] = MemoryPropertyFlags.DeviceLocalBit | coherent;
            }
            else
            {
                preferences[count++] = MemoryPropertyFlags.DeviceLocalBit | coherent;
                preferences[count++] = coherent | MemoryPropertyFlags.HostCachedBit;
            }
            preferences[count++] = coherent;
            preferences[count++] = MemoryPropertyFlags.HostVisibleBit;

            for (int p = 0; p < count; p++)
            {
                for (uint i = 0; i < props.MemoryTypeCount; i++)
                {
                    if ((typeBits & (1u << (int)i)) == 0)
                        continue;
                    if ((props.MemoryTypes[(int)i].PropertyFlags & preferences[p]) == preferences[p])
                        return i;
                }
            }
            throw new InvalidOperationException("No suitable Vulkan memory type found.");
        }

        /// <summary>True when <paramref name="memoryTypeIndex"/> is device-local VRAM.</summary>
        public bool IsDeviceLocalMemoryType(uint memoryTypeIndex)
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);
            if (memoryTypeIndex >= props.MemoryTypeCount)
                return false;
            return (props.MemoryTypes[(int)memoryTypeIndex].PropertyFlags &
                    MemoryPropertyFlags.DeviceLocalBit) != 0;
        }

        /// <summary>
        /// Memory type indices this device exposes, with their flags. Used by the
        /// bandwidth probe to measure every host-visible tier instead of trusting
        /// the driver's ordering (AMD Windows lists the uncached write-combined
        /// heap before the cached one, which is exactly the wrong way round for
        /// per-op upload/download).
        /// </summary>
        public IReadOnlyList<(uint Index, MemoryPropertyFlags Flags)> EnumerateMemoryTypes()
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);
            var list = new List<(uint, MemoryPropertyFlags)>((int)props.MemoryTypeCount);
            for (uint i = 0; i < props.MemoryTypeCount; i++)
                list.Add((i, props.MemoryTypes[(int)i].PropertyFlags));
            return list;
        }

        /// <summary>True when <paramref name="memoryTypeIndex"/> can be mapped by the host.</summary>
        public bool IsHostVisibleMemoryType(uint memoryTypeIndex)
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);
            if (memoryTypeIndex >= props.MemoryTypeCount)
                return false;
            return (props.MemoryTypes[(int)memoryTypeIndex].PropertyFlags &
                    MemoryPropertyFlags.HostVisibleBit) != 0;
        }

        /// <summary>Human-readable property flags for one memory type (diagnostics).</summary>
        public string DescribeMemoryType(uint memoryTypeIndex)
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);
            if (memoryTypeIndex >= props.MemoryTypeCount)
                return "out of range";
            return DescribeFlags(props.MemoryTypes[(int)memoryTypeIndex].PropertyFlags);
        }

        /// <summary>Number of memory types reported by the chosen physical device.</summary>
        public uint MemoryTypeCount
        {
            get
            {
                Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);
                return props.MemoryTypeCount;
            }
        }

        /// <summary>Prints heaps and memory types; useful when a bandwidth problem is suspected.</summary>
        public void LogMemoryProperties()
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);

            Console.WriteLine($"  heaps ({props.MemoryHeapCount}):");
            for (uint i = 0; i < props.MemoryHeapCount; i++)
            {
                var heap = props.MemoryHeaps[(int)i];
                string kind = heap.Flags.HasFlag(MemoryHeapFlags.DeviceLocalBit)
                    ? "DEVICE_LOCAL"
                    : "host        ";
                Console.WriteLine($"    heap {i}: {kind} {heap.Size / 1024.0 / 1024.0,8:F0} MiB");
            }

            Console.WriteLine($"  memory types ({props.MemoryTypeCount}):");
            for (uint i = 0; i < props.MemoryTypeCount; i++)
            {
                var type = props.MemoryTypes[(int)i];
                Console.WriteLine($"    type {i,2}: heap {type.HeapIndex}  {DescribeFlags(type.PropertyFlags)}");
            }
        }

        private static string DescribeFlags(MemoryPropertyFlags flags)
        {
            var parts = new List<string>(4);
            if (flags.HasFlag(MemoryPropertyFlags.DeviceLocalBit)) parts.Add("DEVICE_LOCAL");
            if (flags.HasFlag(MemoryPropertyFlags.HostVisibleBit)) parts.Add("HOST_VISIBLE");
            if (flags.HasFlag(MemoryPropertyFlags.HostCoherentBit)) parts.Add("HOST_COHERENT");
            if (flags.HasFlag(MemoryPropertyFlags.HostCachedBit)) parts.Add("HOST_CACHED");
            if (flags.HasFlag(MemoryPropertyFlags.LazilyAllocatedBit)) parts.Add("LAZILY_ALLOCATED");
            return parts.Count == 0 ? "-" : string.Join("|", parts);
        }

        private void Cleanup()
        {
            if (CommandPool.Handle != 0 && Device.Handle != 0)
            {
                Vk.DestroyCommandPool(Device, CommandPool, null);
                CommandPool = default;
            }
            if (Device.Handle != 0)
            {
                Vk.DestroyDevice(Device, null);
                Device = default;
            }
            if (Instance.Handle != 0)
            {
                Vk.DestroyInstance(Instance, null);
                Instance = default;
            }
            IsAvailable = false;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (IsAvailable)
                Vk.DeviceWaitIdle(Device);
            Cleanup();
        }
    }
}
