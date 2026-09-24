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
        private bool _supportsProperties2;
        private bool _memoryBudgetEnabled;

        //VK_API_VERSION_1_1 = (major << 22) | (minor << 12)
        private const uint ApiVersion11 = (1u << 22) | (1u << 12);

        private const string MemoryBudgetExtensionName = "VK_EXT_memory_budget";

        /// <summary>
        /// Largest DEVICE_LOCAL heap of the chosen device - the card's VRAM
        /// (8 GiB on an RX 5700 XT). 0 when the device reports none.
        /// </summary>
        public ulong DeviceLocalHeapSizeBytes { get; private set; }

        /// <summary>Heap index the VRAM figures refer to; uint.MaxValue when none.</summary>
        public uint DeviceLocalHeapIndex { get; private set; } = uint.MaxValue;

        /// <summary>True when the driver's live budget (VK_EXT_memory_budget) is enabled.</summary>
        public bool MemoryBudgetSupported => _memoryBudgetEnabled;

        /// <summary>
        /// Driver budget for this process on the VRAM heap (0 = unknown). Can sit
        /// below the heap size when other apps or the OS hold memory.
        /// </summary>
        public ulong DeviceLocalBudgetBytes { get; private set; }

        /// <summary>Bytes this process currently holds on the VRAM heap (0 = unknown).</summary>
        public ulong DeviceLocalUsageBytes { get; private set; }

        /// <summary>
        /// True when Resizable BAR is in effect: the main device-local heap is
        /// itself host-visible, so the CPU can map VRAM directly instead of going
        /// through a small aperture.
        /// <para>
        /// Detection is by heap layout rather than a driver-specific extension.
        /// With ReBAR enabled the whole VRAM heap is exposed as
        /// DEVICE_LOCAL|HOST_VISIBLE. With it disabled, the main VRAM heap has no
        /// host-visible memory types at all and the driver instead advertises a
        /// separate, small device-local host-visible heap - the legacy BAR window
        /// (256 MiB on an RX 5700 XT) - which is orders of magnitude slower for
        /// host traffic. That layout difference is what this flag reports.
        /// </para>
        /// </summary>
        public bool ResizableBarEnabled { get; private set; }

        /// <summary>
        /// Total device-local memory the host can actually map, i.e. the BAR
        /// aperture. Equal to the VRAM heap when ReBAR is enabled; a small
        /// fraction of it when ReBAR is off.
        /// </summary>
        public ulong HostVisibleDeviceLocalBytes { get; private set; }

        /// <summary>
        /// True when a working set (activations, workspace, gradients) could
        /// reasonably live in VRAM instead of being shuttled over PCIe each op.
        /// <para>
        /// This requires ReBAR. Without it the only host-mappable device memory is
        /// the small BAR window, which measures in the tens of MiB/s here - three
        /// orders of magnitude slower than cached system RAM - so placing a
        /// per-op working set there would be far worse than leaving it in host
        /// memory. With ReBAR the whole heap is mappable and the trade reverses,
        /// because data can stay resident on the device instead of crossing the
        /// bus repeatedly.
        /// </para>
        /// <para>
        /// Note this is about a *resident* working set. It does not change the
        /// preference for host-cached system RAM for the existing
        /// pack/dispatch/unpack path, which still round-trips through the host
        /// every op and therefore still favours the fastest host tier.
        /// </para>
        /// </summary>
        public bool SupportsVramWorkingSet => ResizableBarEnabled;

        /// <summary>
        /// Device memory a resident working set could use: the mappable device-local
        /// size when ReBAR is on, otherwise 0 because the BAR window is too small
        /// and too slow to be useful for per-op data.
        /// </summary>
        public ulong VramWorkingSetBudgetBytes =>
            ResizableBarEnabled ? HostVisibleDeviceLocalBytes : 0;

        public bool TryInitialize()
        {
            try
            {
                CreateInstance();
                if (!PickPhysicalDevice())
                    return false;
                CreateLogicalDevice();
                CreateCommandPool();
                DetectDeviceMemory();
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
                //Vulkan 1.1 unlocks vkGetPhysicalDeviceMemoryProperties2, which is
                //how the live VRAM budget (VK_EXT_memory_budget) is queried.
                ApiVersion = ApiVersion11
            };

            var createInfo = new InstanceCreateInfo
            {
                SType = StructureType.InstanceCreateInfo,
                PApplicationInfo = &appInfo
            };

            Result result = Vk.CreateInstance(in createInfo, null, out Instance);
            if (result == Result.Success)
            {
                _supportsProperties2 = true;
            }
            else
            {
                //A 1.0-only driver is still useful: the fixed heap size (total VRAM)
                //stays detectable, only the live budget query is lost.
                appInfo.ApiVersion = Vk.Version10;
                result = Vk.CreateInstance(in createInfo, null, out Instance);
            }

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

            //VK_EXT_memory_budget exposes the driver's live per-heap budget/usage
            //(what is actually free right now) instead of only the fixed heap size.
            //It is a device extension, so it must be requested here.
            bool budgetRequested = false;
            Result result;

            if (_supportsProperties2 && DeviceSupportsExtension(MemoryBudgetExtensionName))
            {
                using var extName = SilkMarshal.StringToMemory(MemoryBudgetExtensionName);
                byte** extList = stackalloc byte*[1];
                extList[0] = (byte*)extName.AsPtr<byte>();

                createInfo.EnabledExtensionCount = 1;
                createInfo.PpEnabledExtensionNames = extList;
                budgetRequested = true;

                result = Vk.CreateDevice(PhysicalDevice, in createInfo, null, out Device);
            }
            else
            {
                result = Vk.CreateDevice(PhysicalDevice, in createInfo, null, out Device);
            }

            if (result != Result.Success && budgetRequested)
            {
                //The driver listed the extension but refused the device with it
                //enabled; retry without it rather than losing the GPU entirely.
                createInfo.EnabledExtensionCount = 0;
                createInfo.PpEnabledExtensionNames = null;
                budgetRequested = false;
                result = Vk.CreateDevice(PhysicalDevice, in createInfo, null, out Device);
            }

            if (result != Result.Success)
                throw new InvalidOperationException($"vkCreateDevice failed: {result}");

            _memoryBudgetEnabled = budgetRequested;

            Vk.CurrentDevice = Device;
            Vk.GetDeviceQueue(Device, QueueFamilyIndex, 0, out Queue);
        }

        /// <summary>True when the chosen device reports the named extension.</summary>
        private bool DeviceSupportsExtension(string extensionName)
        {
            uint count = 0;
            //Explicit byte* casts: Silk also offers string-marshalling overloads
            //and a bare null would be ambiguous between the two.
            if (Vk.EnumerateDeviceExtensionProperties(PhysicalDevice, (byte*)null, &count, (ExtensionProperties*)null) != Result.Success ||
                count == 0)
                return false;

            var extensions = new ExtensionProperties[count];
            fixed (ExtensionProperties* ptr = extensions)
            {
                if (Vk.EnumerateDeviceExtensionProperties(PhysicalDevice, (byte*)null, &count, ptr) != Result.Success)
                    return false;

                for (uint i = 0; i < count; i++)
                {
                    //ExtensionName is the first field of ExtensionProperties
                    //(char[VK_MAX_EXTENSION_NAME_SIZE] in vk.xml), so the struct's
                    //address is the string's address.
                    var name = SilkMarshal.PtrToString((nint)(&ptr[i]), NativeStringEncoding.UTF8);
                    if (name == extensionName)
                        return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Detects the device-local (VRAM) heap size and, when VK_EXT_memory_budget
        /// is enabled, the driver's live budget and this process's usage on it.
        /// Called once during initialization after the device is created.
        /// </summary>
        private void DetectDeviceMemory()
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);

            ulong bestSize = 0;
            uint bestIndex = uint.MaxValue;
            for (uint i = 0; i < props.MemoryHeapCount; i++)
            {
                var heap = props.MemoryHeaps[(int)i];
                if (heap.Flags.HasFlag(MemoryHeapFlags.DeviceLocalBit) && heap.Size > bestSize)
                {
                    bestSize = heap.Size;
                    bestIndex = i;
                }
            }

            DeviceLocalHeapSizeBytes = bestSize;
            DeviceLocalHeapIndex = bestIndex;

            DetectResizableBar(props, bestIndex);

            if (!_memoryBudgetEnabled || bestIndex == uint.MaxValue)
                return;

            //VK_EXT_memory_budget: budget/usage are fixed ulong[VK_MAX_MEMORY_HEAPS]
            //arrays inside the extension struct, filled by the chained query.
            var budget = new PhysicalDeviceMemoryBudgetPropertiesEXT
            {
                SType = StructureType.PhysicalDeviceMemoryBudgetPropertiesExt
            };
            var info = new PhysicalDeviceMemoryProperties2
            {
                SType = StructureType.PhysicalDeviceMemoryProperties2,
                PNext = &budget
            };

            Vk.GetPhysicalDeviceMemoryProperties2(PhysicalDevice, &info);

            DeviceLocalBudgetBytes = budget.HeapBudget[(int)bestIndex];
            DeviceLocalUsageBytes = budget.HeapUsage[(int)bestIndex];
        }

        /// <summary>
        /// Works out whether Resizable BAR is in effect from the heap layout, and
        /// how much device-local memory the host can actually map.
        ///
        /// A device-local heap is host-mappable when at least one of its memory
        /// types carries HOST_VISIBLE. ReBAR is considered enabled only when that
        /// is true for the *main* VRAM heap: with ReBAR off, the main heap has no
        /// host-visible types and the driver exposes a separate small aperture
        /// heap instead, which is what makes host access to VRAM so slow.
        /// </summary>
        private void DetectResizableBar(
            PhysicalDeviceMemoryProperties props,
            uint deviceLocalHeapIndex)
        {
            ulong hostVisibleDeviceLocal = 0;
            bool mainHeapIsHostVisible = false;

            for (uint h = 0; h < props.MemoryHeapCount; h++)
            {
                var heap = props.MemoryHeaps[(int)h];
                if (!heap.Flags.HasFlag(MemoryHeapFlags.DeviceLocalBit))
                    continue;

                bool heapHasHostVisibleType = false;
                for (uint m = 0; m < props.MemoryTypeCount; m++)
                {
                    var type = props.MemoryTypes[(int)m];
                    if (type.HeapIndex != h)
                        continue;

                    if ((type.PropertyFlags & MemoryPropertyFlags.HostVisibleBit) != 0)
                        heapHasHostVisibleType = true;
                }

                if (heapHasHostVisibleType)
                    hostVisibleDeviceLocal += heap.Size;

                if (h == deviceLocalHeapIndex)
                    mainHeapIsHostVisible = heapHasHostVisibleType;
            }

            ResizableBarEnabled =
                deviceLocalHeapIndex != uint.MaxValue && mainHeapIsHostVisible;
            HostVisibleDeviceLocalBytes = hostVisibleDeviceLocal;
        }

        /// <summary>
        /// One-line human-readable ReBAR status, e.g.
        /// "Resizable BAR: enabled (7920 MiB of VRAM host-mappable)" or
        /// "Resizable BAR: disabled (256 MiB BAR window of 7920 MiB VRAM mappable)".
        /// </summary>
        public string DescribeResizableBar()
        {
            const double mib = 1024.0 * 1024.0;

            if (DeviceLocalHeapIndex == uint.MaxValue)
                return "Resizable BAR: unknown (no device-local heap)";

            return ResizableBarEnabled
                ? $"Resizable BAR: enabled ({DeviceLocalHeapSizeBytes / mib:F0} MiB of VRAM host-mappable)"
                : $"Resizable BAR: disabled ({HostVisibleDeviceLocalBytes / mib:F0} MiB BAR window " +
                  $"of {DeviceLocalHeapSizeBytes / mib:F0} MiB VRAM mappable)";
        }

        private void CreateCommandPool()
        {
            var poolInfo = new CommandPoolCreateInfo
            {
                SType = StructureType.CommandPoolCreateInfo,
                Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
                QueueFamilyIndex = QueueFamilyIndex
            };

            Result result = Vk.CreateCommandPool(Device, in poolInfo, null, out CommandPool);
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

        /// <summary>
        /// Memory type for a GPU-resident buffer (weights/scratch the GPU reads
        /// many times and the host writes once). Preference:
        ///   1. DEVICE_LOCAL without HOST_VISIBLE - true VRAM, full bandwidth for
        ///      compute reads. Unmapped, so it has to be filled with a staged
        ///      device copy.
        ///   2. DEVICE_LOCAL with HOST_VISIBLE - the resizable-BAR window (256 MiB
        ///      on this RX 5700 XT). Also VRAM-speed for the GPU.
        /// Returns uint.MaxValue when the device has no device-local type at all,
        /// so callers can fall back to ordinary host-visible buffers.
        /// </summary>
        public uint SelectDeviceLocalMemoryType(uint typeBits)
        {
            Vk.GetPhysicalDeviceMemoryProperties(PhysicalDevice, out PhysicalDeviceMemoryProperties props);

            for (int pass = 0; pass < 2; pass++)
            {
                bool requireUnhosted = pass == 0;
                for (uint i = 0; i < props.MemoryTypeCount; i++)
                {
                    if ((typeBits & (1u << (int)i)) == 0)
                        continue;

                    var flags = props.MemoryTypes[(int)i].PropertyFlags;
                    if ((flags & MemoryPropertyFlags.DeviceLocalBit) == 0)
                        continue;

                    bool hostVisible = (flags & MemoryPropertyFlags.HostVisibleBit) != 0;

                    //Pass 0 wants VRAM the host cannot map; pass 1 accepts the
                    //host-visible device-local window.
                    if (hostVisible == requireUnhosted)
                        continue;

                    return i;
                }
            }

            return uint.MaxValue;
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

            Console.WriteLine($"  VRAM: {DeviceLocalHeapSizeBytes / 1024.0 / 1024.0:F0} MiB" +
                (MemoryBudgetSupported
                    ? $", driver budget {DeviceLocalBudgetBytes / 1024.0 / 1024.0:F0} MiB" +
                      $", used by this process {DeviceLocalUsageBytes / 1024.0 / 1024.0:F0} MiB"
                    : " (live budget unavailable: VK_EXT_memory_budget not enabled)"));

            Console.WriteLine($"  {DescribeResizableBar()}");

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
