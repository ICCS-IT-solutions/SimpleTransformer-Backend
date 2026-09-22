using System;
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
