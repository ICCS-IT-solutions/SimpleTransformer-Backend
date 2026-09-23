using Serilog;
using SimpleTransformer.Api;
using SimpleTransformer.Model;

namespace SimpleTransformer
{
    class Program
    {
        static void Main(string[] args)
        {
            // Backend parity self-test (no server start): dotnet run -- --backend-selftest
            if (args.Contains("--backend-selftest"))
            {
                bool ok = SimpleTransformer.AccelerationEngine.BackendSelfTest.RunAndPrint();
                Environment.Exit(ok ? 0 : 1);
            }

            // End-to-end pipeline smoke test (no server start): dotnet run -- --pipeline-smoketest
            if (args.Contains("--pipeline-smoketest"))
            {
                bool ok = SimpleTransformer.AccelerationEngine.PipelineSmokeTest.RunAndPrint();
                Environment.Exit(ok ? 0 : 1);
            }

            // Vulkan backend bring-up (no server start): dotnet run -- --vulkan-selftest
            if (args.Contains("--vulkan-selftest"))
            {
                bool ok = SimpleTransformer.AccelerationEngine.GpuVulkan.VulkanSelfTest.RunAndPrint();
                Environment.Exit(ok ? 0 : 1);
            }

            // Vulkan Phase 4 latency + allocation harness: dotnet run -c Release -- --vulkan-bench
            if (args.Contains("--vulkan-bench"))
            {
                bool ok = SimpleTransformer.AccelerationEngine.GpuVulkan.VulkanBenchmark.RunAndPrint();
                Environment.Exit(ok ? 0 : 1);
            }

            // Vulkan heap/memory-type diagnostic: dotnet run -c Release -- --vulkan-meminfo
            if (args.Contains("--vulkan-meminfo"))
            {
                //Apply the same config override the server would, so the probe
                //reports the effective budget in real use.
                try
                {
                    var cfg = new Config.ConfigManager();
                    cfg.LoadFromFile("config/config.ini");
                    AccelerationEngine.GpuVulkan.VulkanMemorySettings.BudgetBytesOverride =
                        cfg.GetAs<long>("memory_budget_mb", 0, "Vulkan") * 1024L * 1024L;
                    AccelerationEngine.GpuVulkan.VulkanMemorySettings.HostBudgetBytesOverride =
                        cfg.GetAs<long>("host_memory_budget_mb", 0, "Vulkan") * 1024L * 1024L;
                }
                catch
                {
                    //No config available - auto-detect.
                }

                using var probe = new SimpleTransformer.AccelerationEngine.GpuVulkan.GpuVulkanBackend();
                probe.LogMemoryInfo();
                Environment.Exit(0);
            }

            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .WriteTo.Console()
                .WriteTo.File(
                    "logs/server-.log",
                    rollingInterval: RollingInterval.Day)
                .CreateLogger();

            //Inject the model via constructor DI 
            var server = new Server();
            
            //Start the server
            server.Start();
        }
    }
}