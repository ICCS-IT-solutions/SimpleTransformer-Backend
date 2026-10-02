using Serilog;
using SimpleTransformer.Api;
using SimpleTransformer.Model;

namespace SimpleTransformer
{
    class Program
    {
        static void Main(string[] args)
        {
            // Optional accelerator switch for A/B measurement, e.g.
            //   dotnet run -c Release -- --vulkan-bench --vulkan-tier host
            //   dotnet run -c Release -- --vulkan-bench --vulkan-tier device
            // Keeps the accelerator configuration-free: the host process pushes
            // the choice into the settings static before any backend exists.
            int tierIndex = Array.IndexOf(args, "--vulkan-tier");
            if (tierIndex >= 0 && tierIndex + 1 < args.Length)
            {
                AccelerationEngine.GpuVulkan.VulkanMemorySettings.PerOpMemoryTier =
                    args[tierIndex + 1].Equals("host", StringComparison.OrdinalIgnoreCase)
                        ? AccelerationEngine.GpuVulkan.VulkanPerOpMemoryTier.HostCached
                        : AccelerationEngine.GpuVulkan.VulkanPerOpMemoryTier.DeviceLocal;
            }
            int tierMbIndex = Array.IndexOf(args, "--vulkan-tier-min-mb");
            if (tierMbIndex >= 0 && tierMbIndex + 1 < args.Length)
            {
                if (ulong.TryParse(args[tierMbIndex + 1], out ulong minMb))
                {
                    AccelerationEngine.GpuVulkan.VulkanMemorySettings.DeviceLocalThresholdBytes =
                        minMb * 1024UL * 1024UL;
                }
            }

            if (args.Contains("--vulkan-profile"))
            {
                AccelerationEngine.GpuVulkan.VulkanPhaseProfile.Enabled = true;
            }

            // Per-step training profiler: dotnet run -- --train-profile
            // Enables VulkanPhaseProfile plus one [train-profile] log line per
            // TrainStep (phase split + Vulkan deltas). Off by default.
            if (args.Contains("--train-profile"))
            {
                AccelerationEngine.GpuVulkan.VulkanPhaseProfile.Enabled = true;
                SimpleTransformer.Model.TrainingStepProfile.Enabled = true;

                // Diagnostic flags exit below before the server path configures
                // Serilog, and Serilog's default logger drops everything, so the
                // profile lines would vanish. Bring the sinks up here instead.
                ConfigureLogging();
            }


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

            // Token-cache (.stbin) round-trip + streaming parity:
            // dotnet run -- --tokencache-selftest
            if (args.Contains("--tokencache-selftest"))
            {
                bool ok = SimpleTransformer.AccelerationEngine.TokenCacheSelfTest.RunAndPrint();
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

            // Memory pressure valve self-test (no server start):
            // dotnet run -- --memory-valve-selftest
            // Exercises the quota clamp and the whole escalation ladder from
            // synthetic samples, so the policy is verified without having to
            // drive a real heap to 95% first.
            if (args.Contains("--memory-valve-selftest"))
            {
                try
                {
                    // Apply the real config when present, so the self-test
                    // exercises the same bounds the server would run with.
                    var cfg = new Config.ConfigManager();
                    cfg.LoadFromFile("config/config.ini");
                    Server.ApplyMemoryPressureSettings(cfg);
                }
                catch
                {
                    // No config available - fall back to the defaults.
                }

                bool ok = SimpleTransformer.Model.MemoryPressureSelfTest.RunAndPrint();
                Environment.Exit(ok ? 0 : 1);
            }

            // Dropout layer self-test (no server start): dotnet run -- --dropout-selftest
            if (args.Contains("--dropout-selftest"))
            {
                bool ok = SimpleTransformer.Model.DropoutSelfTest.RunAndPrint();
                Environment.Exit(ok ? 0 : 1);
            }
            ConfigureLogging();

            //Inject the model via constructor DI 
            var server = new Server();
            
            //Start the server
            server.Start();
        }

        private static bool _loggingConfigured;

        /// <summary>
        /// Single place that decides where log lines go: console plus a rolling
        /// daily file. Used by the server start path and by diagnostic flags
        /// (<c>--train-profile</c>) that must emit logs before the server runs.
        /// Idempotent, so a diagnostic flag can bring the sinks up early without
        /// the server path opening a second file sink on the same log.
        /// </summary>
        private static void ConfigureLogging()
        {
            if (_loggingConfigured)
                return;

            _loggingConfigured = true;
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .WriteTo.Console()
                .WriteTo.File(
                    "logs/server-.log",
                    rollingInterval: RollingInterval.Day)
                .CreateLogger();
        }
    }
}