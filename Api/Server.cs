using SimpleTransformer.Api.Endpoints.Services.Extensions;
using SimpleTransformer.Model;
using Serilog;
using SimpleTransformer.Model.Tokenizer;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Config;
using SimpleTransformer.AppDb;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;
using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Api.ManagementEngine;
using MemSettings = SimpleTransformer.Model.MemoryPressureSettings;

namespace SimpleTransformer.Api
{
    // Host-memory pressure relief: the valve gates on max(process, system)
    // load (see MemoryPressureValve), preemptively releasing pooled resources
    // and letting the GC reclaim before the box reaches swap.
    public class Server
    {
        private static ConfigManager _configManager = new ConfigManager();

        public void Start()
        {
            try
            {
                _configManager.LoadFromFile("config/config.ini");

                //GPU (VRAM) budget for the Vulkan backend: 0/absent = auto-detect
                //from the driver when the first backend is constructed. The host
                //staging budget guards system RAM the same way.
                AccelerationEngine.GpuVulkan.VulkanMemorySettings.BudgetBytesOverride =
                    _configManager.GetAs<long>("memory_budget_mb", 0, "Vulkan") * 1024L * 1024L;
                AccelerationEngine.GpuVulkan.VulkanMemorySettings.HostBudgetBytesOverride =
                    _configManager.GetAs<long>("host_memory_budget_mb", 0, "Vulkan") * 1024L * 1024L;

                //How long a recorded Vulkan batch may take to complete before the
                //waiting caller gives up. A wedged batch then surfaces as a clear
                //error in seconds instead of stalling a training job for minutes.
                //Raise it on devices where first-run shader compilation or very
                //large batches legitimately take longer.
                int dispatchTimeoutMs = _configManager.GetAs<int>("dispatch_timeout_ms", 30000, "Vulkan");
                if (dispatchTimeoutMs > 0)
                {
                    AccelerationEngine.GpuVulkan.VulkanKernelLauncher.DispatchTimeoutMs =
                        dispatchTimeoutMs;
                }

                //Host-memory pressure relief for the managed heap. The [Vulkan]
                //budgets above cover unmanaged device buffers only; model weights,
                //AdamW moments, gradients and pooled activations all live on the
                //managed heap and are what a long run actually exhausts. Pushed
                //here, before any model is constructed, so the statics are set
                //before the first TrainStep samples them.
                ApplyMemoryPressureSettings(_configManager);

                SQLitePCL.Batteries.Init();

                var builder = WebApplication.CreateBuilder();

                builder.Host.UseSerilog();

                builder.WebHost.ConfigureKestrel(options =>
                {
                    options.ListenAnyIP(5000);
                    // Large corpus uploads (.jsonl up to 2GB) stream line-by-line;
                    // Kestrel's default 30MB cap would reject them with 413.
                    options.Limits.MaxRequestBodySize = CorpusStreamPipeline.MaxRequestBytes;
                });  

                builder.Services.AddSingleton<ConfigManager>(_configManager); 

                // 1. Database Contexts
                // NOTE: Only the factory is registered here.
                builder.Services.AddDbContextFactory<AppDbContext>(options =>
                    DbContextConfiguration.ConfigureDbContext(options, _configManager));

                // 2. Factories and Core Components (Scoped / Transient)
                builder.Services.AddSingleton<ITransformerModelFactory, TransformerModelFactory>();
                builder.Services.AddSingleton<ModelManager>();

                // 3. MVC & Open API
                builder.Services.AddControllers();
                builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
                {
                    // Multipart section cap must match the Kestrel body cap or
                    // large corpus uploads fail binding with
                    // "Multipart body length limit 134217728 exceeded".
                    options.MultipartBodyLengthLimit = CorpusStreamPipeline.MaxRequestBytes;
                    options.ValueLengthLimit = int.MaxValue;
                    options.MultipartHeadersLengthLimit = int.MaxValue;
                });
                builder.Services.AddEndpointsApiExplorer();
                builder.Services.AddSwaggerGen();

                builder.Services.AddCors(options =>
                {
                    options.AddDefaultPolicy(policy =>
                    {
                        policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader();
                    });
                    options.AddPolicy("Frontend", pol =>
                    {
                        pol.WithOrigins("http://localhost:5173","https://192.168.3.3:5173","http://127.0.0.1:5173").AllowAnyMethod().AllowAnyHeader();
                    });
                });

                // 4. Tokenizer & Vocab (Singletons)
                builder.Services.AddSingleton<Vocabulary>(provider =>
                {
                    const string vocabularyFile = "vocabulary.json";
                    if (!File.Exists(vocabularyFile)) 
                        throw new FileNotFoundException("Vocabulary file not found.", vocabularyFile);

                    var loader = new JsonVocabularyLoader();
                    return loader.LoadFromFile(vocabularyFile);
                });

                builder.Services.AddSingleton<IVocabularyCompiler, SentencePieceVocabularyCompiler>();

                builder.Services.AddSingleton<ITokenizer>(provider =>
                {
                    var vocab = provider.GetRequiredService<Vocabulary>();
                    return new SentencePieceTokenizer(vocab);
                });

                // 5. Stateful Managers (Must be Singletons)
                builder.Services.AddSingleton<TrainingJobManager>();

                // 6. Request API Services (Should be Scoped)
                builder.Services.AddScoped<VocabularyService>();
                builder.Services.AddScoped<TrainingService>();
                builder.Services.AddScoped<TrainingCorpusService>();
                builder.Services.AddScoped<InferenceService>();
                builder.Services.AddScoped<TransformerModelService>();
                builder.Services.AddScoped<ConfigService>();

                var app = builder.Build();

                // 7. Safe Initialization via the singleton DbContext factory
                var dbFactory = app.Services.GetRequiredService<IDbContextFactory<AppDbContext>>();
                using (var dbContext = dbFactory.CreateDbContext())
                {
                    DbContextConfiguration.InitializeDatabase(dbContext);
                }

                //A process restart loses every in-memory guard (training loops,
                //loaded model), so move stale rows back to a state the UI can act
                //on instead of leaving jobs reading Running forever.
                TrainingJobExtensions.ReconcileDbStateOnStartupAsync(dbFactory)
                    .GetAwaiter().GetResult();

                app.UseCors("Frontend");

                if (app.Environment.IsDevelopment())
                {
                    app.UseSwagger();
                    app.UseSwaggerUI();
                }

                Log.Information("REST API ready.");
                Log.Information("Listening for requests...");

                app.MapControllers();
                app.Run();    
            }
            catch (Exception ex)
            {
                Log.Warning($"{ex.Message}\nStack trace: {ex.StackTrace}");
            }
        }

        /// <summary>
        /// Reads the <c>[Memory]</c> section into <see cref="Model.MemoryPressureSettings"/>.
        /// Public and static so the diagnostic entry points
        /// (<c>--memory-valve-selftest</c>) apply the same config the server would
        /// rather than testing defaults.
        /// </summary>
        public static void ApplyMemoryPressureSettings(ConfigManager configManager)
        {
            const long mb = 1024L * 1024L;

            MemSettings.Enabled = configManager.GetAs<bool>("enabled", true, "Memory");
            MemSettings.MonitorSystemPressure = configManager.GetAs<bool>("monitor_system_pressure", true, "Memory");
            MemSettings.MaxQuotaBytes = Math.Max(0L, configManager.GetAs<long>("max_quota_mb", 0, "Memory")) * mb;
            MemSettings.MinQuotaBytes = Math.Max(0L, configManager.GetAs<long>("min_quota_mb", 0, "Memory")) * mb;
            MemSettings.LowerBoundPercent = configManager.GetAs<double>("lower_bound_percent", 70.0, "Memory");
            MemSettings.UpperBoundPercent = configManager.GetAs<double>("upper_bound_percent", 85.0, "Memory");
            MemSettings.CheckEveryNSteps = configManager.GetAs<int>("check_every_n_steps", 25, "Memory");
            MemSettings.MinCooldownSteps = configManager.GetAs<int>("min_cooldown_steps", 250, "Memory");
            MemSettings.WorkspaceRetainPercent = configManager.GetAs<double>("workspace_retain_percent", 50.0, "Memory");
            MemSettings.WorkspaceCapFractionOfQuota = configManager.GetAs<double>("workspace_cap_fraction_of_quota", 0.35, "Memory");
            MemSettings.AllowBlockingCompact = configManager.GetAs<bool>("allow_blocking_compact", true, "Memory");

            long physical = System.GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long resolved = Model.MemoryPressureValve.ResolveQuotaBytes(
                MemSettings.MaxQuotaBytes,
                MemSettings.MinQuotaBytes,
                physical);

            Log.Information(
                "Memory valve: {State} (system monitor {SysMon}), quota {Quota:F0} MiB (configured {Configured:F0}, machine {Physical:F0} MiB), band {Lower:F0}-{Upper:F0}% of max(proc, sys), every {Every} steps, cooldown {Cooldown}, pool cap {PoolCap:P0} of quota.",
                MemSettings.Enabled ? "enabled" : "disabled",
                MemSettings.MonitorSystemPressure ? "on" : "off",
                resolved / mb,
                MemSettings.MaxQuotaBytes / mb,
                physical / mb,
                MemSettings.LowerBoundPercent,
                MemSettings.UpperBoundPercent,
                MemSettings.CheckEveryNSteps,
                MemSettings.MinCooldownSteps,
                MemSettings.WorkspaceCapFractionOfQuota);
        }

        public class AppDbContextFactory : IDesignTimeDbContextFactory<AppDbContext>
        {
            // Change return type from IDesignTimeDbContextFactory<AppDbContext> to AppDbContext
            public AppDbContext CreateDbContext(string[] args)
            {
                var configManager = new ConfigManager();
                var optionsBuilder = new DbContextOptionsBuilder<AppDbContext>();

                DbContextConfiguration.ConfigureDbContext(optionsBuilder, configManager);
                
                return new AppDbContext(optionsBuilder.Options);
            }
        }
    }
}