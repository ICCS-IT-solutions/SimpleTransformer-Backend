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

namespace SimpleTransformer.Api
{
    public class Server
    {
        private static ConfigManager _configManager = new ConfigManager();

        public void Start()
        {
            try
            {
                _configManager.LoadFromFile("config/config.ini");

                SQLitePCL.Batteries.Init();

                var builder = WebApplication.CreateBuilder();

                builder.Host.UseSerilog();

                builder.WebHost.ConfigureKestrel(options =>
                {
                    options.ListenAnyIP(5000);
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
                        pol.WithOrigins("http://localhost:5173").AllowAnyMethod().AllowAnyHeader();
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