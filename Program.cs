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