using System;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Verifies the temp-checkpoint cadence policy in
    /// <see cref="CheckpointCadenceSettings"/> without starting a server.
    /// The policy is a pure function of (steps since last write, time since last
    /// write), so it is tested directly rather than through a training loop:
    /// the master switch, each trigger, the zero-interval cases, and the epoch
    /// arithmetic that motivated the change.
    /// <para>
    /// Run with: dotnet run -- --checkpoint-cadence-selftest
    /// </para>
    /// </summary>
    public static class CheckpointCadenceSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Checkpoint Cadence Self-Test ===");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            bool origEnabled = CheckpointCadenceSettings.Enabled;
            int origSteps = CheckpointCadenceSettings.TempCheckpointEverySteps;
            double origMinutes = CheckpointCadenceSettings.TempCheckpointEveryMinutes;
            int origEpochInterval = CheckpointCadenceSettings.EpochCheckpointInterval;

            bool Should(int steps, double minutes) =>
                CheckpointCadenceSettings.ShouldWriteTempCheckpoint(
                    steps, TimeSpan.FromMinutes(minutes));

            try
            {
                CheckpointCadenceSettings.Enabled = true;
                CheckpointCadenceSettings.TempCheckpointEverySteps = 50;
                CheckpointCadenceSettings.TempCheckpointEveryMinutes = 5.0;

                Console.WriteLine("-- Master switch --");
                CheckpointCadenceSettings.Enabled = false;
                Check("disabled suppresses every write",
                      !Should(1000, 600) && !Should(1, 600));
                CheckpointCadenceSettings.Enabled = true;
                Check("enabled honours the intervals", Should(50, 0));

                Console.WriteLine();
                Console.WriteLine("-- Step trigger (interval 50) --");
                Check("below the interval does not fire", !Should(49, 0));
                Check("at the interval fires", Should(50, 0));
                Check("well past the interval fires", Should(500, 0));

                Console.WriteLine();
                Console.WriteLine("-- Time trigger (interval 5 min) --");
                Check("just under the interval does not fire", !Should(0, 4.9));
                Check("just over the interval fires", Should(0, 5.1));

                Console.WriteLine();
                Console.WriteLine("-- Either trigger --");
                Check("steps trigger fires before time does", Should(50, 0.5));
                Check("time trigger fires before steps do", Should(3, 5.1));

                Console.WriteLine();
                Console.WriteLine("-- Zeroed intervals disable one trigger each --");
                CheckpointCadenceSettings.TempCheckpointEverySteps = 0;
                Check("steps=0 kills the step trigger", !Should(10_000, 1));
                Check("steps=0 leaves the time trigger alive", Should(0, 5.1));
                CheckpointCadenceSettings.TempCheckpointEverySteps = 50;

                CheckpointCadenceSettings.TempCheckpointEveryMinutes = 0;
                Check("minutes=0 kills the time trigger", !Should(10, 99_999));
                Check("minutes=0 leaves the step trigger alive", Should(50, 99_999));
                CheckpointCadenceSettings.TempCheckpointEveryMinutes = 5.0;

                CheckpointCadenceSettings.TempCheckpointEverySteps = 0;
                CheckpointCadenceSettings.TempCheckpointEveryMinutes = 0;
                Check("both intervals zero means never", !Should(10_000_000, 100_000));
                CheckpointCadenceSettings.TempCheckpointEverySteps = 50;
                CheckpointCadenceSettings.TempCheckpointEveryMinutes = 5.0;
Console.WriteLine();
                Console.WriteLine("-- Epoch checkpoint interval floor --");
                Check("minimum is 5", CheckpointCadenceSettings.MinimumEpochCheckpointInterval == 5,
                      $"floor = {CheckpointCadenceSettings.MinimumEpochCheckpointInterval}");
                Check("a larger configured value is kept",
                      CheckpointCadenceSettings.ClampEpochCheckpointInterval(25) == 25, "25 -> 25");
                Check("exactly the floor is kept",
                      CheckpointCadenceSettings.ClampEpochCheckpointInterval(5) == 5, "5 -> 5");
                Check("below the floor is raised",
                      CheckpointCadenceSettings.ClampEpochCheckpointInterval(1) == 5, "1 -> 5");
                Check("zero and negative are raised",
                      CheckpointCadenceSettings.ClampEpochCheckpointInterval(0) == 5
                      && CheckpointCadenceSettings.ClampEpochCheckpointInterval(-7) == 5,
                      "0 -> 5, -7 -> 5");
                Check("default is the floor",
                      CheckpointCadenceSettings.EpochCheckpointInterval ==
                      CheckpointCadenceSettings.MinimumEpochCheckpointInterval,
                      $"default = {CheckpointCadenceSettings.EpochCheckpointInterval}");

                Console.WriteLine();
                Console.WriteLine("-- Epoch arithmetic (the reason this exists) --");
                const int epochSteps = 560;          // ~4.5M tokens at batch*seq ~8000
                const double checkpointMiB = 3340.0; // 208.8M params x 16 bytes

                int newWrites = 0;
                int lastAt = 0;
                for (int step = 1; step <= epochSteps; step++)
                {
                    if (Should(step - lastAt, 0))
                    {
                        newWrites++;
                        lastAt = step;
                    }
                }
                int oldWrites = epochSteps / 8;      // old: one per outer batch, capped at 8

                Console.WriteLine($"  old (every 8 steps) : {oldWrites,3} writes/epoch = " +
                                  $"{oldWrites * checkpointMiB / 1024.0,7:F0} GB written");
                Console.WriteLine($"  new (every 50 steps): {newWrites,3} writes/epoch = " +
                                  $"{newWrites * checkpointMiB / 1024.0,7:F0} GB written");

                Check("cuts checkpoint writes for a long epoch",
                      newWrites * 3 < oldWrites,
                      $"{oldWrites} -> {newWrites} per epoch " +
                      $"({(double)oldWrites / newWrites:F1}x less I/O)");
                Check("a crash still costs at most one interval",
                      50 < epochSteps,
                      $"worst case 50 of {epochSteps} steps " +
                      $"({100.0 * 50 / epochSteps:F1}% of the epoch)");
            }
            finally
            {
                CheckpointCadenceSettings.Enabled = origEnabled;
                CheckpointCadenceSettings.TempCheckpointEverySteps = origSteps;
                CheckpointCadenceSettings.TempCheckpointEveryMinutes = origMinutes;
                CheckpointCadenceSettings.EpochCheckpointInterval = origEpochInterval;
            }

            Console.WriteLine();
            if (failed == 0)
                Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, {failed} failed.");
            else
                Console.WriteLine($"FAILURES: {passed} passed, {failed} failed.");

            return failed == 0;
        }
    }
}