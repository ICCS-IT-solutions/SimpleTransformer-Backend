using System;
using System.Collections.Generic;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Proves that a resumed run continues rather than restarts.
    /// <para>
    /// Optimizer state (AdamW's first/second moments and bias-correction step,
    /// SGD's momentum velocity) was previously never persisted, so a process
    /// restart reset every moment to zero and restarted bias correction at t=1
    /// while the LR schedule carried on. These checks cover the export/import
    /// round-trip and the resulting trajectory in isolation; the model-level
    /// replay check is added once the v5 checkpoint trailer exists.
    /// </para>
    ///
    /// Run with: dotnet run -- --resume-exactness-selftest
    /// </summary>
    public static class ResumeExactnessSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Resume Exactness Self-Test ===");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-54} {detail}");
            }

            // ---------------------------------------------------------------
            // AdamW: a restored optimizer must continue the trajectory exactly,
            // and a cold optimizer must NOT (or the check proves nothing).
            // ---------------------------------------------------------------
            Console.WriteLine("-- AdamW optimizer state --");
            {
                var original = new AdamWOptimizer(learningRate: 0.01f);
                var trained = MakeParameter("weight", rows: 4, cols: 3, seed: 11);

                for (int step = 0; step < 5; step++)
                {
                    FillGradient(trained, seed: 100 + step);
                    original.Step(new[] { trained });
                }

                OptimizerState snapshot = original.ExportState();
                Check("export captures an entry per stepped parameter",
                    snapshot.Kind == OptimizerStateKinds.AdamW && snapshot.Parameters.Count == 1,
                    $"kind={snapshot.Kind}, entries={snapshot.Parameters.Count}");
                Check("export captures the bias-correction step",
                    snapshot.StepCount == 5, $"step={snapshot.StepCount}");

                // Restored: same weights, optimizer state imported.
                var restored = new AdamWOptimizer(learningRate: 0.01f);
                var restoredParam = CopyOf(trained);
                int skipped = restored.ImportState(snapshot, new[] { restoredParam });
                Check("import reports nothing skipped", skipped == 0);
                Check("import restores the step count",
                    restored.StepCount == original.StepCount, $"{restored.StepCount}");

                // Cold: same weights, empty optimizer state.
                var cold = new AdamWOptimizer(learningRate: 0.01f);
                var coldParam = CopyOf(trained);

                // All three take the identical next step.
                FillGradient(trained, seed: 777);
                FillGradient(restoredParam, seed: 777);
                FillGradient(coldParam, seed: 777);
                var nextStep = new[] { trained };
                original.Step(nextStep);
                restored.Step(new[] { restoredParam });
                cold.Step(new[] { coldParam });

                Check("a restored AdamW continues bit-for-bit",
                    ValuesEqual(trained, restoredParam), "weight match after the next step");
                Check("a cold AdamW would diverge (test is meaningful)",
                    !ValuesEqual(trained, coldParam),
                    $"cold step differs (t=0 vs t={snapshot.StepCount})");
            }
// ---------------------------------------------------------------
            // AdamW edge cases
            // ---------------------------------------------------------------
            Console.WriteLine();
            Console.WriteLine("-- AdamW edge cases --");
            {
                var neverStepped = new AdamWOptimizer(learningRate: 0.01f);
                OptimizerState untouched = neverStepped.ExportState();
                Check("a never-stepped AdamW exports empty state",
                    untouched.Parameters.Count == 0 && untouched.StepCount == 0,
                    $"entries={untouched.Parameters.Count}, step={untouched.StepCount}");

                var fresh = new AdamWOptimizer(learningRate: 0.01f);
                var param = MakeParameter("weight", rows: 2, cols: 2, seed: 3);
                int skipped = fresh.ImportState(untouched, new[] { param });
                Check("importing empty state is a clean no-op",
                    skipped == 0 && fresh.StepCount == 0);

                bool wrongKindRejected = false;
                try
                {
                    fresh.ImportState(
                        new OptimizerState { Kind = OptimizerStateKinds.Sgd },
                        new[] { param });
                }
                catch (InvalidDataException) { wrongKindRejected = true; }
                Check("AdamW rejects an SGD snapshot", wrongKindRejected);

                // An entry whose shape does not match the live parameter must be
                // skipped, never installed: the update loop would index past it.
                var mismatched = new OptimizerState
                {
                    Kind = OptimizerStateKinds.AdamW,
                    StepCount = 2,
                    Parameters = new List<OptimizerParamState>
                    {
                        new()
                        {
                            Name = "weight",
                            FirstMoment = new TensorData { Shape = new[] { 9, 9 }, Data = new float[81] },
                            SecondMoment = new TensorData { Shape = new[] { 9, 9 }, Data = new float[81] }
                        },
                        new()
                        {
                            Name = "not_a_real_parameter",
                            FirstMoment = new TensorData { Shape = new[] { 2, 2 }, Data = new float[4] },
                            SecondMoment = new TensorData { Shape = new[] { 2, 2 }, Data = new float[4] }
                        }
                    }
                };
                var guarded = new AdamWOptimizer(learningRate: 0.01f);
                int mismatchedSkipped = guarded.ImportState(mismatched, new[] { param });
                Check("shape-mismatched and unknown entries are skipped",
                    mismatchedSkipped == 2, $"skipped={mismatchedSkipped}");
            }
// ---------------------------------------------------------------
            // SGD: same continuation contract, via momentum velocity
            // ---------------------------------------------------------------
            Console.WriteLine();
            Console.WriteLine("-- SGD optimizer state --");
            {
                var original = new SgdOptimizer(learningRate: 0.05f, momentum: 0.9f, weightDecay: 0.01f);
                var trained = MakeParameter("weight", rows: 4, cols: 3, seed: 21);

                for (int step = 0; step < 5; step++)
                {
                    FillGradient(trained, seed: 300 + step);
                    original.Step(new[] { trained });
                }

                OptimizerState snapshot = original.ExportState();
                Check("SGD export captures velocity per parameter",
                    snapshot.Kind == OptimizerStateKinds.Sgd && snapshot.Parameters.Count == 1,
                    $"kind={snapshot.Kind}, entries={snapshot.Parameters.Count}");

                var restored = new SgdOptimizer(learningRate: 0.05f, momentum: 0.9f, weightDecay: 0.01f);
                var restoredParam = CopyOf(trained);
                int skipped = restored.ImportState(snapshot, new[] { restoredParam });
                Check("SGD import reports nothing skipped", skipped == 0);

                var cold = new SgdOptimizer(learningRate: 0.05f, momentum: 0.9f, weightDecay: 0.01f);
                var coldParam = CopyOf(trained);

                FillGradient(trained, seed: 888);
                FillGradient(restoredParam, seed: 888);
                FillGradient(coldParam, seed: 888);
                original.Step(new[] { trained });
                restored.Step(new[] { restoredParam });
                cold.Step(new[] { coldParam });

                Check("a restored SGD continues bit-for-bit",
                    ValuesEqual(trained, restoredParam), "weight match after the next step");
                Check("a cold SGD would diverge (test is meaningful)",
                    !ValuesEqual(trained, coldParam), "momentum buffer lost");

                // Vanilla SGD keeps no state, so its snapshot is legitimately empty.
                var vanilla = new SgdOptimizer(learningRate: 0.05f);
                var vanillaParam = MakeParameter("weight", rows: 2, cols: 2, seed: 5);
                OptStep(vanilla, vanillaParam, seed: 1);
                Check("vanilla SGD exports empty state",
                    vanilla.ExportState().Parameters.Count == 0);

                bool wrongKindRejected = false;
                try
                {
                    restored.ImportState(
                        new OptimizerState { Kind = OptimizerStateKinds.AdamW },
                        new[] { restoredParam });
                }
                catch (InvalidDataException) { wrongKindRejected = true; }
                Check("SGD rejects an AdamW snapshot", wrongKindRejected);
            }

            Console.WriteLine();
            Console.WriteLine($"{(failed == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT")}: {passed} passed, {failed} failed.");
            return failed == 0;
        }
private static TrainableParameter MakeParameter(string name, int rows, int cols, int seed)
        {
            var parameter = new TrainableParameter(name, new Tensor(rows, cols), new Tensor(rows, cols));
            var random = new Random(seed);
            for (int i = 0; i < parameter.Value.Data.Length; i++)
                parameter.Value.Data[i] = (float)(random.NextDouble() - 0.5);
            return parameter;
        }

        private static void FillGradient(TrainableParameter parameter, int seed)
        {
            var random = new Random(seed);
            for (int i = 0; i < parameter.Gradient!.Data.Length; i++)
                parameter.Gradient.Data[i] = (float)(random.NextDouble() - 0.5);
        }

        private static void OptStep(IOptimizer optimizer, TrainableParameter parameter, int seed)
        {
            FillGradient(parameter, seed);
            optimizer.Step(new[] { parameter });
        }

        private static TrainableParameter CopyOf(TrainableParameter source)
        {
            var copy = new TrainableParameter(
                source.Name,
                new Tensor(source.Value.Shape),
                new Tensor(source.Gradient!.Shape));
            Array.Copy(source.Value.Data, copy.Value.Data, source.Value.Data.Length);
            Array.Copy(source.Gradient!.Data, copy.Gradient!.Data, source.Gradient!.Data.Length);
            return copy;
        }

        private static bool ValuesEqual(TrainableParameter a, TrainableParameter b)
        {
            if (a.Value.Data.Length != b.Value.Data.Length) return false;
            for (int i = 0; i < a.Value.Data.Length; i++)
            {
                if (a.Value.Data[i] != b.Value.Data[i]) return false;
            }
            return true;
        }
    }
}