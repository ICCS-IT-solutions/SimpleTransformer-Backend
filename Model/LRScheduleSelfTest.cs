namespace SimpleTransformer.Model
{
    /// <summary>
    /// Pins the LR-schedule horizon invariant that a long-running training project
    /// regressed on repeatedly.
    /// <para>
    /// The defect: the horizon was computed as <c>totalEpochs * numBatches</c>,
    /// where <c>totalEpochs</c> is the ABSOLUTE end epoch
    /// (<c>startEpoch + config.Epochs</c>) rather than the steps this invocation
    /// will run. On any resume the horizon landed far below the model's
    /// checkpointed GlobalStep, and LRScheduler.GetLR - which returns the floor
    /// once <c>step &gt;= TotalSteps</c> - pinned the learning rate at
    /// <c>MinLearningRate</c> for the entire remainder of the run. Observed in the
    /// field as <c>LR schedule: 120 total steps ... resuming at step 982</c>:
    /// 862 steps past the end of the curve, so the LR sat at the floor for the whole
    /// run. It was invisible because the log line printed the SETTINGS, which looked
    /// perfectly healthy while none of them were in effect.
    /// <para>
    /// Scope, stated precisely because it is easy to overstate: the peak is a one-time
    /// event at the warmup boundary, and the cosine only decays after it. This fix does
    /// NOT make the peak reachable again for a model already past warmup - no horizon
    /// can. It stops the schedule being clamped to the floor, so the LR actually
    /// follows the curve and MinLearningRate stops being the de-facto rate.
    /// <para>
    /// Run with: dotnet run -- --lr-schedule-selftest
    /// </para>
    /// </summary>
    public static class LRScheduleSelfTest
    {
        /// <summary>
        /// The horizon calculation under test, mirrored from
        /// TrainingJobExtensions.RunTrainingLoop. Pure, so the invariant can be
        /// pinned without driving a training loop.
        /// </summary>
        private static int ComputeHorizon(int globalStep, int configEpochs, int numBatches)
        {
            int stepsThisRun = Math.Max(1, configEpochs) * Math.Max(1, numBatches);
            int scheduleSteps = Math.Max(1, globalStep + stepsThisRun);

            // Defensive guard, also mirrored: the horizon must stay ahead of the
            // step counter or the peak and warmup become unreachable.
            if (scheduleSteps <= globalStep)
                scheduleSteps = globalStep + Math.Max(1, numBatches);

            return scheduleSteps;
        }

        /// <summary>The pre-fix expression, kept as the regression oracle.</summary>
        private static int ComputeHorizonOld(int startEpoch, int configEpochs, int numBatches)
            => Math.Max(1, (startEpoch + configEpochs) * Math.Max(1, numBatches));

        public static bool RunAndPrint()
        {
            Console.WriteLine("=== LR Schedule Horizon Self-Test ===");
            Console.WriteLine();

            int passed = 0, failed = 0;
            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            // 1. THE REGRESSION: the exact field case. 982 steps already banked,
            // resuming for 1 epoch at 2 batches/epoch. The old horizon was 120 -
            // 862 steps short of where the counter already was.
            const int globalStep = 982, configEpochs = 1, numBatches = 2;
            int oldHorizon = ComputeHorizonOld(59, configEpochs, numBatches);
            int newHorizon = ComputeHorizon(globalStep, configEpochs, numBatches);

            Check("old horizon WAS broken for the reported case",
                oldHorizon <= globalStep,
                $"old={oldHorizon} vs step={globalStep} (short by {globalStep - oldHorizon})");

            Check("new horizon exceeds GlobalStep",
                newHorizon > globalStep,
                $"new={newHorizon} > step={globalStep}");

            // 2. The invariant itself, across a spread of resume shapes.
            (int gs, int ep, int nb)[] cases =
            {
                (0, 100, 2),      // fresh run
                (0, 10, 1),       // fresh, single batch
                (200, 1, 2),      // resume, one epoch
                (982, 1, 2),      // the reported failure
                (5000, 100, 8),   // deep resume, large epoch count
                (1, 1, 1),        // minimal
                (0, 1, 1),        // absolute minimum
            };

            bool allAhead = true;
            string firstFailure = "";
            foreach (var (gs, ep, nb) in cases)
            {
                int h = ComputeHorizon(gs, ep, nb);
                if (h <= gs)
                {
                    allAhead = false;
                    firstFailure = $"step={gs} epochs={ep} batches={nb} -> horizon={h}";
                    break;
                }
            }
            Check("horizon always exceeds GlobalStep", allAhead,
                allAhead ? $"{cases.Length} shapes" : firstFailure);

            // 3. Fresh runs are UNCHANGED - the fix must only affect resumes.
            Check("fresh runs unaffected (globalStep 0)",
                ComputeHorizon(0, 100, 2) == ComputeHorizonOld(0, 100, 2),
                $"{ComputeHorizon(0, 100, 2)} == {ComputeHorizonOld(0, 100, 2)}");

            // 4. The horizon covers the banked steps plus this run's steps.
            Check("horizon = banked + this run's steps",
                ComputeHorizon(982, 100, 2) == 982 + 200,
                $"{ComputeHorizon(982, 100, 2)} == 982 + 200");

            // 5. Degenerate inputs cannot produce a horizon at or below the step.
            Check("zero epochs still yields a usable horizon",
                ComputeHorizon(50, 0, 2) > 50, $"{ComputeHorizon(50, 0, 2)} > 50");

            Check("zero batches still yields a usable horizon",
                ComputeHorizon(50, 10, 0) > 50, $"{ComputeHorizon(50, 10, 0)} > 50");

            // 6. END-TO-END through the real LRScheduler: GetLR must not return
            //    the floor for a resumed run. This is the assertion that would
            //    have caught the original bug.
            const float peak = 0.0003f, floor = 0.00005f;
            const int warmup = 12;

            int horizon = ComputeHorizon(globalStep, configEpochs, numBatches);
            var sched = new LRScheduler(peak, floor, warmup, horizon);

            float lrNow = sched.GetLR(globalStep);
            Check("GetLR at the resume step is not clamped to the floor",
                lrNow > floor, $"step {globalStep}: LR={lrNow:E3} > floor {floor:E3}");

            // The peak is a ONE-TIME event at the warmup boundary: after warmup
            // the cosine only ever decays. So "peak reachable after a deep
            // resume" is false by construction for ANY horizon - an earlier
            // version of this test wrongly asserted it. What matters is that the
            // LR is whatever the cosine says at that step, rather than being
            // clamped to the floor by a horizon shorter than the work already
            // done. A longer horizon must therefore give a strictly higher LR.
            float lrLong = new LRScheduler(peak, floor, warmup, globalStep + 600)
                .GetLR(globalStep);
            Check("a longer horizon yields a higher LR at the same step",
                lrLong > lrNow, $"step {globalStep}: {lrNow:E3} (short) < {lrLong:E3} (long)");

            Check("clamped (old) horizon gives exactly the floor",
                MathF.Abs(new LRScheduler(peak, floor, warmup, 120).GetLR(globalStep) - floor) < 1e-9f,
                $"old horizon 120 -> {new LRScheduler(peak, floor, warmup, 120).GetLR(globalStep):E3}");

            // The old horizon really did pin it - prove the test can fail.
            var brokenSched = new LRScheduler(peak, floor, warmup, oldHorizon);
            Check("old horizon demonstrably pinned LR at the floor (oracle)",
                MathF.Abs(brokenSched.GetLR(globalStep) - floor) < 1e-9f,
                $"old: step {globalStep} of {oldHorizon} -> {brokenSched.GetLR(globalStep):E3}");

            // 7. Fresh-run behaviour is untouched.
            var fresh = new LRScheduler(peak, floor, warmup, 1000);
            Check("fresh run: step 0 LR is below peak",
                fresh.GetLR(0) < peak, $"{fresh.GetLR(0):E3}");
            Check("fresh run: peak reached at end of warmup",
                MathF.Abs(fresh.GetLR(warmup) - peak) < 1e-6f,
                $"step {warmup}: {fresh.GetLR(warmup):E3}");
            Check("fresh run: LR decays to the floor past the horizon",
                MathF.Abs(fresh.GetLR(1000) - floor) < 1e-6f,
                $"step 1000: {fresh.GetLR(1000):E3}");
            Console.WriteLine();
            if (failed == 0)
            {
                Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, 0 failed.");
                return true;
            }

            Console.WriteLine($"FAILURES: {passed} passed, {failed} FAILED.");
            return false;
        }
    }
}