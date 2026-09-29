using System;
using System.Linq;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.Config;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Exercises the memory pressure valve without a server and without
    /// allocating a real multi-gigabyte heap: the quota clamp and the whole
    /// escalation ladder are driven from synthetic samples through the pure
    /// <see cref="MemoryPressureValve.Evaluate"/>, so the policy that governs
    /// whether a training run stalls is verified deterministically.
    ///
    /// Run with: dotnet run -- --memory-valve-selftest
    /// </summary>
    public static class MemoryPressureSelfTest
    {
        private const long Gib = 1L << 30;
        private const long Mib = 1L << 20;

        private static long TotalFootprint(IEnumerable<TensorBase> tensors)
        {
            long total = 0;
            foreach (var t in tensors)
            {
                var buffer = t.Buffer;
                if (buffer != null)
                    total += (long)buffer.Length * sizeof(float);
            }
            return total;
        }
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Host Memory Pressure Valve Self-Test ===");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            // Synthetic machine: a 32 GiB host, so the clamps are checkable by hand.
            const long physical = 32 * Gib;

            Console.WriteLine("-- Quota clamping (32 GiB machine) --");
            {
                // Auto: 80% of 32 GiB = 25.6 GiB, inside the [1,48] GiB band.
                long auto = MemoryPressureValve.ResolveQuotaBytes(0, 0, physical);
                Check("auto quota = 80% of physical",
                    auto == (long)(physical * 0.80), $"{auto / (double)Gib:F1} GiB");

                // A configured quota below the ceiling passes through untouched.
                long small = MemoryPressureValve.ResolveQuotaBytes(4 * Gib, 0, physical);
                Check("configured quota below ceiling is kept",
                    small == 4 * Gib, $"{small / (double)Gib:F1} GiB");

                // The headline rule: a quota larger than the machine is a cap,
                // never a licence - it must clamp to 95%, not to 64 GiB.
                long huge = MemoryPressureValve.ResolveQuotaBytes(64 * Gib, 0, physical);
                Check("oversized quota clamps to 95% of physical",
                    huge == (long)(physical * 0.95), $"{huge / (double)Gib:F1} GiB");

                // A min_quota larger than the machine must not push the result
                // back through the ceiling.
                long floored = MemoryPressureValve.ResolveQuotaBytes(0, 128 * Gib, physical);
                Check("min_quota larger than machine still clamps",
                    floored == (long)(physical * 0.95), $"{floored / (double)Gib:F1} GiB");

                // An explicit min above an explicit max wins within the band.
                long raised = MemoryPressureValve.ResolveQuotaBytes(2 * Gib, 6 * Gib, physical);
                Check("min_quota raises a smaller configured quota",
                    raised == 6 * Gib, $"{raised / (double)Gib:F1} GiB");
            }

            Console.WriteLine();
            Console.WriteLine("-- Escalation ladder (band 70-85%) --");
            {
                var valve = new MemoryPressureValve();
                MemoryPressureSettings.LowerBoundPercent = 70.0;
                MemoryPressureSettings.UpperBoundPercent = 85.0;
                MemoryPressureSettings.MinCooldownSteps = 5;
                MemoryPressureSettings.AllowBlockingCompact = true;

                MemoryPressureSample At(double pct) =>
                    new((long)(physical * pct / 100.0), 0, physical, physical);

                Check("below band does nothing",
                    valve.Evaluate(At(50.0)) == MemoryReliefLevel.None);

                Check("just above lower bound trims device pools",
                    valve.Evaluate(At(72.0)) == MemoryReliefLevel.TrimDevicePools);

                // Mid-band needs the latch released first, otherwise the
                // cooldown from the previous rung suppresses it - which is
                // itself the behaviour under test.
                valve.Evaluate(At(10.0));
                Check("mid band trims the workspace pool",
                    valve.Evaluate(At(78.0)) == MemoryReliefLevel.TrimWorkspace);

                valve.Evaluate(At(10.0));
                Check("at upper bound permits compaction",
                    valve.Evaluate(At(90.0)) == MemoryReliefLevel.Compact);
            }

            Console.WriteLine();
            Console.WriteLine("-- Anti-thrash latch and cooldown --");
            {
                var valve = new MemoryPressureValve();
                MemoryPressureSettings.MinCooldownSteps = 5;

                MemoryPressureSample At(double pct) =>
                    new((long)(physical * pct / 100.0), 0, physical, physical);

                MemoryReliefLevel first = valve.Evaluate(At(90.0));
                Check("first excursion compacts", first == MemoryReliefLevel.Compact);

                // Still above the band, inside the cooldown: must stay silent.
                int suppressed = 0;
                for (int i = 0; i < 4; i++)
                {
                    if (valve.Evaluate(At(90.0)) == MemoryReliefLevel.None)
                        suppressed++;
                }
                Check("stays quiet during cooldown", suppressed == 4, $"{suppressed}/4 suppressed");

                // Cooldown elapsed, still hot: allowed to act again.
                Check("acts again once the cooldown expires",
                    valve.Evaluate(At(90.0)) == MemoryReliefLevel.Compact);

                // Drop back under the band, then re-excise: the latch must
                // clear so the ladder is available again immediately.
                valve.Evaluate(At(10.0));
                Check("latch clears below the band",
                    valve.Evaluate(At(90.0)) == MemoryReliefLevel.Compact);
            }

            Console.WriteLine();
            Console.WriteLine("-- Blocking compact gate --");
            {
                MemoryPressureSettings.AllowBlockingCompact = false;
                var valve = new MemoryPressureValve();

                MemoryPressureSample At(double pct) =>
                    new((long)(physical * pct / 100.0), 0, physical, physical);

                Check("never compacts when the gate is closed",
                    valve.Evaluate(At(99.0)) == MemoryReliefLevel.TrimWorkspace);
                MemoryPressureSettings.AllowBlockingCompact = true;
            }

            Console.WriteLine();
            Console.WriteLine("-- Activation pool cap and trim --");
            {
                using var backend = BackendSelector.SelectBackend(BackendSelector.BackendType.CpuReference);
                using var workspace = new TensorWorkspace(backend);

                // ~64 MiB of distinct-shaped activations across 16 buckets.
                // Distinct shapes are the point: a real run churns sequence and
                // batch shapes, and the old pool had no cap at all.
                for (int i = 1; i <= 16; i++)
                {
                    var t = workspace.Borrow2D(512, i * 512);
                    workspace.Release(t);
                }

                long retained = workspace.RetainedBytes;
                Check("pooled bytes are accounted for",
                    retained > 0, $"{retained / (double)Mib:F1} MiB across {workspace.PooledShapeCount} shapes");

                // A borrow of an already-pooled shape must decrement the
                // retained count, or the cap would free the same bytes twice.
                var reused = workspace.Borrow2D(512, 16 * 512);
                Check("borrowing a pooled shape frees its bytes from the count",
                    workspace.RetainedBytes == retained - (512L * 16 * 512 * 4),
                    $"{workspace.RetainedBytes / (double)Mib:F1} MiB");
                workspace.Release(reused);
                Check("releasing restores the count",
                    workspace.RetainedBytes == retained,
                    $"{workspace.RetainedBytes / (double)Mib:F1} MiB");

                // Trim to half: must shrink, and the accounting must match what
                // the caller can observe.
                int dropped = workspace.TrimRetainedTo(retained / 2);
                Check("trim drops tensors", dropped > 0, $"{dropped} dropped");

                long afterTrim = workspace.RetainedBytes;
                Check("trim brings the pool under target",
                    afterTrim <= retained / 2,
                    $"{afterTrim / (double)Mib:F1} of {retained / (double)Mib:F1} MiB");

                // Every path that removes from a bag must decrement the counter,
                // so a full drain must land on exactly zero.
                workspace.ReleasePooledMemory();
                Check("draining the pool zeroes the retained count",
                    workspace.RetainedBytes == 0,
                    $"{workspace.RetainedBytes} bytes, {workspace.PooledTensorCount} tensors left");

                // The cap must actually BIND, not merely be measurable. Pin a
                // known small quota so the cap is deterministic, then push far
                // more than it allows through the real bulk path and assert the
                // pool is held down.
                long savedMax = MemoryPressureSettings.MaxQuotaBytes;
                double savedFrac = MemoryPressureSettings.WorkspaceCapFractionOfQuota;
                int savedEvery = MemoryPressureSettings.CheckEveryNSteps;

                const long capQuota = 8 * Mib;
                const long expectedCap = capQuota / 2;
                MemoryPressureSettings.MaxQuotaBytes = capQuota;
                MemoryPressureSettings.WorkspaceCapFractionOfQuota = 0.5;
                MemoryPressureSettings.CheckEveryNSteps = 1; // recompute the cap every borrow

                workspace.ReleasePooledMemory();

                // ~33 MiB of distinct-shaped activations (16 KiB .. 1 MiB each),
                // all borrowed at once and then handed back through Reset() -
                // exactly the shape of one training step. Every tensor is well
                // under the cap, so the cap is genuinely able to bind.
                var held = new List<TensorBase>();
                for (int i = 1; i <= 64; i++)
                    held.Add(workspace.Borrow2D(64, i * 64));

                Check("a single step can exceed the cap while borrowed",
                    TotalFootprint(held) > expectedCap,
                    $"{TotalFootprint(held) / (double)Mib:F1} MiB held");

                workspace.Reset();

                Check("cap holds the pool under its byte limit after a step",
                    workspace.RetainedBytes <= expectedCap,
                    $"{workspace.RetainedBytes / (double)Mib:F2} MiB pooled against a {expectedCap / (double)Mib:F0} MiB cap");

                MemoryPressureSettings.MaxQuotaBytes = savedMax;
                MemoryPressureSettings.WorkspaceCapFractionOfQuota = savedFrac;
                MemoryPressureSettings.CheckEveryNSteps = savedEvery;
                // The pool must still be usable after a trim - survivors have to
                // be handed back correctly or every later step re-allocates.
                var afterTrimBorrow = workspace.Borrow2D(512, 4 * 512);
                Check("workspace still usable after trimming",
                    afterTrimBorrow.Shape.SequenceEqual(new[] { 512, 4 * 512 }),
                    $"borrowed [{string.Join(",", afterTrimBorrow.Shape)}]");
                workspace.Release(afterTrimBorrow);
            }
            Console.WriteLine("-- Live sample --");
            {
                var valve = new MemoryPressureValve();
                MemoryPressureSample sample = valve.Sample();

                Check("reports a usable memory container", sample.IsUsable,
                    $"{sample.PrivateBytes / (double)Gib:F2} of {sample.PhysicalBytes / (double)Gib:F2} GiB " +
                    $"({sample.UsedFraction * 100:F1}%), quota {sample.QuotaBytes / (double)Gib:F1} GiB");
                Check("used fraction is a sane proportion",
                    sample.UsedFraction >= 0.0 && sample.UsedFraction <= 1.0,
                    $"{sample.UsedFraction:P1}");

                Console.WriteLine();
                Console.WriteLine($"  valve telemetry: {valve.Describe()}");
            }

            Console.WriteLine();
            Console.WriteLine($"{(failed == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT")}: {passed} passed, {failed} failed.");
            return failed == 0;
        }
    }
}
