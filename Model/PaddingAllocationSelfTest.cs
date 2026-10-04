using System;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Guards the ONE property that behavioural padding tests cannot see: that a
    /// padded forward does not allocate outside the workspace pool.
    /// <para>
    /// The padding mask is batch x seq x seq floats - 8 MiB at batch 8 / seq
    /// 512, far past the Large Object Heap threshold. It was built with
    /// `new Tensor(...)`, so every padded step produced multi-megabyte garbage
    /// that only a gen2 collection reclaims. Nothing went wrong functionally,
    /// which is exactly why it survived: the existing padding test uses an 8x8
    /// model where that mask is 2 KB and the whole failure mode is below its
    /// resolution. It also stayed invisible to the pool cap and the memory
    /// valve, which track pooled tensors only.
    /// <para>
    /// So this test asserts ALLOCATION, measured with
    /// GC.GetAllocatedBytesForCurrentThread across repeated padded forwards, at a
    /// sequence length large enough that the mask genuinely lands on the LOH. It
    /// also runs the same measurement with the leak deliberately reintroduced, so
    /// the threshold is known to separate the two behaviours.
    /// <para>
    /// Run with: dotnet run -- --padding-alloc-selftest
    /// </para>
    /// </summary>
    public static class PaddingAllocationSelfTest
    {
        // seq 256 x batch 2 = 512 KiB mask: comfortably past the 85,000-byte
        // LOH threshold, while the forward itself stays cheap.
        private const int Seq = 256;
        private const int Batch = 2;
        private const int Real = 160;
        private const int Iterations = 12;

        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Padding Allocation Self-Test ===");
            Console.WriteLine();

            int passed = 0, failed = 0;
            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            long maskBytes = (long)Batch * Seq * Seq * sizeof(float);
            Console.WriteLine($"-- batch {Batch}, seq {Seq}, first {Real} real --");
            Console.WriteLine($"   mask is {maskBytes / 1024.0:F0} KiB " +
                              $"(LOH threshold ~85 KB)");
            Console.WriteLine();

            long pooled = MeasureBytesPerIteration();
            long leaked = MeasureBytesPerIteration(reintroduceLeak: true);
            long unpadded = MeasureBytesPerIteration(padded: false);

            Console.WriteLine($"   pooled forward : {pooled / 1024.0,10:F1} KiB allocated/iteration");
            Console.WriteLine($"   leaked forward : {leaked / 1024.0,10:F1} KiB allocated/iteration");
            Console.WriteLine($"   UNPADDED fwd  : {unpadded / 1024.0,10:F1} KiB allocated/iteration  (baseline, no mask)");
            Console.WriteLine();

            // Assert the DIFFERENCE, not an absolute near-zero. The forward has
            // allocation unrelated to the mask (TensorView slices per batch item,
            // Parallel.For bookkeeping), so "allocates nothing" is not a
            // claim this test can honestly make. What it CAN prove is that the
            // mask stopped being allocated off the pool: the gap between the
            // leaked and pooled runs is the mask, and that gap must be at least
            // the whole mask.
            long gap = leaked - pooled;

            Check("pooled padded forward no longer allocates the mask",
                  gap >= maskBytes * 9 / 10,
                  $"pooled is {gap / 1024.0:F0} KiB below leaked; mask is {maskBytes / 1024.0:F0} KiB");

            Check("pooled run costs less than one mask",
                  pooled < maskBytes / 2,
                  $"{pooled / 1024.0:F1} KiB < {maskBytes / 2048.0:F1} KiB");

            Check("padding costs nothing beyond the shared baseline",
                  pooled <= unpadded + maskBytes / 10,
                  $"padded {pooled / 1024.0:F1} KiB vs unpadded {unpadded / 1024.0:F1} KiB");

            Console.WriteLine();
            if (failed == 0) Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, {failed} failed.");
            else Console.WriteLine($"FAILURES: {passed} passed, {failed} failed.");
            return failed == 0;
        }
        /// <summary>
        /// Bytes allocated per padded forward, after warm-up so the pool is
        /// already populated and first-touch costs are excluded.
        /// </summary>
        private static long MeasureBytesPerIteration(bool reintroduceLeak = false, bool padded = true)
        {
            var config = new TransformerConfig
            {
                VocabSize = 64, EmbeddingSize = 32, NumLayers = 1,
                NumHeads = 2, FeedForwardSize = 64, MaxSequenceLength = Seq
            };

            using var model = new TransformerModel(
                Guid.NewGuid(), config,
                new TrainingConfig { Optimizer = OptimizerType.AdamW, LearningRate = 0.01f },
                useQLora: false,
                backendType: BackendSelector.BackendType.CpuSimd);

            var rng = new Random(4242);
            foreach (var prm in model.Parameters)
            {
                float[] d = prm.Value.Data;
                for (int i = 0; i < d.Length; i++)
                    d[i] = (float)((rng.NextDouble() - 0.5) * 0.4);
            }

            var input = new Tensor(Batch, Seq);
            var target = new Tensor(Batch, Seq);
            for (int b = 0; b < Batch; b++)
            {
                for (int c = 0; c < Seq; c++)
                {
                    bool isReal = !padded || c < Real;
                    input[b, c] = isReal ? rng.Next(config.VocabSize) : 0;
                    target[b, c] = isReal
                        ? rng.Next(config.VocabSize)
                        : TransformerModel.IgnoreIndex;
                }
            }

            // Warm-up: populates the pool and pays every first-touch cost, so the
            // measured window contains only steady-state behaviour.
            for (int i = 0; i < 3; i++)
            {
                if (reintroduceLeak)
                    model.ForwardForAllocationTest(input, target, poolAttentionMask: false);
                else
                    model.Forward(input, target);
                model.Workspace.Reset();
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < Iterations; i++)
            {
                if (reintroduceLeak)
                    model.ForwardForAllocationTest(input, target, poolAttentionMask: false);
                else
                    model.Forward(input, target);
                model.Workspace.Reset();
            }
            long after = GC.GetAllocatedBytesForCurrentThread();

            return (after - before) / Iterations;
        }
    }
}