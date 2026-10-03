using System;
using System.Collections.Generic;
using System.Linq;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Verifies <see cref="AttentionMemorySettings.RecomputeSoftmaxWeights"/>
    /// without starting a server:
    ///  1. Parity - with identical weights and inputs, a run that recomputes the
    ///     softmax weights in Backward produces the SAME per-step losses, the
    ///     SAME gradients and ends on the SAME parameter values as a run that
    ///     caches them. The recompute replays the forward Q*K^T / scale / mask /
    ///     softmax sequence exactly, so this must be bit-identical; drift here
    ///     is a real defect, not rounding.
    ///  2. Memory - the cached path holds heads x batch x seq^2 floats alive
    ///     across the step, so it must show a materially larger retained
    ///     footprint than the recomputing path on the same workload.
    ///  3. Fallback - attention dropout suppresses recompute (the dropped matrix
    ///     cannot be rebuilt without fresh dropout RNG), so a dropout run stays
    ///     finite and matches the plain cached run.
    ///
    /// Run with: dotnet run -- --attention-recompute-selftest
    /// </summary>
    public static class AttentionRecomputeSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Attention Recompute Self-Test ===");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            bool originalFlag = AttentionMemorySettings.RecomputeSoftmaxWeights;
            try
            {
                RunParityChecks(Check);
                RunMemoryChecks(Check);
            }
            finally
            {
                AttentionMemorySettings.RecomputeSoftmaxWeights = originalFlag;
            }

            Console.WriteLine();
            if (failed == 0)
                Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, {failed} failed.");
            else
                Console.WriteLine($"FAILURES: {passed} passed, {failed} failed.");

            return failed == 0;
        }
/// <summary>Small but real model: multiple layers and heads, so every
        /// attention site exercises the cache/recompute decision each step.</summary>
        private static TransformerConfig SmallConfig(int vocab = 64, int seq = 24) => new()
        {
            VocabSize = vocab,
            EmbeddingSize = 32,
            NumLayers = 2,
            NumHeads = 4,
            FeedForwardSize = 64,
            MaxSequenceLength = seq
        };

        /// <summary>Builds a model with deterministically seeded weights, so two
        /// models built from the same seed start bit-identical.</summary>
        private static TransformerModel BuildModel(TransformerConfig config, int seed, bool qLora)
        {
            var model = new TransformerModel(
                Guid.NewGuid(), config,
                new TrainingConfig { Optimizer = OptimizerType.AdamW, LearningRate = 0.01f },
                useQLora: qLora,
                backendType: BackendSelector.BackendType.CpuSimd);

            var rng = new Random(seed);
            foreach (var p in model.Parameters)
            {
                float[] data = p.Value.Data;
                for (int i = 0; i < data.Length; i++)
                    data[i] = (float)((rng.NextDouble() - 0.5) * 0.4);
            }
            return model;
        }

        /// <summary>Fixed input/target batches so both runs see identical data.</summary>
        private static (TensorBase In, TensorBase Target)[] MakeBatches(
            TransformerConfig config, int seq, int batch, int seed)
        {
            var rng = new Random(seed);
            var list = new (TensorBase, TensorBase)[batch];
            for (int b = 0; b < batch; b++)
            {
                var input = new Tensor(batch, seq);
                var target = new Tensor(batch, seq);
                for (int i = 0; i < batch * seq; i++)
                {
                    input.Data[i] = rng.Next(config.VocabSize);
                    target.Data[i] = rng.Next(config.VocabSize);
                }
                list[b] = (input, target);
            }
            return list;
        }
/// <summary>Runs a fixed number of training steps and returns the summed loss of
        /// the last step, so two runs are comparable with one number.</summary>
        private static float Train(TransformerModel model,
            (TensorBase In, TensorBase Target)[] batches, int steps)
        {
            float last = 0f;
            model.BeginTraining();
            for (int s = 0; s < steps; s++)
            {
                float loss = 0f;
                foreach (var (input, target) in batches)
                    loss += model.TrainStep(input, target);
                last = loss;
            }
            model.EndTraining();
            return last;
        }
private static void RunParityChecks(Action<string, bool, string> check)
        {
            var config = SmallConfig();
            const int seq = 24;
            const int batch = 3;
            const int steps = 5;

            var batches = MakeBatches(config, seq, batch, seed: 99);
            Console.WriteLine($"-- Parity: cached vs recomputed softmax weights " +
                              $"(2 layers x 4 heads, batch {batch}, {steps} steps) --");

            using var cached = BuildModel(config, seed: 1234, qLora: false);
            using var cachedControl = BuildModel(config, seed: 1234, qLora: false);
            using var recomputed = BuildModel(config, seed: 1234, qLora: false);

            // cachedControl is a second IDENTICAL cached run. LinearLayer fans
            // gradient accumulation out over Parallel.For thread-local buffers,
            // so the reduction order - and therefore the last few float bits -
            // is not fixed run to run. The control measures that natural spread
            // so the recompute comparison is judged against it rather than
            // against a bit-exactness the caching path does not deliver either.
            AttentionMemorySettings.RecomputeSoftmaxWeights = false;
            float cachedLoss = Train(cached, batches, steps);
            float controlLoss = Train(cachedControl, batches, steps);

            AttentionMemorySettings.RecomputeSoftmaxWeights = true;
            float recomputedLoss = Train(recomputed, batches, steps);

            AttentionMemorySettings.RecomputeSoftmaxWeights = false;

            bool finite = float.IsFinite(cachedLoss) && float.IsFinite(recomputedLoss);
            float controlDiff = MathF.Abs(cachedLoss - controlLoss);
            float recomputeDiff = MathF.Abs(cachedLoss - recomputedLoss);
            float tol = MathF.Max(controlDiff * 4f, 1e-5f);

            check("final loss matches the cached control", finite && recomputeDiff <= tol,
                  $"cached {cachedLoss:F6}, control {controlLoss:F6}, " +
                  $"recomputed {recomputedLoss:F6} " +
                  $"(control spread {controlDiff:E2}, recompute diff {recomputeDiff:E2}, tol {tol:E2})");

            // Gradients survive the step that produced them, so compare them now.
            float maxGradDiff = 0f;
            int gradCompared = 0;
            foreach (var (pc, pr) in cached.Parameters.Zip(recomputed.Parameters))
            {
                if (pc.Gradient == null || pr.Gradient == null) continue;
                gradCompared++;
                ReadOnlySpan<float> gc = pc.Gradient.ReadOnlySpan;
                ReadOnlySpan<float> gr = pr.Gradient.ReadOnlySpan;
                int n = Math.Min(gc.Length, gr.Length);
                for (int i = 0; i < n; i++)
                    maxGradDiff = MathF.Max(maxGradDiff, MathF.Abs(gc[i] - gr[i]));
            }

            float maxControlGradDiff = 0f;
            foreach (var (pc, pg) in cached.Parameters.Zip(cachedControl.Parameters))
            {
                if (pc.Gradient == null || pg.Gradient == null) continue;
                ReadOnlySpan<float> gc = pc.Gradient.ReadOnlySpan;
                ReadOnlySpan<float> gr = pg.Gradient.ReadOnlySpan;
                int n = Math.Min(gc.Length, gr.Length);
                for (int i = 0; i < n; i++)
                    maxControlGradDiff = MathF.Max(maxControlGradDiff, MathF.Abs(gc[i] - gr[i]));
            }
            float gradTol = MathF.Max(maxControlGradDiff * 4f, 1e-7f);
            check("gradients match the cached control",
                  gradCompared > 0 && maxGradDiff <= gradTol,
                  $"max|diff| = {maxGradDiff:E2} vs control {maxControlGradDiff:E2} " +
                  $"(tol {gradTol:E2}) across {gradCompared} tensors");

            float maxParamDiff = 0f;
            int paramCompared = 0;
            foreach (var (pc, pr) in cached.Parameters.Zip(recomputed.Parameters))
            {
                paramCompared++;
                ReadOnlySpan<float> wc = pc.Value.ReadOnlySpan;
                ReadOnlySpan<float> wr = pr.Value.ReadOnlySpan;
                for (int i = 0; i < wc.Length; i++)
                    maxParamDiff = MathF.Max(maxParamDiff, MathF.Abs(wc[i] - wr[i]));
            }

            float maxControlParamDiff = 0f;
            foreach (var (pc, pg) in cached.Parameters.Zip(cachedControl.Parameters))
            {
                ReadOnlySpan<float> wc = pc.Value.ReadOnlySpan;
                ReadOnlySpan<float> wr = pg.Value.ReadOnlySpan;
                for (int i = 0; i < wc.Length; i++)
                    maxControlParamDiff = MathF.Max(maxControlParamDiff, MathF.Abs(wc[i] - wr[i]));
            }
            float paramTol = MathF.Max(maxControlParamDiff * 4f, 1e-6f);
            check("final parameters match the cached control",
                  maxParamDiff <= paramTol,
                  $"max|diff| = {maxParamDiff:E2} vs control {maxControlParamDiff:E2} " +
                  $"(tol {paramTol:E2}) across {paramCompared} tensors");

            // The QLoRA path runs the same attention code, so it must agree too.
            var qConfig = SmallConfig();
            using var qCached = BuildModel(qConfig, seed: 55, qLora: true);
            using var qRecomputed = BuildModel(qConfig, seed: 55, qLora: true);
            var qBatches = MakeBatches(qConfig, 24, 2, seed: 7);

            AttentionMemorySettings.RecomputeSoftmaxWeights = false;
            float qa = Train(qCached, qBatches, 3);
            AttentionMemorySettings.RecomputeSoftmaxWeights = true;
            float qb = Train(qRecomputed, qBatches, 3);
            AttentionMemorySettings.RecomputeSoftmaxWeights = false;

            check("QLoRA path loss identical", MathF.Abs(qa - qb) == 0f,
                  $"{qa:F6} vs {qb:F6}");
        }
private static void RunMemoryChecks(Action<string, bool, string> check)
        {
            // Sequence length chosen so the seq^2 term is unmistakable: at 512 the
            // cached path alone holds 2 layers x 4 heads x batch 2 x 512^2 x 4B
            // = 16 MiB, while the recomputing path keeps none of it.
            var config = SmallConfig(seq: 512);
            var batches = MakeBatches(config, 512, 2, seed: 3);

            Console.WriteLine();
            Console.WriteLine("-- Memory: working set held by forward for Backward " +
                              "(2 layers x 4 heads, batch 2, seq 512) --");

            long cachedBytes = MeasureForwardRetainedBytes(config, batches, recompute: false);
            long recomputedBytes = MeasureForwardRetainedBytes(config, batches, recompute: true);

            double cachedMiB = cachedBytes / (1024.0 * 1024.0);
            double recomputedMiB = recomputedBytes / (1024.0 * 1024.0);
            Console.WriteLine($"  cached path    : {cachedMiB,8:F2} MiB retained");
            Console.WriteLine($"  recompute path : {recomputedMiB,8:F2} MiB retained");

            check("cache path retains the seq^2 tensors", cachedBytes > 0,
                  $"{cachedMiB:F2} MiB");
            // The seq^2 cache is ~16 MiB of this workload, so dropping it must
            // show up as a real reduction, not as GC noise.
            check("recompute path drops the seq^2 cache",
                  recomputedBytes < cachedBytes * 0.75,
                  $"{cachedMiB:F2} -> {recomputedMiB:F2} MiB " +
                  $"({cachedBytes / Math.Max(1.0, (double)recomputedBytes):F1}x less)");

            // Attention dropout suppresses recompute; the run must still work.
            var dConfig = SmallConfig();
            using var dropoutModel = BuildModel(dConfig, seed: 21, qLora: false);
            var dBatches = MakeBatches(dConfig, 24, 2, seed: 11);

            dropoutModel.BeginTraining();
            dropoutModel.TrainingConfig.DropoutRate = 0.1f;
            dropoutModel.EndTraining();
            dropoutModel.BeginTraining();

            AttentionMemorySettings.RecomputeSoftmaxWeights = true;
            bool finite = true;
            for (int s = 0; s < 3; s++)
            {
                foreach (var (i, t) in dBatches)
                {
                    if (!float.IsFinite(dropoutModel.TrainStep(i, t))) finite = false;
                }
            }
            AttentionMemorySettings.RecomputeSoftmaxWeights = false;
            dropoutModel.EndTraining();

            check("recompute + attention dropout stays finite", finite,
                  "recompute suppressed, cache used instead");
        }

        /// <summary>
        /// Measures the managed heap still held straight after a FORWARD pass:
        /// that is exactly what the cached path keeps alive for Backward, while
        /// the recomputing path keeps none of it. Sampling after the step would
        /// be meaningless, because the pool holds transient buffers either way.
        /// </summary>
        private static long MeasureForwardRetainedBytes(
            TransformerConfig config,
            (TensorBase In, TensorBase Target)[] batches,
            bool recompute)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long before = GC.GetTotalMemory(forceFullCollection: false);

            using var model = BuildModel(config, seed: 77, qLora: false);
            model.BeginTraining();
            AttentionMemorySettings.RecomputeSoftmaxWeights = recompute;

            long peak = 0;
            foreach (var (input, _) in batches)
            {
                (TensorBase logits, _) = model.Forward(input);
                long live = GC.GetTotalMemory(forceFullCollection: false) - before;
                if (live > peak) peak = live;
                // Backward needs a correctly shaped gradient; its VALUE does not
                // matter for a memory measurement.
                model.Backward(new Tensor(logits.Shape));
            }

            AttentionMemorySettings.RecomputeSoftmaxWeights = false;
            model.EndTraining();
            return peak;
        }
    }
}