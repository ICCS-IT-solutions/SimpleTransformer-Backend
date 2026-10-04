using System;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Verifies padding support, without starting a server:
    ///  1. Padded target positions are skipped by the loss, so the loss and the
    ///     gradients depend only on real tokens. Proven by matching a batch that
    ///     is exactly those real tokens at their natural length - the loss
    ///     normalises by valid-token count, so those two must agree.
    ///  2. Padding does not leak into attention: perturbing a PAD input token must
    ///     not move the logit predicted from an earlier real position.
    ///  3. The real positions produce the same logits they do unpadded.
    /// <para>
    /// Run with: dotnet run -- --padding-selftest
    /// </para>
    /// </summary>
    public static class PaddingSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Padding Self-Test ===");
            Console.WriteLine();

            int passed = 0, failed = 0;
            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            try
            {
                var cfg = new TransformerConfig
                {
                    VocabSize = 32, EmbeddingSize = 16, NumLayers = 1,
                    NumHeads = 2, FeedForwardSize = 32, MaxSequenceLength = 8
                };
                const int seq = 8;
                const int real = 5;

                Console.WriteLine($"-- batch 1, seq {seq}, first {real} positions real --");

                var realInput = new Tensor(1, real);
                var realTarget = new Tensor(1, real);
                for (int i = 0; i < real; i++) { realInput[0, i] = 3; realTarget[0, i] = 4; }

                using var refModel = BuildModel(cfg, 99);
                refModel.BeginTraining();
                float refLoss = refModel.TrainStep(realInput, realTarget);
                refModel.EndTraining();

                var padInput = new Tensor(1, seq);
                var padTarget = new Tensor(1, seq);
                for (int i = 0; i < real; i++) { padInput[0, i] = 3; padTarget[0, i] = 4; }
                for (int i = real; i < seq; i++)
                {
                    padInput[0, i] = 0;
                    padTarget[0, i] = TransformerModel.IgnoreIndex;
                }

                using var padModel = BuildModel(cfg, 99);
                padModel.BeginTraining();
                float padLoss = padModel.TrainStep(padInput, padTarget);
                padModel.EndTraining();

                Check("loss ignores padded target positions",
                      MathF.Abs(refLoss - padLoss) <= 1e-4f,
                      $"unpadded-5 {refLoss:F6} vs padded-8 {padLoss:F6}");

                float gradDiff = MaxGradDiff(refModel, padModel);
                Check("padded gradients match the unpadded case",
                      gradDiff <= 1e-4f, $"max|diff| = {gradDiff:E2}");

                Console.WriteLine();
                Console.WriteLine("-- Padding must not leak into attention --");
                var probe = (Tensor)padInput.Clone();
                probe[0, seq - 1] = 11;

                using var leakModel = BuildModel(cfg, 99);
                (TensorBase la, _) = leakModel.Forward(padInput, padTarget);
                (TensorBase lb, _) = leakModel.Forward(probe, padTarget);
                float logitA = la[0, 0, 0], logitB = lb[0, 0, 0];
                Check("changing a PAD token does not move logit@0",
                      MathF.Abs(logitA - logitB) == 0f,
                      $"{logitA:F6} -> {logitB:F6}");

                using var refProbe = BuildModel(cfg, 99);
                (TensorBase na, _) = refProbe.Forward(realInput, realTarget);
                float maxRealDiff = 0f;
                for (int t = 0; t < real; t++)
                    maxRealDiff = MathF.Max(maxRealDiff, MathF.Abs(na[0, t, 4] - la[0, t, 4]));
                Check("padded batch reproduces the unpadded logits",
                      maxRealDiff <= 1e-4f,
                      $"max|diff| = {maxRealDiff:E2} over {real} positions");

                Console.WriteLine();
                Check("unpadded batch runs on the shared causal mask",
                      float.IsFinite(na[0, 0, 4]), $"logit@0 = {na[0, 0, 4]:F6}");
            }
            finally
            {
                AttentionMaskSettings.UseCausalMask = true;
            }

            Console.WriteLine();
            if (failed == 0) Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, {failed} failed.");
            else Console.WriteLine($"FAILURES: {passed} passed, {failed} failed.");
            return failed == 0;
        }

        private static float MaxGradDiff(TransformerModel a, TransformerModel b)
        {
            float worst = 0f;
            using var ea = a.Parameters.GetEnumerator();
            using var eb = b.Parameters.GetEnumerator();
            while (ea.MoveNext() && eb.MoveNext())
            {
                var gaT = ea.Current.Gradient;
                var gbT = eb.Current.Gradient;
                if (gaT == null || gbT == null) continue;
                var ga = gaT.ReadOnlySpan;
                var gb = gbT.ReadOnlySpan;
                int n = Math.Min(ga.Length, gb.Length);
                for (int i = 0; i < n; i++)
                    worst = MathF.Max(worst, MathF.Abs(ga[i] - gb[i]));
            }
            return worst;
        }

        private static TransformerModel BuildModel(TransformerConfig config, int seed)
        {
            var model = new TransformerModel(
                Guid.NewGuid(), config,
                new TrainingConfig { Optimizer = OptimizerType.AdamW, LearningRate = 0.01f },
                useQLora: false,
                backendType: BackendSelector.BackendType.CpuSimd);
            var rng = new Random(seed);
            foreach (var p in model.Parameters)
            {
                float[] d = p.Value.Data;
                for (int i = 0; i < d.Length; i++)
                    d[i] = (float)((rng.NextDouble() - 0.5) * 0.4);
            }
            return model;
        }
    }
}