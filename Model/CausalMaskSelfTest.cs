using System;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Verifies causal (autoregressive) masking, without starting a server:
    /// the mask admits the lower triangle and blocks the future; and - the
    /// property that actually matters - perturbing only the LAST token does not
    /// move the logit predicted from position 0. Checked in BOTH directions, so
    /// the unmasked control has to move and the test is not a tautology.
    /// <para>
    /// Run with: dotnet run -- --causal-mask-selftest
    /// </para>
    /// </summary>
    public static class CausalMaskSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Causal Mask Self-Test ===");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            bool origMask = AttentionMaskSettings.UseCausalMask;
            try
            {
                Console.WriteLine("-- Settings --");
                Check("default is masked", origMask, $"UseCausalMask = {origMask}");

                Console.WriteLine();
                Console.WriteLine("-- Generated mask geometry --");
                using (var backend = BackendSelector.SelectBackend(BackendSelector.BackendType.CpuSimd))
                using (var ws = new TensorWorkspace(backend))
                {
                    const int seq = 6;
                    var mask = MaskUtilitiesSimd.CreateCausalMask(seq);

                    bool lowerOk = true, futureBlocked = true;
                    for (int r = 0; r < seq; r++)
                    {
                        for (int c = 0; c < seq; c++)
                        {
                            float v = mask[r, c];
                            if (c <= r && v != 1f) lowerOk = false;
                            if (c > r && v != 0f) futureBlocked = false;
                        }
                    }
                    Check("lower triangle incl. diagonal attends", lowerOk, $"{seq}x{seq}");
                    Check("strictly-future positions are blocked", futureBlocked, $"{seq}x{seq}");

                    var scores = new Tensor(seq, seq);
                    scores.Span.Fill(1f);
                    backend.ApplyMaskInPlace(scores, mask);
                    bool applied = true;
                    for (int r = 0; r < seq; r++)
                        for (int c = 0; c < seq; c++)
                            if (c > r && scores[r, c] > -1e8f) applied = false;
                    Check("apply drives the future to the sentinel", applied, "-1e9 pre-softmax");
                }
Console.WriteLine();
                Console.WriteLine("-- Future independence (the real property) --");
                (float mb, float ma) = ProbeLastToken(masked: true);
                Check("masked: changing the last token does NOT move logit@0",
                      MathF.Abs(mb - ma) == 0f, $"{mb:F6} -> {ma:F6}");

                (float fb, float fa) = ProbeLastToken(masked: false);
                Check("unmasked control: changing it DOES move logit@0",
                      MathF.Abs(fb - fa) > 0f,
                      $"{fb:F6} -> {fa:F6} (control proves the test bites)");

                Console.WriteLine();
                Console.WriteLine("-- Training path with masking on --");
                var cfg = TinyConfig();
                using (var model = BuildModel(cfg, 1234))
                {
                    AttentionMaskSettings.UseCausalMask = true;
                    var input = new Tensor(2, cfg.MaxSequenceLength);
                    var target = new Tensor(2, cfg.MaxSequenceLength);
                    for (int i = 0; i < input.Data.Length; i++)
                    {
                        input.Data[i] = i % cfg.VocabSize;
                        target.Data[i] = (i + 1) % cfg.VocabSize;
                    }
                    model.BeginTraining();
                    bool finite = true;
                    for (int s = 0; s < 3; s++)
                        if (!float.IsFinite(model.TrainStep(input, target))) finite = false;
                    model.EndTraining();
                    Check("3 masked training steps stay finite", finite, "loss not NaN/Inf");
                }
            }
            finally
            {
                AttentionMaskSettings.UseCausalMask = origMask;
            }

            Console.WriteLine();
            if (failed == 0)
                Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, {failed} failed.");
            else
                Console.WriteLine($"FAILURES: {passed} passed, {failed} failed.");

            return failed == 0;
        }

        private static TransformerConfig TinyConfig() => new TransformerConfig
        {
            VocabSize = 32,
            EmbeddingSize = 16,
            NumLayers = 1,
            NumHeads = 2,
            FeedForwardSize = 32,
            MaxSequenceLength = 8
        };

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

        /// <summary>
        /// Perturbs only the LAST token and returns the logit predicted from
        /// position 0, before and after. Under causal masking the two must be
        /// identical; without it they must differ.
        /// </summary>
        private static (float Before, float After) ProbeLastToken(bool masked)
        {
            var cfg = TinyConfig();
            int seq = cfg.MaxSequenceLength;

            using var model = BuildModel(cfg, 4321);
            AttentionMaskSettings.UseCausalMask = masked;

            var a = new Tensor(seq);
            for (int i = 0; i < seq; i++) a[i] = 1;

            var b = (Tensor)a.Clone();
            b[seq - 1] = 2;      // perturb ONLY the final position

            (TensorBase la, _) = model.Forward(a);
            (TensorBase lb, _) = model.Forward(b);
            return (la[0, 0], lb[0, 0]);
        }
    }
}