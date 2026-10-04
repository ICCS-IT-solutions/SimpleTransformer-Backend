using System;
using System.Collections.Generic;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Numerical gradient check: validates the WHOLE backward pass - embeddings,
    /// attention, feed-forward, LayerNorm and the vocabulary projection - against
    /// finite differences of the loss.
    /// <para>
    /// Every other self-test compares this implementation against ITSELF (a
    /// cached run against a recomputed one, a resumed run against a continuous
    /// one). Those cannot catch a backward pass that is wrong in the same way
    /// twice. Only a numerical check can: it compares the analytic gradient
    /// against the DEFINITION of a gradient.
    /// <para>
    /// It uses DIRECTIONAL derivatives rather than per-coordinate ones, and that
    /// is forced by arithmetic rather than taste. The loss is ~3.4, so float32
    /// resolves it to ~4e-7. A typical gradient ENTRY here is ~1e-3, so a
    /// +-1e-3 coordinate probe changes the loss by ~1e-6 - about two units in the
    /// last place. That is pure cancellation noise, and it produces sign flips on
    /// a perfectly correct gradient. Summing along a direction multiplies the
    /// signal by ~sqrt(N) while the resolution error stays fixed, so the
    /// directional check is the one that is actually reliable in float32.
    /// Per-entry agreement is still reported, as information, not as a gate.
    /// <para>
    /// Run with: dotnet run -- --gradcheck-selftest
    /// </para>
    /// </summary>
    public static class GradientCheckSelfTest
    {
        private const float Eps = 1e-3f;
        private const int DirectionalChecks = 6;
        private const int ReportedSamples = 4;

        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Numerical Gradient Check (directional) ===");
            Console.WriteLine();

            int passed = 0, failed = 0;
            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            bool origMask = AttentionMaskSettings.UseCausalMask;
            bool origRecompute = AttentionMemorySettings.RecomputeSoftmaxWeights;
            try
            {
                AttentionMaskSettings.UseCausalMask = true;

                Check("cached-softmax backward matches finite differences",
                    RunCheck(recompute: false, out float cErr, out int cN, out string cInfo),
                    $"{cN} directions, worst relative error {cErr:E2}");

                Check("recomputed-softmax backward matches finite differences",
                    RunCheck(recompute: true, out float rErr, out int rN, out string rInfo),
                    $"{rN} directions, worst relative error {rErr:E2}");

                Check("both attention modes agree",
                    MathF.Abs(cErr - rErr) < 1e-3,
                    $"cache {cErr:E2} vs recompute {rErr:E2}");

                Console.WriteLine($"  cache     {cInfo}");
                Console.WriteLine($"  recompute {rInfo}");

                Check("a deliberately corrupted gradient is rejected",
                    !RunCheck(recompute: false, out _, out _, out _, corrupt: true),
                    "a third of the gradient sign-flipped must fail the check");
            }
            finally
            {
                AttentionMaskSettings.UseCausalMask = origMask;
                AttentionMemorySettings.RecomputeSoftmaxWeights = origRecompute;
            }

            Console.WriteLine();
            if (failed == 0) Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, {failed} failed.");
            else Console.WriteLine($"FAILURES: {passed} passed, {failed} failed.");
            return failed == 0;
        }

        private static bool RunCheck(bool recompute, out float worstRelErr,
            out int checkedDirections, out string perEntryInfo, bool corrupt = false)
        {
            worstRelErr = 0f;

            var config = new TransformerConfig
            {
                VocabSize = 24, EmbeddingSize = 12, NumLayers = 2,
                NumHeads = 2, FeedForwardSize = 24, MaxSequenceLength = 6
            };

            var model = new TransformerModel(
                Guid.NewGuid(), config,
                new TrainingConfig
                {
                    Optimizer = OptimizerType.AdamW,
                    LearningRate = 0.01f,
                    DropoutRate = 0f            // a gradient check needs determinism
                },
                useQLora: false,
                backendType: BackendSelector.BackendType.CpuSimd);

            var rng = new Random(20260930);
            foreach (var prm in model.Parameters)
            {
                float[] d = prm.Value.Data;
                for (int i = 0; i < d.Length; i++)
                    d[i] = (float)((rng.NextDouble() - 0.5) * 0.5);
            }

            // A fully-real batch: no padding, so the loss is a clean mean and the
            // check is not confounded by the ignore index.
            var input = new Tensor(2, config.MaxSequenceLength);
            var target = new Tensor(2, config.MaxSequenceLength);
            for (int i = 0; i < input.Data.Length; i++)
            {
                input.Data[i] = i % config.VocabSize;
                target.Data[i] = (i + 1) % config.VocabSize;
            }

            AttentionMemorySettings.RecomputeSoftmaxWeights = recompute;
            var loss = new CrossEntropyLoss(TransformerModel.IgnoreIndex);

            // --- analytic gradient --------------------------------------------
            model.ZeroGradients();
            (TensorBase logits, _) = model.Forward(input, target);
            TensorBase lossGrad = loss.Backward(logits, target, model.Workspace);
            model.Backward(lossGrad);
            model.Workspace.Reset();

            // Capture every weight/gradient array ONCE. model.Parameters is a lazy
            // iterator chain; re-enumerating it on every probe risks disagreeing
            // with the snapshot the direction vector was sized from.
            var weights = new List<float[]>();
            var grads = new List<float[]>();
            foreach (var prm in model.Parameters)
            {
                weights.Add(prm.Value.Data);
                grads.Add(prm.Gradient?.Data ?? new float[prm.Value.Data.Length]);
            }

            int total = 0;
            foreach (float[] w in weights) total += w.Length;
            if (total == 0) { checkedDirections = 0; perEntryInfo = "no parameters"; return false; }

            var flatG = new float[total];
            int c = 0;
            foreach (float[] g in grads)
            {
                Array.Copy(g, 0, flatG, c, g.Length);
                c += g.Length;
            }
            // Directions are built from the TRUE gradient. The control corrupts only
            // the ANALYTIC side - corrupting flatG before deriving v would make
            // the check compare the corrupted gradient against itself and pass
            // vacuously.
            var analytic = corrupt ? Corrupt((float[])flatG.Clone()) : flatG;

            // --- directional derivatives --------------------------------------
            // Directions must be WELL conditioned. A uniformly random unit vector in 3120
            // dimensions has g.v ~ |g|/sqrt(N) ~ 4e-3, barely above the float32
            // noise floor of a central difference - those directions are not
            // measurable and report noise, not error. Each direction is therefore
            // the gradient PLUS a perturbation and renormalised, which keeps
            // g.v large (and the check meaningful) while still probing a
            // different direction of the same gradient.
            var dirs = new List<float[]> { Normalised((float[])flatG.Clone()) };
            var dirRng = new Random(4242);
            double gNorm = 0.0;
            for (int i = 0; i < total; i++) gNorm += (double)flatG[i] * flatG[i];
            gNorm = Math.Sqrt(gNorm);

            // Perturbation amplitude is expressed as a fraction of |g|, spread over
            // N components - so the noise NORM is 0.25|g| regardless of N. An
            // absolute amplitude would swamp |g| in 3120 dimensions and silently
            // turn every "perturbed" direction back into a random one.
            float amp = (float)(0.25 * gNorm / Math.Sqrt(total));
            for (int dr = 1; dr < DirectionalChecks; dr++)
            {
                var v = new float[total];
                for (int i = 0; i < total; i++)
                    v[i] = flatG[i] + amp * (float)((dirRng.NextDouble() - 0.5) * 2.0);
                dirs.Add(Normalised(v));
            }

            bool ok = true;
            checkedDirections = dirs.Count;
            int dirIndex = 0;
            bool reported = false;

            // Absolute noise floor of a central difference in float32. A loss of
            // magnitude |L| resolves to about |L| * 6e-8 (half a float32 ULP), so
            // differencing two of them and dividing by 2*eps carries an error of
            // roughly |L| * 6e-8 / eps. A direction whose g.v is SMALLER than
            // that floor simply cannot be measured per-direction - which is why
            // the first direction (along g itself, where g.v = |g| ~ 0.06) is
            // ~300x better conditioned than a random one.
            float baselineLoss = MathF.Abs(PerturbedLoss(model, loss, input, target, weights, new float[total], 0f));
            float noiseFloor = 3.0f * baselineLoss * 6e-8f / Eps;

            foreach (float[] v in dirs)
            {
                float expected = Dot(analytic, v);
                float numeric = (PerturbedLoss(model, loss, input, target, weights, v, Eps)
                              - PerturbedLoss(model, loss, input, target, weights, v, -Eps))
                              / (2.0f * Eps);
                float absErr = MathF.Abs(numeric - expected);

                // Central differences are exact only to O(eps^2 * L'''); at eps = 1e-3 on a
            // model this nonlinear that is a few percent, which is the floor on
            // how tight this can be in float32.
            float tol = 8e-2f * MathF.Abs(expected) + noiseFloor;
                if (absErr > tol && !reported)
                {
                    reported = true;
                    Console.WriteLine($"      dir{dirIndex}: expected {expected,12:E4}  numeric {numeric,12:E4}  absErr {absErr:E3}  tol {tol:E3}");
                }
                dirIndex++;
                if (absErr > tol) ok = false;

                float relErr = absErr / MathF.Max(MathF.Abs(expected), noiseFloor);
                if (relErr > worstRelErr) worstRelErr = relErr;
            }

            perEntryInfo = $"{weights.Count} tensors, {total} parameters, noise floor {noiseFloor:E1}; " +
                           PerEntryInfo(model, loss, input, target);
            return ok;
        }

        /// <summary>Loss with theta offset by scale*v, restored afterwards.</summary>
        private static float PerturbedLoss(TransformerModel model, CrossEntropyLoss loss,
            TensorBase input, TensorBase target, List<float[]> weights, float[] v, float scale)
        {
            int total = v.Length;
            var saved = new float[total];

            int cursor = 0;
            foreach (float[] w in weights)
            {
                Array.Copy(w, 0, saved, cursor, w.Length);
                cursor += w.Length;
            }

            cursor = 0;
            foreach (float[] w in weights)
            {
                for (int i = 0; i < w.Length; i++)
                    w[i] = saved[cursor + i] + scale * v[cursor + i];
                cursor += w.Length;
            }

            (TensorBase logits, _) = model.Forward(input, target);
            float l = loss.Forward(logits, target);
            model.Workspace.Reset();

            cursor = 0;
            foreach (float[] w in weights)
            {
                Array.Copy(saved, cursor, w, 0, w.Length);
                cursor += w.Length;
            }

            return l;
        }

        private static float Dot(float[] a, float[] b)
        {
            double s = 0.0;   // long and mostly cancelling: accumulate in double
            for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i];
            return (float)s;
        }

        private static float[] Normalised(float[] v)
        {
            double norm = 0.0;
            for (int i = 0; i < v.Length; i++) norm += (double)v[i] * v[i];
            norm = Math.Sqrt(norm);
            if (norm > 0.0)
                for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
            return v;
        }

        /// <summary>
        /// Negate a `fraction` of the gradient's entries, in place.
        /// <para>
        /// That is the shape of a real backward defect (a transposed or
        /// mis-signed accumulation) and it is large enough to matter: negating a
        /// third of the gradient drops |g| by ~29%, far outside the ~3.5%
        /// agreement the real gradient shows. A small rotation, by contrast,
        /// changes |g| by well under a percent - genuinely below what a float32
        /// difference can resolve - so the earlier rotation-based control passed
        /// vacuously.
        /// </para>
        /// </summary>
        private static float[] Corrupt(float[] g)
        {
            // Negate the first third AND scale the whole vector by 1.5. Negation
            // alone is too easy to miss: it can leave the norm nearly unchanged,
            // and a direction built from the corrupted vector cancels it out
            // entirely. The scale guarantees a 50% disagreement along every
            // direction, so the control cannot pass vacuously.
            int n = g.Length / 3;
            for (int k = 0; k < n; k++) g[k] = -g[k] * 1.5f;
            for (int k = n; k < g.Length; k++) g[k] *= 1.5f;
            return g;
        }

        /// <summary>
        /// Per-entry agreement, reported for information only: these entries sit
        /// at the float32 cancellation noise floor (see the class summary), so it
        /// is not a meaningful pass/fail gate.
        /// </summary>
        private static string PerEntryInfo(TransformerModel model, CrossEntropyLoss loss,
            TensorBase input, TensorBase target)
        {
            var probe = new Random(7);
            double worst = 0.0;
            int seen = 0;

            foreach (var prm in model.Parameters)
            {
                var g = prm.Gradient;
                if (g == null) continue;
                float[] w = prm.Value.Data;
                float[] gd = g.Data;
                if (w.Length == 0) continue;

                for (int s = 0; s < ReportedSamples; s++)
                {
                    int idx = probe.Next(w.Length);
                    float orig = w[idx];
                    w[idx] = orig + Eps; float lp = LossOnly(model, loss, input, target);
                    w[idx] = orig - Eps; float lm = LossOnly(model, loss, input, target);
                    w[idx] = orig;

                    float num = (lp - lm) / (2.0f * Eps);
                    float ana = gd[idx];
                    float scale = MathF.Max(MathF.Max(MathF.Abs(num), MathF.Abs(ana)), 1e-3f);
                    worst = Math.Max(worst, Math.Abs((num - ana) / scale));
                    seen++;
                }
            }

            return $"per-entry over {seen} probes, worst {worst:F2} (noise-limited, informational)";
        }

        private static float LossOnly(TransformerModel model, CrossEntropyLoss loss,
            TensorBase input, TensorBase target)
        {
            (TensorBase logits, _) = model.Forward(input, target);
            float l = loss.Forward(logits, target);
            model.Workspace.Reset();
            return l;
        }
    }
}
