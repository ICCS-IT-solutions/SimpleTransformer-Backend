using System;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Verifies the inverted-dropout layer and the train/eval plumbing that
    /// switches it on, without starting a server:
    ///  1. The mask drops approximately <c>rate</c> of the units and the
    ///     inverted scaling (1/(1-rate)) keeps the mean equal to the input's,
    ///     so no rescaling is needed at inference.
    ///  2. The backward pass sends each gradient through the exact mask Forward
    ///     used, so dropped units receive zero gradient.
    ///  3. Disabled, and enabled-at-rate-0, are byte-exact pass-throughs.
    ///  4. A seeded layer is reproducible; an out-of-range rate is rejected.
    ///  5. DropoutRng is deterministic, uniform in [0,1), and Mix is injective.
    ///  6. Masks are a pure function of (site salt, step, batch item): the same
    ///     key reproduces a mask, a changed step does not, item 0 is unaffected
    ///     by batch size, and an interleaved forward cannot perturb the stream.
    ///  7. Model level: BeginTraining/EndTraining toggle dropout, the rate is
    ///     re-read from TrainingConfig on each entry, and a dropout-enabled
    ///     training step stays finite.
    ///
    /// Run with: dotnet run -- --dropout-selftest
    /// </summary>
    public static class DropoutSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Dropout Layer Self-Test ===");
            Console.WriteLine();

            int passed = 0;
            int failed = 0;

            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            using var backend = BackendSelector.SelectBackend(BackendSelector.BackendType.CpuSimd);
            using var workspace = new TensorWorkspace(backend);

            const int rows = 25;
            const int cols = 40;              // 1000 elements: good statistics
            const float rate = 0.5f;
            const float tol = 1e-6f;

            var input = new Tensor(rows, cols);
            input.Span.Fill(1.0f);            // all ones, so output == mask exactly
            var gradient = new Tensor(rows, cols);
            gradient.Span.Fill(1.0f);

            Console.WriteLine($"-- DropoutLayer (backend: {backend.Name}, {rows}x{cols}, rate {rate:P0}) --");

            // 1. Statistics + inverted scaling
            var layer = new DropoutLayer(rate, "test_dropout", seed: 1234) { Enabled = true };
            TensorBase output = layer.Forward(input, workspace);

            ReadOnlySpan<float> outSpan = output.ReadOnlySpan;
            int dropped = 0;
            bool allSurvivorsScaled = true;
            double sum = 0.0;
            for (int i = 0; i < outSpan.Length; i++)
            {
                float v = outSpan[i];
                if (v == 0f) dropped++;
                else
                {
                    sum += v;
                    if (MathF.Abs(v - 2.0f) > tol) allSurvivorsScaled = false;
                }
            }

            float droppedFraction = dropped / (float)outSpan.Length;
            Check("dropped fraction is close to the rate",
                MathF.Abs(droppedFraction - rate) < 0.05f, $"{droppedFraction:P1} vs {rate:P0}");

            float mean = (float)(sum / outSpan.Length);
            Check("inverted scaling preserves the mean",
                MathF.Abs(mean - 1.0f) < 0.08f, $"{mean:F4} vs 1.0");

            Check("survivors are scaled by 1/(1-rate)",
                allSurvivorsScaled, "every kept value == 2.0");

            // 2. Backward applies the exact forward mask
            TensorBase inputGradient = layer.Backward(gradient, workspace);
            ReadOnlySpan<float> gradSpan = inputGradient.ReadOnlySpan;

            bool masksMatch = true;
            for (int i = 0; i < gradSpan.Length; i++)
            {
                // input was all ones, so output == mask and dInput == grad * mask == mask.
                if (MathF.Abs(gradSpan[i] - outSpan[i]) > tol) { masksMatch = false; break; }
            }
            Check("backward zeroes exactly the dropped units", masksMatch,
                "dInput == forward output elementwise");

            workspace.Release(output);
            workspace.Release(inputGradient);

            // 3. Seeded reproducibility
            var seededA = new DropoutLayer(rate, "seed_a", seed: 99) { Enabled = true };
            var seededB = new DropoutLayer(rate, "seed_b", seed: 99) { Enabled = true };
            TensorBase outA = seededA.Forward(input, workspace);
            TensorBase outB = seededB.Forward(input, workspace);
            Check("same seed produces an identical mask",
                SpansEqual(outA.ReadOnlySpan, outB.ReadOnlySpan, tol), "two layers, seed 99");
            workspace.Release(outA);
            workspace.Release(outB);

            // 4. Pass-through paths (disabled / rate 0)
            var disabled = new DropoutLayer(rate, "disabled", seed: 7) { Enabled = false };
            TensorBase disabledOut = disabled.Forward(input, workspace);
            Check("disabled layer is an exact forward pass-through",
                SpansEqual(disabledOut.ReadOnlySpan, input.ReadOnlySpan, 0f), "no scaling");

            TensorBase disabledGrad = disabled.Backward(gradient, workspace);
            Check("disabled layer is an exact backward pass-through",
                SpansEqual(disabledGrad.ReadOnlySpan, gradient.ReadOnlySpan, 0f));
            workspace.Release(disabledOut);
            workspace.Release(disabledGrad);

            var zeroRate = new DropoutLayer(0.0f, "zero_rate") { Enabled = true };
            TensorBase zeroOut = zeroRate.Forward(input, workspace);
            Check("rate 0 is an exact pass-through even when enabled",
                SpansEqual(zeroOut.ReadOnlySpan, input.ReadOnlySpan, 0f));
            workspace.Release(zeroOut);

            // 5. Validation + DropoutSite propagation
            bool rejectedHigh = false, rejectedLow = false;
            try { _ = new DropoutLayer(1.0f); } catch (ArgumentOutOfRangeException) { rejectedHigh = true; }
            try { _ = new DropoutLayer(-0.1f); } catch (ArgumentOutOfRangeException) { rejectedLow = true; }
            Check("rate 1.0 is rejected", rejectedHigh);
            Check("negative rate is rejected", rejectedLow);

            // DropoutSite is the persisted unit, so rate/enable must reach the
            // per-item layers it hands out.
            var controlled = new DropoutSite("controlled", rate);
            controlled.SetRate(0.25f);
            controlled.SetEnabled(true);
            DropoutLayer controlledLayer = controlled.ForItem(0);
            Check("DropoutSite pushes enable + rate to its layers",
                controlledLayer.Enabled && MathF.Abs(controlledLayer.DropoutRate - 0.25f) < tol,
                $"enabled={controlledLayer.Enabled}, rate={controlledLayer.DropoutRate}");

            // 6. DropoutRng: the properties the resume guarantee depends on
            Console.WriteLine();
            Console.WriteLine("-- DropoutRng --");

            {
                var rngA = DropoutRng.FromSeed(123UL);
                var rngB = DropoutRng.FromSeed(123UL);
                bool sameSequence = true;
                for (int i = 0; i < 64; i++)
                {
                    if (rngA.NextUInt64() != rngB.NextUInt64()) { sameSequence = false; break; }
                }
                Check("same seed yields the same sequence", sameSequence);

                var rngC = DropoutRng.FromSeed(124UL);
                Check("a different seed yields a different sequence",
                    rngA.NextUInt64() != rngC.NextUInt64());

                var rngRange = DropoutRng.FromSeed(7UL);
                bool inRange = true;
                double drawSum = 0.0;
                const int draws = 200_000;
                for (int i = 0; i < draws; i++)
                {
                    float v = rngRange.NextSingle();
                    if (!(v >= 0f && v < 1f)) { inRange = false; break; }
                    drawSum += v;
                }
                Check("NextSingle stays in [0,1)", inRange);
                double meanDraw = drawSum / draws;
                Check("NextSingle is centred near 0.5",
                    MathF.Abs((float)meanDraw - 0.5f) < 0.01f, $"{meanDraw:F4}");

                // Mix must be injective in each argument, or two different
                // (step, batch item) pairs could collide on the same mask.
                ulong m00 = DropoutRng.Mix(1UL, 2UL);
                ulong m01 = DropoutRng.Mix(1UL, 3UL);
                ulong m10 = DropoutRng.Mix(2UL, 2UL);
                Check("Mix is injective in both arguments",
                    m00 != m01 && m00 != m10 && m01 != m10);

                // Known-answer test: pins the algorithm. An accidental change
                // would silently alter every mask sequence and invalidate the
                // reproducibility of existing checkpoints, so it must fail loudly.
                var rngKa = DropoutRng.FromSeed(0UL);
                ulong k0 = rngKa.NextUInt64();
                ulong k1 = rngKa.NextUInt64();
                ulong k2 = rngKa.NextUInt64();
                ulong k3 = rngKa.NextUInt64();
                Check("DropoutRng known-answer (seed 0)",
                    k0 == 0x99EC5F36CB75F2B4UL && k1 == 0xBF6E1F784956452AUL &&
                    k2 == 0x1A5F849D4933E6E0UL && k3 == 0x6AA594F1262D2D2CUL,
                    $"{k0:X16} {k1:X16} {k2:X16} {k3:X16}");
            }


            // 7. Step-keyed site derivation: the property that makes resume exact
            Console.WriteLine();
            Console.WriteLine("-- DropoutSite step keying --");

            {
                var siteA = new DropoutSite("step_a", rate, salt: 42UL);
                siteA.SetEnabled(true);

                siteA.PrepareForStep(7);
                TensorBase a1 = siteA.ForItem(0).Forward(input, workspace);
                siteA.PrepareForStep(7);
                TensorBase a2 = siteA.ForItem(0).Forward(input, workspace);
                Check("same (salt, step, item) reproduces the mask",
                    SpansEqual(a1.ReadOnlySpan, a2.ReadOnlySpan, 0f), "re-keyed to the same step");

                siteA.PrepareForStep(8);
                TensorBase a3 = siteA.ForItem(0).Forward(input, workspace);
                Check("a different step changes the mask",
                    !SpansEqual(a1.ReadOnlySpan, a3.ReadOnlySpan, 0f));

                // Batch-size independence: item 0's mask must not depend on how
                // many items were built before it.
                var siteSolo = new DropoutSite("solo", rate, salt: 99UL);
                siteSolo.SetEnabled(true);
                siteSolo.PrepareForStep(3);
                TensorBase soloItem0 = siteSolo.ForItem(0).Forward(input, workspace);

                var siteBatch = new DropoutSite("batch", rate, salt: 99UL);
                siteBatch.SetEnabled(true);
                siteBatch.PrepareForStep(3);
                siteBatch.ForItem(3);                       // build items 0..3 first
                TensorBase batchItem0 = siteBatch.ForItem(0).Forward(input, workspace);
                Check("item 0's mask is independent of batch size",
                    SpansEqual(soloItem0.ReadOnlySpan, batchItem0.ReadOnlySpan, 0f),
                    "batch of 1 vs batch of 4");

                // Interleaved-forward immunity: a Forward between steps (an
                // inference Predict, say) must not perturb the next step.
                var siteInterleaved = new DropoutSite("interleaved", rate, salt: 5UL);
                siteInterleaved.SetEnabled(true);
                siteInterleaved.PrepareForStep(11);
                siteInterleaved.ForItem(0).Forward(input, workspace);
                siteInterleaved.ForItem(0).Forward(input, workspace);   // interleaved pass
                siteInterleaved.PrepareForStep(12);
                TensorBase interleaved = siteInterleaved.ForItem(0).Forward(input, workspace);

                var siteClean = new DropoutSite("clean", rate, salt: 5UL);
                siteClean.SetEnabled(true);
                siteClean.PrepareForStep(12);
                TensorBase clean = siteClean.ForItem(0).Forward(input, workspace);
                Check("an interleaved forward cannot perturb the next step",
                    SpansEqual(interleaved.ReadOnlySpan, clean.ReadOnlySpan, 0f));

                workspace.Release(a1);
                workspace.Release(a2);
                workspace.Release(a3);
                workspace.Release(soloItem0);
                workspace.Release(batchItem0);
                workspace.Release(interleaved);
                workspace.Release(clean);
            }


            // 8. Model level: train/eval toggle, rate re-read, finite step
            Console.WriteLine();
            Console.WriteLine("-- Model integration --");

            var config = new TransformerConfig
            {
                VocabSize = 64,
                EmbeddingSize = 16,
                NumLayers = 2,
                NumHeads = 2,
                FeedForwardSize = 32,
                MaxSequenceLength = 8
            };

            var trainingConfig = new TrainingConfig
            {
                Optimizer = OptimizerType.AdamW,
                LearningRate = 0.001f,
                DropoutRate = 0.5f
            };

            const int sequenceLength = 8;
            var random = new Random(11);
            var sequence = new Tensor(sequenceLength);
            var target = new Tensor(sequenceLength);
            for (int i = 0; i < sequenceLength; i++)
            {
                sequence.Span[i] = random.Next(config.VocabSize);
                target.Span[i] = random.Next(config.VocabSize);
            }

            using var model = new TransformerModel(
                Guid.NewGuid(), config, trainingConfig, useQLora: false);

            Check("a freshly built model is in inference mode",
                model.CanInfer && !model.IsTraining);

            model.BeginTraining();
            Check("BeginTraining switches to training mode",
                model.IsTraining && !model.CanInfer);

            Tensor t1 = CloneForwardLogits(model, sequence);
            Tensor t2 = CloneForwardLogits(model, sequence);
            Check("training forwards differ (dropout is active)",
                !SpansEqual(t1.ReadOnlySpan, t2.ReadOnlySpan, 0f), "independent masks per pass");

            model.EndTraining();
            Tensor e1 = CloneForwardLogits(model, sequence);
            Tensor e2 = CloneForwardLogits(model, sequence);
            Check("inference forwards are identical (dropout off)",
                SpansEqual(e1.ReadOnlySpan, e2.ReadOnlySpan, 0f), "deterministic in eval");

            // Rate re-read: at 0.0 a training run must still be deterministic.
            model.TrainingConfig.DropoutRate = 0.0f;
            model.BeginTraining();
            Tensor z1 = CloneForwardLogits(model, sequence);
            Tensor z2 = CloneForwardLogits(model, sequence);
            Check("rate 0.0 in training is deterministic (rate re-read)",
                SpansEqual(z1.ReadOnlySpan, z2.ReadOnlySpan, 0f));

            // Raise the rate after construction: BeginTraining must pick it up.
            model.TrainingConfig.DropoutRate = 0.5f;
            model.BeginTraining();
            Tensor r1 = CloneForwardLogits(model, sequence);
            Tensor r2 = CloneForwardLogits(model, sequence);
            Check("rate raised post-build is applied at BeginTraining",
                !SpansEqual(r1.ReadOnlySpan, r2.ReadOnlySpan, 0f));

            // A dropout-enabled training step must not diverge.
            bool stepsFinite = true;
            for (int step = 0; step < 5; step++)
            {
                float loss = model.TrainStep(sequence, target);
                if (!float.IsFinite(loss)) { stepsFinite = false; break; }
            }
            Check("5 dropout training steps stay finite",
                stepsFinite, "no NaN/Inf loss from the masked path");

            model.EndTraining();
            Check("EndTraining returns to inference mode", model.CanInfer && !model.IsTraining);

            Console.WriteLine();
            Console.WriteLine($"{(failed == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT")}: {passed} passed, {failed} failed.");
            return failed == 0;
        }

        /// <summary>
        /// Runs one forward pass and returns an OWNED copy of the logits, so the
        /// caller can compare values after the workspace recycles the borrowed
        /// tensors on the next pass. Forward's borrowed tensors are reclaimed by
        /// the workspace Reset inside TrainStep or on model Dispose.
        /// </summary>
        private static Tensor CloneForwardLogits(TransformerModel model, TensorBase input)
        {
            var (logits, _) = model.Forward(input);
            var copy = new Tensor(logits.Shape);
            logits.ReadOnlySpan.CopyTo(copy.Span);
            return copy;
        }

        /// <summary>
        /// Element-wise comparison that treats NaN as "not equal" (so a diverged
        /// pass can never masquerade as a reproducible one).
        /// </summary>
        private static bool SpansEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float tolerance)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (!(MathF.Abs(a[i] - b[i]) <= tolerance)) return false;
            }
            return true;
        }
    }
}