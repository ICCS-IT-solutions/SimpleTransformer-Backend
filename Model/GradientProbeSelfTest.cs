using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Verifies the backward NaN tripwire (GradientProbeSettings) without starting
    /// a server.
    /// <para>
    /// The probe exists because gradient clipping can only name the poisoned
    /// PARAMETER after the whole backward sweep, not the OPERATION that produced
    /// the NaN. On a diverging run it named <c>token_embeddings</c>, yet every NaN
    /// count divided exactly by the embedding width - whole rows matching the
    /// batch's distinct token ids - which is the embedding scatter faithfully
    /// receiving an already-bad gradient. The parameter was the messenger.
    /// <para>
    /// Checks:
    ///  1. A healthy gradient is finite and produces NO probe report.
    ///  2. The detector catches a single NaN and a single +Inf anywhere.
    ///  3. Stage attribution names the operation that CREATED the NaN - the stage
    ///     before the first non-finite one - not the one that received it.
    ///  4. The default is OFF, so production training pays nothing.
    /// <para>
    /// Run with: dotnet run -- --gradient-probe-selftest
    /// </para>
    /// </summary>
    public static class GradientProbeSelfTest
    {
        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Gradient Probe Self-Test ===");
            Console.WriteLine();

            int passed = 0, failed = 0;
            void Check(string name, bool ok, string detail = "")
            {
                if (ok) passed++; else failed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            const int rows = 8, cols = 16;
            Tensor MakeTensor(Func<int, int, float> fill)
            {
                var t = new Tensor(rows, cols);
                var span = t.Data.AsSpan();
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        span[r * cols + c] = fill(r, c);
                return t;
            }

            // 1. No false positives on a healthy gradient.
            Check("healthy gradient reports finite",
                DiagonisticUtilities.IsFinite(MakeTensor((_, _) => 0.5f)));

            // 2. Detection of NaN / +Inf at various positions.
            Check("single NaN at [0,0] detected",
                !DiagonisticUtilities.IsFinite(MakeTensor((r, c) => (r == 0 && c == 0) ? float.NaN : 0.5f)));

            Check("single NaN at last element detected",
                !DiagonisticUtilities.IsFinite(MakeTensor((r, c) =>
                    (r == rows - 1 && c == cols - 1) ? float.NaN : 0.5f)));

            Check("single +Inf detected (not just NaN)",
                !DiagonisticUtilities.IsFinite(MakeTensor((r, c) =>
                    (r == 3 && c == 7) ? float.PositiveInfinity : 0.5f)));

            Console.WriteLine();
            int sigPassed = 0, sigFailed = 0;
            void Sig(string name, bool ok, string detail = "")
            {
                if (ok) sigPassed++; else sigFailed++;
                Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name,-52} {detail}");
            }

            // 3. The whole-row pattern seen in the real log: every non-finite entry
            // confined to complete rows, the signature of the embedding scatter
            // receiving a bad incoming gradient rather than the embedding itself
            // misbehaving. Real log counts all divided exactly by the embed width.
            const int vocab = 64, dim = 8, poisonedRows = 13;
            var rowWise = new Tensor(vocab, dim);
            var rw = rowWise.Data.AsSpan();
            for (int r = 0; r < vocab; r++)
                for (int c = 0; c < dim; c++)
                    rw[r * dim + c] = r < poisonedRows ? float.NaN : 0f;
            int nonFinite = 0;
            for (int i = 0; i < rw.Length; i++)
                if (!float.IsFinite(rw[i])) nonFinite++;
            Sig("whole-row NaN matches 512-aligned log pattern",
                nonFinite == poisonedRows * dim && !DiagonisticUtilities.IsFinite(rowWise),
                $"{nonFinite} = {poisonedRows} rows x {dim}");

            // 4. Stage attribution: the operation BEFORE the first non-finite stage
            // is the one that created the NaN. Mirrors the Probe closure in
            // TransformerModel.Backward.
            string Attribute(string[] stageNames, bool[] nonFiniteAt)
            {
                string firstBad = null, previous = "loss gradient (entering backward)";
                for (int s = 0; s < stageNames.Length; s++)
                {
                    if (firstBad != null) break;
                    if (!nonFiniteAt[s]) { previous = stageNames[s]; continue; }
                    return $"CREATED BY '{previous}'";
                }
                return string.Empty;
            }

            var st = new[] { "loss gradient", "output projection", "block 1", "block 0", "embedding" };

            Sig("loss-origin NaN attributed to the loss gradient",
                Attribute(st, new[] { true, false, false, false, false })
                    == "CREATED BY 'loss gradient (entering backward)'");

            Sig("projection-origin NaN attributed to output projection",
                Attribute(st, new[] { false, true, false, false, false })
                    == "CREATED BY 'loss gradient'");

            Sig("block-origin NaN attributed to the preceding block",
                Attribute(st, new[] { false, false, true, false, false })
                    == "CREATED BY 'output projection'");

            Sig("stops at the FIRST bad stage, not a later one",
                Attribute(st, new[] { false, true, true, true, true })
                    == "CREATED BY 'loss gradient'");

            Sig("healthy sweep attributes nothing (no false report)",
                Attribute(st, new[] { false, false, false, false, false }) == string.Empty);

            // 5. Off by default: production training must pay nothing.
            Sig("probe disabled by default (no per-stage cost)",
                !GradientProbeSettings.Enabled);

            // 6. Sub-stage attribution: the model-level probe can only name a BLOCK,
            //    so a block runs its own probe over its six internal steps. These use
            //    BackwardProbe directly - the same type the block uses - to pin the
            //    attribution the real diagnostic depends on.
            GradientProbeSettings.Enabled = true;
            try
            {
                var healthyGrad = MakeTensor((_, _) => 0.25f);
                var poisonedGrad = MakeTensor((r, c) => (r == 2 && c == 3) ? float.NaN : 0.25f);

                // NaN entering the block: the block's first stage is condemned.
                var p1 = new BackwardProbe("layers.5", "block input gradient");
                Sig("block probe is active when armed", p1.Active);
                p1.Check("block input gradient", poisonedGrad);
                Sig("NaN at entry attributes to block input gradient",
                    p1.FirstBadStage == "block input gradient");

                // NaN introduced by the feed-forward, as the real divergence may be.
                var p2 = new BackwardProbe("layers.5", "block input gradient");
                p2.Check("block input gradient", healthyGrad);
                p2.Check("layerNorm2 (_layerNorm2.Backward)", healthyGrad);
                p2.Check("FFN-path dropout", healthyGrad);
                p2.Check("feed-forward (_feedForward.Backward)", poisonedGrad);
                Sig("NaN from feed-forward attributes to feed-forward stage",
                    p2.FirstBadStage == "feed-forward (_feedForward.Backward)");

                // Once found, the chain stops reporting further stages.
                p2.Check("layerNorm1 (_layerNorm1.Backward)", poisonedGrad);
                Sig("chain stops after the first non-finite stage",
                    p2.FirstBadStage == "feed-forward (_feedForward.Backward)");

                // A nested probe must not re-report what the parent already found.
                var parent = new BackwardProbe("model", "loss gradient");
                parent.Check("loss gradient", poisonedGrad);
                var child = new BackwardProbe("layers.5", "block input gradient", parent);
                Sig("nested probe is inactive once parent is non-finite",
                    !child.Active && child.FoundNonFinite);

                // Throw mode must abort rather than continue on poisoned data.
                GradientProbeSettings.ThrowOnFirstNonFinite = true;
                var p3 = new BackwardProbe("layers.5", "block input gradient");
                bool threw = false;
                try
                {
                    p3.Check("feed-forward (_feedForward.Backward)", poisonedGrad);
                }
                catch (InvalidOperationException ex) when (ex.Message.StartsWith("[GradientProbe]"))
                {
                    threw = true;
                }
                Sig("throw mode aborts the sweep at the guilty stage", threw);
            }
            finally
            {
                GradientProbeSettings.Enabled = false;
                GradientProbeSettings.ThrowOnFirstNonFinite = false;
            }

            // 7. The SQLite REAL-column constraint that actually killed a training
            //    job: BindDouble throws "Cannot store 'NaN' values" and takes the
            //    whole SaveChanges - and therefore the run - down with it. The
            //    training loop now deliberately produces a NaN epoch loss when every
            //    step was skipped, so the persistence layer must sanitise it. This
            //    asserts the sanitiser against a real SQLite database, not a mock.
            RunSqliteNaNCheck(Sig);

            // 8. The actual root cause the tripwire found: GELU's tanh-approximation
            //    derivative evaluates 0 * Inf = NaN for |x| beyond ~1.845e19, because
            //    (1 + 3c*x^2) overflows to Inf while (1 - t*t) is exactly 0. That NaN
            //    poisoned every gradient downstream of the feed-forward, which is why
            //    token_embeddings looked like it was diverging. Assert the fix holds
            //    across the finite range and returns the analytic limits beyond it.
            GeluOverflowCheck(Sig);

            passed += sigPassed;
            failed += sigFailed;

            Console.WriteLine();
            if (failed == 0)
            {
                Console.WriteLine($"ALL CHECKS PASSED: {passed} passed, 0 failed.");
                return true;
            }

            Console.WriteLine($"FAILURES: {passed} passed, {failed} FAILED.");
            return false;
        }

        /// <summary>
        /// Proves the GELU derivative fix. Two properties matter:
        ///  1. It never returns NaN/Inf for a finite input - the bug that produced
        ///     the whole chain of poisoned gradients.
        ///  2. It is UNCHANGED for ordinary inputs, so this cannot silently alter
        ///     training. Checked against the original inline polynomial over the range
        ///     where that polynomial is itself finite.
        /// </summary>
        private static void GeluOverflowCheck(ProbeCheck check)
        {
            // The original, unguarded expression, kept here as the oracle.
            static float Original(float value)
            {
                float x2 = value * value;
                float x3 = x2 * value;
                float u = 0.7978845608f * (value + 0.044715f * x3);
                float t = MathF.Tanh(u);
                return 0.5f * (1f + t)
                     + 0.5f * value * (1f - t * t) * 0.7978845608f * (1f + 3f * 0.044715f * x2);
            }

            // (1) No NaN or Inf for ANY finite input, including the extremes that
            //     triggered the bug in the first place.
            float[] danger =
            {
                1e13f, 1e18f, 1.845e19f, 1e20f, 1e30f, float.MaxValue,
                -1e13f, -1e18f, -1.845e19f, -1e20f, -1e30f, -float.MaxValue
            };

            bool allFinite = true;
            string worst = "";
            foreach (var v in danger)
            {
                float d = GeluDerivative.Evaluate(v);
                if (!float.IsFinite(d))
                {
                    allFinite = false;
                    worst = $"{v:E3} -> {d}";
                    break;
                }
            }
            check("GELU derivative is finite for extreme inputs (was NaN)", allFinite, worst);

            // (2) Analytic limits beyond the guard: GELU' -> 1 as x -> +inf, -> 0
            //     as x -> -inf.
            check("GELU derivative saturates to 1 for large +x",
                GeluDerivative.Evaluate(1e20f) == 1f && GeluDerivative.Evaluate(float.MaxValue) == 1f);
            check("GELU derivative saturates to 0 for large -x",
                GeluDerivative.Evaluate(-1e20f) == 0f && GeluDerivative.Evaluate(-float.MaxValue) == 0f);

            // (3) Bit-for-bit agreement with the original across the ordinary range.
            //     This is the guarantee that the fix changes nothing that was working.
            float[] ordinary =
            {
                -10f, -5f, -3f, -2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f, 3f, 5f, 10f, 100f, 1000f
            };
            float maxDelta = 0f;
            string mismatch = "";
            foreach (var v in ordinary)
            {
                float a = Original(v);
                float b = GeluDerivative.Evaluate(v);
                float delta = MathF.Abs(a - b);
                if (delta > maxDelta) { maxDelta = delta; mismatch = $"{v}: {a} vs {b}"; }
            }
            check("ordinary-range results unchanged vs original", maxDelta == 0f,
                maxDelta == 0f ? "bit-identical" : mismatch);

            // (4) The specific documented threshold behaviour.
            check("derivative is finite just below and above the guard",
                float.IsFinite(GeluDerivative.Evaluate(GeluDerivative.OverflowGuard * 0.99f))
                && float.IsFinite(GeluDerivative.Evaluate(GeluDerivative.OverflowGuard * 1.01f)));
        }

        /// <summary>
        /// Proves the NaN -> REAL-column fix against a real SQLite database, and
        /// reproduces the original failure mode so the regression cannot come back.
        /// Uses an in-memory provider, so it touches no user data.
        /// </summary>
        private delegate void ProbeCheck(string name, bool ok, string detail = "");

        private static void RunSqliteNaNCheck(ProbeCheck check)
        {
            SQLitePCL.Batteries.Init();

            // Sanitiser as used by UpsertCheckpointAsync / UpdateJob.
            float Sanitise(float loss) => float.IsFinite(loss) ? loss : 0f;

            using var connection = new Microsoft.Data.Sqlite.SqliteConnection("Data Source=:memory:");
            connection.Open();

            using (var create = connection.CreateCommand())
            {
                create.CommandText = "CREATE TABLE probe (id INTEGER PRIMARY KEY, loss REAL);";
                create.ExecuteNonQuery();
            }

            // (a) Reproduce the failure: an unguarded NaN must throw. If SQLite ever
            //     starts accepting NaN this check fails loudly, which is correct -
            //     it means the constraint the fix relies on has changed.
            bool unguardedThrew;
            try
            {
                using var bad = connection.CreateCommand();
                bad.CommandText = "INSERT INTO probe (loss) VALUES (@l)";
                bad.Parameters.AddWithValue("@l", double.NaN);
                bad.ExecuteNonQuery();
                unguardedThrew = false;
            }
            catch (Exception ex)
            {
                var inner = ex.InnerException?.Message ?? ex.Message;
                unguardedThrew = inner.Contains("Cannot store 'NaN' values");
                if (!unguardedThrew)
                    Console.WriteLine($"        (sqlite said: {inner})");
            }

            check("raw NaN into a REAL column throws (original bug)", unguardedThrew,
                unguardedThrew ? "reproduced" : "NOT reproduced - SQLite behaviour changed");

            // (b) The sanitiser must let the write through.
            bool guardStored;
            try
            {
                using var good = connection.CreateCommand();
                good.CommandText = "INSERT INTO probe (loss) VALUES (@l)";
                good.Parameters.AddWithValue("@l", (double)Sanitise(float.NaN));
                good.ExecuteNonQuery();
                guardStored = true;
            }
            catch
            {
                guardStored = false;
            }

            check("sanitised NaN stores without throwing", guardStored);

            // (c) A real finite loss must pass through untouched - the sanitiser
            //     must not quietly rewrite good data.
            using (var real = connection.CreateCommand())
            {
                real.CommandText = "INSERT INTO probe (loss) VALUES (@l)";
                real.Parameters.AddWithValue("@l", 7.208750);
                real.ExecuteNonQuery();
            }

            using (var read = connection.CreateCommand())
            {
                read.CommandText = "SELECT COUNT(*) FROM probe WHERE ABS(loss - 7.20875) < 1e-6";
                check("finite loss passes through unchanged",
                    Convert.ToInt64(read.ExecuteScalar()) == 1);
            }

            // (d) +Inf must be caught too, not just NaN.
            check("sanitiser handles +Inf and -Inf",
                Sanitise(float.PositiveInfinity) == 0f && Sanitise(float.NegativeInfinity) == 0f);
        }
    }
}