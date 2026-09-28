using SimpleTransformer.Api.Endpoints.Services.Extensions;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.AccelerationEngine
{
    /// <summary>
    /// Round-trip + parity check for the .stbin token-cache streaming path
    /// (dotnet run -- --tokencache-selftest). Needs no DB, GPU or model: it
    /// rebuilds the legacy windowing maths inline so the streaming path can be
    /// compared against exactly the semantics of TrainingDataExtensions.
    /// </summary>
    public static class TokenCacheSelfTest
    {
        public static bool RunAndPrint()
        {
            try
            {
                var vocab = new Vocabulary(new Dictionary<string, int>
                {
                    [SpecialTokens.Pad] = 0,
                    [SpecialTokens.Unknown] = 1,
                    [SpecialTokens.BeginningOfSequence] = 2,
                    [SpecialTokens.EndOfSequence] = 3,
                    [SpecialTokens.Mask] = 4,
                    ["hello"] = 5, ["world"] = 6, ["foo"] = 7, ["bar"] = 8,
                });
                var tokenizer = new WordLevelTokenizer(vocab);
                var vocabId = Guid.NewGuid();
                string dir = Path.Combine(Path.GetTempPath(), "stbin-" + Guid.NewGuid());
                Directory.CreateDirectory(dir);

                try
                {
                    HeaderAndDtype(dir, vocab, tokenizer, vocabId);
                    StaleCacheIsRejected(dir, vocab, tokenizer, vocabId);
                    DocBoundariesArePreserved(dir, tokenizer, vocabId);
                    WindowsMatchLegacySlicing(dir, vocab, tokenizer, vocabId);
                    StreamingMatchesLegacyBatching(dir, vocab, tokenizer, vocabId);
                    InputTargetAlignment(dir);
                    BufferedEpochIsASnapshot(dir);
                    U32FallbackRoundTrips(dir);
                    ReadDocumentsSplitsOnBlankLines(dir);
                    GreedyAdapterRoundTrips();
                }
                finally
                {
                    Directory.Delete(dir, recursive: true);
                }

                Console.WriteLine("TokenCacheSelfTest: PASS");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("TokenCacheSelfTest: FAIL: " + ex);
                return false;
            }
        }

        // ---- legacy-equivalent maths (mirrors TrainingDataExtensions) ----

        private const int Window = 4;

        /// <summary>
        /// Legacy loop is: for (i = 0; i &lt;= n - window - 1; i += window).
        /// That yields (n - 1) / window samples whenever n &gt;= window + 1.
        /// </summary>
        private static long LegacySampleCount(long n, int window) =>
            n >= window + 1 ? (n - 1) / window : 0;

        /// <summary>Legacy CreateMiniBatches limit/batch count with DropLast.</summary>
        private static (int batches, long used) LegacyBatching(
            long samples, int batchSize, bool dropLast)
        {
            long full = samples / batchSize;
            long rem = samples % batchSize;
            bool drop = dropLast && rem != 0 && full > 0;
            long used = drop ? full * batchSize : samples;
            int batches = (int)(full + (rem == 0 || drop ? 0 : 1));
            return (batches, used);
        }

        // ---- small assertion helpers ------------------------------------

        private static void Check(bool condition, string what)
        {
            if (!condition) throw new Exception("check failed: " + what);
            Console.WriteLine("  [ok] " + what);
        }

        private static void Throws<T>(Action action, string what) where T : Exception
        {
            try
            {
                action();
            }
            catch (T)
            {
                Console.WriteLine("  [ok] " + what);
                return;
            }
            catch (Exception ex)
            {
                throw new Exception($"check failed: {what} (threw {ex.GetType().Name})");
            }
            throw new Exception($"check failed: {what} (no exception thrown)");
        }

        private static string PathIn(string dir, string name) => Path.Combine(dir, name);

        private static TokenCache.Header WriteCache(
            string bin, IEnumerable<string> docs, Func<string, int[]> encode,
            TokenizerType type, int vocabSize, Guid vocabId)
        {
            var header = TokenCacheIO.Write(bin, docs, encode, (int)type, vocabSize, vocabId);
            Console.WriteLine($"  wrote {bin}: {header.TokenCount} ids, {header.Dtype}, " +
                $"{header.DocCount} docs, crc 0x{header.BodyCrc32:X8}");
            return header;
        }

        /// <summary>Read the whole body back through the streaming reader.</summary>
        private static int[] ReadAllIds(string bin, TokenCache.Header h)
        {
            using var fs = new FileStream(bin, FileMode.Open, FileAccess.Read, FileShare.Read);
            var dst = new float[h.TokenCount];
            TokenCacheIO.ReadWindowAsFloat(fs, h, 0, (int)h.TokenCount, dst);
            var ids = new int[dst.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                int v = (int)dst[i];
                if (dst[i] != v) throw new Exception("non-integral token id " + dst[i]);
                ids[i] = v;
            }
            return ids;
        }


        // ---- tests ------------------------------------------------------

        private static void HeaderAndDtype(
            string dir, Vocabulary vocab, ITokenizer tokenizer, Guid vocabId)
        {
            Console.WriteLine("header/dtype:");
            string bin = PathIn(dir, "header.stbin");
            var docs = new[] { "hello world", "foo bar hello" };
            var h = WriteCache(bin, docs, tokenizer.Encode,
                tokenizer.Type, vocab.Count, vocabId);

            Check(h.Dtype == TokenCache.DType.U16, "u16 dtype for vocab <= 65535");
            Check(TokenCache.DtypeFor(ushort.MaxValue) == TokenCache.DType.U16,
                "65535 boundary stays u16");
            Check(TokenCache.DtypeFor(ushort.MaxValue + 1) == TokenCache.DType.U32,
                "65536 boundary switches to u32");
            Check(TokenCache.BytesPerToken(TokenCache.DType.U16) == 2, "u16 = 2 bytes");
            Check(h.VocabSize == vocab.Count, "vocab size recorded");
            Check(h.VocabId == vocabId, "vocab id recorded");
            Check(h.TokenizerType == (int)TokenizerType.WordLevel, "tokenizer type recorded");
            Check(h.DocCount == docs.Length, "doc count recorded");
            // "hello world" -> 4 ids, "foo bar hello" -> 5 (bos/eos per doc).
            Check(h.TokenCount == 9, "token count = sum of per-doc encodes");
            Check(h.BodyCrc32 != 0, "body crc computed");

            long expectedBytes = TokenCache.HeaderBytes +
                h.TokenCount * TokenCache.BytesPerToken(h.Dtype);
            Check(new FileInfo(bin).Length == expectedBytes,
                "file length = 64-byte header + token body");

            var reread = TokenCache.ReadHeader(bin);
            Check(reread.TokenCount == h.TokenCount && reread.Dtype == h.Dtype &&
                reread.VocabId == h.VocabId && reread.BodyCrc32 == h.BodyCrc32,
                "header round-trips through EncodeHeader/ReadHeader");
        }


        private static void StaleCacheIsRejected(
            string dir, Vocabulary vocab, ITokenizer tokenizer, Guid vocabId)
        {
            Console.WriteLine("stale-cache guard:");
            string bin = PathIn(dir, "stale.stbin");
            WriteCache(bin, new[] { "hello world" }, tokenizer.Encode,
                tokenizer.Type, vocab.Count, vocabId);
            var h = TokenCache.ReadHeader(bin);

            // The happy path must NOT throw.
            TokenCache.Validate(h, vocabId, vocab.Count, (int)tokenizer.Type);
            Check(true, "matching vocab/tokenizer validates clean");

            Throws<InvalidOperationException>(
                () => TokenCache.Validate(h, Guid.NewGuid(), vocab.Count, (int)tokenizer.Type),
                "different vocab id rejected (would train wrong ids)");
            Throws<InvalidOperationException>(
                () => TokenCache.Validate(h, vocabId, vocab.Count + 1, (int)tokenizer.Type),
                "vocab size drift rejected");
            Throws<InvalidOperationException>(
                () => TokenCache.Validate(h, vocabId, vocab.Count, (int)TokenizerType.Bpe),
                "tokenizer type drift rejected");

            // Corrupt magic, version and a chopped body must all be caught
            // before a single id is trusted.
            string bad = PathIn(dir, "bad.stbin");
            File.Copy(bin, bad, overwrite: true);
            using (var fs = new FileStream(bad, FileMode.Open, FileAccess.Write))
            {
                fs.WriteByte(0x00);
            }
            Throws<InvalidDataException>(
                () => TokenCache.ReadHeader(bad), "clobbered magic rejected");

            File.Copy(bin, bad, overwrite: true);
            using (var fs = new FileStream(bad, FileMode.Open, FileAccess.Write))
            {
                fs.Seek(4, SeekOrigin.Begin);
                fs.WriteByte(0x7F);
            }
            Throws<InvalidDataException>(
                () => TokenCache.ReadHeader(bad), "unknown version rejected");

            File.Copy(bin, bad, overwrite: true);
            using (var fs = new FileStream(bad, FileMode.Open, FileAccess.Write))
            {
                fs.SetLength(TokenCache.HeaderBytes + 2);
            }
            Throws<InvalidDataException>(
                () => TokenCache.ReadHeader(bad), "truncated body rejected");

            using (var fs = new FileStream(bad, FileMode.Open, FileAccess.Write))
            {
                fs.SetLength(TokenCache.HeaderBytes - 8);
            }
            Throws<InvalidDataException>(
                () => TokenCache.ReadHeader(bad), "header shorter than 64 bytes rejected");
        }


        private static void DocBoundariesArePreserved(
            string dir, ITokenizer tokenizer, Guid vocabId)
        {
            Console.WriteLine("doc boundaries:");
            string bin = PathIn(dir, "docs.stbin");
            var docs = new[] { "hello world", "foo bar hello", "world foo" };
            var h = WriteCache(bin, docs, tokenizer.Encode,
                tokenizer.Type, 9, vocabId);

            var expected = docs.SelectMany(tokenizer.Encode).ToArray();
            var actual = ReadAllIds(bin, h);
            Check(actual.SequenceEqual(expected),
                "body == concatenated per-doc Encode()");

            // bos/eos are wrapped per document by the caller, which is what
            // marks doc boundaries without a separate .idx sidecar.
            int firstLen = tokenizer.Encode(docs[0]).Length;
            Check(actual[0] == 2 && actual[firstLen - 1] == 3,
                "first doc starts bos / ends eos");
            Check(actual[firstLen] == 2 && actual[firstLen + tokenizer.Encode(docs[1]).Length - 1] == 3,
                "second doc starts bos / ends eos");
            Check(actual[^1] == 3, "last doc ends eos");

            // Empty/whitespace docs contribute nothing and are not counted.
            string bin2 = PathIn(dir, "docs-skip.stbin");
            var h2 = WriteCache(bin2, new[] { "hello world", "", "   " }, tokenizer.Encode,
                tokenizer.Type, 9, vocabId);
            Check(h2.DocCount == 1 && h2.TokenCount == 4,
                "blank docs skipped (not counted)");
        }

        private static void WindowsMatchLegacySlicing(
            string dir, Vocabulary vocab, ITokenizer tokenizer, Guid vocabId)
        {
            Console.WriteLine("window slicing == CreateTrainingSamples:");
            string bin = PathIn(dir, "windows.stbin");
            var docs = new[] { "hello world", "foo bar hello", "world foo hello bar" };
            var h = WriteCache(bin, docs, tokenizer.Encode, tokenizer.Type, vocab.Count, vocabId);
            var flat = ReadAllIds(bin, h);

            long legacyCount = LegacySampleCount(flat.Length, Window);
            using var src = new StreamingBatchSource(bin, Window, batchSize: 4, dropLast: false);
            Check(src.SampleCount == legacyCount,
                $"sample count {src.SampleCount} == legacy {legacyCount}");

            using var fs = new FileStream(bin, FileMode.Open, FileAccess.Read, FileShare.Read);
            var row = new float[Window + 1];
            int samples = 0;
            for (int i = 0; i <= flat.Length - Window - 1; i += Window)
            {
                TokenCacheIO.ReadWindowAsFloat(fs, h, i, Window + 1, row);
                for (int c = 0; c < Window; c++)
                {
                    if ((int)row[c] != flat[i + c]) throw new Exception($"input mismatch @{i}+{c}");
                    if ((int)row[c + 1] != flat[i + c + 1]) throw new Exception($"target mismatch @{i}+{c}");
                }
                samples++;
            }
            Check(samples == legacyCount, "every legacy window paged back identically");

            // The hot path reuses ONE scratch buffer for every window: a fresh
            // page per window would be ~1KB of garbage per window, per epoch.
            var scratch = new byte[(Window + 1) * TokenCache.BytesPerToken(h.Dtype)];
            using var fs2 = new FileStream(bin, FileMode.Open, FileAccess.Read, FileShare.Read);
            var row2 = new float[Window + 1];
            for (int i = 0; i <= flat.Length - Window - 1; i += Window)
            {
                TokenCacheIO.ReadWindowAsFloat(fs2, h, i, Window + 1, row2, scratch);
                for (int c = 0; c <= Window; c++)
                    if ((int)row2[c] != flat[i + c])
                        throw new Exception($"scratch-overload mismatch @{i}+{c}");
            }
            Check(true, "scratch-buffer overload pages identical windows (no per-window alloc)");

            Throws<ArgumentException>(
                () => TokenCacheIO.ReadWindowAsFloat(fs2, h, 0, Window + 1, row2, new byte[4]),
                "undersized scratch buffer rejected instead of over-reading");

            // A corrupt/short cache must fail loudly rather than train on zeros.
            Throws<EndOfStreamException>(
                () => TokenCacheIO.ReadWindowAsFloat(
                    fs2, h, flat.Length - 1, Window + 1, row2, scratch),
                "window reading past the end of the body is detected");
        }


        private static void StreamingMatchesLegacyBatching(
            string dir, Vocabulary vocab, ITokenizer tokenizer, Guid vocabId)
        {
            Console.WriteLine("StreamingBatchSource batch accounting:");
            string bin = PathIn(dir, "batches.stbin");
            // Deliberately not a whole multiple of batchSize, to exercise the
            // trailing-partial-batch / DropLast rules.
            var docs = new[] { string.Join(' ', Enumerable.Repeat("hello world foo bar", 7)) };
            var h = WriteCache(bin, docs, tokenizer.Encode, tokenizer.Type, vocab.Count, vocabId);
            var flat = ReadAllIds(bin, h);
            long legacySamples = LegacySampleCount(flat.Length, Window);

            const int batchSize = 4;
            foreach (bool dropLast in new[] { false, true })
            {
                var (legacyBatches, legacyUsed) = LegacyBatching(legacySamples, batchSize, dropLast);
                using var src = new StreamingBatchSource(bin, Window, batchSize, dropLast);

                Check(src.SampleCount == legacySamples,
                    $"dropLast={dropLast}: sample count {src.SampleCount} == legacy {legacySamples}");
                Check(src.BatchCount == legacyBatches,
                    $"dropLast={dropLast}: batch count {src.BatchCount} == legacy {legacyBatches}");

                var epoch = src.StreamEpoch(20260928).ToList();
                Check(epoch.Count == legacyBatches,
                    $"dropLast={dropLast}: enumerated {epoch.Count} batches == {legacyBatches}");

                long used = epoch.Sum(b => (long)b.BatchSize);
                Check(used == legacyUsed,
                    $"dropLast={dropLast}: used samples {used} == legacy {legacyUsed}");

                bool shapesOk = epoch.All(b => b.Inputs.Rows == b.BatchSize &&
                    b.Inputs.Cols == Window && b.Targets.Rows == b.BatchSize &&
                    b.Targets.Cols == Window);
                Check(shapesOk, $"dropLast={dropLast}: every batch is [row, {Window}]");
            }

            // Same seed => identical shuffle (resume/debug reproducibility);
            // different seed => different order.
            using var srcA = new StreamingBatchSource(bin, Window, batchSize, false);
            var first = srcA.StreamEpoch(1234).Select(b => (int)b.Inputs[0, 0]).ToArray();
            using var srcB = new StreamingBatchSource(bin, Window, batchSize, false);
            var again = srcB.StreamEpoch(1234).Select(b => (int)b.Inputs[0, 0]).ToArray();
            using var srcC = new StreamingBatchSource(bin, Window, batchSize, false);
            var other = srcC.StreamEpoch(4321).Select(b => (int)b.Inputs[0, 0]).ToArray();
            Check(first.SequenceEqual(again), "same seed => identical order");
            Check(!first.SequenceEqual(other), "different seed => different order");

            // Two concurrent epoch enumerations must not corrupt each other:
            // both pages the file independently (own FileStream + buffers).
            using var srcD = new StreamingBatchSource(bin, Window, batchSize, false);
            var e1 = srcD.StreamEpoch(7).Select(b => (int)b.Inputs[0, 0]).ToArray();
            var e2 = srcD.StreamEpoch(7).Select(b => (int)b.Inputs[0, 0]).ToArray();
            Check(e1.SequenceEqual(e2), "re-enumerating the same source is repeatable");

            // StreamingBatchSource.Validate must agree with TokenCache.Validate.
            using var srcE = new StreamingBatchSource(bin, Window, batchSize, false);
            srcE.Validate(vocabId, tokenizer);
            Check(true, "StreamingBatchSource.Validate accepts the matching tokenizer");
            Throws<InvalidOperationException>(
                () => srcE.Validate(Guid.NewGuid(), tokenizer),
                "StreamingBatchSource.Validate rejects a different vocab id");
        }


        /// <summary>
        /// A synthetic, globally increasing id stream makes every window
        /// provably identifiable: input[r,c] must be k*Window + c and the
        /// target must be exactly one id later, for ANY shuffle order.
        /// </summary>
        private static void InputTargetAlignment(string dir)
        {
            Console.WriteLine("input/target alignment (synthetic increasing ids):");
            int counter = 0;
            int[] EncodeSeq(string _)
            {
                var ids = new int[Window * 5];
                for (int i = 0; i < ids.Length; i++) ids[i] = counter++;
                return ids;
            }

            string bin = PathIn(dir, "align.stbin");
            var h = WriteCache(bin, new[] { "ignored" }, EncodeSeq,
                TokenizerType.WordLevel, 1000, Guid.NewGuid());
            Check(h.TokenCount == Window * 5, "synthetic encode length honoured");

            const int batchSize = 3;
            using var src = new StreamingBatchSource(bin, Window, batchSize, false);
            int rowsSeen = 0;
            long maxId = -1;
            foreach (var batch in src.StreamEpoch(99))
            {
                for (int r = 0; r < batch.BatchSize; r++)
                {
                    for (int c = 0; c < Window; c++)
                    {
                        int input = (int)batch.Inputs[r, c];
                        int target = (int)batch.Targets[r, c];
                        if (target != input + 1)
                            throw new Exception($"target {target} != input {input} + 1 at [{r},{c}]");
                        if (input % Window != c)
                            throw new Exception($"row/col misaligned: {input} % {Window} != {c}");
                        maxId = Math.Max(maxId, target);
                    }
                    rowsSeen++;
                }
            }
            Check(rowsSeen == src.SampleCount, "every sample row materialised exactly once");
            Check(maxId <= h.TokenCount - 1, "no read past the end of the token body");
        }

        private static void U32FallbackRoundTrips(string dir)
        {
            Console.WriteLine("u32 fallback:");
            string bin = PathIn(dir, "u32.stbin");
            var bigIds = new[] { 65536, 70000, 1234567, 0 };
            var h = WriteCache(bin, new[] { "ignored" }, _ => bigIds,
                TokenizerType.Bpe, 1_000_000, Guid.NewGuid());

            Check(h.Dtype == TokenCache.DType.U32, "vocab > 65535 selects u32");
            Check(h.TokenizerType == (int)TokenizerType.Bpe, "tokenizer type recorded for u32");
            var roundTripped = ReadAllIds(bin, h);
            Check(roundTripped.SequenceEqual(bigIds), "u32 ids beyond 65535 round-trip exactly");

            // A u16 cache must refuse an out-of-range id rather than wrap it.
            Throws<InvalidOperationException>(
                () => TokenCacheIO.Write(PathIn(dir, "oob.stbin"), new[] { "ignored" },
                    _ => new[] { 70000 }, (int)TokenizerType.WordLevel, 1000, Guid.NewGuid()),
                "u16 writer rejects an id above 65535 instead of wrapping");
        }

        private static void ReadDocumentsSplitsOnBlankLines(string dir)
        {
            Console.WriteLine("ReadDocuments doc splitting:");
            string txt = PathIn(dir, "corpus.txt");
            File.WriteAllText(txt,
                "hello world\r\n\r\nfoo bar hello\r\nand more\r\n\r\nworld foo");

            var docs = TokenCacheIO.ReadDocuments(txt).ToArray();
            Check(docs.Length == 3, $"3 docs recovered from 3 blank-line-separated blocks");
            Check(docs[0] == "hello world", "doc 1 = single line");
            Check(docs[1] == "foo bar hello and more", "doc 2 = continuation lines joined by a space");
            Check(docs[2] == "world foo", "doc 3 = trailing block without a final blank line");

            // Round-trip: what ReadDocuments yields is what the cache stores.
            string bin = PathIn(dir, "fromtxt.stbin");
            var vocab = new Vocabulary(new Dictionary<string, int>
            {
                [SpecialTokens.Pad] = 0, [SpecialTokens.Unknown] = 1,
                [SpecialTokens.BeginningOfSequence] = 2, [SpecialTokens.EndOfSequence] = 3,
                [SpecialTokens.Mask] = 4,
                ["hello"] = 5, ["world"] = 6, ["foo"] = 7, ["bar"] = 8,
                ["and"] = 9, ["more"] = 10,
            });
            var tokenizer = new WordLevelTokenizer(vocab);
            var h = WriteCache(bin, TokenCacheIO.ReadDocuments(txt), tokenizer.Encode,
                tokenizer.Type, vocab.Count, Guid.NewGuid());
            Check(h.DocCount == 3, "cache built from ReadDocuments has 3 docs");

            var expected = TokenCacheIO.ReadDocuments(txt).SelectMany(tokenizer.Encode).ToArray();
            Check(ReadAllIds(bin, h).SequenceEqual(expected),
                "cache body matches ReadDocuments + Encode exactly");

            string empty = PathIn(dir, "empty.stbin");
            var h2 = WriteCache(empty, Enumerable.Empty<string>(), tokenizer.Encode,
                tokenizer.Type, vocab.Count, Guid.NewGuid());
            Check(h2.TokenCount == 0 && h2.DocCount == 0, "empty corpus writes an empty body");
            Throws<InvalidOperationException>(
                () => TokenCache.Validate(h2, h2.VocabId, vocab.Count, (int)tokenizer.Type),
                "empty cache is rejected by Validate (nothing to train on)");
        }

        /// <summary>
        /// The regression that matters most: StreamEpoch hands out two reusable
        /// tensors, so anything that buffers an epoch (the &lt;=8 batch path, or
        /// any .ToList()/Chunk()) must go through MaterializeEpoch, otherwise
        /// every buffered MiniBatch aliases the LAST batch's data and the epoch
        /// silently trains the same sample over and over.
        /// </summary>
        private static void BufferedEpochIsASnapshot(string dir)
        {
            Console.WriteLine("buffered epoch snapshots (aliasing regression):");
            int counter = 0;
            int[] EncodeSeq(string _)
            {
                // 12 full batches' worth of samples plus the one extra token the
                // windowing needs, so there is no trailing partial batch and
                // every batch exercises the reused-buffer path.
                var ids = new int[Window * 12 + 1];
                for (int i = 0; i < ids.Length; i++) ids[i] = counter++;
                return ids;
            }

            string bin = PathIn(dir, "buffered.stbin");
            WriteCache(bin, new[] { "ignored" }, EncodeSeq,
                TokenizerType.WordLevel, 1000, Guid.NewGuid());

            const int batchSize = 4;
            const int seed = 4242;

            using var src = new StreamingBatchSource(bin, Window, batchSize, false);
            var buffered = src.MaterializeEpoch(seed);
            Check(buffered.Count == src.BatchCount && buffered.Count > 1,
                $"materialized epoch has {buffered.Count} batches (>1 to be meaningful)");

            // Every batch must own its own tensors...
            bool shared = false;
            for (int i = 0; i < buffered.Count && !shared; i++)
                for (int j = i + 1; j < buffered.Count; j++)
                    if (ReferenceEquals(buffered[i].Inputs, buffered[j].Inputs) ||
                        ReferenceEquals(buffered[i].Targets, buffered[j].Targets))
                    {
                        shared = true;
                        break;
                    }
            Check(!shared, "no buffered batch shares a tensor with another");

            // ...and must NOT all hold the final batch's data.
            int distinctFirstIds = buffered.Select(b => (int)b.Inputs[0, 0]).Distinct().Count();
            Check(distinctFirstIds == buffered.Count,
                "buffered batches hold distinct data (not one aliased buffer)");

            // The zero-copy stream and the snapshots must agree in order+content.
            using var src2 = new StreamingBatchSource(bin, Window, batchSize, false);
            int idx = 0;
            foreach (var live in src2.StreamEpoch(seed))
            {
                for (int r = 0; r < live.BatchSize; r++)
                    for (int c = 0; c < Window; c++)
                    {
                        if (live.Inputs[r, c] != buffered[idx].Inputs[r, c] ||
                            live.Targets[r, c] != buffered[idx].Targets[r, c])
                            throw new Exception($"MaterializeEpoch batch {idx} != StreamEpoch batch {idx}");
                    }
                idx++;
            }
            Check(idx == buffered.Count, "MaterializeEpoch order matches StreamEpoch");

            // The .ToList() trap itself: prove it aliases, which is exactly why
            // the training loop must not do this on StreamEpoch. Every batch of
            // a buffered epoch is the SAME tensors, all showing the last batch's
            // values - an epoch that silently trains one sample N times.
            using var src3 = new StreamingBatchSource(bin, Window, batchSize, false);
            var trapped = src3.StreamEpoch(seed).ToList();
            bool aliased = trapped.Count > 1 && trapped
                .All(b => ReferenceEquals(b.Inputs, trapped[0].Inputs) &&
                          ReferenceEquals(b.Targets, trapped[0].Targets));
            Check(aliased,
                "documented: .ToList() over StreamEpoch aliases one buffer (must use MaterializeEpoch)");
            Check(trapped.All(b => b.Inputs[0, 0] == trapped[0].Inputs[0, 0]),
                "aliased batches also report identical values (would be a silent no-op epoch)");
        }


        /// <summary>
        /// The BPE fallback used when the merge table was not persisted: exact
        /// whole-word hits must win, otherwise greedy longest-match, and Decode
        /// must reverse it through the reverse map (not an O(vocab) scan).
        /// </summary>
        private static void GreedyAdapterRoundTrips()
        {
            Console.WriteLine("greedy subword adapter (BPE fallback):");
            var vocab = new Vocabulary(new Dictionary<string, int>
            {
                [SpecialTokens.Pad] = 0, [SpecialTokens.Unknown] = 1,
                [SpecialTokens.BeginningOfSequence] = 2, [SpecialTokens.EndOfSequence] = 3,
                ["he"] = 4, ["llo"] = 5, ["hello"] = 6, ["world"] = 7,
            });
            var adapter = new GreedySubwordTokenizerAdapter(vocab);

            Check(adapter.Type == TokenizerType.Bpe, "adapter reports the Bpe tokenizer type");
            Check(adapter.VocabularySize == 8, "adapter vocabulary size");
            Check(adapter.EosTokenId == 3, "adapter eos id");

            var wholeWords = adapter.Encode("hello world");
            Check(wholeWords.SequenceEqual(new[] { 2, 6, 7, 3 }),
                "whole-word vocab hits win (bos hello world eos)");

            var greedy = adapter.Encode("llohe");
            Check(greedy.SequenceEqual(new[] { 2, 5, 4, 3 }),
                "non-vocab word falls back to greedy longest-match (llo + he)");

            var unmatched = adapter.Encode("hellos");
            Check(unmatched.SequenceEqual(new[] { 2, 6, 1, 3 }),
                "unmatched trailing char becomes a single <unk> rather than aborting");

            Check(adapter.Decode(wholeWords) == "hello world",
                "Decode reverses Encode via the reverse map");
            Check(adapter.Decode(new[] { 0, 2, 3 }) == "",
                "bos/eos/pad decode to nothing");
            Check(adapter.Decode(new[] { 6, 6, 7 }) == "hello hello world",
                "repeated ids decode in order");
        }


    }
}
