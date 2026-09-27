using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Streaming extract + clean pipeline for large corpus uploads. Instead of
    /// buffering every document in memory, each upload is read once and each
    /// cleaned document is written straight to the output file. Memory stays
    /// flat (one line + dedup hashes) regardless of corpus size.
    ///
    /// Deduplication uses 64-bit FNV-1a hashes of the cleaned text rather than
    /// the full strings: ~1M documents cost ~8MB instead of ~1GB. Hash
    /// collisions are theoretically possible (2^-64 per pair) but negligible
    /// for a training corpus. Disable dedup for a byte-exact guarantee.
    ///
    /// Warnings are capped (default 200) so a pathological file cannot grow
    /// the report without bound; the cap itself is reported.
    /// </summary>
    public static class CorpusStreamPipeline
    {
        /// <summary>Kestrel cap for corpus uploads: 2GB.</summary>
        public const long MaxRequestBytes = 2147483648L;

        /// <summary>
        /// ASP.NET buffers multipart sections to disk past ~64KB, but small
        /// single-document .txt/.json uploads are still faster buffered.
        /// Anything above this size streams line-by-line instead.
        /// </summary>
        public const long StreamThresholdBytes = 64L * 1024 * 1024;

        public const int DefaultMaxWarnings = 200;

        public sealed class StreamStats
        {
            public int DocumentsIn;
            public int DocumentsOut;
            public long CharsIn;
            public long CharsOut;
            public int DuplicatesRemoved;
            public int FilteredByLength;
            public int FilteredEmpty;
            /// <summary>64-bit content hashes of kept documents (see class docs).</summary>
            public HashSet<ulong> DedupHashes { get; } = new();
            public List<CorpusFileStats> Files { get; } = new();
            public List<string> Warnings { get; } = new();
            public List<CorpusSampleDoc> Samples { get; } = new();

            public CorpusPreprocessReport ToReport() => new()
            {
                DocumentsIn = DocumentsIn,
                DocumentsOut = DocumentsOut,
                CharsIn = CharsIn,
                CharsOut = CharsOut,
                DuplicatesRemoved = DuplicatesRemoved,
                FilteredByLength = FilteredByLength,
                FilteredEmpty = FilteredEmpty,
                Files = Files,
                Warnings = Warnings,
                Samples = Samples
            };
        }

        public sealed class StreamOptions
        {
            public CorpusSourceFormat Format { get; init; } = CorpusSourceFormat.Auto;
            public string? TextField { get; init; }
            public string? Template { get; init; }
            public bool NormalizeWhitespace { get; init; } = true;
            public bool StripHtml { get; init; }
            public bool Deduplicate { get; init; } = true;
            public int MinChars { get; init; }
            public int MaxChars { get; init; }
            public int MaxWarnings { get; init; } = DefaultMaxWarnings;
            public int MaxSamples { get; init; } = 5;
            public int SampleTruncateLength { get; init; } = 500;
        }

        /// <summary>
        /// Stream one upload into the shared output writer. Returns per-file
        /// stats; cumulative totals accumulate in <paramref name="stats"/>.
        /// </summary>
        public static async Task<CorpusFileStats> ProcessUploadAsync(
            Stream uploadStream,
            string fileName,
            StreamWriter output,
            StreamStats stats,
            StreamOptions options,
            CancellationToken cancellationToken = default)
        {
            var resolved = options.Format == CorpusSourceFormat.Auto
                ? TrainingCorpusExtractor.ResolveFromExtension(fileName)
                : options.Format;

            var fileStats = new CorpusFileStats
            {
                FileName = fileName,
                ResolvedFormat = resolved.ToString()
            };

            int docsOutBefore = stats.DocumentsOut;
            var cleanOptions = new CorpusPreprocessOptions
            {
                NormalizeWhitespace = options.NormalizeWhitespace,
                StripHtml = options.StripHtml,
                Deduplicate = false, // dedup is hash-based here, not set-based
                MinChars = options.MinChars,
                MaxChars = options.MaxChars,
                MaxSamples = options.MaxSamples,
                SampleTruncateLength = options.SampleTruncateLength
            };
            var seenHashes = stats.DedupHashes;

            if (resolved == CorpusSourceFormat.Jsonl)
            {
                await ProcessJsonLinesAsync(
                    uploadStream, fileName, output, stats, fileStats,
                    cleanOptions, options, seenHashes, cancellationToken);
            }
            else if (uploadStream.CanSeek && uploadStream.Length > StreamThresholdBytes)
            {
                throw new InvalidDataException(
                    $"File '{fileName}' is {(uploadStream.Length / (1024 * 1024))}MB; " +
                    $"single-document .{resolved.ToString().ToLowerInvariant()} uploads are capped at " +
                    $"{StreamThresholdBytes / (1024 * 1024)}MB because they must fit in memory. " +
                    "Convert to .jsonl (one record per line) to stream arbitrarily large files.");
            }
            else
            {
                await ProcessBufferedAsync(
                    uploadStream, fileName, resolved, output, stats, fileStats,
                    cleanOptions, options, seenHashes, cancellationToken);
            }

            fileStats.DocumentsOut = stats.DocumentsOut - docsOutBefore;
            stats.Files.Add(fileStats);
            return fileStats;
        }

        /// <summary>
        /// Line-delimited path: constant memory per line, the right choice for
        /// GB-scale .jsonl. Each line is parsed, cleaned and written (or
        /// counted as dropped) before the next is read.
        /// </summary>
        private static async Task ProcessJsonLinesAsync(
            Stream uploadStream,
            string fileName,
            StreamWriter output,
            StreamStats stats,
            CorpusFileStats fileStats,
            CorpusPreprocessOptions cleanOptions,
            StreamOptions options,
            HashSet<ulong> seenHashes,
            CancellationToken cancellationToken)
        {
            using var reader = new StreamReader(uploadStream, leaveOpen: true);
            string? line;
            int lineNumber = 0;

            while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
            {
                lineNumber++;
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string? document = null;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    document = TrainingCorpusExtractor.ExtractDocumentFromElement(
                        doc.RootElement, $":{lineNumber}",
                        options.TextField, options.Template, out var warning);
                    if (warning != null)
                    {
                        AddWarning(stats, options, $"{fileName}{warning}");
                    }
                }
                catch (JsonException ex)
                {
                    AddWarning(stats, options,
                        $"{fileName}:{lineNumber}: skipped malformed JSON line ({ex.Message}).");
                    continue;
                }

                if (document == null)
                {
                    continue;
                }

                fileStats.DocumentsIn++;
                await WriteDocumentAsync(
                    document, fileName, output, stats, fileStats,
                    cleanOptions, options, seenHashes);
            }
        }

        /// <summary>
        /// Single-document path for .txt and whole-file .json. The whole upload
        /// is one document; required because a .txt corpus has no record
        /// boundaries to stream on. Guarded by size by the caller.
        /// </summary>
        private static async Task ProcessBufferedAsync(
            Stream uploadStream,
            string fileName,
            CorpusSourceFormat resolved,
            StreamWriter output,
            StreamStats stats,
            CorpusFileStats fileStats,
            CorpusPreprocessOptions cleanOptions,
            StreamOptions options,
            HashSet<ulong> seenHashes,
            CancellationToken cancellationToken)
        {
            string content;
            using (var reader = new StreamReader(uploadStream, leaveOpen: true))
            {
                content = await reader.ReadToEndAsync(cancellationToken);
            }

            if (resolved == CorpusSourceFormat.Txt)
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    AddWarning(stats, options, $"{fileName}: file is empty, nothing extracted.");
                    return;
                }

                fileStats.DocumentsIn++;
                await WriteDocumentAsync(
                    content, fileName, output, stats, fileStats,
                    cleanOptions, options, seenHashes);
                return;
            }

            // Whole-document .json: reuse the buffered extractor, then stream
            // each extracted document through the same cleaning path.
            CorpusExtractionResult extraction;
            try
            {
                extraction = TrainingCorpusExtractor.ExtractFromJsonString(
                    content, fileName, options.TextField, options.Template);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException(
                    $"File '{fileName}' is not valid JSON: {ex.Message}", ex);
            }

            foreach (var warning in extraction.Warnings)
            {
                AddWarning(stats, options, warning);
            }

            fileStats.DocumentsIn += extraction.Documents.Count;
            foreach (var document in extraction.Documents)
            {
                await WriteDocumentAsync(
                    document, fileName, output, stats, fileStats,
                    cleanOptions, options, seenHashes);
            }
        }

        /// <summary>
        /// Clean one document, apply filters + hash dedup, write survivors.
        /// Identical filtering semantics to the buffered preprocessor.
        /// </summary>
        private static async Task WriteDocumentAsync(
            string document,
            string fileName,
            StreamWriter output,
            StreamStats stats,
            CorpusFileStats fileStats,
            CorpusPreprocessOptions cleanOptions,
            StreamOptions options,
            HashSet<ulong> seenHashes)
        {
            stats.DocumentsIn++;
            stats.CharsIn += document.Length;

            var text = CorpusPreprocessor.CleanDocument(document, cleanOptions);
            if (string.IsNullOrEmpty(text))
            {
                stats.FilteredEmpty++;
                return;
            }

            if ((cleanOptions.MinChars > 0 && text.Length < cleanOptions.MinChars) ||
                (cleanOptions.MaxChars > 0 && text.Length > cleanOptions.MaxChars))
            {
                stats.FilteredByLength++;
                return;
            }

            if (options.Deduplicate && !seenHashes.Add(Hash64(text)))
            {
                stats.DuplicatesRemoved++;
                return;
            }

            if (stats.Samples.Count < options.MaxSamples)
            {
                stats.Samples.Add(new CorpusSampleDoc
                {
                    FileName = fileName,
                    Original = Truncate(document, options.SampleTruncateLength),
                    Cleaned = Truncate(text, options.SampleTruncateLength)
                });
            }

            await output.WriteAsync(text);
            await output.WriteLineAsync();
            await output.WriteLineAsync();

            stats.DocumentsOut++;
            stats.CharsOut += text.Length;
        }

        private static void AddWarning(StreamStats stats, StreamOptions options, string warning)
        {
            if (stats.Warnings.Count >= options.MaxWarnings)
            {
                return;
            }

            stats.Warnings.Add(warning);

            if (stats.Warnings.Count == options.MaxWarnings)
            {
                stats.Warnings.Add(
                    $"Further warnings suppressed (limit {options.MaxWarnings}). " +
                    "Counts above remain exact.");
            }
        }

        /// <summary>
        /// 64-bit FNV-1a over UTF-16 code units. Fast, allocation-free, and
        /// ~8 bytes per document in the dedup set instead of the full text.
        /// </summary>
        public static ulong Hash64(string text)
        {
            const ulong offset = 14695981039346656037UL;
            const ulong prime = 1099511628211UL;

            ulong hash = offset;
            foreach (char c in text)
            {
                hash ^= (ulong)c;
                hash *= prime;
            }

            return hash;
        }

        private static string Truncate(string? value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value ?? string.Empty;
            }

            return value.Substring(0, maxLength) + "…";
        }
// __PART3__
    }
}
