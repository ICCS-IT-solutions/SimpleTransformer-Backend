using System.Text;
using System.Text.RegularExpressions;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Pure, deterministic cleaning pass over extracted documents: Unicode NFC,
    /// control-character removal, optional HTML stripping, whitespace collapse,
    /// length filtering and exact deduplication. Returns the cleaned documents
    /// plus a full audit report.
    /// </summary>
    public static class CorpusPreprocessor
    {
        private static readonly Regex WhitespaceRun = new(@"\s+", RegexOptions.Compiled);
        private static readonly Regex HtmlTag = new(@"<[^>]+>", RegexOptions.Compiled);

        public static (List<string> Cleaned, CorpusPreprocessReport Report) Process(
            IReadOnlyList<CorpusExtractionResult> extractions,
            CorpusPreprocessOptions? options = null)
        {
            options ??= CorpusPreprocessOptions.Defaults;

            var cleaned = new List<string>();
            var report = new CorpusPreprocessReport();
            var seen = new HashSet<string>(StringComparer.Ordinal);

            foreach (var extraction in extractions)
            {
                var stats = new CorpusFileStats
                {
                    FileName = extraction.FileName,
                    ResolvedFormat = extraction.ResolvedFormat.ToString(),
                    DocumentsIn = extraction.Documents.Count
                };

                report.Warnings.AddRange(extraction.Warnings);

                foreach (var doc in extraction.Documents)
                {
                    report.DocumentsIn++;
                    report.CharsIn += doc?.Length ?? 0;

                    var text = CleanDocument(doc, options);
                    if (string.IsNullOrEmpty(text))
                    {
                        report.FilteredEmpty++;
                        continue;
                    }

                    if ((options.MinChars > 0 && text.Length < options.MinChars) ||
                        (options.MaxChars > 0 && text.Length > options.MaxChars))
                    {
                        report.FilteredByLength++;
                        continue;
                    }

                    if (options.Deduplicate && !seen.Add(text))
                    {
                        report.DuplicatesRemoved++;
                        continue;
                    }

                    if (report.Samples.Count < options.MaxSamples)
                    {
                        report.Samples.Add(new CorpusSampleDoc
                        {
                            FileName = extraction.FileName,
                            Original = Truncate(doc, options.SampleTruncateLength),
                            Cleaned = Truncate(text, options.SampleTruncateLength)
                        });
                    }

                    cleaned.Add(text);
                    report.CharsOut += text.Length;
                }

                stats.DocumentsOut = cleaned.Count - report.DocumentsOut;
                report.DocumentsOut = cleaned.Count;
                report.Files.Add(stats);
            }

            return (cleaned, report);
        }

        /// <summary>
        /// Cleaning order matters: normalize encoding first, turn every control
        /// character into a space (never delete, to avoid fusing words), then
        /// collapse whitespace. HTML is stripped before whitespace handling.
        /// </summary>
        public static string CleanDocument(string? doc, CorpusPreprocessOptions options)
        {
            if (string.IsNullOrWhiteSpace(doc))
            {
                return string.Empty;
            }

            // Unicode NFC so canonically-equivalent strings dedupe and tokenize alike.
            string text = doc.Normalize(NormalizationForm.FormC);

            if (options.StripHtml)
            {
                text = System.Net.WebUtility.HtmlDecode(text);
                text = HtmlTag.Replace(text, " ");
            }

            if (!options.NormalizeWhitespace)
            {
                return text.Trim();
            }

            var sb = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                // All control chars (including \n, \r, \t, zero-width spaces,
                // BOM) become spaces; deleting them would fuse adjacent words.
                if (char.IsControl(c) || c == '\u200B' || c == '\u200C' || c == '\uFEFF')
                {
                    sb.Append(' ');
                }
                else
                {
                    sb.Append(c);
                }
            }

            return WhitespaceRun.Replace(sb.ToString(), " ").Trim();
        }

        private static string Truncate(string? value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value ?? string.Empty;
            }

            return value.Substring(0, maxLength) + "…";
        }
    }
}
