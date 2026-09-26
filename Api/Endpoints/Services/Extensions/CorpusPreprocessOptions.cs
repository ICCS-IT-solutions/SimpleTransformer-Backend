namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Cleaning policy for extracted corpus documents. All options are optional
    /// and default to the conservative set: whitespace normalization and exact
    /// deduplication on, HTML stripping off, no length limits.
    /// </summary>
    public sealed class CorpusPreprocessOptions
    {
        public bool NormalizeWhitespace { get; set; } = true;
        public bool StripHtml { get; set; } = false;
        public bool Deduplicate { get; set; } = true;
        public int MinChars { get; set; } = 0;
        /// <summary>Maximum document length; 0 means unlimited.</summary>
        public int MaxChars { get; set; } = 0;
        /// <summary>How many before/after samples to keep in the report.</summary>
        public int MaxSamples { get; set; } = 5;
        /// <summary>Truncation length applied to report samples only.</summary>
        public int SampleTruncateLength { get; set; } = 500;

        public static CorpusPreprocessOptions Defaults => new();
    }

    public sealed class CorpusFileStats
    {
        public string FileName { get; set; } = string.Empty;
        public string ResolvedFormat { get; set; } = string.Empty;
        public int DocumentsIn { get; set; }
        public int DocumentsOut { get; set; }
    }

    public sealed class CorpusSampleDoc
    {
        public string FileName { get; set; } = string.Empty;
        public string Original { get; set; } = string.Empty;
        public string Cleaned { get; set; } = string.Empty;
    }

    /// <summary>
    /// Audit trail for an extract + preprocess run. Returned by the preview
    /// endpoint and written beside the merged corpus as preprocess-report.json.
    /// </summary>
    public sealed class CorpusPreprocessReport
    {
        public int DocumentsIn { get; set; }
        public int DocumentsOut { get; set; }
        public long CharsIn { get; set; }
        public long CharsOut { get; set; }
        public int DuplicatesRemoved { get; set; }
        public int FilteredByLength { get; set; }
        public int FilteredEmpty { get; set; }
        public List<CorpusFileStats> Files { get; set; } = new();
        public List<string> Warnings { get; set; } = new();
        public List<CorpusSampleDoc> Samples { get; set; } = new();
    }
}
