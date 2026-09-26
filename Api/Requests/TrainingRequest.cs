using SimpleTransformer.Api.Endpoints.Services.Extensions;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Requests
{
    /// <summary>
    /// Optional extract + preprocess controls shared by file-based training job
    /// creation and the dry-run preview endpoint. Everything is optional and
    /// defaulted so plain .txt uploads behave exactly as before.
    /// </summary>
    public class CorpusPreprocessRequest
    {
        /// <summary>Auto, Txt, Json or Jsonl. Auto resolves from the file extension.</summary>
        public string? Format { get; set; }
        /// <summary>JSON object field holding the training text (default: text, content, body).</summary>
        public string? TextField { get; set; }
        /// <summary>Optional "{prompt}\n\n{completion}" style template for multi-field records.</summary>
        public string? Template { get; set; }
        public bool NormalizeWhitespace { get; set; } = true;
        public bool StripHtml { get; set; } = false;
        public bool Deduplicate { get; set; } = true;
        public int MinChars { get; set; } = 0;
        /// <summary>Maximum document length; 0 means unlimited.</summary>
        public int MaxChars { get; set; } = 0;

        public CorpusSourceFormat ResolveFormat()
        {
            if (string.IsNullOrWhiteSpace(Format) ||
                string.Equals(Format, "auto", StringComparison.OrdinalIgnoreCase))
            {
                return CorpusSourceFormat.Auto;
            }

            if (Enum.TryParse<CorpusSourceFormat>(Format, ignoreCase: true, out var parsed))
            {
                return parsed;
            }

            throw new ArgumentException(
                $"Unknown corpus format '{Format}'. Expected Auto, Txt, Json or Jsonl.");
        }

        public CorpusPreprocessOptions ToOptions() => new()
        {
            NormalizeWhitespace = NormalizeWhitespace,
            StripHtml = StripHtml,
            Deduplicate = Deduplicate,
            MinChars = MinChars,
            MaxChars = MaxChars
        };
    }

    public class TrainingRequest
    {
        public required string InputText { get; set; }
        //When using a file, store the path here in order to pass it to the training service
        public string? InputFilePath { get; set; } = string.Empty;
        public required Guid TransformerModelId { get; set; }
        public Guid VocabularyId { get; set; }
        public Guid? PreviousCheckpointId { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
    public class TrainingFileRequest : CorpusPreprocessRequest
    {
        //One or more corpus files. They are extracted, cleaned and concatenated
        //server-side into a single training corpus while the original names are
        //kept for tracking. Alternatively, set TrainingCorpusId to train on a
        //saved corpus instead of uploading files.
        public List<IFormFile> TextFiles { get; set; } = new();
        public Guid? TrainingCorpusId { get; set; }
        public required Guid TransformerModelId { get; set; }
        public Guid VocabularyId { get; set; }
        public Guid? PreviousCheckpointId { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }

    /// <summary>Dry-run extract + preprocess over uploads. Creates no job and writes no files.</summary>
    public class CorpusPreviewRequest : CorpusPreprocessRequest
    {
        public List<IFormFile> TextFiles { get; set; } = new();
    }
}