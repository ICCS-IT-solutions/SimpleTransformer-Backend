using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Responses
{
    public class VocabularyLoaderResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; } = InteractionStatus.Success;
    }

    public class VocabularyCompilationResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; } = InteractionStatus.Success;
        public Vocabulary? Vocabulary { get; set; } = null;
        public int RequestedVocabSize { get; set; }
        public int ActualVocabSize { get; set; }
        public string TokenizerType { get; set; } = string.Empty;
        /// <summary>Distinct types observed in the source text.</summary>
        public long TypesSeen { get; set; }
        /// <summary>
        /// Fraction of running tokens covered by the kept vocabulary.
        /// Meaningful for frequency-truncated (word-level) compiles.
        /// </summary>
        public double Coverage { get; set; }
        /// <summary>First tokens by id (specials, then most frequent).</summary>
        public List<string> SampleTokens { get; set; } = new();
    }
}