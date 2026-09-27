namespace SimpleTransformer.Model.Tokenizer
{
    public class VocabularyCompilationResult
    {
        public Vocabulary Vocabulary { get; }
        
        /// <summary>
        /// Optional merge rules for subword algorithms like BPE. 
        /// Null for word-level compilers.
        /// </summary>
        public IReadOnlyList<(string First, string Second)>? Merges { get; }

        /// <summary>Distinct types observed in the source text.</summary>
        public long TypesSeen { get; }

        /// <summary>Total running token occurrences observed.</summary>
        public long TokenOccurrences { get; }

        /// <summary>Occurrences covered by the kept vocabulary.</summary>
        public long KeptOccurrences { get; }

        /// <summary>
        /// Fraction of running tokens covered by the kept vocabulary.
        /// Meaningful for frequency-truncated compilers (word-level);
        /// 1.0 when occurrence stats were not collected.
        /// </summary>
        public double Coverage => TokenOccurrences > 0
            ? (double)KeptOccurrences / TokenOccurrences
            : 1.0;

        public VocabularyCompilationResult(
            Vocabulary vocabulary, 
            IReadOnlyList<(string First, string Second)>? merges = null,
            long typesSeen = 0,
            long tokenOccurrences = 0,
            long keptOccurrences = 0)
        {
            Vocabulary = vocabulary;
            Merges = merges;
            TypesSeen = typesSeen;
            TokenOccurrences = tokenOccurrences;
            KeptOccurrences = keptOccurrences;
        }
    }
}