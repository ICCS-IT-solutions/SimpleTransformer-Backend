using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.AppDb
{
    public class VocabularyEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        //Infer from the tokenizer used to create this, store as a json string in the db
        [JsonConverter(typeof(JsonStringEnumConverter))]
        public TokenizerType TokenizerType { get; set; } 
        public DateTime DateCreated { get; set; }
        public int NumTokens { get; set; }
        public required string Filename { get; set; }
        public required string Filepath { get; set; }

        /// <summary>
        /// Names of the source files the vocabulary was compiled from, so the
        /// compile provenance survives a restart. Empty for entries compiled
        /// before this was recorded.
        /// </summary>
        public string SourceFileNames { get; set; } = string.Empty;

        /// <summary>
        /// Size requested at compile time (including special tokens). 0 when
        /// unknown. <see cref="NumTokens"/> is what was actually produced, which
        /// is lower when the source had fewer distinct types than requested.
        /// </summary>
        public int RequestedSize { get; set; }

        /// <summary>
        /// Distinct types observed in the source text. 0 when the alphabet did not
        /// collect statistics (BPE and SentencePiece cover every character, so
        /// they report nothing here).
        /// </summary>
        public long TypesSeen { get; set; }

        /// <summary>
        /// Fraction (0-1) of running tokens covered by the kept vocabulary. Only
        /// meaningful when <see cref="TypesSeen"/> is greater than 0; 1.0 means
        /// "not measured" rather than "perfect coverage".
        /// </summary>
        public double Coverage { get; set; } = 1.0;
    }
}