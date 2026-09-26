using System.ComponentModel.DataAnnotations;

namespace SimpleTransformer.AppDb
{
    /// <summary>
    /// A named, reusable training corpus: cleaned plain-text documents prepared
    /// from uploaded sources. Corpora are immutable snapshots — changing options
    /// means creating a new corpus. The preprocess options used are stored as
    /// JSON so any corpus can be recreated or audited.
    /// </summary>
    public class TrainingCorpusEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public required string Filename { get; set; }
        public required string Filepath { get; set; }
        //Original upload filenames, comma separated, for provenance.
        public string SourceFileNames { get; set; } = string.Empty;
        //Source format requested at creation (Auto, Txt, Json, Jsonl).
        public string Format { get; set; } = "Auto";
        //Serialized CorpusPreprocessOptions used to build this corpus.
        public string OptionsJson { get; set; } = string.Empty;
        //Denormalized stats so listing is a single query.
        public int DocumentsIn { get; set; }
        public int DocumentsOut { get; set; }
        public long CharsIn { get; set; }
        public long CharsOut { get; set; }
        public int DuplicatesRemoved { get; set; }
        public int FilteredByLength { get; set; }
        public int FilteredEmpty { get; set; }
        public long FileSize { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public DateTime DateUpdated { get; set; }
    }

    /*
    //Create the training corpus entry class as a typescript data class
    export type TrainingCorpusEntry = {
        entryId: string;
        name: string;
        filename: string;
        filepath: string;
        sourceFileNames: string;
        format: string;
        optionsJson: string;
        documentsIn: number;
        documentsOut: number;
        charsIn: number;
        charsOut: number;
        duplicatesRemoved: number;
        filteredByLength: number;
        filteredEmpty: number;
        fileSize: number;
        dateCreated: Date;
        dateUpdated: Date;
    }
    */
}