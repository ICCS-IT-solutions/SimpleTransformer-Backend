using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace SimpleTransformer.AppDb
{
    public class TrainingCheckpointEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public Guid TransformerModelId { get; set; }
        public TransformerModelEntry? TransformerModel { get; set; }
        public required string Filename { get; set; }
        public required string Filepath { get; set; }
        public string? Sha256 { get; set; }
        public long FileSize { get; set; }
        public int Epoch { get; set; }
        public float Loss { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public Guid? TrainingRunId { get; set; }
    }

    /*
    export type TrainingCheckpointEntry = {
        entryId: string;
        filename: string;
        filepath: string;
        sha256?: string;
        filesize: number;
        epoch: number;
        loss: number;
        dateCreated: Date;
        trainingRunId?: string;   
    }
    */
}
