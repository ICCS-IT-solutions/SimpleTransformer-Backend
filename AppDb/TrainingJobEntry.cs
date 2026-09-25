using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using SimpleTransformer.Api.Responses;

namespace SimpleTransformer.AppDb
{
    public class TrainingJobEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public string Message { get; set; } = string.Empty;
        //The model this job trains. Kept alongside the config ids so a job can be
        //resolved back to the exact model definition it targeted, not just the
        //configs that were in effect when it was created.
        public Guid TransformerModelId { get; set; }
        //Associated model, training and vocabulary
        public Guid TransformerConfigId { get; set; }
        public Guid TrainingConfigId { get; set; }
        public Guid VocabularyId { get; set; }

        //Associated model, training and vocabulary
        public TransformerModelEntry? TransformerModel { get; set; }
        public TransformerConfigEntry? TransformerConfig { get; set; }
        public TrainingConfigEntry? TrainingConfig { get; set; }
        public VocabularyEntry? Vocabulary { get; set; }

        //Training sources
        public string? InputText { get; set; } //For live training
        public string InputFilePath { get; set; } = string.Empty; //For batch training using files
        //Original filenames uploaded for file based jobs, comma separated so a
        //job can always be traced back to the exact corpus it was trained on.
        public string SourceFileNames { get; set; } = string.Empty;

        // Optional checkpoint to resume from
        public Guid? PreviousCheckpointId { get; set; }

        //Status
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public DateTime DateUpdated { get; set; }
        public DateTime? DateStarted { get; set; }
        public DateTime? DateCompleted { get; set; }

        public TrainingJobStatus Status { get; set; } = TrainingJobStatus.Pending;
        // Epoch progress
        public int CurrentEpoch { get; set; }
        public int EpochsCompleted { get; set; }
        public int TotalEpochs { get; set; }

        // Batch progress
        public int CurrentBatch { get; set; }
        public int BatchesCompleted { get; set; }
        public int TotalBatches { get; set; }

        // Sub-batch progress
        public int CurrentSubBatch { get; set; }
        public int SubBatchesCompleted { get; set; }
        public int TotalSubBatches { get; set; }

        // Metrics
        public float CurrentLoss { get; set; }

        // Latest checkpoint
        public Guid? TrainingCheckpointId { get; set; }
        public string CheckpointFilename { get; set; } = string.Empty;
        public TrainingCheckpointEntry? TrainingCheckpoint { get; set; }
        //Other properties and statuses
        public string Error { get; set; } = string.Empty;
    }
    /*
    //Create the training job entry class as a typescript data class
    export type TrainingJobEntry = {
        entryId: string;
        name: string;
        message: string;
        transformerConfigId: string;
        transformerModelId: string;
        trainingConfigId: string;
        vocabularyId: string;
        inputText?: string;
        inputFilePath: string;
        previousCheckpointId?: string;
        previousCheckpoint?: string;
        dateCreated: Date;
        dateUpdated: Date;
        dateStarted?: Date;
        dateCompleted?: Date;
        status: TrainingJobStatus;
        currentEpoch: number;
        epochsCompleted: number;
        totalEpochs: number;
        currentBatch: number;
        batchesCompleted: number;
        totalBatches: number;
        currentSubBatch: number;
        subBatchesCompleted: number;
        totalSubBatches: number;
        currentLoss: number;
        trainingCheckpointId?: string;
        checkpointFilename: string;
        trainingCheckpoint?: TrainingCheckpointEntry;
        error: string;
    }
    */
}