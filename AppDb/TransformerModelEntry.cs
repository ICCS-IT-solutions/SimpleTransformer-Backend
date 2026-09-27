using System.ComponentModel.DataAnnotations;
using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    public class TransformerModelEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public required string Description { get; set; }
        public bool IsLoaded { get; set; }
        public required Guid TransformerConfigId { get; set; }
        public required Guid TrainingConfigId { get; set; }

        /// <summary>
        /// Vocabulary this model tokenises with. Nullable so models created before
        /// the reference existed keep working (they fall back to the vocabulary the
        /// server loaded at startup).
        /// <para>
        /// The token id space is decided by the vocabulary, and checkpoints record
        /// <see cref="Model.TransformerConfig.VocabSize"/>, so this is fixed in
        /// practice: <see cref="TransformerModelService"/> ignores it on update, and
        /// a training job pins a null value the first time it runs. Training with a
        /// different vocabulary requires a new model.
        /// </para>
        /// </summary>
        public Guid? VocabularyId { get; set; }

        /// <summary>
        /// Convenience navigation for responses. Only populated where the query
        /// explicitly includes it.
        /// </summary>
        public VocabularyEntry? Vocabulary { get; set; }

        /// <summary>
        /// Acceleration backend used by this model. One of the
        /// <see cref="AccelerationEngine.BackendSelector.BackendType"/> names.
        /// Defaults to "Auto" (best available at load time).
        /// </summary>
        public string AccelerationBackend { get; set; } = "Auto";

        /// <summary>
        /// True when the model uses quantised LoRA (frozen 4-bit base weights plus
        /// trainable LoRA adapters); false for a "raw" model where every weight is
        /// a dense fp32 trainable parameter.
        /// <para>
        /// This is a structural property of the model, fixed at creation: a QLoRA
        /// model and a raw model expose completely different trainable parameter
        /// sets, so a checkpoint written by one can never be loaded into the
        /// other. Changing this on an existing model would orphan its checkpoints,
        /// so the update path deliberately ignores it.
        /// </para>
        /// </summary>
        public bool UseQLora { get; set; } = true;

        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public DateTime? DateUpdated { get; set; }
    }

    /*
    export type TransformerModelEntry = {
        entryId: string;
        name: string;
        description: string;
        isLoaded: boolean;
        transformerConfigId: string;
        trainingConfigId: string;
        accelerationBackend: string;
        dateCreated: Date;
        dateUpdated?: Date;
    }
    */
}

