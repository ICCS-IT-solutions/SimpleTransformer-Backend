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
        /// Acceleration backend used by this model. One of the
        /// <see cref="AccelerationEngine.BackendSelector.BackendType"/> names.
        /// Defaults to "Auto" (best available at load time).
        /// </summary>
        public string AccelerationBackend { get; set; } = "Auto";

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

