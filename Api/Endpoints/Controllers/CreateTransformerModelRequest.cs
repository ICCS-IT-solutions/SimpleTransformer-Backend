using SimpleTransformer.AppDb;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    public class CreateTransformerModelRequest
    {
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required TransformerConfigEntry TransformerConfig { get; set; }
        public required TrainingConfigEntry TrainingConfig { get; set; }

        /// <summary>
        /// Acceleration backend name (BackendSelector.BackendType). "Auto" by default.
        /// </summary>
        public string AccelerationBackend { get; set; } = "Auto";

        /// <summary>
        /// True to build a quantised LoRA model (frozen 4-bit base weights plus
        /// trainable adapters), false for a "raw" model where every weight is a
        /// dense fp32 trainable parameter. Defaults to true. This is fixed at
        /// creation and ignored on update: QLoRA and raw models have different
        /// trainable parameter sets, so their checkpoints are not interchangeable.
        /// </summary>
        public bool UseQLora { get; set; } = true;

        /// <summary>
        /// Vocabulary the model should tokenise with. Null leaves the model
        /// unpinned: it runs on the vocabulary the server loaded at startup until a
        /// training job pins one. When set, creation is refused if the artifact is
        /// missing or its token count disagrees with the transformer config, and the
        /// value is ignored on update because a vocabulary change would orphan every
        /// saved checkpoint (they record VocabSize).
        /// </summary>
        public Guid? VocabularyId { get; set; }
    }

    /*
    export type CreateTransformerModelRequest = {
        name: string;
        description: string;
        transformerConfig: TransformerConfigEntry;
        trainingConfig: TrainingConfigEntry;
        accelerationBackend: string;
        useQLora: boolean;
    }
    */
}