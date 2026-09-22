using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Requests
{
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
    public class TrainingFileRequest
    {
        public required IFormFile TextFile { get; set; }
        public required Guid TransformerModelId { get; set; }
        public Guid VocabularyId { get; set; }
        public Guid? PreviousCheckpointId { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
}