namespace SimpleTransformer.Api.Responses
{
    public class TrainingProgressResponse
    {
        public string JobId { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        //The model this job targets (id + name) so the frontend can show and
        //resolve the exact model definition the job belongs to.
        public string TransformerModelId { get; set; } = string.Empty;
        public string TransformerModelName { get; set; } = string.Empty;
        public TrainingJobStatus Status { get; set; }
        public int CurrentEpoch { get; set; }
        public int TotalEpochs { get; set; }
        public float CurrentLoss { get; set; }
        public int CurrentBatch { get; set; }
        public int TotalBatches { get; set; }
        public int NumSubBatches { get; set; }
        public int CurrentSubBatch { get; set; }
        public DateTime? StartedAt { get; set; }
        public DateTime? CompletedAt { get; set; }
        public DateTime LastUpdatedAt { get; set; }
        public float Loss { get; set; }
        public string? Checkpoint { get; set; }
        public string Error { get; set; } = string.Empty;
    }
}