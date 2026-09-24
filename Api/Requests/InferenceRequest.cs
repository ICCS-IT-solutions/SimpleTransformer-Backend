namespace SimpleTransformer.Api.Requests
{
    public class InferenceRequest
    {
        public required string InputText { get; set; }
        public required Guid TransformerModelId { get; set; }
        public Guid? TrainingCheckpointId { get; set; }
        public GenerationParameters GenerationParameters { get; set; } = new GenerationParameters();
    }

    public class GenerationParameters
    {
        public int MaxTokens { get; set; } = 20;
        public float Temperature { get; set; } = 0.8f;
        public float Penalty { get; set; } = 1.2f;
        public int TopK { get; set; } = 10;
        public float TopP { get; set; } = 0.9f;        
    }

    public class LoadVocabularyRequest
    {
        public required string File { get; set; }
    }
    public class CompileVocabularyRequest
    {
        public required List<string> Files { get; set; }
    }    
}