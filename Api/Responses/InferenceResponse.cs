namespace SimpleTransformer.Api.Responses
{
    //What comes back from the model.
    public class InferenceResponse
    {
        public InteractionStatus Status { get; set; }
        public string OutputText { get; set; } = string.Empty;

        //Provenance of the weights that produced OutputText, so callers can
        //verify a selected checkpoint was actually used rather than trusting
        //silent in-memory state.
        public string WeightsSource { get; set; } = string.Empty;
        public string? CheckpointFilename { get; set; }
        public int? CheckpointEpoch { get; set; }
        public float? CheckpointLoss { get; set; }
    }
}