using SimpleTransformer.AppDb;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Responses
{
    public class TransformerModelResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; } = InteractionStatus.Success;
        public TransformerModelEntry? Model { get; set; }
        public List<TransformerModelEntry>? Models { get; set; }

        /// <summary>
        /// The acceleration backend actually resolved at runtime by the loaded model
        /// (e.g. "GpuVulkan (AMD Radeon RX 5700 XT)"). Only populated by the load
        /// and active-model endpoints.
        /// </summary>
        public string? ActiveBackend { get; set; }
    }
}