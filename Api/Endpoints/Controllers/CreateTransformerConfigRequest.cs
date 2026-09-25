using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class CreateTransformerConfigRequest
    {
        public string? DisplayName { get; set; }
        public required string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public required TransformerConfig Config { get; set; }
    }

    public class UpdateTransformerConfigRequest
    {
        public required Guid ConfigId { get; set; }
        public string? DisplayName { get; set; }
        public required string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public required TransformerConfig Config { get; set; }
    }
}