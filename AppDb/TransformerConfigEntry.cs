using System.ComponentModel.DataAnnotations;
using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    public class TransformerConfigEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public string? DisplayName { get; set; }
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required TransformerConfig Config { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
    }

    public class TransformerConfigPresetEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public string? DisplayName { get; set; }
        public required string Name { get; set; }
        public required Guid TransformerConfigId { get; set; }
        public required TransformerConfig Config { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
    }
}