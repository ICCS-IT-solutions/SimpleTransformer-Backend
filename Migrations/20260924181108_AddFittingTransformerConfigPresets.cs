using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddFittingTransformerConfigPresets : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            //Presets sized for an 8 GiB VRAM / 32 GiB RAM machine. The figures in
            //each description are measured for the current architecture, not
            //guesses: VRAM holds the dequantized fp32 base weights of every
            //QLoRA linear layer, and host RAM holds the trainable tensors, the
            //AdamW state and the cached attention scores (H x Seq x Seq per
            //layer, which is what grows with sequence length).
            //
            //Inserts are guarded by name so re-running this migration is a no-op
            //rather than a duplicate-key failure.
            migrationBuilder.Sql(@"
            INSERT INTO ""TransformerConfigs"" (""EntryId"", ""Name"", ""Description"", ""Config"", ""DateCreated"")
            SELECT '11111111-1111-4111-8111-111111111101', 'E1024-L12-H16-F4096-S1024',
                'E1024 L12 H16 F4096 Seq1024 - 13.1M trainable (QLoRA), 172.5M raw, ~615 MiB VRAM, ~6.5 GiB RAM at batch 8. Best quality-to-memory balance.',
                '{""VocabSize"":10000,""EmbeddingSize"":1024,""NumLayers"":12,""HiddenSize"":1024,""NumHeads"":16,""FeedForwardSize"":4096,""MaxSequenceLength"":1024}',
                CURRENT_TIMESTAMP
            WHERE NOT EXISTS (SELECT 1 FROM ""TransformerConfigs"" WHERE ""Name"" = 'E1024-L12-H16-F4096-S1024');

            INSERT INTO ""TransformerConfigs"" (""EntryId"", ""Name"", ""Description"", ""Config"", ""DateCreated"")
            SELECT '11111111-1111-4111-8111-111111111102', 'E768-L12-H8-F1536-S2048',
                'E768 L12 H8 F1536 Seq2048 - 10.3M trainable (QLoRA), 73.6M raw, ~245 MiB VRAM, ~12.5 GiB RAM at batch 8. Long context; lower BatchSize to save RAM.',
                '{""VocabSize"":10000,""EmbeddingSize"":768,""NumLayers"":12,""HiddenSize"":768,""NumHeads"":8,""FeedForwardSize"":1536,""MaxSequenceLength"":2048}',
                CURRENT_TIMESTAMP
            WHERE NOT EXISTS (SELECT 1 FROM ""TransformerConfigs"" WHERE ""Name"" = 'E768-L12-H8-F1536-S2048');

            INSERT INTO ""TransformerConfigs"" (""EntryId"", ""Name"", ""Description"", ""Config"", ""DateCreated"")
            SELECT '11111111-1111-4111-8111-111111111103', 'E2048-L24-H16-F8192-S1024',
                'E2048 L24 H16 F8192 Seq1024 - 29.9M trainable (QLoRA), 1251M raw, ~4686 MiB VRAM, ~13.5 GiB RAM at batch 8. Large; uses two thirds of the VRAM budget.',
                '{""VocabSize"":10000,""EmbeddingSize"":2048,""NumLayers"":24,""HiddenSize"":2048,""NumHeads"":16,""FeedForwardSize"":8192,""MaxSequenceLength"":1024}',
                CURRENT_TIMESTAMP
            WHERE NOT EXISTS (SELECT 1 FROM ""TransformerConfigs"" WHERE ""Name"" = 'E2048-L24-H16-F8192-S1024');

            INSERT INTO ""TransformerConfigs"" (""EntryId"", ""Name"", ""Description"", ""Config"", ""DateCreated"")
            SELECT '11111111-1111-4111-8111-111111111104', 'E256-L4-H4-F1024-S256',
                'E256 L4 H4 F1024 Seq256 - 2.8M trainable (QLoRA), 8.3M raw, ~22 MiB VRAM, ~74 MiB RAM at batch 8. Smallest useful config, good for smoke tests.',
                '{""VocabSize"":10000,""EmbeddingSize"":256,""NumLayers"":4,""HiddenSize"":256,""NumHeads"":4,""FeedForwardSize"":1024,""MaxSequenceLength"":256}',
                CURRENT_TIMESTAMP
            WHERE NOT EXISTS (SELECT 1 FROM ""TransformerConfigs"" WHERE ""Name"" = 'E256-L4-H4-F1024-S256');
            ");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.Sql(@"
            DELETE FROM ""TransformerConfigs""
            WHERE ""EntryId"" IN (
                '11111111-1111-4111-8111-111111111101',
                '11111111-1111-4111-8111-111111111102',
                '11111111-1111-4111-8111-111111111103',
                '11111111-1111-4111-8111-111111111104');
            ");
        }


    }
}
