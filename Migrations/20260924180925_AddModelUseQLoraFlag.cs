using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddModelUseQLoraFlag : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "UseQLora",
                table: "TransformerModels",
                type: "INTEGER",
                nullable: false,
                //Every model that existed before this column was created was a
                //QLoRA model (raw training was not reachable), so existing rows
                //must default to true rather than EF's false.
                defaultValue: true);

            // -----------------------------------------------------------------
            // Rename configurations and models to describe the architecture they
            // actually use. The old names advertised parameter counts (73M/29M/85M)
            // that were roughly 8x too high: these are QLoRA models, where the
            // figure counted frozen 4-bit base weights while only LoRA adapters
            // and embeddings are trainable. Architecture-based names cannot drift.
            // -----------------------------------------------------------------

            migrationBuilder.Sql(@"
            UPDATE ""TransformerConfigs""
            SET ""Name"" = 'E768-L12-H8-F1536-S768',
                ""Description"" = 'E768 L12 H8 F1536 Seq768 - 9.3M trainable (QLoRA), 72.6M raw, ~245 MiB VRAM, ~1.9 GiB RAM at batch 8'
            WHERE ""Name"" = 'Medium (73M)';

            UPDATE ""TransformerConfigs""
            SET ""Name"" = 'E512-L6-H8-F1024-S512',
                ""Description"" = 'E512 L6 H8 F1024 Seq512 - 5.7M trainable (QLoRA), 23.1M raw, ~56 MiB VRAM, ~0.5 GiB RAM at batch 8'
            WHERE ""Name"" = 'Base (29M)';

            UPDATE ""TransformerConfigs""
            SET ""Name"" = 'E768-L12-H8-F3072-S2048',
                ""Description"" = 'E768 L12 H8 F3072 Seq2048 - 10.6M trainable (QLoRA), 101.9M raw, ~299 MiB VRAM, ~12.3 GiB RAM at batch 8'
            WHERE ""Name"" = 'Medium (85M)';
            ");

            //Models follow their config. The checkpoint directory on disk is keyed
            //by the model name (checkpoints/{modelName}/), so the folders have to be
            //renamed to match or every existing checkpoint becomes unreachable.
            migrationBuilder.Sql(@"
            UPDATE ""TransformerModels""
            SET ""Name"" = 'E768-L12-H8-F1536-S768',
                ""Description"" = 'QLoRA model on the E768-L12-H8-F1536-S768 config. 9.3M trainable params, ~245 MiB VRAM.',
                ""DateUpdated"" = CURRENT_TIMESTAMP
            WHERE ""Name"" = 'Transformer-73M-VK';

            UPDATE ""TransformerModels""
            SET ""Name"" = 'E512-L6-H8-F1024-S512',
                ""Description"" = 'QLoRA model on the E512-L6-H8-F1024-S512 config. 5.7M trainable params, ~56 MiB VRAM.',
                ""DateUpdated"" = CURRENT_TIMESTAMP
            WHERE ""Name"" = 'Transformer-29M';

            UPDATE ""TransformerModels""
            SET ""Name"" = 'E768-L12-H8-F3072-S2048',
                ""Description"" = 'QLoRA model on the E768-L12-H8-F3072-S2048 config. 10.6M trainable params, ~299 MiB VRAM.',
                ""DateUpdated"" = CURRENT_TIMESTAMP
            WHERE ""Name"" = 'Transformer-73M';

            UPDATE ""TransformerModels""
            SET ""Name"" = 'E768-L12-H8-F3072-S2048-LONGCTX',
                ""Description"" = 'QLoRA model, long context variant. 10.6M trainable params, ~299 MiB VRAM. Heavy RAM use at Seq2048.',
                ""DateUpdated"" = CURRENT_TIMESTAMP
            WHERE ""Name"" = 'Transformer-85M';

            --Checkpoint rows point at checkpoints/{modelName}/. Retarget them so
            --already-saved checkpoints stay reachable under the new names.
            UPDATE ""TrainingCheckpoints""
            SET ""Filepath"" = 'checkpoints/E768-L12-H8-F1536-S768/'
            WHERE ""Filepath"" = 'checkpoints/Transformer-73M-VK/';

            UPDATE ""TrainingCheckpoints""
            SET ""Filepath"" = 'checkpoints/E512-L6-H8-F1024-S512/'
            WHERE ""Filepath"" = 'checkpoints/Transformer-29M/';

            UPDATE ""TrainingCheckpoints""
            SET ""Filepath"" = 'checkpoints/E768-L12-H8-F3072-S2048/'
            WHERE ""Filepath"" = 'checkpoints/Transformer-73M/';

            UPDATE ""TrainingCheckpoints""
            SET ""Filepath"" = 'checkpoints/E768-L12-H8-F3072-S2048-LONGCTX/'
            WHERE ""Filepath"" = 'checkpoints/Transformer-85M/';

            --Orphan rows: checkpoints recorded at the checkpoints/ root belong to
            --no model folder and can never be found by the resume lookup, which
            --matches checkpoints/{modelName}/ exactly.
            DELETE FROM ""TrainingCheckpoints""
            WHERE ""Filepath"" = 'checkpoints/' OR ""Filepath"" = 'checkpoints';
            ");
        }
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "UseQLora",
                table: "TransformerModels");
        }
    }
}
