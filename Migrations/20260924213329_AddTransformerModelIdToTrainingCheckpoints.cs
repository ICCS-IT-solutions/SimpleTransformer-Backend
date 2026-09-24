using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddTransformerModelIdToTrainingCheckpoints : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<Guid>(
                name: "TransformerModelId",
                table: "TrainingCheckpoints",
                type: "TEXT",
                nullable: false,
                defaultValue: new Guid("00000000-0000-0000-0000-000000000000"));

            // Backfill existing checkpoints with matching TransformerModel based on Filepath
            migrationBuilder.Sql(@"
                UPDATE TrainingCheckpoints
                SET TransformerModelId = (
                    SELECT m.EntryId
                    FROM TransformerModels m
                    WHERE TrainingCheckpoints.Filepath LIKE '%' || m.Name || '%'
                    LIMIT 1
                )
                WHERE TransformerModelId = '00000000-0000-0000-0000-000000000000'
                   OR TransformerModelId = '00000000-0000-0000-0000-000000000000';
            ");

            migrationBuilder.CreateIndex(
                name: "IX_TrainingCheckpoints_TransformerModelId",
                table: "TrainingCheckpoints",
                column: "TransformerModelId");

            migrationBuilder.AddForeignKey(
                name: "FK_TrainingCheckpoints_TransformerModels_TransformerModelId",
                table: "TrainingCheckpoints",
                column: "TransformerModelId",
                principalTable: "TransformerModels",
                principalColumn: "EntryId",
                onDelete: ReferentialAction.Cascade);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_TrainingCheckpoints_TransformerModels_TransformerModelId",
                table: "TrainingCheckpoints");

            migrationBuilder.DropIndex(
                name: "IX_TrainingCheckpoints_TransformerModelId",
                table: "TrainingCheckpoints");

            migrationBuilder.DropColumn(
                name: "TransformerModelId",
                table: "TrainingCheckpoints");
        }
    }
}
