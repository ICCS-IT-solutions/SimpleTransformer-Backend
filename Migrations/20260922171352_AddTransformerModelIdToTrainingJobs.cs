using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddTransformerModelIdToTrainingJobs : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<Guid>(
                name: "TransformerModelId",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: false,
                defaultValue: new Guid("00000000-0000-0000-0000-000000000000"));

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_TransformerModelId",
                table: "TrainingJobs",
                column: "TransformerModelId");

            migrationBuilder.AddForeignKey(
                name: "FK_TrainingJobs_TransformerModels_TransformerModelId",
                table: "TrainingJobs",
                column: "TransformerModelId",
                principalTable: "TransformerModels",
                principalColumn: "EntryId",
                onDelete: ReferentialAction.Restrict);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_TrainingJobs_TransformerModels_TransformerModelId",
                table: "TrainingJobs");

            migrationBuilder.DropIndex(
                name: "IX_TrainingJobs_TransformerModelId",
                table: "TrainingJobs");

            migrationBuilder.DropColumn(
                name: "TransformerModelId",
                table: "TrainingJobs");
        }
    }
}
