using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformerBackend.Migrations
{
    /// <inheritdoc />
    public partial class AddVocabularyModelLink : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<double>(
                name: "Coverage",
                table: "Vocabularies",
                type: "REAL",
                nullable: false,
                defaultValue: 0.0);

            migrationBuilder.AddColumn<int>(
                name: "RequestedSize",
                table: "Vocabularies",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "SourceFileNames",
                table: "Vocabularies",
                type: "TEXT",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<long>(
                name: "TypesSeen",
                table: "Vocabularies",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0L);

            migrationBuilder.AddColumn<Guid>(
                name: "VocabularyId",
                table: "TransformerModels",
                type: "TEXT",
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_TransformerModels_VocabularyId",
                table: "TransformerModels",
                column: "VocabularyId");

            migrationBuilder.AddForeignKey(
                name: "FK_TransformerModels_Vocabularies_VocabularyId",
                table: "TransformerModels",
                column: "VocabularyId",
                principalTable: "Vocabularies",
                principalColumn: "EntryId",
                onDelete: ReferentialAction.Restrict);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_TransformerModels_Vocabularies_VocabularyId",
                table: "TransformerModels");

            migrationBuilder.DropIndex(
                name: "IX_TransformerModels_VocabularyId",
                table: "TransformerModels");

            migrationBuilder.DropColumn(
                name: "Coverage",
                table: "Vocabularies");

            migrationBuilder.DropColumn(
                name: "RequestedSize",
                table: "Vocabularies");

            migrationBuilder.DropColumn(
                name: "SourceFileNames",
                table: "Vocabularies");

            migrationBuilder.DropColumn(
                name: "TypesSeen",
                table: "Vocabularies");

            migrationBuilder.DropColumn(
                name: "VocabularyId",
                table: "TransformerModels");
        }
    }
}
