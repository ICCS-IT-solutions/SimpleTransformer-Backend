using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformerBackend.Migrations
{
    /// <inheritdoc />
    public partial class AddTrainingCorpora : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<Guid>(
                name: "TrainingCorpusId",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: true);

            migrationBuilder.CreateTable(
                name: "TrainingCorpora",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Filename = table.Column<string>(type: "TEXT", nullable: false),
                    Filepath = table.Column<string>(type: "TEXT", nullable: false),
                    SourceFileNames = table.Column<string>(type: "TEXT", nullable: false),
                    Format = table.Column<string>(type: "TEXT", nullable: false),
                    OptionsJson = table.Column<string>(type: "TEXT", nullable: false),
                    DocumentsIn = table.Column<int>(type: "INTEGER", nullable: false),
                    DocumentsOut = table.Column<int>(type: "INTEGER", nullable: false),
                    CharsIn = table.Column<long>(type: "INTEGER", nullable: false),
                    CharsOut = table.Column<long>(type: "INTEGER", nullable: false),
                    DuplicatesRemoved = table.Column<int>(type: "INTEGER", nullable: false),
                    FilteredByLength = table.Column<int>(type: "INTEGER", nullable: false),
                    FilteredEmpty = table.Column<int>(type: "INTEGER", nullable: false),
                    FileSize = table.Column<long>(type: "INTEGER", nullable: false),
                    DateCreated = table.Column<DateTime>(type: "TEXT", nullable: false),
                    DateUpdated = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainingCorpora", x => x.EntryId);
                });

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_TrainingCorpusId",
                table: "TrainingJobs",
                column: "TrainingCorpusId");

            migrationBuilder.CreateIndex(
                name: "IX_TrainingCorpora_Name",
                table: "TrainingCorpora",
                column: "Name",
                unique: true);

            migrationBuilder.AddForeignKey(
                name: "FK_TrainingJobs_TrainingCorpora_TrainingCorpusId",
                table: "TrainingJobs",
                column: "TrainingCorpusId",
                principalTable: "TrainingCorpora",
                principalColumn: "EntryId",
                onDelete: ReferentialAction.SetNull);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_TrainingJobs_TrainingCorpora_TrainingCorpusId",
                table: "TrainingJobs");

            migrationBuilder.DropTable(
                name: "TrainingCorpora");

            migrationBuilder.DropIndex(
                name: "IX_TrainingJobs_TrainingCorpusId",
                table: "TrainingJobs");

            migrationBuilder.DropColumn(
                name: "TrainingCorpusId",
                table: "TrainingJobs");
        }
    }
}
