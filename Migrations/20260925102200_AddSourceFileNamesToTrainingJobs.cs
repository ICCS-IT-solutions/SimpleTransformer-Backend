using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddSourceFileNamesToTrainingJobs : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "SourceFileNames",
                table: "TrainingJobs",
                type: "TEXT",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "SourceFileNames",
                table: "TrainingJobs");
        }
    }
}
