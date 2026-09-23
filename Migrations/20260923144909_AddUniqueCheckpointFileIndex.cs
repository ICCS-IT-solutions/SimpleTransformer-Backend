using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddUniqueCheckpointFileIndex : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            //One record per checkpoint file is now the invariant. Older rows were
            //inserted once per training batch for the same temp file, so collapse
            //each (Filepath, Filename) group down to its newest row first, then
            //repoint any training jobs that referenced a duplicate.
            migrationBuilder.Sql(
                """
                UPDATE TrainingJobs
                SET TrainingCheckpointId = (
                    SELECT surviving.EntryId
                    FROM TrainingCheckpoints AS current
                    JOIN TrainingCheckpoints AS surviving
                      ON surviving.Filepath = current.Filepath
                     AND surviving.Filename = current.Filename
                    WHERE current.EntryId = TrainingJobs.TrainingCheckpointId
                      AND surviving.rowid = (
                          SELECT MAX(candidate.rowid)
                          FROM TrainingCheckpoints AS candidate
                          WHERE candidate.Filepath = current.Filepath
                            AND candidate.Filename = current.Filename
                      )
                )
                WHERE TrainingCheckpointId IS NOT NULL;
                """);

            migrationBuilder.Sql(
                """
                UPDATE TrainingJobs
                SET PreviousCheckpointId = (
                    SELECT surviving.EntryId
                    FROM TrainingCheckpoints AS current
                    JOIN TrainingCheckpoints AS surviving
                      ON surviving.Filepath = current.Filepath
                     AND surviving.Filename = current.Filename
                    WHERE current.EntryId = TrainingJobs.PreviousCheckpointId
                      AND surviving.rowid = (
                          SELECT MAX(candidate.rowid)
                          FROM TrainingCheckpoints AS candidate
                          WHERE candidate.Filepath = current.Filepath
                            AND candidate.Filename = current.Filename
                      )
                )
                WHERE PreviousCheckpointId IS NOT NULL;
                """);

            migrationBuilder.Sql(
                """
                DELETE FROM TrainingCheckpoints
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM TrainingCheckpoints
                    GROUP BY Filepath, Filename
                );
                """);

            migrationBuilder.CreateIndex(
                name: "IX_TrainingCheckpoints_Filepath_Filename",
                table: "TrainingCheckpoints",
                columns: new[] { "Filepath", "Filename" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_TrainingCheckpoints_Filepath_Filename",
                table: "TrainingCheckpoints");
        }
    }
}
