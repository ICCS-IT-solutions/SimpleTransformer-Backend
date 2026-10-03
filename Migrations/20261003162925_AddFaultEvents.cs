using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformerBackend.Migrations
{
    /// <inheritdoc />
    public partial class AddFaultEvents : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "FaultEvents",
                columns: table => new
                {
                    EntryId = table.Column<Guid>(type: "TEXT", nullable: false),
                    FaultType = table.Column<string>(type: "TEXT", nullable: false),
                    Component = table.Column<string>(type: "TEXT", nullable: false),
                    Message = table.Column<string>(type: "TEXT", nullable: false),
                    StackTrace = table.Column<string>(type: "TEXT", nullable: true),
                    JobId = table.Column<string>(type: "TEXT", nullable: true),
                    ModelId = table.Column<string>(type: "TEXT", nullable: true),
                    Fingerprint = table.Column<string>(type: "TEXT", nullable: false),
                    OccurrenceCount = table.Column<int>(type: "INTEGER", nullable: false),
                    FirstSeenAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    LastSeenAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_FaultEvents", x => x.EntryId);
                });

            migrationBuilder.CreateIndex(
                name: "IX_FaultEvents_Fingerprint",
                table: "FaultEvents",
                column: "Fingerprint");

            migrationBuilder.CreateIndex(
                name: "IX_FaultEvents_LastSeenAt",
                table: "FaultEvents",
                column: "LastSeenAt");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "FaultEvents");
        }
    }
}
