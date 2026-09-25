using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace SimpleTransformer.Migrations
{
    /// <inheritdoc />
    public partial class AddDisplayNameToTransformerConfigs : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "DisplayName",
                table: "TransformerConfigs",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DisplayName",
                table: "TransformerConfigPresets",
                type: "TEXT",
                nullable: true);

            migrationBuilder.Sql(@"
                UPDATE TransformerConfigs
                SET DisplayName = CASE Name
                    WHEN 'E256-L4-H4-F1024-S256' THEN 'Pico Smoke Test'
                    WHEN 'E512-L6-H8-F1024-S512' THEN 'Micro (512-ctx)'
                    WHEN 'E768-L12-H8-F1536-S768' THEN 'Small Standard'
                    WHEN 'E768-L12-H8-F1536-S2048' THEN 'Small Long-Context'
                    WHEN 'E768-L12-H8-F3072-S2048' THEN 'Medium Long-Context'
                    WHEN 'E1024-L12-H16-F4096-S1024' THEN 'Base Balanced'
                    WHEN 'E2048-L24-H16-F8192-S1024' THEN 'Large High-Capacity'
                    ELSE DisplayName
                END;
            ");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "DisplayName",
                table: "TransformerConfigs");

            migrationBuilder.DropColumn(
                name: "DisplayName",
                table: "TransformerConfigPresets");
        }
    }
}
