using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    //For better future-proofing and being able to stop and resume training, along with register multiple jobs, I am going to create a database.
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
            
        }
        public DbSet<TrainingConfigEntry> TrainingConfigs { get; set; } = null!;
        public DbSet<TransformerConfigEntry> TransformerConfigs { get; set; } = null!;
        public DbSet<TrainingCheckpointEntry> TrainingCheckpoints { get; set; } = null!;
        public DbSet<VocabularyEntry> Vocabularies { get; set; } = null!;
        public DbSet<TrainingConfigPresetEntry> TrainingConfigPresets { get; set; } = null!;
        public DbSet<TransformerConfigPresetEntry> TransformerConfigPresets { get; set; } = null!;
        public DbSet<TransformerModelEntry> TransformerModels { get; set; } = null!;
        public DbSet<TrainingJobEntry> TrainingJobs { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Ignore<TrainingConfig>();
            modelBuilder.Ignore<TransformerConfig>();

            //Primary keys per table point to EntryId
            modelBuilder.Entity<TrainingConfigEntry>().HasKey(x => x.EntryId);
            modelBuilder.Entity<TransformerConfigEntry>().HasKey(x => x.EntryId);
            modelBuilder.Entity<TrainingCheckpointEntry>().HasKey(x => x.EntryId);

            //One record per checkpoint file on disk: the temp checkpoint is
            //overwritten in place, so (Filepath, Filename) must be unique.
            modelBuilder.Entity<TrainingCheckpointEntry>()
                .HasIndex(x => new { x.Filepath, x.Filename })
                .IsUnique();
            modelBuilder.Entity<VocabularyEntry>().HasKey(x => x.EntryId);
            modelBuilder.Entity<TrainingConfigPresetEntry>().HasKey(x => x.EntryId);
            modelBuilder.Entity<TransformerConfigPresetEntry>().HasKey(x => x.EntryId);
            modelBuilder.Entity<TransformerModelEntry>().HasKey(x => x.EntryId);
            modelBuilder.Entity<TrainingJobEntry>().HasKey(x => x.EntryId);

            // ---------------------------------------------------------------------
            // Training configuration
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingConfigEntry>()
                .Property(x => x.Config)
                .HasConversion<JsonStringValueConverter<TrainingConfig>>();

            // ---------------------------------------------------------------------
            // Transformer configuration
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TransformerConfigEntry>()
                .Property(x => x.Config)
                .HasConversion<JsonStringValueConverter<TransformerConfig>>();

            // ---------------------------------------------------------------------
            // Configuration presets
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingConfigPresetEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();

            modelBuilder.Entity<TransformerConfigPresetEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();
                
            // ---------------------------------------------------------------------
            // Transformer models
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TransformerModelEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();

            modelBuilder.Entity<TransformerModelEntry>()
                .HasIndex(x => x.TransformerConfigId);

            modelBuilder.Entity<TransformerModelEntry>()
                .HasIndex(x => x.TrainingConfigId);
            // ---------------------------------------------------------------------
            // Training jobs
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.Name)
                .IsUnique();

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TransformerModelId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TransformerConfigId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TrainingConfigId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.VocabularyId);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasIndex(x => x.TrainingCheckpointId);

            // ---------------------------------------------------------------------
            // Relationships
            // ---------------------------------------------------------------------

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TransformerModel)
                .WithMany()
                .HasForeignKey(x => x.TransformerModelId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TransformerConfig)
                .WithMany()
                .HasForeignKey(x => x.TransformerConfigId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TrainingConfig)
                .WithMany()
                .HasForeignKey(x => x.TrainingConfigId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.Vocabulary)
                .WithMany()
                .HasForeignKey(x => x.VocabularyId)
                .OnDelete(DeleteBehavior.Restrict);

            modelBuilder.Entity<TrainingJobEntry>()
                .HasOne(x => x.TrainingCheckpoint)
                .WithMany()
                .HasForeignKey(x => x.TrainingCheckpointId)
                .OnDelete(DeleteBehavior.SetNull);

        }
    }
}