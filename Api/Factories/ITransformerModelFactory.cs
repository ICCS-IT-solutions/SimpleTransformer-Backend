using Microsoft.EntityFrameworkCore;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Factories
{
    public interface ITransformerModelFactory
    {
        /// <param name="modelId">Model to construct.</param>
        /// <param name="useQLora">
        /// Overrides the model's stored UseQLora flag when supplied. Omit it to use
        /// the persisted value, which is what production paths should do.
        /// </param>
        Task<TransformerModel> CreateModelAsync(Guid modelId, bool? useQLora = null);
    }

    public class TransformerModelFactory : ITransformerModelFactory
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        public TransformerModelFactory(IDbContextFactory<AppDbContext> dbFactory)
        {
            _dbFactory = dbFactory;
        }

        public async Task<TransformerModel> CreateModelAsync(Guid modelId, bool? useQLora = null)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            //Check to see if the model exists in the database and return it if it does
            var model = await db.TransformerModels.FirstOrDefaultAsync(x => x.EntryId == modelId);
            
            if (model == null)
            {
                throw new InvalidOperationException($"Model with id {modelId} not found in database.");
            }

            var transformerConfig = await db.TransformerConfigs.FirstOrDefaultAsync(x => x.EntryId == model.TransformerConfigId);

            if (transformerConfig == null)
            {
                throw new InvalidOperationException($"Transformer config with id {model.TransformerConfigId} not found in database.");
            }
        
            var trainingConfig = await db.TrainingConfigs.FirstOrDefaultAsync(x => x.EntryId == model.TrainingConfigId);

            if (trainingConfig == null)
            {
                throw new InvalidOperationException($"Training config with id {model.TrainingConfigId} not found in database.");
            }

            //A QLoRA model and a raw model have structurally different trainable
            //parameter sets, so the persisted flag is the source of truth.
            bool effectiveUseQLora = useQLora ?? model.UseQLora;

            //The vocabulary decides the token id space, and checkpoints record the
            //configuration's VocabSize, so resolve and check it before the embedding
            //matrix is built rather than failing later with an unloadable checkpoint.
            if (model.VocabularyId.HasValue)
            {
                var vocabulary = await db.Vocabularies
                    .AsNoTracking()
                    .FirstOrDefaultAsync(x => x.EntryId == model.VocabularyId.Value);

                if (vocabulary == null)
                {
                    throw new InvalidOperationException(
                        $"Model '{model.Name}' references vocabulary {model.VocabularyId.Value}, " +
                        "which no longer exists in the database.");
                }

                var problem = VocabularyArtifacts.DescribeModelVocabularyProblem(
                    model.Name,
                    transformerConfig.Config.VocabSize,
                    vocabulary,
                    VocabularyArtifacts.TryReadTokenCount(vocabulary));

                if (problem != null)
                {
                    throw new InvalidOperationException(problem);
                }
            }

            return new TransformerModel(modelId, transformerConfig.Config, trainingConfig.Config, effectiveUseQLora,
                ParseBackendType(model.AccelerationBackend));
        }

        /// <summary>
        /// Parses the stored backend name; falls back to Auto for unknown or legacy values.
        /// </summary>
        private static BackendSelector.BackendType ParseBackendType(string? name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return BackendSelector.BackendType.Auto;

            return Enum.TryParse<BackendSelector.BackendType>(name, ignoreCase: true, out var type)
                ? type
                : BackendSelector.BackendType.Auto;
        }
    }
}