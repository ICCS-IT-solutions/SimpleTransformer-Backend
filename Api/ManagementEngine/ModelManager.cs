using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.ManagementEngine
{
    public class ModelManager
    {
        private TransformerModel? _loadedModel;
        private ITransformerModelFactory _modelFactory;
        public TransformerModel? LoadedModel => _loadedModel;

                public Guid? LoadedModelId =>
            _loadedModel?.TransformerModelId;

        public ModelManager(ITransformerModelFactory modelFactory)
        {
            _modelFactory = modelFactory;
        }

        public async Task<TransformerModel?> LoadModelAsync(Guid modelId)
        {
            // If this model is already loaded, simply return it.
            if (_loadedModel?.TransformerModelId == modelId)
            {
                return _loadedModel;
            }

            // Construct the new model first.
            var model = await _modelFactory.CreateModelAsync(modelId, useQLora: true);

            if (model == null)
            {
                return null;
            }

            // New model was successfully created.
            // Now it is safe to dispose the previous model - unless a training job
            // is still running on it. Disposing mid-run tears the workspace out from
            // under the loop, which surfaces as "Cannot access a disposed object"
            // inside the job. The service refuses the switch up front; this is the
            // backstop.
            if (_loadedModel is { IsTraining: true })
            {
                throw new InvalidOperationException(
                    "Cannot replace the loaded model while a training job is running on it. " +
                    "Stop the job first.");
            }

            _loadedModel?.Dispose();

            _loadedModel = model;

            return _loadedModel;
        }

        public TransformerModel? GetModel(Guid modelId)
        {
            if (_loadedModel?.TransformerModelId != modelId)
            {
                return null;
            }

            return _loadedModel;
        }

        /// <summary>
        /// True while the loaded model is owned by a running training job. Callers
        /// must not unload or replace the model in that state.
        /// </summary>
        public bool IsLoadedModelTraining => _loadedModel is { IsTraining: true };

        /// <summary>
        /// Disposes and drops the loaded model. Returns false (leaving the model
        /// intact) when a training job is running on it, because disposing the
        /// model mid-run invalidates the job's tensors and workspace.
        /// </summary>
        public bool UnloadModel()
        {
            if (_loadedModel is { IsTraining: true })
                return false;

            _loadedModel?.Dispose();
            _loadedModel = null;
            return true;
        }
    }
}