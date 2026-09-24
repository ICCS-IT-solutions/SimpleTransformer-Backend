using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Api.Endpoints.Controllers;
using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Api.ManagementEngine;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TransformerModelService
    {
        private IDbContextFactory<AppDbContext> _dbFactory;
        private ITransformerModelFactory _transformerModelFactory;
        private ModelManager _modelManager;

        public TransformerModelService(ITransformerModelFactory transformerModelFactory, IDbContextFactory<AppDbContext> dbFactory, ModelManager modelManager)
        {
            _transformerModelFactory = transformerModelFactory;
            _dbFactory = dbFactory;
            _modelManager = modelManager;
        }
        //Create and return a model from the database using its id
        public async Task<ApiResponse<TransformerModelResponse>> GetModel(Guid modelId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var model = await db.TransformerModels
                .AsNoTracking()
                .FirstOrDefaultAsync(x => x.EntryId == modelId);

            if (model == null)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Model not found in database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }
            return new ApiResponse<TransformerModelResponse>()
            {
                Message = "Model fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Model fetched successfully.",
                    Status = InteractionStatus.Success,
                    Model = model
                }
            };
        }

        public async Task<ApiResponse<TransformerModelResponse>> GetModels()
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var models = await db.TransformerModels.ToListAsync();
            return new ApiResponse<TransformerModelResponse>()
            {
                Message = "Models fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Models fetched successfully.",
                    Status = InteractionStatus.Success,
                    Models = models
                }
            };
        }

        public async Task<ApiResponse<TransformerModel?>> CreateRuntimeModel(Guid modelId)
        {
            var model = await _transformerModelFactory.CreateModelAsync(modelId);
            return new ApiResponse<TransformerModel?>
            {
                Message = "Model created successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = model
            };
        }

        //Create a model definition to store in the database
        public async Task<ApiResponse<TransformerModelResponse>> CreateTransformerModel(CreateTransformerModelRequest req)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var model = new TransformerModelEntry
            {
                Name = req.Name,
                Description = req.Description,
                TransformerConfigId = req.TransformerConfig.EntryId,
                TrainingConfigId = req.TrainingConfig.EntryId,
                AccelerationBackend = req.AccelerationBackend,
                UseQLora = req.UseQLora
            };

            await db.TransformerModels.AddAsync(model);
            await db.SaveChangesAsync();
            return new ApiResponse<TransformerModelResponse>
            {
                Message = "Model created successfully",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Model created successfully",
                    Status = InteractionStatus.Success,
                    Model = model
                }
            };
        }

        public async Task<ApiResponse<TransformerModelResponse>> LoadModel(Guid modelId)
        {
            using var db = await _dbFactory.CreateDbContextAsync();
            var model = await db.TransformerModels.FirstOrDefaultAsync(x => x.EntryId == modelId);

            if (model == null)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Model not found in database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            var runtimeLoadedId = _modelManager.LoadedModelId;

            if (model.IsLoaded)
            {
                if (runtimeLoadedId == model.EntryId)
                {
                    return new ApiResponse<TransformerModelResponse>()
                    {
                        Message = "Model already loaded.",
                        Status = ResponseStatus.Failure,
                        StatusCode = 400
                    };
                }

                // Stale flag (e.g. the server restarted while the DB still says the
                // model is loaded): clear it so the load can proceed.
                var stale = await db.TransformerModels.Where(x => x.IsLoaded).ToListAsync();
                foreach (var entry in stale)
                {
                    entry.IsLoaded = false;
                }
            }

            //Switching models disposes the outgoing runtime model, so refuse while a
            //training job owns it rather than corrupting the run.
            if (_modelManager.IsLoadedModelTraining && runtimeLoadedId != model.EntryId)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Another model is in use by a training job. Stop that job and wait for it to finish before loading a different model.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            // Construct the runtime model first: if construction fails the database
            // is left untouched and any previously loaded runtime model stays intact
            // (ModelManager only swaps after a successful build).
            try
            {
                await _modelManager.LoadModelAsync(modelId);
            }
            catch (Exception ex)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = $"Failed to load model: {ex.Message}",
                    Status = ResponseStatus.Failure,
                    StatusCode = 500
                };
            }

            //Unmark any other model flagged as loaded; only one runtime model is active.
            var otherModels = await db.TransformerModels
                .Where(x => x.IsLoaded && x.EntryId != model.EntryId)
                .ToListAsync();
            foreach (var other in otherModels)
            {
                other.IsLoaded = false;
            }

            model.IsLoaded = true;
            model.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TransformerModelResponse>()
            {
                Message = "Model loaded successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Model loaded successfully.",
                    Status = InteractionStatus.Success,
                    Model = model,
                    ActiveBackend = _modelManager.LoadedModel?.Backend.Name
                }
            };
        }

        public async Task<ApiResponse<TransformerModelResponse>> UnloadModel(Guid modelId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var model = await db.TransformerModels.FirstOrDefaultAsync(x => x.EntryId == modelId);

            if (model == null)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Model not found in database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (!model.IsLoaded && _modelManager.LoadedModelId != model.EntryId)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Model is not loaded.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            //Dispose the runtime model when it is the one being unloaded. A live
            //training job keeps it alive: unloading then would leave the loop
            //holding disposed tensors and the job would die mid-step.
            if (_modelManager.LoadedModelId == model.EntryId)
            {
                if (!_modelManager.UnloadModel())
                {
                    return new ApiResponse<TransformerModelResponse>()
                    {
                        Message = "This model is still in use by a training job. Stop the job and wait for it to finish before unloading it.",
                        Status = ResponseStatus.Failure,
                        StatusCode = 409
                    };
                }
            }

            model.IsLoaded = false;
            model.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TransformerModelResponse>()
            {
                Message = "Model unloaded successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Model unloaded successfully.",
                    Status = InteractionStatus.Success,
                    Model = model
                }
            };
        }

        public async Task<ApiResponse<TransformerModelResponse>> GetActiveModel()
        {
            var loadedId = _modelManager.LoadedModelId;

            if (loadedId == null)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "No model is currently loaded.",
                    Status = ResponseStatus.Success,
                    StatusCode = 200,
                    Data = new TransformerModelResponse
                    {
                        Message = "No model is currently loaded.",
                        Status = InteractionStatus.Success
                    }
                };
            }

            await using var db = await _dbFactory.CreateDbContextAsync();
            var model = await db.TransformerModels
                .AsNoTracking()
                .FirstOrDefaultAsync(x => x.EntryId == loadedId);

            if (model == null)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Active model entry could not be found in the database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 500
                };
            }

            return new ApiResponse<TransformerModelResponse>()
            {
                Message = "Active model fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Active model fetched successfully.",
                    Status = InteractionStatus.Success,
                    Model = model,
                    ActiveBackend = _modelManager.LoadedModel?.Backend.Name
                }
            };
        }

        public async Task<ApiResponse<TransformerModelResponse>> UpdateTransformerModel(Guid modelId, CreateTransformerModelRequest req)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var model = await db.TransformerModels.FirstOrDefaultAsync(x => x.EntryId == modelId);

            if (model == null)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Model not found in database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //A loaded runtime model would desync from the changed definition.
            if (model.IsLoaded)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Unload the model before updating it.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            //Validate that the referenced configs exist before persisting the update.
            var transformerConfigExists = await db.TransformerConfigs.AnyAsync(x => x.EntryId == req.TransformerConfig.EntryId);
            if (!transformerConfigExists)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = $"Transformer config with id {req.TransformerConfig.EntryId} not found in database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            var trainingConfigExists = await db.TrainingConfigs.AnyAsync(x => x.EntryId == req.TrainingConfig.EntryId);
            if (!trainingConfigExists)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = $"Training config with id {req.TrainingConfig.EntryId} not found in database.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            model.Name = req.Name;
            model.Description = req.Description;
            model.TransformerConfigId = req.TransformerConfig.EntryId;
            model.TrainingConfigId = req.TrainingConfig.EntryId;
            model.AccelerationBackend = req.AccelerationBackend;
            model.DateUpdated = DateTime.UtcNow;

            //UseQLora is intentionally NOT updated here. It decides which trainable
            //parameters the model exposes, so flipping it on an existing model would
            //make every saved checkpoint unloadable. A new model is required to
            //train the other way.

            await db.SaveChangesAsync();

            return new ApiResponse<TransformerModelResponse>()
            {
                Message = "Model updated successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TransformerModelResponse
                {
                    Message = "Model updated successfully.",
                    Status = InteractionStatus.Success,
                    Model = model
                }
            };
        }
    }
}