using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Api.Endpoints.Controllers;
using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TransformerModelService
    {
        private IDbContextFactory<AppDbContext> _dbFactory;
        private ITransformerModelFactory _transformerModelFactory;

        public TransformerModelService(ITransformerModelFactory transformerModelFactory, IDbContextFactory<AppDbContext> dbFactory)
        {
            _transformerModelFactory = transformerModelFactory;
            _dbFactory = dbFactory;
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
                AccelerationBackend = req.AccelerationBackend
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

            if (model.IsLoaded)
            {
                return new ApiResponse<TransformerModelResponse>()
                {
                    Message = "Model already loaded.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }
            //If another model is loaded, unload it
            var otherModel = await db.TransformerModels.FirstOrDefaultAsync(x => x.IsLoaded == true && x.EntryId != model.EntryId);
            if (otherModel != null)
            {
                otherModel.IsLoaded = false;
            }

            model.IsLoaded = true;

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
                    Model = model
                }
            };
        }
    }
}