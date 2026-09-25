using Microsoft.EntityFrameworkCore;
using SimpleTransformer.AppDb;
using SimpleTransformer.Config;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class ConfigService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly ConfigManager _configManager;
        public ConfigService(ConfigManager configManager, IDbContextFactory<AppDbContext> dbFactory)
        {
            _configManager = configManager;
            _dbFactory = dbFactory;
        }

        public async Task<ApiResponse<ConfigManagerResponse>> CreateTrainingConfig(CreateTrainingConfigRequest req)
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var configToAdd = new TrainingConfigEntry
            {
                Name = req.Name,
                Description = req.Description,
                //Map each of the config variables to the ones coming from the request -> Config object
                Config = req.Config
            };
            
            bool exists = await _db.TrainingConfigs.AnyAsync(x => x.Name == req.Name);

            if (exists)
            {
                return new ApiResponse<ConfigManagerResponse>
                {
                    Message = "Config with this name already exists, please choose a different name.",
                    Status = ResponseStatus.Error,
                    StatusCode = 500
                };
            }
            
            await _db.TrainingConfigs.AddAsync(configToAdd);
            await _db.SaveChangesAsync();

            return new ApiResponse<ConfigManagerResponse>
            {
                Message = "Config created successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<ConfigManagerResponse>> CreateTransformerConfig(CreateTransformerConfigRequest req)
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var configToAdd = new TransformerConfigEntry
            {
                DisplayName = req.DisplayName,
                Name = req.Name,
                Description = req.Description,
                //Map each of the config variables to the ones coming from the request -> Config object
                Config = req.Config
            };

            bool exists = await _db.TransformerConfigs.AnyAsync(x => x.Name == req.Name);
            
            if (exists)
            {
                return new ApiResponse<ConfigManagerResponse>
                {
                    Message = "Config with this name already exists, please choose a different name.",
                    Status = ResponseStatus.Error,
                    StatusCode = 500
                };
            }
            
            await _db.TransformerConfigs.AddAsync(configToAdd);
            await _db.SaveChangesAsync();

            return new ApiResponse<ConfigManagerResponse>
            {
                Message = "Config created successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        //Update existing configs - reuse the same DTO's as for creating, only though update the existing one in the database
        public async Task<ApiResponse<ConfigManagerResponse>> UpdateTrainingConfig(
            UpdateTrainingConfigRequest req)
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var config = await _db.TrainingConfigs
                .FirstOrDefaultAsync(x => x.EntryId == req.ConfigId);

            if (config == null)
            {
                return new ApiResponse<ConfigManagerResponse>
                {
                    Message = "Training configuration not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            var duplicateName = await _db.TrainingConfigs
                .AnyAsync(x => x.Name == req.Name && x.EntryId != req.ConfigId);

            if (duplicateName)
            {
                return new ApiResponse<ConfigManagerResponse>
                {
                    Message = "Another configuration already uses this name.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            config.Name = req.Name;
            config.Description = req.Description;
            config.Config = req.Config;

            await _db.SaveChangesAsync();

            return new ApiResponse<ConfigManagerResponse>
            {
                Message = "Training configuration updated successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<ConfigManagerResponse>> UpdateTransformerConfig(
            UpdateTransformerConfigRequest req)
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var config = await _db.TransformerConfigs
                .FirstOrDefaultAsync(x => x.EntryId == req.ConfigId);

            if (config == null)
            {
                return new ApiResponse<ConfigManagerResponse>
                {
                    Message = "Transformer configuration not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            var duplicateName = await _db.TransformerConfigs
                .AnyAsync(x => x.Name == req.Name && x.EntryId != req.ConfigId);

            if (duplicateName)
            {
                return new ApiResponse<ConfigManagerResponse>
                {
                    Message = "Another configuration already uses this name.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            config.DisplayName = req.DisplayName;
            config.Name = req.Name;
            config.Description = req.Description;
            config.Config = req.Config;

            await _db.SaveChangesAsync();

            return new ApiResponse<ConfigManagerResponse>
            {
                Message = "Transformer configuration updated successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<ConfigManagerTransformerConfigResponse>> GetTransformerConfigs()
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var configs = await _db.TransformerConfigs.ToListAsync();
            return new ApiResponse<ConfigManagerTransformerConfigResponse> 
            {
                Message = "Transformer configs fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new ConfigManagerTransformerConfigResponse 
                { 
                    Message = "Transformer configs fetched successfully.",
                    Status = Responses.InteractionStatus.Success,
                    TransformerConfigs = configs 
                } 
            };
        }

        public async Task<ApiResponse<ConfigManagerTrainingConfigResponse>> GetTrainingConfigs()
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var configs = await _db.TrainingConfigs.ToListAsync();
            return new ApiResponse<ConfigManagerTrainingConfigResponse> 
            {
                Message = "Training configs fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new ConfigManagerTrainingConfigResponse 
                { 
                    Message = "Training configs fetched successfully.",
                    Status = Responses.InteractionStatus.Success,
                    TrainingConfigs = configs 
                } 
            };
        }
        //Get individual configurations using their entry id passed in
        public async Task<ApiResponse<ConfigManagerTrainingConfigResponse>> GetTrainingConfig(Guid id)
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var config = await _db.TrainingConfigs.FirstOrDefaultAsync(x => x.EntryId == id);
            return new ApiResponse<ConfigManagerTrainingConfigResponse>
            {
                Message = "Training config fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new ConfigManagerTrainingConfigResponse
                {
                    Message = "Training config fetched successfully.",
                    Status = Responses.InteractionStatus.Success,
                    TrainingConfig = config
                }
            };
        }

        public async Task<ApiResponse<ConfigManagerTransformerConfigResponse>> GetTransformerConfig(Guid id)
        {
            using var _db = await _dbFactory.CreateDbContextAsync();
            var config = await _db.TransformerConfigs.FirstOrDefaultAsync(x => x.EntryId == id);
            return new ApiResponse<ConfigManagerTransformerConfigResponse>
            {
                Message = "Transformer config fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new ConfigManagerTransformerConfigResponse
                {
                    Message = "Transformer config fetched successfully.",
                    Status = Responses.InteractionStatus.Success,
                    TransformerConfig = config
                }
            };
        }
    }
}