using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.AccelerationEngine.GpuVulkan;

namespace SimpleTransformer.Api.Endpoints.Services
{
    [ApiController]
    [Route("")]
    public class ConfigController : ControllerBase
    {
        private readonly ConfigService _configService;
        public ConfigController(ConfigService configService)
        {
            _configService = configService;
        }

        /// <summary>
        /// Lists the acceleration backends the frontend can offer for model creation,
        /// with live availability (probes each backend without starting the server flow).
        /// </summary>
        [HttpGet("api/v1/backends")]
        public ApiResponse<object> GetBackends()
        {
            var backends = Enum.GetNames<BackendSelector.BackendType>()
                .Where(n => n != nameof(BackendSelector.BackendType.Auto))
                .Select(n =>
                {
                    bool available;
                    string description;
                    try
                    {
                        using var backend = BackendSelector.SelectBackend(Enum.Parse<BackendSelector.BackendType>(n));
                        available = backend.IsAvailable;
                        description = backend.Name;

                        //Surface the detected VRAM/budget so the frontend shows
                        //how much GPU memory this machine actually has.
                        if (backend is GpuVulkanBackend gpu && gpu.IsAvailable)
                            description = $"{backend.Name} | {gpu.GpuMemoryInfo} | {gpu.HostStagingInfo}";
                    }
                    catch
                    {
                        available = false;
                        description = "Unavailable (initialization failed)";
                    }

                    return new
                    {
                        name = n,
                        available,
                        description
                    };
                })
                .ToList();

            // Auto is always offered; it resolves to the best available backend at model load time.
            backends.Insert(0, new
            {
                name = nameof(BackendSelector.BackendType.Auto),
                available = true,
                description = "Automatically select the best available backend"
            });

            return new ApiResponse<object>
            {
                Message = "Acceleration backends fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = backends
            };
        }

        //Existing configurations
        [HttpPost("api/v1/config/update/training")]
        public async Task<ApiResponse<ConfigManagerResponse>> UpdateTrainingConfig([FromBody] UpdateTrainingConfigRequest req)
        {
            return await _configService.UpdateTrainingConfig(req);
        }

        [HttpPost("api/v1/config/update/transformer")]
        public async Task<ApiResponse<ConfigManagerResponse>> UpdateTransformerConfig([FromBody] UpdateTransformerConfigRequest req)
        {
            return await _configService.UpdateTransformerConfig(req);
        }
        //Create new configurations
        [HttpPost("api/v1/config/create/training")]
        public async Task<ApiResponse<ConfigManagerResponse>> CreateTrainingConfig([FromBody] CreateTrainingConfigRequest req)
        {
            return await _configService.CreateTrainingConfig(req);
        }

        [HttpPost("api/v1/config/create/transformer")]
        public async Task<ApiResponse<ConfigManagerResponse>> CreateTransformerConfig([FromBody] CreateTransformerConfigRequest req)
        {
            return await _configService.CreateTransformerConfig(req);
        }

        [HttpGet("api/v1/config/training/list")]
        public async Task<ApiResponse<ConfigManagerTrainingConfigResponse>> GetTrainingConfigs()
        {
            return await _configService.GetTrainingConfigs();
        }

        [HttpGet("api/v1/config/transformer/list")]
        public async Task<ApiResponse<ConfigManagerTransformerConfigResponse>> GetTransformerConfigs()
        {
            return await _configService.GetTransformerConfigs();
        }
        //Get individual configs using their id
        [HttpGet("api/v1/config/training/{configId}")]
        public async Task<ApiResponse<ConfigManagerTrainingConfigResponse>> GetTrainingConfig(Guid configId)
        {
            return await _configService.GetTrainingConfig(configId);
        }

        [HttpGet("api/v1/config/transformer/{configId}")] 
        public async Task<ApiResponse<ConfigManagerTransformerConfigResponse>> GetTransformerConfig(Guid configId)
        {
            return await _configService.GetTransformerConfig(configId);
        }
    }

    public class ConfigManagerResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; }
    }
}