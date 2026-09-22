using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("")]
    public class TransformerModelController : ControllerBase
    {
        private TransformerModelService _transformerModelService;

        public TransformerModelController(TransformerModelService transformerModelService)
        {
            _transformerModelService = transformerModelService;
        }
        [HttpPost("api/v1/models/create")]
        public async Task<ApiResponse<TransformerModelResponse>> CreateTransformerModel([FromBody] CreateTransformerModelRequest req)
        {
            return await _transformerModelService.CreateTransformerModel(req);
        }

        [HttpGet("api/v1/models/{modelId}")] 
        public async Task<ApiResponse<TransformerModelResponse>> GetModel([FromRoute] Guid modelId)
        {
            return await _transformerModelService.GetModel(modelId);
        }

        [HttpGet("api/v1/models/list")] 
        public async Task<ApiResponse<TransformerModelResponse>> GetModels()
        {
            return await _transformerModelService.GetModels();
        }

        [HttpPost("api/v1/models/{modelId}/load")]
        public async Task<ApiResponse<TransformerModelResponse>> LoadModel([FromRoute] Guid modelId)
        {
            return await _transformerModelService.LoadModel(modelId);
        }

        [HttpPost("api/v1/models/{modelId}/unload")]
        public async Task<ApiResponse<TransformerModelResponse>> UnloadModel([FromRoute] Guid modelId)
        {
            return await _transformerModelService.UnloadModel(modelId);
        }

        //The runtime model currently held in memory by the ModelManager (if any),
        //including the acceleration backend that was actually resolved at load time.
        [HttpGet("api/v1/models/active")]
        public async Task<ApiResponse<TransformerModelResponse>> GetActiveModel()
        {
            return await _transformerModelService.GetActiveModel();
        }

        [HttpPost("api/v1/models/{modelId}/update")]
        public async Task<ApiResponse<TransformerModelResponse>> UpdateTransformerModel([FromRoute] Guid modelId, [FromBody] CreateTransformerModelRequest req)
        {
            return await _transformerModelService.UpdateTransformerModel(modelId, req);
        }

        //Whether I will need this endpoint, I don't yet know, but it does help for testing to see if the backend can trigger a model load from a GUID.
        [HttpPost("api/v1/models/{modelId}/create-runtime-model")]
        public async Task<ApiResponse<TransformerModel?>> CreateRuntimeModel([FromRoute] Guid modelId)
        {
            return await _transformerModelService.CreateRuntimeModel(modelId);
        }
    }

    /*
    export type CreateTransformerModelRequest = {
        name: string;
        description: string;
        transformerConfig: TransformerConfigEntry;
        trainingConfig: TrainingConfigEntry;
    }
    */
}