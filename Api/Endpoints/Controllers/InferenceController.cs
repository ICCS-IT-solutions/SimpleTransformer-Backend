using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;


namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("")]
    public class InferenceController : ControllerBase
    {
        private readonly InferenceService _service;

        public InferenceController(
            InferenceService service)
        {
            _service = service;
        }
        [HttpPost("api/v1/infer")]
        public async Task<ApiResponse<InferenceResponse>> Infer([FromBody] InferenceRequest req)
        {
            //Validate: Input must not be empty or null, but the other two props do have auto properties assigned.
            if(string.IsNullOrEmpty(req.InputText)) throw new ArgumentException("Input must not be empty or null.");

            //Get the model and tokenizer from the injected instances in the constructor
            return await _service.Infer(req);
        }
        //Alias of infer, uses the same service.
        [HttpPost("api/v1/predict")]
        public async Task<ApiResponse<InferenceResponse>> Predict([FromBody] InferenceRequest req)
        {
            //Validate: Input must not be empty or null, but the other two props do have auto properties assigned.
            if(string.IsNullOrEmpty(req.InputText)) throw new ArgumentException("Input must not be empty or null.");

            //Get the model and tokenizer from the injected instances in the constructor
            return await _service.Infer(req);
        } 

        [HttpPost("api/v1/checkpoints/{transformerModelId}")]
        public async Task<ApiResponse<List<TrainingCheckpointEntry>>> GetAvailableCheckpoints(Guid transformerModelId)
        {
            return await _service.GetCheckpointsForModel(transformerModelId);
        }
    }
}