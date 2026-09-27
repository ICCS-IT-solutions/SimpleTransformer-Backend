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
    public class VocabularyController : ControllerBase
    {
        private readonly VocabularyService _vocabularyService;

        public VocabularyController(VocabularyService vocabularyService) => _vocabularyService = vocabularyService;

        [HttpPost("api/v1/vocabulary/load")]
        public async Task<ApiResponse<VocabularyLoaderResponse>> LoadFromFile([FromBody] LoadVocabularyRequest req)
        {
            return await _vocabularyService.LoadFromFile(req);
        }

        [HttpPost("api/v1/vocabulary/compile")]
        public async Task<ApiResponse<VocabularyCompilationResponse>> Compile([FromBody] CompileVocabularyRequest req)
        {
            return await _vocabularyService.Compile(req); 
        }

        [HttpGet("api/v1/vocabulary/sources")]
        public async Task<ApiResponse<List<VocabularySourceFile>>> GetLoadVocabularySources()
        {
            return await _vocabularyService.GetLoadVocabularySources();
        }

        [HttpGet("api/v1/vocabulary/properties")]
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetCurrentVocabularyProperties()
        {
            return await _vocabularyService.GetCurrentVocabularyProperties();
        }

        [HttpGet("api/v1/vocabulary/properties/{modelId:guid}")]
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetActiveVocabularyProperties(Guid modelId)
        {
            return await _vocabularyService.GetActiveVocabularyProperties(modelId);
        }

        [HttpGet("api/v1/vocabulary/active")]
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetActiveVocabulary()
        {
            return await _vocabularyService.GetActiveVocabularyProperties(null);
        }

        [HttpGet("api/v1/vocabulary/available")]
        public async Task<ApiResponse<AvailableVocabulariesResponse>> GetAvailableVocabularies()
        {
            return await _vocabularyService.GetAvailableVocabulariesAsync();
        }

        [HttpPost("api/v1/vocabulary/upload")]
        public async Task<ApiResponse<VocabularyLoaderResponse>> UploadFiles(
            [FromForm] VocabularyUploadRequest req)
        {
            return await _vocabularyService.UploadFiles(req.Files);
        }
    }
    public class VocabularyUploadRequest
    {
        public List<IFormFile> Files { get; set; } = [];
        public string? Name { get; set; }
    }   

    public class VocabularySourceFile
    {
        public string? Name { get; set; }
        public float FileSize { get; set; }
    } 

    public class AvailableVocabulariesResponse
    {
        public List<VocabularyEntry> Vocabularies { get; set; } = [];
        public bool IsLoaded { get; set; }
    }
}   