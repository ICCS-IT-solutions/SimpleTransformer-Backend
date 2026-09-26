using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Requests;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("")]
    public class TrainingCorpusController : ControllerBase
    {
        private readonly TrainingCorpusService _corpusService;

        public TrainingCorpusController(TrainingCorpusService corpusService) => _corpusService = corpusService;

        [HttpPost("api/v1/corpora/create")]
        public async Task<ApiResponse<CorpusDetailResponse>> Create([FromForm] CorpusCreateRequest req)
        {
            return await _corpusService.CreateAsync(req);
        }

        [HttpGet("api/v1/corpora/available")]
        public async Task<ApiResponse<List<CorpusListItem>>> GetAvailable()
        {
            return await _corpusService.GetAvailableAsync();
        }

        [HttpGet("api/v1/corpora/{corpusId}")]
        public async Task<ApiResponse<CorpusDetailResponse>> GetOne(Guid corpusId)
        {
            return await _corpusService.GetOneAsync(corpusId);
        }

        [HttpPost("api/v1/corpora/{corpusId}/delete")]
        public async Task<ApiResponse<CorpusDetailResponse>> Delete(Guid corpusId)
        {
            return await _corpusService.DeleteAsync(corpusId);
        }
    }

    /// <summary>Save uploaded sources as a named, reusable corpus.</summary>
    public class CorpusCreateRequest : CorpusPreprocessRequest
    {
        public required string Name { get; set; }
        public List<IFormFile> TextFiles { get; set; } = new();
    }

    public class CorpusListItem
    {
        public Guid EntryId { get; set; }
        public string Name { get; set; } = string.Empty;
        public string SourceFileNames { get; set; } = string.Empty;
        public string Format { get; set; } = string.Empty;
        public int DocumentsIn { get; set; }
        public int DocumentsOut { get; set; }
        public long CharsIn { get; set; }
        public long CharsOut { get; set; }
        public int DuplicatesRemoved { get; set; }
        public int FilteredByLength { get; set; }
        public int FilteredEmpty { get; set; }
        public long FileSize { get; set; }
        public int UsedByJobs { get; set; }
        public DateTime DateCreated { get; set; }
    }

    public class CorpusDetailResponse : CorpusListItem
    {
        public string OptionsJson { get; set; } = string.Empty;
        public JsonElement? Options { get; set; }
        public List<string> Warnings { get; set; } = new();
    }
}
