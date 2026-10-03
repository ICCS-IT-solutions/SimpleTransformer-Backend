using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    /// <summary>
    /// Durable fault history, backed by the FaultEvents table. Complements the
    /// rotating log files: a fault recorded here outlives the seven-day rotation
    /// and survives deletion of the job that produced it.
    /// </summary>
    [ApiController]
    [Route("")]
    public class FaultController : ControllerBase
    {
        private readonly FaultService _faultService;

        public FaultController(FaultService faultService)
        {
            _faultService = faultService;
        }

        /// <summary>Recent fault signatures, newest first.</summary>
        [HttpGet("api/v1/faults")]
        public async Task<ApiResponse<List<FaultSummary>>> GetFaults(
            [FromQuery] int take = 100,
            [FromQuery] string? component = null)
        {
            return new ApiResponse<List<FaultSummary>>
            {
                Message = "Fault events retrieved.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = await _faultService.GetRecentAsync(take, component),
            };
        }

        /// <summary>
        /// Occurrence totals per fault type - the "what keeps happening" rollup.
        /// </summary>
        [HttpGet("api/v1/faults/summary")]
        public async Task<ApiResponse<List<FaultCount>>> GetSummary()
        {
            return new ApiResponse<List<FaultCount>>
            {
                Message = "Fault summary retrieved.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = await _faultService.GetCountsAsync(),
            };
        }

        /// <summary>One signature in full, stack trace included.</summary>
        [HttpGet("api/v1/faults/{entryId:guid}")]
        public async Task<ApiResponse<FaultDetail>> GetOne(Guid entryId)
        {
            var detail = await _faultService.GetOneAsync(entryId);

            if (detail == null)
            {
                return new ApiResponse<FaultDetail>
                {
                    Message = $"No fault event with id {entryId}.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404,
                };
            }

            return new ApiResponse<FaultDetail>
            {
                Message = "Fault event retrieved.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = detail,
            };
        }
    }
}