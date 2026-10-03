using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    /// <summary>
    /// Read-only access to the rolling Serilog files in <c>logs/</c>. The files
    /// themselves are written by the sink in Program.ConfigureLogging; this
    /// exposes them to the Diagnostics view.
    ///
    /// Admin-gated in the frontend router, but note the backend performs no
    /// authorization of its own yet - the same seam every other controller has.
    /// </summary>
    [ApiController]
    [Route("")]
    public class LogsController : ControllerBase
    {
        private readonly LogService _logService;

        public LogsController(LogService logService)
        {
            _logService = logService;
        }

        /// <summary>
        /// Retained log files, newest first. Seven are kept by the sink's
        /// retainedFileCountLimit; this reports whatever is actually on disk,
        /// which also reflects files not yet pruned if the server was idle.
        /// </summary>
        [HttpGet("api/v1/logs/files")]
        public ApiResponse<List<LogFileEntry>> GetFiles()
        {
            return new ApiResponse<List<LogFileEntry>>
            {
                Message = "Log files retrieved.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = _logService.GetFiles(),
            };
        }

        /// <summary>
        /// Tail of the log, newest last. Supplying <c>file</c> reads one day;
        /// omitting it searches every retained day (slower, and reported as
        /// truncated when the cap is hit). Reads from the end of each file and is
        /// capped, so a 12 MB day costs the same as a small one.
        /// </summary>
        [HttpGet("api/v1/logs")]
        public ApiResponse<LogSearchResult> GetLogs(
            [FromQuery] string? file,
            [FromQuery] string? search,
            [FromQuery] string? level,
            [FromQuery] int lines = 200)
        {
            if (!string.IsNullOrWhiteSpace(file))
            {
                //Surface a bad name as a failed envelope rather than a 500: the
                //guard throws ArgumentException for anything outside logs/.
                try
                {
                    var single = _logService.Read(file, search, level, lines);
                    return new ApiResponse<LogSearchResult>
                    {
                        Message = single.Truncated
                            ? "Log entries retrieved (partial: the source was larger than the read window)."
                            : "Log entries retrieved.",
                        Status = ResponseStatus.Success,
                        StatusCode = 200,
                        Data = single,
                    };
                }
                catch (ArgumentException ex)
                {
                    return new ApiResponse<LogSearchResult>
                    {
                        Message = $"Invalid log file name: {ex.Message}",
                        Status = ResponseStatus.Failure,
                        StatusCode = 400,
                        Data = new LogSearchResult(),
                    };
                }
            }

            var all = _logService.Read(null, search, level, lines);
            return new ApiResponse<LogSearchResult>
            {
                Message = all.Truncated
                    ? "Log entries retrieved across all retained days (partial: results were capped)."
                    : "Log entries retrieved across all retained days.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = all,
            };
        }
    }
}