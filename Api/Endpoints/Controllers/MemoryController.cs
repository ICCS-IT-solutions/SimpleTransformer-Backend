using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.ManagementEngine;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    /// <summary>
    /// Host-memory pressure valve status and frontend-triggered reset. The
    /// valve itself lives one-per-model inside TransformerModel (see
    /// MemoryPressureValve); these endpoints surface it so the UI can show
    /// live pressure and re-arm the ladder after a run failed under OOM.
    /// </summary>
    [ApiController]
    [Route("")]
    public class MemoryController : ControllerBase
    {
        private readonly ModelManager _modelManager;

        public MemoryController(ModelManager modelManager)
        {
            _modelManager = modelManager;
        }

        /// <summary>
        /// Live host-memory pressure plus the loaded model's valve telemetry.
        /// Safe to poll: one process sample, no allocations of consequence.
        /// </summary>
        [HttpGet("api/v1/memory")]
        public ApiResponse<object> GetMemoryStatus()
        {
            const double mib = 1024.0 * 1024.0;

            var model = _modelManager.LoadedModel;

            // Live sample from a throwaway valve: the loaded model's valve only
            // samples once every CheckEveryNSteps training steps, so its
            // LastSample may be stale when no job is running.
            var sample = new MemoryPressureValve().Sample();

            return new ApiResponse<object>
            {
                Message = "Memory pressure status fetched successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new
                {
                    modelLoaded = model != null,
                    modelId = model?.TransformerModelId,
                    isTraining = model?.IsTraining ?? false,
                    valve = model?.DescribeMemoryPressure()
                        ?? "no model loaded (valve state is per-model)",
                    host = new
                    {
                        sample.PrivateBytes,
                        sample.WorkingSetBytes,
                        sample.ManagedHeapBytes,
                        sample.PhysicalBytes,
                        sample.QuotaBytes,
                        //The valve gates on max(proc, sys); the Process line is
                        //the process share alone, so the two are distinct figures.
                        usedPercent = Math.Round(sample.UsedFraction * 100.0, 1),
                        systemUsedPercent = Math.Round(sample.SystemUsedFraction * 100.0, 1),
                        availableMiB = Math.Round(sample.PhysicalBytes / mib, 0)
                    }
                }
            };
        }

        /// <summary>
        /// Re-arms the loaded model's memory pressure valve and, by default,
        /// performs an immediate relief pass (idle device pool trim, activation
        /// pool trim, blocking compacting collection) - the valve's Compact
        /// rung on demand. Send {"relieveNow": false} to clear only the
        /// latch/cooldown without stalling a running job. The body may be
        /// omitted entirely; the default applies. Returns 400 when no model is
        /// loaded, because valve state is per-model and dies with the model.
        /// </summary>
        [HttpPost("api/v1/memory/reset")]
        public ApiResponse<object> ResetMemory(
            [FromBody(EmptyBodyBehavior = Microsoft.AspNetCore.Mvc.ModelBinding.EmptyBodyBehavior.Allow)]
            ResetMemoryRequest? req)
        {
            var model = _modelManager.LoadedModel;
            if (model == null)
            {
                return new ApiResponse<object>
                {
                    Message = "No model is loaded, so there is no memory valve to reset.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400,
                    Data = new { reset = false }
                };
            }

            bool relieveNow = req?.RelieveNow ?? true;
            string telemetry = model.ResetMemoryPressure(relieveNow);

            return new ApiResponse<object>
            {
                Message = relieveNow
                    ? "Memory pressure valve reset and memory reclaimed."
                    : "Memory pressure valve reset.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new { reset = true, relieveNow, valve = telemetry }
            };
        }
    }
}