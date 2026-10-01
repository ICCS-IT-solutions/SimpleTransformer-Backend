namespace SimpleTransformer.Api.Requests
{
    /// <summary>Body for POST api/v1/memory/reset. All members optional.</summary>
    public class ResetMemoryRequest
    {
        /// <summary>
        /// True (default) to also trim the activation/device pools and run a
        /// blocking compacting collection; false to clear only the valve's
        /// latch and cooldown without stalling a running job.
        /// </summary>
        public bool RelieveNow { get; set; } = true;
    }
}