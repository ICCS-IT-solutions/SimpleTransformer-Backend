using System.Collections.Concurrent;

namespace SimpleTransformer.Api.ManagementEngine
{
    public class TrainingJobManager
    {
        private readonly ConcurrentDictionary<Guid, TrainingJobControl> _jobs = new();

        public TrainingJobControl GetOrCreate(Guid jobId)
        {
            return _jobs.GetOrAdd(
                jobId,
                _ => new TrainingJobControl());
        }

        /// <summary>
        /// Discards a job's guard and returns a fresh one. A stopped or cancelled
        /// control carries a cancelled token and a finished task, so relaunching a
        /// job must not inherit it.
        /// </summary>
        public TrainingJobControl Reset(Guid jobId)
        {
            if (_jobs.TryRemove(jobId, out var previous))
            {
                previous.Stop();
            }

            return GetOrCreate(jobId);
        }

        public TrainingJobControl GetControl(Guid jobId)
        {
            return _jobs[jobId];
        }

        public bool TryGet(
            Guid jobId,
            out TrainingJobControl? control)
        {
            return _jobs.TryGetValue(jobId, out control);
        }

        public bool Remove(Guid jobId)
        {
            return _jobs.TryRemove(jobId, out _);
        }
    }
}