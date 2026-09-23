using System.Collections.Concurrent;

namespace SimpleTransformer.Api.ManagementEngine
{
    public class TrainingJobManager
    {
        private readonly ConcurrentDictionary<Guid, TrainingJobControl> _jobs = new();

        //Jobs currently being prepared by a start/resume request (model and
        //checkpoint loading), used to reject duplicate launches.
        private readonly ConcurrentDictionary<Guid, byte> _launching = new();

        public TrainingJobControl GetOrCreate(Guid jobId)
        {
            return _jobs.GetOrAdd(
                jobId,
                _ => new TrainingJobControl());
        }

        /// <summary>
        /// Returns the guard a new run should launch with. A live loop already owns
        /// the job and must never be replaced (two loops would fight over the shared
        /// model instance and the job row), while a guard left behind by a stopped or
        /// finished run carries a cancelled token and has to be replaced.
        /// </summary>
        public bool TryGetLaunchControl(Guid jobId, out TrainingJobControl control)
        {
            var existing = GetOrCreate(jobId);

            if (existing.HasLiveLoop)
            {
                control = existing;
                return false;
            }

            control = existing.IsStopped || existing.Cancellation.IsCancellationRequested
                ? Reset(jobId)
                : existing;

            return true;
        }

        /// <summary>
        /// Reserves a job while a start/resume request prepares it. Loading a model
        /// and reading a checkpoint can take tens of seconds, so a duplicate click
        /// must not begin a second preparation for the same job.
        /// </summary>
        public bool TryBeginLaunch(Guid jobId)
        {
            return _launching.TryAdd(jobId, 0);
        }

        public void EndLaunch(Guid jobId)
        {
            _launching.TryRemove(jobId, out _);
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