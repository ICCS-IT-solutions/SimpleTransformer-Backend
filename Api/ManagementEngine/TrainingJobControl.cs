namespace SimpleTransformer.Api.ManagementEngine
{
    public sealed class TrainingJobControl
{
    private readonly object _lock = new();

    private TaskCompletionSource<bool> _resumeSource =
        CreateResumeSource();

    public CancellationTokenSource Cancellation { get; } =
        new CancellationTokenSource();

    public bool IsPaused { get; private set; }

    public bool IsStopped { get; private set; }

    public Task? RunningTask { get; set; }

    /// <summary>
    /// True while a training loop is still executing for this guard. After a
    /// restart, a stop or a cancel there is no live loop, so the job has to be
    /// relaunched (rebuilt from its checkpoint) instead of being un-paused.
    /// </summary>
    public bool HasLiveLoop => RunningTask is { IsCompleted: false };

    private static TaskCompletionSource<bool> CreateResumeSource()
    {
        return new TaskCompletionSource<bool>(
            TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public void Pause()
    {
        lock (_lock)
        {
            if (IsStopped || Cancellation.IsCancellationRequested)
                return;

            IsPaused = true;

            _resumeSource =
                CreateResumeSource();
        }
    }

    public void Resume()
    {
        TaskCompletionSource<bool> source;

        lock (_lock)
        {
            if (!IsPaused)
                return;

            IsPaused = false;
            source = _resumeSource;
        }

        source.TrySetResult(true);
    }

    public void Stop()
    {
        lock (_lock)
        {
            IsStopped = true;
            IsPaused = false;

            _resumeSource.TrySetResult(true);
            Cancellation.Cancel();
        }
    }

    public async Task WaitIfPausedAsync(CancellationToken cancellationToken = default)
    {
        Task waitTask;

        lock (_lock)
        {
            if (!IsPaused)
                return;

            waitTask = _resumeSource.Task;
        }

        //Waiting only on the resume signal would leave a paused loop deaf to a
        //stop or cancel, so cancellation releases the wait as well.
        await waitTask.WaitAsync(cancellationToken);
    }
}
}