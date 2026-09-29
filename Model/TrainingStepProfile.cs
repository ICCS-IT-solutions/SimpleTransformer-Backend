using System.Diagnostics;
using Serilog;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.AccelerationEngine.GpuVulkan;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Per-step training profiler. Gated behind <see cref="Enabled"/> (off by
    /// default): when off, <see cref="StepScope"/> costs one bool check and no
    /// allocation. When on (CLI: <c>--train-profile</c>, which also enables
    /// <see cref="VulkanPhaseProfile"/>), each <c>TrainStep</c> logs one line
    /// splitting wall-clock into forward / loss-backward / model-backward /
    /// clip+validate / optimizer / refresh-tail, plus Vulkan deltas
    /// (dispatches, submits, MiB up/down, phase ms) when the model runs on the
    /// Vulkan backend. That split is the decision input for all further
    /// optimisation: StageIn+StageOut dominance means fewer round-trips
    /// (batching/residency), DispatchWait dominance means bigger dispatches.
    /// </summary>
    public static class TrainingStepProfile
    {
        public static volatile bool Enabled;

        public struct StepScope : IDisposable
        {
            /// <summary>
            /// A no-op scope used when profiling is disabled or the caller has no
            /// backend handle yet. Disposing it short-circuits before any logging.
            /// </summary>
            public static readonly StepScope Disabled =
                new(null, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
            private readonly bool _active;
            private readonly IAccelerationBackend? _backend;
            private readonly long _t0;
            private readonly long _dispatch0;
            private readonly long _submit0;
            private readonly long _up0;
            private readonly long _down0;
            private readonly double _phaseStageIn0;
            private readonly double _phaseDispatch0;
            private readonly double _phaseWait0;
            private readonly double _phaseStageOut0;
            private readonly double _phasePool0;
            private long _tForward;
            private long _tLossBackward;
            private long _tModelBackward;
            private long _tClipValidate;
            private long _tOptimizer;
            private bool _disposed;

            public StepScope(IAccelerationBackend backend)
                : this(backend, Stopwatch.GetTimestamp(), true)
            {
            }

            private StepScope(
                IAccelerationBackend? backend, long t0, long dispatch0, long submit0,
                long up0, long down0, double phaseStageIn0, double phaseDispatch0,
                double phaseWait0, double phaseStageOut0, double phasePool0, bool active)
            {
                _active = active;
                _backend = backend;
                _t0 = t0;
                _dispatch0 = dispatch0;
                _submit0 = submit0;
                _up0 = up0;
                _down0 = down0;
                _phaseStageIn0 = phaseStageIn0;
                _phaseDispatch0 = phaseDispatch0;
                _phaseWait0 = phaseWait0;
                _phaseStageOut0 = phaseStageOut0;
                _phasePool0 = phasePool0;
                _tForward = _tLossBackward = _tModelBackward = _tClipValidate = _tOptimizer = 0;
                _disposed = false;
            }

            private StepScope(IAccelerationBackend? backend, long t0, bool active)
            {
                _active = active;
                _backend = backend;
                _t0 = t0;

                long dispatch0 = 0, submit0 = 0, up0 = 0, down0 = 0;
                if (active && backend is GpuVulkanBackend gpu)
                {
                    dispatch0 = gpu.DispatchCount;
                    submit0 = gpu.SubmitCount;
                    up0 = GpuVulkanBackend.UploadedBytes;
                    down0 = GpuVulkanBackend.DownloadedBytes;
                }
                _dispatch0 = dispatch0;
                _submit0 = submit0;
                _up0 = up0;
                _down0 = down0;

                if (active && VulkanPhaseProfile.Enabled)
                {
                    _phaseStageIn0 = VulkanPhaseProfile.StageInMs;
                    _phaseDispatch0 = VulkanPhaseProfile.DispatchMs;
                    _phaseWait0 = VulkanPhaseProfile.DispatchWaitMs;
                    _phaseStageOut0 = VulkanPhaseProfile.StageOutMs;
                    _phasePool0 = VulkanPhaseProfile.PoolMs;
                }
                else
                {
                    _phaseStageIn0 = _phaseDispatch0 = _phaseWait0 = _phaseStageOut0 = _phasePool0 = 0;
                }

                _tForward = _tLossBackward = _tModelBackward = _tClipValidate = _tOptimizer = 0;
                _disposed = false;
            }

            public void MarkForward(long ticks) { if (_active) _tForward = ticks; }
            public void MarkLossBackward(long ticks) { if (_active) _tLossBackward = ticks; }
            public void MarkModelBackward(long ticks) { if (_active) _tModelBackward = ticks; }
            public void MarkClipValidate(long ticks) { if (_active) _tClipValidate = ticks; }
            public void MarkOptimizer(long ticks) { if (_active) _tOptimizer = ticks; }

            public void Dispose()
            {
                if (!_active || _disposed) return;
                _disposed = true;
                double freq = Stopwatch.Frequency;
                double totalMs = (Stopwatch.GetTimestamp() - _t0) * 1000.0 / freq;
                double fMs = _tForward * 1000.0 / freq;
                double lbMs = _tLossBackward * 1000.0 / freq;
                double mbMs = _tModelBackward * 1000.0 / freq;
                double cvMs = _tClipValidate * 1000.0 / freq;
                double optMs = _tOptimizer * 1000.0 / freq;
                // Refresh + trim + workspace reset + disposal: total minus phases.
                double tailMs = Math.Max(0, totalMs - (fMs + lbMs + mbMs + cvMs + optMs));

                if (_backend is GpuVulkanBackend gpu)
                {
                    long dispatches = gpu.DispatchCount - _dispatch0;
                    long submits = gpu.SubmitCount - _submit0;
                    double avg = submits > 0 ? (double)dispatches / submits : 0;
                    double upMiB = (GpuVulkanBackend.UploadedBytes - _up0) / 1024.0 / 1024.0;
                    double downMiB = (GpuVulkanBackend.DownloadedBytes - _down0) / 1024.0 / 1024.0;
                    double vkSum = (VulkanPhaseProfile.StageInMs - _phaseStageIn0)
                                 + (VulkanPhaseProfile.DispatchMs - _phaseDispatch0)
                                 + (VulkanPhaseProfile.DispatchWaitMs - _phaseWait0)
                                 + (VulkanPhaseProfile.StageOutMs - _phaseStageOut0)
                                 + (VulkanPhaseProfile.PoolMs - _phasePool0);
                    // vkSum only covers instrumented kernels, so the remainder is
                    // host-side tensor work / uninstrumented GPU paths: the
                    // pointer to what to attack next when no single phase wins.
                    double vkGap = Math.Max(0, totalMs - vkSum);
                    string phases = VulkanPhaseProfile.Enabled
                        ? string.Format(
                            " | vk stageIn {0:F1} dispatch {1:F1} wait {2:F1} stageOut {3:F1} pool {4:F1} ms" +
                            " (accounted {5:F1} ms, uninstrumented {6:F1} ms)",
                            VulkanPhaseProfile.StageInMs - _phaseStageIn0,
                            VulkanPhaseProfile.DispatchMs - _phaseDispatch0,
                            VulkanPhaseProfile.DispatchWaitMs - _phaseWait0,
                            VulkanPhaseProfile.StageOutMs - _phaseStageOut0,
                            VulkanPhaseProfile.PoolMs - _phasePool0,
                            vkSum,
                            vkGap)
                        : string.Empty;
                    Log.Information(
                        "[train-profile] step {Total:F1} ms (fwd {Fwd:F1} lossBwd {LossBwd:F1} modelBwd {ModelBwd:F1} clipVal {Clip:F1} opt {Opt:F1} tail {Tail:F1})" +
                        " | dispatches {Disp} in {Sub} submits ({Avg:F1}/submit) up {Up:F2} MiB down {Down:F2} MiB{Phases}",
                        totalMs, fMs, lbMs, mbMs, cvMs, optMs, tailMs,
                        dispatches, submits, avg, upMiB, downMiB, phases);
                }
                else
                {
                    Log.Information(
                        "[train-profile] step {Total:F1} ms (fwd {Fwd:F1} lossBwd {LossBwd:F1} modelBwd {ModelBwd:F1} clipVal {Clip:F1} opt {Opt:F1} tail {Tail:F1}) on {Backend}",
                        totalMs, fMs, lbMs, mbMs, cvMs, optMs, tailMs, _backend?.Name ?? "unknown");
                }
            }
        }

        /// <summary>
        /// Starts a profiling scope for one <c>TrainStep</c>. Returns
        /// <see cref="StepScope.Disabled"/> (a no-op) when profiling is off.
        /// </summary>
        public static StepScope Begin(IAccelerationBackend backend) =>
            Enabled ? new StepScope(backend) : StepScope.Disabled;
    }
}
