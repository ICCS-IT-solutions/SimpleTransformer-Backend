using Serilog;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Per-stage NaN tripwire for the backward sweep.
    /// <para>
    /// Gradient clipping only detects a non-finite gradient AFTER the whole sweep,
    /// so it can name the poisoned PARAMETER but never the OPERATION that produced
    /// the NaN. The evidence that motivated this: on a diverging run
    /// <c>token_embeddings</c> reported hundreds of thousands of NaN entries, yet
    /// every count divided exactly by the embedding width - whole rows, matching the
    /// distinct token ids in the batch. That is the embedding scatter faithfully
    /// receiving an already-NaN gradient: the parameter was the messenger.
    /// <para>
    /// A <see cref="BackwardProbe"/> is a linear chain of stages. Each stage checks
    /// the gradient it receives; the first stage that goes non-finite is recorded,
    /// and the stage BEFORE it is reported as the origin, because the stage that
    /// SEES the NaN is downstream of the operation that MADE it.
    /// <para>
    /// Nested probes are supported: the model-level probe covers loss gradient,
    /// output projection and each block, and the block-level probe covers a block's
    /// six internal steps. A block that is already known bad skips its own
    /// sub-stages, so the output stays one clear line rather than a cascade.
    /// <para>
    /// Cost: one finite-scan per stage per step. Diagnostic only - see
    /// <see cref="GradientProbeSettings"/>.
    /// </para>
    /// </summary>
    public sealed class BackwardProbe
    {
        private string? _firstBadStage;
        private string _previousStage;
        private readonly bool _enabled;
        private readonly string _owner;

        /// <summary>Stage name of the first non-finite gradient seen, if any.</summary>
        public string? FirstBadStage => _firstBadStage;

        /// <summary>True once any stage in this chain has gone non-finite.</summary>
        public bool FoundNonFinite => _firstBadStage != null;

        /// <summary>
        /// True when this chain should be active: probing is enabled, and (for a
        /// nested probe) the parent has not already localised the failure.
        /// </summary>
        public bool Active => _enabled && _firstBadStage == null;

        /// <param name="owner">Label for the log line, e.g. "block 5".</param>
        /// <param name="firstStage">Name of the gradient entering this chain.</param>
        /// <param name="parent">Optional parent probe; this one is skipped once the
        /// parent has already found a non-finite stage, to avoid a cascade.</param>
        public BackwardProbe(string owner, string firstStage, BackwardProbe? parent = null)
        {
            _enabled = GradientProbeSettings.Enabled;
            _owner = owner;
            _previousStage = firstStage;

            // Inherit "already found" from the parent so a nested chain does not
            // re-report stages the parent has already condemned.
            if (parent != null && parent.FoundNonFinite)
                _firstBadStage = parent.FirstBadStage;
        }

        /// <summary>
        /// Records the stage named <paramref name="stage"/>. Returns true once the
        /// chain has seen a non-finite gradient. Throws when
        /// <see cref="GradientProbeSettings.ThrowOnFirstNonFinite"/> is set, so the
        /// sweep stops at the exact operation instead of running on poisoned data.
        /// </summary>
        public bool Check(string stage, TensorBase gradient)
        {
            if (!Active)
                return _firstBadStage != null;

            if (DiagonisticUtilities.IsFinite(gradient))
            {
                _previousStage = stage;
                return false;
            }

            _firstBadStage = stage;
            var message =
                $"[{_owner}] first non-finite gradient AFTER '{stage}'; it was finite " +
                $"after '{_previousStage}', so '{_previousStage}' is where the NaN was created.";

            if (GradientProbeSettings.ThrowOnFirstNonFinite)
                throw new InvalidOperationException("[GradientProbe] " + message);

            Log.Error("[GradientProbe] {Message}", message);
            return true;
        }
    }

    /// <summary>
    /// Controls the backward-pass NaN tripwire. Off by default.
    /// <para>
    /// Set <c>gradient_probe = true</c> in config.ini to arm it. When armed, every
    /// stage of every backward pass finite-scans its gradient, so the first
    /// non-finite one - and therefore the operation that created it - is named
    /// directly instead of being inferred from a downstream parameter.
    /// </para>
    /// </summary>
    public static class GradientProbeSettings
    {
        /// <summary>
        /// Default false: the tripwire scans every stage of every step, so it is
        /// opt-in. Set <c>gradient_probe = true</c> in config.ini to arm it.
        /// </summary>
        public static bool Enabled { get; set; }

        /// <summary>
        /// When true, the first non-finite stage aborts that step. TrainStep catches
        /// it, logs and skips the optimizer step. False lets the sweep finish so you
        /// can see every affected stage, not just the first.
        /// </summary>
        public static bool ThrowOnFirstNonFinite { get; set; }
    }
}