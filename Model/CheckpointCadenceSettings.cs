using System;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// How often the training loop writes its mid-epoch "temp" checkpoint, pushed
    /// in by the host process (Server / CLI) from config.ini before any model is
    /// constructed. The loop itself stays configuration-free: it only reads this
    /// static and asks <see cref="ShouldWriteTempCheckpoint"/>.
    /// <para>
    /// The temp checkpoint is the job's resume pointer, and every one is the FULL
    /// parameter block - weights, gradients, and the v5 trailer carrying AdamW's
    /// m and v - so it costs 16 bytes per parameter. At 208.8M parameters that is
    /// 3.34 GB per write. It used to be written once per outer batch, and the
    /// outer batch is capped at 8 optimiser steps, which on a 4.5M-token corpus
    /// is ~560 steps/epoch and therefore ~70 writes/epoch: ~234 GB of disk
    /// traffic per epoch, and none of it scales better than the step count does.
    /// <para>
    /// Cadence is therefore decoupled from the progress-display grouping: the
    /// loop writes on whichever of the step/time interval comes first, plus
    /// unconditionally at the end of every epoch and when a run is stopped or
    /// cancelled. Graceful stops therefore lose nothing, and only a hard crash
    /// (kill, power loss, OOM) can cost up to one interval of work.
    /// </para>
    /// </summary>
    public static class CheckpointCadenceSettings
    {
        /// <summary>
        /// Master switch for mid-epoch temp checkpoints. False leaves only the
        /// periodic end-of-epoch named checkpoints, so a crash can cost a whole
        /// epoch. Defaults to true.
        /// </summary>
        public static bool Enabled { get; set; } = true;

        /// <summary>
        /// Write a temp checkpoint after this many optimiser steps. 0 disables the
        /// step trigger. Default 50: on the 208.8M-parameter model that is one
        /// 3.34 GB write every ~50 steps instead of every 8, cutting a 560-step
        /// epoch from ~70 writes to ~11.
        /// </summary>
        public static int TempCheckpointEverySteps { get; set; } = 50;

        /// <summary>
        /// Also write after this many minutes, whichever trigger comes first.
        /// 0 disables the time trigger. A step interval alone does not bound the
        /// COST of a write: at 3.34 GB the serialisation and disk flush can take
        /// tens of seconds, so a slow step rate needs a clock as well as a count.
        /// Default 5 minutes.
        /// </summary>
        public static double TempCheckpointEveryMinutes { get; set; } = 5.0;

        /// <summary>
        /// Hard floor on <see cref="EpochCheckpointInterval"/>, enforced here so
        /// every caller clamps identically rather than each inventing its own
        /// minimum.
        /// </summary>
        public const int MinimumEpochCheckpointInterval = 5;

        /// <summary>
        /// Epochs between the PERSISTENT named checkpoints (always also written
        /// on the final epoch). These are NOT overwritten - each filename embeds
        /// its epoch and loss - so this interval, not the temp cadence, is what
        /// bounds how many full parameter-block files accumulate on disk.
        /// Clamped to at least <see cref="MinimumEpochCheckpointInterval"/>:
        /// anything tighter is redundant, because the temp checkpoint already
        /// bounds crash loss to a handful of steps.
        /// </summary>
        public static int EpochCheckpointInterval { get; set; } = MinimumEpochCheckpointInterval;

        /// <summary>
        /// Clamps a configured interval to the supported floor. Exposed so the
        /// value can be validated in one place.
        /// </summary>
        public static int ClampEpochCheckpointInterval(int configured) =>
            Math.Max(MinimumEpochCheckpointInterval, configured);

        /// <summary>
        /// The one place the policy is decided, kept pure so it can be tested
        /// without running a training loop. Both arguments are measured since
        /// the previous temp checkpoint (or since the run started).
        /// </summary>
        public static bool ShouldWriteTempCheckpoint(int stepsSinceLast, TimeSpan elapsedSinceLast)
        {
            if (!Enabled)
                return false;

            bool bySteps = TempCheckpointEverySteps > 0
                           && stepsSinceLast >= TempCheckpointEverySteps;
            bool byTime = TempCheckpointEveryMinutes > 0
                          && elapsedSinceLast >= TimeSpan.FromMinutes(TempCheckpointEveryMinutes);

            return bySteps || byTime;
        }
    }
}