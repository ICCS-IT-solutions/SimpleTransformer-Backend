using System;
using System.Diagnostics;
using System.Threading;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Per-phase profiling for Vulkan GPU operations.
    /// Tracks elapsed time (in Stopwatch ticks) and counts across:
    /// - StageIn: tensor pack + buffer upload
    /// - Dispatch: kernel launcher submission / dispatch queueing
    /// - DispatchWait: sync / fence wait
    /// - StageOut: buffer download + tensor unpack
    /// - Pool: buffer renting and returning
    ///
    /// Enabled via <see cref="Enabled"/>. When disabled, recording methods
    /// perform a single boolean check and return immediately (zero overhead).
    /// </summary>
    public static class VulkanPhaseProfile
    {
        public static volatile bool Enabled;

        private static long _stageInTicks;
        private static long _dispatchTicks;
        private static long _dispatchWaitTicks;
        private static long _stageOutTicks;
        private static long _poolTicks;
        private static long _opCount;

        public static void Reset()
        {
            Interlocked.Exchange(ref _stageInTicks, 0);
            Interlocked.Exchange(ref _dispatchTicks, 0);
            Interlocked.Exchange(ref _dispatchWaitTicks, 0);
            Interlocked.Exchange(ref _stageOutTicks, 0);
            Interlocked.Exchange(ref _poolTicks, 0);
            Interlocked.Exchange(ref _opCount, 0);
        }

        public static void RecordStageIn(long ticks)
        {
            if (Enabled) Interlocked.Add(ref _stageInTicks, ticks);
        }

        public static void RecordDispatch(long ticks)
        {
            if (Enabled) Interlocked.Add(ref _dispatchTicks, ticks);
        }

        public static void RecordDispatchWait(long ticks)
        {
            if (Enabled) Interlocked.Add(ref _dispatchWaitTicks, ticks);
        }

        public static void RecordStageOut(long ticks)
        {
            if (Enabled) Interlocked.Add(ref _stageOutTicks, ticks);
        }

        public static void RecordPool(long ticks)
        {
            if (Enabled) Interlocked.Add(ref _poolTicks, ticks);
        }

        public static void RecordOp()
        {
            if (Enabled) Interlocked.Increment(ref _opCount);
        }

        public static double StageInMs => _stageInTicks * 1000.0 / Stopwatch.Frequency;
        public static double DispatchMs => _dispatchTicks * 1000.0 / Stopwatch.Frequency;
        public static double DispatchWaitMs => _dispatchWaitTicks * 1000.0 / Stopwatch.Frequency;
        public static double StageOutMs => _stageOutTicks * 1000.0 / Stopwatch.Frequency;
        public static double PoolMs => _poolTicks * 1000.0 / Stopwatch.Frequency;
        public static long OpCount => Interlocked.Read(ref _opCount);

        public static string Summary()
        {
            long ops = OpCount;
            if (ops == 0) return "No GPU operations profiled.";

            double totalMs = StageInMs + DispatchMs + DispatchWaitMs + StageOutMs + PoolMs;
            return $"Profiled {ops} ops ({totalMs:F2} ms total):\n" +
                   $"  StageIn (Pack/Up)     : {StageInMs,7:F2} ms ({StageInMs / totalMs * 100,5:F1}%)  avg {StageInMs / ops * 1000,6:F1} µs/op\n" +
                   $"  Dispatch (Record/Sub) : {DispatchMs,7:F2} ms ({DispatchMs / totalMs * 100,5:F1}%)  avg {DispatchMs / ops * 1000,6:F1} µs/op\n" +
                   $"  DispatchWait (GPU/Sync):{DispatchWaitMs,7:F2} ms ({DispatchWaitMs / totalMs * 100,5:F1}%)  avg {DispatchWaitMs / ops * 1000,6:F1} µs/op\n" +
                   $"  StageOut (Dn/Unpack)  : {StageOutMs,7:F2} ms ({StageOutMs / totalMs * 100,5:F1}%)  avg {StageOutMs / ops * 1000,6:F1} µs/op\n" +
                   $"  Pool (Rent/Return)    : {PoolMs,7:F2} ms ({PoolMs / totalMs * 100,5:F1}%)  avg {PoolMs / ops * 1000,6:F1} µs/op";
        }
    }
}
