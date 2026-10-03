using System.ComponentModel.DataAnnotations;

namespace SimpleTransformer.AppDb
{
    /// <summary>
    /// A fault worth remembering beyond the rolling log file: one row per distinct
    /// failure signature, with a running occurrence count rather than one row per
    /// throw. Log files rotate away after seven days; this table is the durable
    /// record, and the count answers "how often has this actually happened?"
    /// without unbounded growth during a fault storm.
    ///
    /// Rollups are keyed by <see cref="Fingerprint"/> (hash of exception type plus
    /// the top frames), so the same OOM from two different jobs lands on one row
    /// and OccurrenceCount climbs.
    /// </summary>
    public class FaultEventEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();

        /// <summary>Exception type name, e.g. "OutOfMemoryException".</summary>
        public string FaultType { get; set; } = string.Empty;

        /// <summary>Which subsystem faulted: training, inference, server, api.</summary>
        public string Component { get; set; } = string.Empty;

        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// Truncated on write. A full .NET stack is routinely 10-50 KB, and this
        /// column is only ever read one row at a time in the fault viewer.
        /// </summary>
        public string? StackTrace { get; set; }

        /// <summary>Set when the fault happened inside a training run.</summary>
        public string? JobId { get; set; }

        public string? ModelId { get; set; }

        /// <summary>Rollup key: hash of type plus the top stack frames.</summary>
        public string Fingerprint { get; set; } = string.Empty;

        /// <summary>Times this signature has been seen since FirstSeenAt.</summary>
        public int OccurrenceCount { get; set; } = 1;

        public DateTime FirstSeenAt { get; set; } = DateTime.UtcNow;
        public DateTime LastSeenAt { get; set; } = DateTime.UtcNow;
    }
}