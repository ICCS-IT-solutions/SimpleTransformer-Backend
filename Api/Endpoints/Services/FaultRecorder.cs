using System.Security.Cryptography;
using System.Text;
using Microsoft.EntityFrameworkCore;
using Serilog;
using SimpleTransformer.AppDb;

namespace SimpleTransformer.Api.Endpoints.Services
{
    /// <summary>
    /// Durable record of faults, independent of the rotating log file.
    ///
    /// Writes are rollups: a signature is hashed (type plus the top stack
    /// frames) and the existing row's OccurrenceCount is incremented rather than
    /// inserting a new row per throw. A fault that fires on every training step
    /// therefore costs one row and one increment.
    ///
    /// Every entry point is wrapped so a failure while recording a failure can
    /// never mask the original fault - the worst outcome is a lost fault record,
    /// not a second exception thrown from a catch block.
    /// </summary>
    public static class FaultRecorder
    {
        //Stacks are truncated: a full .NET stack runs 10-50 KB and this is only
        //ever read a row at a time in the fault viewer.
        private const int MaxStackChars = 8000;
        private const int MaxMessageChars = 2000;

        /// <summary>Number of leading stack frames folded into the fingerprint.</summary>
        private const int FingerprintFrames = 3;

        /// <summary>
        /// Persist a fault signature, rolling it up onto an existing row.
        ///
        /// <paramref name="dbFactory"/> is nullable on purpose: a startup fault can
        /// happen before the container is built, and losing the record of the one
        /// failure that explains why nothing else is running would be the worst
        /// possible outcome. A null factory simply skips the write.
        /// </summary>
        public static void Record(
            IDbContextFactory<AppDbContext>? dbFactory,
            Exception ex,
            string component,
            Guid? jobId = null,
            Guid? modelId = null)
        {
            if (ex == null)
                return;

            if (dbFactory == null)
            {
                Log.Error(
                    ex,
                    "Fault in {Component} could not be persisted: the database was not available.",
                    component);
                return;
            }

            //OutOfMemoryException is recorded only to the log. A database write
            //allocates, and competing with the run that just exhausted the heap
            //is the one case most likely to fail or to trigger a second OOM. The
            //job row already records the failure, so skip the write entirely.
            if (ex is OutOfMemoryException)
            {
                Log.Warning(
                    "Fault ({Component}) from job {JobId} not persisted: OutOfMemoryException is never written to the database.",
                    component, jobId);
                return;
            }

            try
            {
                string fingerprint = Fingerprint(ex);
                var now = DateTime.UtcNow;

                using var db = dbFactory.CreateDbContext();

                var existing = db.FaultEvents
                    .FirstOrDefault(x => x.Fingerprint == fingerprint);

                if (existing != null)
                {
                    existing.OccurrenceCount += 1;
                    existing.LastSeenAt = now;
                    //Keep the newest context: the same signature usually means the
                    //same job, but a rolled-up signature can outlive the job that
                    //first produced it.
                    if (jobId != null) existing.JobId = jobId.ToString();
                    if (modelId != null) existing.ModelId = modelId.ToString();

                    db.FaultEvents.Update(existing);
                }
                else
                {
                    db.FaultEvents.Add(new FaultEventEntry
                    {
                        FaultType = ex.GetType().Name,
                        Component = component,
                        Message = Truncate(ex.Message, MaxMessageChars),
                        StackTrace = Truncate(ex.StackTrace, MaxStackChars),
                        JobId = jobId?.ToString(),
                        ModelId = modelId?.ToString(),
                        Fingerprint = fingerprint,
                        OccurrenceCount = 1,
                        FirstSeenAt = now,
                        LastSeenAt = now,
                    });
                }

                db.SaveChanges();
            }
            catch (Exception recordEx)
            {
                //Swallowed deliberately: the original fault is what matters, and
                //throwing from here would replace it with an unrelated one.
                Log.Warning(
                    recordEx,
                    "Could not persist fault record for {Component} ({FaultType}).",
                    component, ex.GetType().Name);
            }
        }

        /// <summary>
        /// Stable identity for a failure shape. Uses the type plus the top frames
        /// so two OOMs from different jobs collapse to one row, while a different
        /// failure site stays distinct. The message is excluded because it
        /// carries volatile values (ids, sizes, counts) that would make every
        /// occurrence look unique.
        /// </summary>
        private static string Fingerprint(Exception ex)
        {
            var builder = new StringBuilder();
            builder.Append(ex.GetType().FullName);

            var frames = (ex.StackTrace ?? string.Empty)
                .Split('\n', StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < Math.Min(frames.Length, FingerprintFrames); i++)
            {
                //Strip line numbers - they shift whenever the file is edited and
                //would fragment an otherwise identical signature.
                string frame = frames[i].Trim();
                int at = frame.IndexOf(" in ", StringComparison.Ordinal);
                if (at > 0) frame = frame[..at];
                builder.Append('|').Append(frame);
            }

            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
            return Convert.ToHexString(hash);
        }

        private static string? Truncate(string? value, int max)
        {
            if (string.IsNullOrEmpty(value)) return value;
            return value.Length <= max ? value : value[..max];
        }
    }
}