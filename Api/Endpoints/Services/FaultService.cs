using Microsoft.EntityFrameworkCore;
using SimpleTransformer.AppDb;

namespace SimpleTransformer.Api.Endpoints.Services
{
    /// <summary>
    /// Read side of the fault table. Writes go through <see cref="FaultRecorder"/>,
    /// which is static so the detached training loop can record without a
    /// dependency on DI.
    ///
    /// Faults are never pruned here on a schedule: rows are bounded by the number
    /// of distinct failure signatures, not by volume, so a fault storm rolls up
    /// onto one row instead of filling the table.
    /// </summary>
    public class FaultService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;

        public FaultService(IDbContextFactory<AppDbContext> dbFactory)
        {
            _dbFactory = dbFactory;
        }

        /// <summary>Most recent faults first, heaviest occurrence count first.</summary>
        public async Task<List<FaultSummary>> GetRecentAsync(int take, string? component)
        {
            take = Math.Clamp(take <= 0 ? 100 : take, 1, 500);

            await using var db = await _dbFactory.CreateDbContextAsync();

            var query = db.FaultEvents.AsNoTracking();

            if (!string.IsNullOrWhiteSpace(component))
                query = (IQueryable<FaultEventEntry>)query.Where(
                    x => x.Component == component);

            return await query
                .OrderByDescending(x => x.LastSeenAt)
                .Take(take)
                .Select(x => new FaultSummary
                {
                    EntryId = x.EntryId,
                    FaultType = x.FaultType,
                    Component = x.Component,
                    Message = x.Message,
                    JobId = x.JobId,
                    ModelId = x.ModelId,
                    OccurrenceCount = x.OccurrenceCount,
                    FirstSeenAt = x.FirstSeenAt,
                    LastSeenAt = x.LastSeenAt,
                })
                .ToListAsync();
        }

        /// <summary>Total occurrences grouped by type, for the "what keeps happening" view.</summary>
        public async Task<List<FaultCount>> GetCountsAsync()
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            return await db.FaultEvents
                .AsNoTracking()
                .GroupBy(x => new { x.FaultType, x.Component })
                .Select(g => new FaultCount
                {
                    FaultType = g.Key.FaultType,
                    Component = g.Key.Component,
                    Count = g.Sum(x => x.OccurrenceCount),
                    Signatures = g.Count(),
                    LastSeenAt = g.Max(x => x.LastSeenAt),
                })
                .OrderByDescending(x => x.Count)
                .ToListAsync();
        }

        /// <summary>Full detail for one signature, including the stack trace.</summary>
        public async Task<FaultDetail?> GetOneAsync(Guid entryId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            return await db.FaultEvents
                .AsNoTracking()
                .Where(x => x.EntryId == entryId)
                .Select(x => new FaultDetail
                {
                    EntryId = x.EntryId,
                    FaultType = x.FaultType,
                    Component = x.Component,
                    Message = x.Message,
                    StackTrace = x.StackTrace,
                    JobId = x.JobId,
                    ModelId = x.ModelId,
                    OccurrenceCount = x.OccurrenceCount,
                    FirstSeenAt = x.FirstSeenAt,
                    LastSeenAt = x.LastSeenAt,
                })
                .FirstOrDefaultAsync();
        }
    }

    /// <summary>A fault signature without its stack trace.</summary>
    public class FaultSummary
    {
        public Guid EntryId { get; set; }
        public string FaultType { get; set; } = string.Empty;
        public string Component { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string? JobId { get; set; }
        public string? ModelId { get; set; }
        public int OccurrenceCount { get; set; }
        public DateTime FirstSeenAt { get; set; }
        public DateTime LastSeenAt { get; set; }
    }

    /// <summary>One signature's total occurrences.</summary>
    public class FaultCount
    {
        public string FaultType { get; set; } = string.Empty;
        public string Component { get; set; } = string.Empty;

        /// <summary>Summed occurrences across every signature of this type.</summary>
        public int Count { get; set; }

        /// <summary>How many distinct call sites produced this type.</summary>
        public int Signatures { get; set; }

        public DateTime LastSeenAt { get; set; }
    }

    /// <summary>Full fault detail, stack trace included.</summary>
    public class FaultDetail : FaultSummary
    {
        public string? StackTrace { get; set; }
    }
}