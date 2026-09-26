using Microsoft.EntityFrameworkCore;
using Serilog;
using SimpleTransformer.Api.Endpoints.Controllers;
using SimpleTransformer.Api.Endpoints.Services.Extensions;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.AppDb;
using System.Text.Json;

namespace SimpleTransformer.Api.Endpoints.Services
{
    /// <summary>
    /// Named, reusable training corpora. Uploads are extracted + cleaned with
    /// the same primitives as job creation, then persisted under
    /// training-corpora/&lt;id&gt;/ alongside a report and the original sources.
    /// Corpora are immutable snapshots: no edit endpoint, only create/delete.
    /// </summary>
    public class TrainingCorpusService
    {
        private const string CorpusRoot = "training-corpora";
        private const string CorpusFileName = "corpus.txt";
        private const string ReportFileName = "preprocess-report.json";

        private readonly IDbContextFactory<AppDbContext> _dbFactory;

        public TrainingCorpusService(IDbContextFactory<AppDbContext> dbFactory)
        {
            _dbFactory = dbFactory;
        }

        public async Task<ApiResponse<CorpusDetailResponse>> CreateAsync(CorpusCreateRequest req)
        {
            var name = req.Name?.Trim() ?? string.Empty;
            if (string.IsNullOrEmpty(name))
            {
                return Fail<CorpusDetailResponse>("Corpus name must not be empty.", 400);
            }

            var files = req.TextFiles?
                .Where(f => f != null && f.Length > 0)
                .ToList() ?? new List<IFormFile>();

            if (files.Count == 0)
            {
                return Fail<CorpusDetailResponse>(
                    "At least one non-empty training file must be provided.", 400);
            }

            CorpusSourceFormat requestedFormat;
            try
            {
                requestedFormat = req.ResolveFormat();
            }
            catch (ArgumentException ex)
            {
                return Fail<CorpusDetailResponse>(ex.Message, 400);
            }

            await using var db = await _dbFactory.CreateDbContextAsync();
            if (await db.TrainingCorpora.AnyAsync(x => x.Name == name))
            {
                return Fail<CorpusDetailResponse>(
                    $"A corpus named '{name}' already exists. Choose a different name.", 409);
            }

            var extractions = new List<CorpusExtractionResult>();
            var sourceFileNames = new List<string>();
            foreach (var file in files)
            {
                var uploadName = Path.GetFileName(file!.FileName);
                sourceFileNames.Add(uploadName);

                try
                {
                    await using var uploadStream = file.OpenReadStream();
                    extractions.Add(await TrainingCorpusExtractor.ExtractAsync(
                        uploadStream, uploadName, requestedFormat, req.TextField, req.Template));
                }
                catch (JsonException ex)
                {
                    return Fail<CorpusDetailResponse>(
                        $"File '{uploadName}' is not valid JSON: {ex.Message}", 400);
                }
            }

            var (cleaned, report) = CorpusPreprocessor.Process(extractions, req.ToOptions());
            if (cleaned.Count == 0)
            {
                return Fail<CorpusDetailResponse>(
                    "No usable training text could be extracted from the uploaded files. " +
                    string.Join(" ", report.Warnings.Take(5)), 400);
            }

            var corpusId = Guid.NewGuid();
            var corpusDirectory = Path.Combine(CorpusRoot, corpusId.ToString());
            Directory.CreateDirectory(corpusDirectory);
            Directory.CreateDirectory(Path.Combine(corpusDirectory, "sources"));

            // Keep the original uploads as evidence.
            foreach (var file in files)
            {
                var dest = Path.Combine(corpusDirectory, "sources", Path.GetFileName(file!.FileName));
                await using var destStream = new FileStream(dest, FileMode.Create);
                await using var uploadStream = file.OpenReadStream();
                await uploadStream.CopyToAsync(destStream);
            }

            var corpusPath = Path.Combine(corpusDirectory, CorpusFileName);
            await using (var writer = new StreamWriter(corpusPath, append: false))
            {
                foreach (var document in cleaned)
                {
                    await writer.WriteAsync(document);
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync();
                }
            }

            var reportJson = JsonSerializer.Serialize(
                report, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(Path.Combine(corpusDirectory, ReportFileName), reportJson);

            var optionsJson = JsonSerializer.Serialize(req.ToOptions());
            var entry = new TrainingCorpusEntry
            {
                EntryId = corpusId,
                Name = name,
                Filename = CorpusFileName,
                Filepath = corpusDirectory,
                SourceFileNames = string.Join(", ", sourceFileNames),
                Format = requestedFormat.ToString(),
                OptionsJson = optionsJson,
                DocumentsIn = report.DocumentsIn,
                DocumentsOut = report.DocumentsOut,
                CharsIn = report.CharsIn,
                CharsOut = report.CharsOut,
                DuplicatesRemoved = report.DuplicatesRemoved,
                FilteredByLength = report.FilteredByLength,
                FilteredEmpty = report.FilteredEmpty,
                FileSize = new FileInfo(corpusPath).Length,
                DateUpdated = DateTime.UtcNow
            };

            await db.TrainingCorpora.AddAsync(entry);
            await db.SaveChangesAsync();

            Log.Information(
                "Training corpus '{Name}' created: {DocsOut} of {DocsIn} documents usable.",
                name, report.DocumentsOut, report.DocumentsIn);

            return Ok(ToDetail(entry, optionsJson, report.Warnings, usedByJobs: 0),
                $"Corpus '{name}' created: {report.DocumentsOut} of {report.DocumentsIn} documents usable.");
        }
        public async Task<ApiResponse<List<CorpusListItem>>> GetAvailableAsync()
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var entries = await db.TrainingCorpora
                .AsNoTracking()
                .OrderByDescending(x => x.DateCreated)
                .ToListAsync();

            // Only entries whose file still exists on disk are selectable.
            entries = entries
                .Where(x => File.Exists(Path.Combine(x.Filepath, x.Filename)))
                .ToList();

            var usage = await db.TrainingJobs
                .AsNoTracking()
                .Where(x => x.TrainingCorpusId != null)
                .GroupBy(x => x.TrainingCorpusId!.Value)
                .Select(g => new { CorpusId = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.CorpusId, x => x.Count);

            return Ok(entries
                .Select(x => ToListItem(x, usage.GetValueOrDefault(x.EntryId, 0)))
                .ToList(), "Corpora found.");
        }

        public async Task<ApiResponse<CorpusDetailResponse>> GetOneAsync(Guid corpusId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var entry = await db.TrainingCorpora
                .AsNoTracking()
                .FirstOrDefaultAsync(x => x.EntryId == corpusId);

            if (entry == null)
            {
                return Fail<CorpusDetailResponse>("Corpus not found.", 404);
            }

            var usedByJobs = await db.TrainingJobs
                .AsNoTracking()
                .CountAsync(x => x.TrainingCorpusId == corpusId);

            var warnings = await ReadReportWarnings(entry);
            return Ok(ToDetail(entry, entry.OptionsJson, warnings, usedByJobs), "Corpus found.");
        }
        public async Task<ApiResponse<CorpusDetailResponse>> DeleteAsync(Guid corpusId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var entry = await db.TrainingCorpora
                .FirstOrDefaultAsync(x => x.EntryId == corpusId);

            if (entry == null)
            {
                return Fail<CorpusDetailResponse>("Corpus not found.", 404);
            }

            var usedByJobs = await db.TrainingJobs
                .CountAsync(x => x.TrainingCorpusId == corpusId);

            // Jobs keep their own copy of the corpus text, so deletion only
            // clears their reference (SetNull also guards this at the DB level).
            db.TrainingCorpora.Remove(entry);
            await db.SaveChangesAsync();

            try
            {
                if (Directory.Exists(entry.Filepath))
                {
                    Directory.Delete(entry.Filepath, recursive: true);
                }
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "Corpus directory '{Path}' could not be removed.", entry.Filepath);
            }

            var noun = usedByJobs == 1 ? "job" : "jobs";
            return Ok(ToDetail(entry, entry.OptionsJson, new List<string>(), usedByJobs),
                usedByJobs > 0
                    ? $"Corpus '{entry.Name}' deleted. {usedByJobs} existing {noun} keep their own copy of its text."
                    : $"Corpus '{entry.Name}' deleted.");
        }

        private static async Task<List<string>> ReadReportWarnings(TrainingCorpusEntry entry)
        {
            try
            {
                var reportPath = Path.Combine(entry.Filepath, "preprocess-report.json");
                if (!File.Exists(reportPath))
                {
                    return new List<string>();
                }

                var json = await File.ReadAllTextAsync(reportPath);
                var report = JsonSerializer.Deserialize<CorpusPreprocessReport>(json);
                return report?.Warnings.Take(20).ToList() ?? new List<string>();
            }
            catch
            {
                return new List<string>();
            }
        }
        private static CorpusListItem ToListItem(TrainingCorpusEntry x, int usedByJobs) => new()
        {
            EntryId = x.EntryId,
            Name = x.Name,
            SourceFileNames = x.SourceFileNames,
            Format = x.Format,
            DocumentsIn = x.DocumentsIn,
            DocumentsOut = x.DocumentsOut,
            CharsIn = x.CharsIn,
            CharsOut = x.CharsOut,
            DuplicatesRemoved = x.DuplicatesRemoved,
            FilteredByLength = x.FilteredByLength,
            FilteredEmpty = x.FilteredEmpty,
            FileSize = x.FileSize,
            UsedByJobs = usedByJobs,
            DateCreated = x.DateCreated
        };

        private static CorpusDetailResponse ToDetail(
            TrainingCorpusEntry x, string optionsJson, List<string> warnings, int usedByJobs)
        {
            var detail = new CorpusDetailResponse
            {
                EntryId = x.EntryId,
                Name = x.Name,
                SourceFileNames = x.SourceFileNames,
                Format = x.Format,
                DocumentsIn = x.DocumentsIn,
                DocumentsOut = x.DocumentsOut,
                CharsIn = x.CharsIn,
                CharsOut = x.CharsOut,
                DuplicatesRemoved = x.DuplicatesRemoved,
                FilteredByLength = x.FilteredByLength,
                FilteredEmpty = x.FilteredEmpty,
                FileSize = x.FileSize,
                UsedByJobs = usedByJobs,
                DateCreated = x.DateCreated,
                OptionsJson = optionsJson,
                Warnings = warnings
            };

            try
            {
                detail.Options = JsonSerializer.Deserialize<JsonElement>(optionsJson);
            }
            catch
            {
                detail.Options = null;
            }

            return detail;
        }

        private static ApiResponse<T> Ok<T>(T data, string message) => new()
        {
            Message = message,
            Status = ResponseStatus.Success,
            StatusCode = 200,
            Data = data
        };

        private static ApiResponse<T> Fail<T>(string message, int statusCode) => new()
        {
            Message = message,
            Status = ResponseStatus.Failure,
            StatusCode = statusCode
        };
    }
}
