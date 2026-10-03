using System.Text.RegularExpressions;

namespace SimpleTransformer.Api.Endpoints.Services
{
    /// <summary>
    /// Reads the rolling Serilog files in <c>logs/</c> for the diagnostics view.
    ///
    /// Two things this deliberately does NOT do: it never returns a whole file
    /// (a busy day is ~12 MB and the UI only shows the tail), and it never trusts
    /// a client-supplied filename (see ResolveLogPath).
    /// </summary>
    public class LogService
    {
        // A log line is: 2026-10-02 18:37:55.948 +02:00 [INF] message...
        private static readonly Regex LinePattern = new(
            @"^(?<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} [+-]\d{2}:\d{2}) \[(?<lvl>[A-Z]{3})\] ?(?<msg>.*)$",
            RegexOptions.Compiled);

        /// <summary>Upper bound on bytes read from the end of a file per request.</summary>
        private const int MaxReadBytes = 2 * 1024 * 1024;

        /// <summary>Upper bound on entries returned.</summary>
        private const int MaxEntries = 2000;

        private readonly string _logDirectory;

        public LogService()
        {
            //Relative, matching the sink path in Program.ConfigureLogging so both
            //resolve to the same directory regardless of the working directory.
            _logDirectory = Path.GetFullPath("logs");
        }

        /// <summary>Log files currently on disk, newest first.</summary>
        public List<LogFileEntry> GetFiles()
        {
            var files = new List<LogFileEntry>();

            if (!Directory.Exists(_logDirectory))
                return files;

            foreach (var path in Directory.EnumerateFiles(_logDirectory, "server-*.log"))
            {
                var info = new FileInfo(path);
                files.Add(new LogFileEntry
                {
                    Name = info.Name,
                    SizeBytes = info.Length,
                    LastModified = info.LastWriteTimeUtc,
                });
            }

            //The name carries the date stamp, so ordering by name descending is
            //also ordering by date descending.
            return files
                .OrderByDescending(x => x.Name, StringComparer.Ordinal)
                .ToList();
        }

        /// <summary>
        /// Tail (and optionally filter) one file, or every retained file when
        /// <paramref name="file"/> is null/empty - the "search all days" mode.
        /// </summary>
        public LogSearchResult Read(string? file, string? search, string? level, int lines)
        {
            //Clamped before use: the limit is client-controlled and an unbounded
            //value would be an easy way to pin the server's memory.
            lines = Math.Clamp(lines <= 0 ? 200 : lines, 1, MaxEntries);

            var targets = string.IsNullOrWhiteSpace(file)
                ? GetFiles()
                : new List<LogFileEntry> { ResolveLogFile(file) };

            var entries = new List<LogEntry>();
            var truncated = false;

            foreach (var target in targets)
            {
                string path;
                try
                {
                    path = ResolveLogPath(target.Name);
                }
                catch (ArgumentException)
                {
                    //One unreadable file must not fail the whole request.
                    continue;
                }

                if (!File.Exists(path))
                    continue;

                var room = MaxEntries - entries.Count;
                if (room <= 0)
                {
                    truncated = true;
                    break;
                }

                var (read, partial) = TailEntries(path, search, level, lines, room);
                entries.AddRange(read);
                truncated |= partial;
            }

            return new LogSearchResult
            {
                Entries = entries,
                TotalMatched = entries.Count,
                Truncated = truncated,
            };
        }

        /// <summary>
        /// Reads at most <paramref name="limit"/> matching entries from the end of
        /// a file, and reports whether the result is partial so the UI can say so
        /// rather than implying completeness.
        /// </summary>
        private static (List<LogEntry> Entries, bool Truncated) TailEntries(
            string path, string? search, string? level, int limit, int room)
        {
            string[] rawLines;
            bool windowed;

            try
            {
                windowed = new FileInfo(path).Length > MaxReadBytes;
                rawLines = ReadTailLines(path, windowed).ToArray();
            }
            catch (IOException)
            {
                return (new List<LogEntry>(), false);
            }
            catch (UnauthorizedAccessException)
            {
                return (new List<LogEntry>(), false);
            }

            var matched = new List<LogEntry>();
            //Entries accumulate oldest-first while scanning forward, so the cap
            //must keep the TAIL: a plain Take() would keep the oldest and drop
            //what the user actually asked to see.
            foreach (var entry in MergeContinuations(rawLines))
            {
                if (!Matches(entry, search, level))
                    continue;

                matched.Add(entry);
                if (matched.Count > limit)
                    matched.RemoveAt(0);
            }

            return (matched, windowed || matched.Count > limit);
        }

        /// <summary>
        /// Reads the tail of a file as raw lines, seeking backwards from the end
        /// when the file exceeds the window. FileShare.ReadWrite because Serilog
        /// holds today's file open for writing and an exclusive open would throw.
        /// </summary>
        private static IEnumerable<string> ReadTailLines(string path, bool windowed)
        {
            using var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite,
                bufferSize: 64 * 1024,
                FileOptions.SequentialScan);

            if (windowed)
                stream.Seek(-MaxReadBytes, SeekOrigin.End);

            using var reader = new StreamReader(stream, System.Text.Encoding.UTF8);
            var lines = new List<string>();
            string? line;
            while ((line = reader.ReadLine()) != null)
                lines.Add(line);

            //A backwards seek can land mid-line, making the first entry a fragment
            //rather than a real record.
            if (windowed && lines.Count > 0)
                lines.RemoveAt(0);

            return lines;
        }

        /// <summary>
        /// Joins continuation lines onto the entry that introduced them. EF Core
        /// writes SQL payloads containing real newlines, so the line after a
        /// timestamped entry is usually a fragment of it, not its own record.
        /// </summary>
        private static IEnumerable<LogEntry> MergeContinuations(IEnumerable<string> lines)
        {
            LogEntry? current = null;

            foreach (var line in lines)
            {
                var match = LinePattern.Match(line);
                if (match.Success)
                {
                    if (current != null)
                        yield return current;

                    current = new LogEntry
                    {
                        Timestamp = match.Groups["ts"].Value,
                        Level = match.Groups["lvl"].Value,
                        Message = match.Groups["msg"].Value,
                    };
                    continue;
                }

                if (current != null)
                {
                    current.Message += Environment.NewLine + line;
                }
                else
                {
                    //Before any timestamp: kept rather than silently dropped.
                    yield return new LogEntry { Timestamp = "", Level = "INF", Message = line };
                }
            }

            if (current != null)
                yield return current;
        }

        private static bool Matches(LogEntry entry, string? search, string? level)
        {
            if (!string.IsNullOrWhiteSpace(level) &&
                !string.Equals(entry.Level, level, StringComparison.OrdinalIgnoreCase))
                return false;

            if (string.IsNullOrWhiteSpace(search))
                return true;

            //Case-insensitive contains over the already-parsed message, which has
            //continuations folded in so a match inside a SQL body still hits.
            return entry.Message.Contains(search, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Resolves a log filename inside the logs directory, or throws.
        ///
        /// Path.GetFileName strips any directory component the client supplied, so
        /// "../../config.ini" collapses to "config.ini" and then fails the
        /// containment check. This is the only thing between the endpoint and
        /// arbitrary file read, so it lives here rather than at the call site.
        /// </summary>
        private string ResolveLogPath(string fileName)
        {
            string safeName = Path.GetFileName(fileName ?? string.Empty);
            if (string.IsNullOrWhiteSpace(safeName))
                throw new ArgumentException("A log file name is required.", nameof(fileName));

            string full = Path.GetFullPath(Path.Combine(_logDirectory, safeName));

            string root = _logDirectory.EndsWith(Path.DirectorySeparatorChar)
                ? _logDirectory
                : _logDirectory + Path.DirectorySeparatorChar;

            if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Log file name escapes the logs directory.", nameof(fileName));

            if (!safeName.EndsWith(".log", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Only .log files can be read.", nameof(fileName));

            return full;
        }

        private LogFileEntry ResolveLogFile(string fileName)
        {
            string full = ResolveLogPath(fileName);
            var info = new FileInfo(full);

            return new LogFileEntry
            {
                Name = info.Name,
                SizeBytes = info.Exists ? info.Length : 0,
                LastModified = info.Exists ? info.LastWriteTimeUtc : DateTime.UtcNow,
            };
        }
    }

    /// <summary>One retained log file on disk.</summary>
    public class LogFileEntry
    {
        public string Name { get; set; } = string.Empty;
        public long SizeBytes { get; set; }
        public DateTime LastModified { get; set; }
    }

    /// <summary>A parsed log entry, with continuations already folded in.</summary>
    public class LogEntry
    {
        public string Timestamp { get; set; } = string.Empty;

        /// <summary>Serilog's three-letter level token: DBG, INF, WRN, ERR, FTL.</summary>
        public string Level { get; set; } = string.Empty;

        public string Message { get; set; } = string.Empty;
    }

    /// <summary>A page of log entries plus whether the result was clipped.</summary>
    public class LogSearchResult
    {
        public List<LogEntry> Entries { get; set; } = new();
        public int TotalMatched { get; set; }

        /// <summary>
        /// True when the source exceeded the read window or the entry cap, so the
        /// UI can label results as partial.
        /// </summary>
        public bool Truncated { get; set; }
    }
}