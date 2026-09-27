using System.Text.Json;

namespace SimpleTransformer.Model.Tokenizer
{
    /// <summary>
    /// Reads vocabulary source files as plain text. .txt/.log pass through;
    /// .json/.jsonl/.ndjson are deserialized with the same field rules as the
    /// training corpus extractor (text/content/body, one record per line for
    /// .jsonl), so a vocabulary can be compiled from the same sources used
    /// for training.
    /// </summary>
    public static class VocabularySourceReader
    {
        private static readonly string[] TextFields = { "text", "content", "body" };

        public static void ValidateExtension(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            bool ok = ext is ".txt" or ".log" or ".json" or ".jsonl" or ".ndjson";
            if (!ok)
            {
                throw new ArgumentException(
                    $"Vocabulary source must be a .txt, .log, .json or .jsonl file: {path}",
                    nameof(path));
            }
        }

        /// <summary>
        /// Guard against inputs an algorithm cannot hold in memory. The message
        /// names the cap and the way out (sample, or use SentencePiece).
        /// </summary>
        public static void ValidateTotalSize(IEnumerable<string> paths, long maxBytes, string algorithm)
        {
            long total = paths.Sum(p => new FileInfo(p).Length);
            if (total > maxBytes)
            {
                throw new ArgumentException(
                    $"{algorithm} vocabulary compilation is capped at {maxBytes / (1024 * 1024)}MB of source text " +
                    $"({total / (1024 * 1024)}MB provided) because it must hold the corpus in memory. " +
                    "Compile from a smaller sample, or use SentencePiece for larger inputs.");
            }
        }

        public static string ReadAllText(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            if (ext is ".json" or ".jsonl" or ".ndjson")
            {
                return ExtractJsonText(path, ext);
            }

            return File.ReadAllText(path);
        }

        private static string ExtractJsonText(string path, string ext)
        {
            if (ext == ".json")
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                var docs = new List<string>();
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var el in doc.RootElement.EnumerateArray())
                    {
                        var d = ExtractElement(el);
                        if (d != null) docs.Add(d);
                    }
                }
                else
                {
                    var d = ExtractElement(doc.RootElement);
                    if (d != null) docs.Add(d);
                }

                return string.Join("\n", docs);
            }

            var lines = new List<string>();
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var d = ExtractElement(doc.RootElement);
                    if (d != null) lines.Add(d);
                }
                catch (JsonException)
                {
                    //Skip malformed lines, matching corpus extraction behavior.
                }
            }

            return string.Join("\n", lines);
        }

        private static string? ExtractElement(JsonElement el)
        {
            if (el.ValueKind == JsonValueKind.String) return el.GetString();
            if (el.ValueKind != JsonValueKind.Object) return null;
            foreach (var f in TextFields)
            {
                if (el.TryGetProperty(f, out var p) && p.ValueKind == JsonValueKind.String)
                {
                    var v = p.GetString();
                    if (!string.IsNullOrWhiteSpace(v)) return v;
                }
            }

            return null;
        }
    }
}
