using System.Text.Json;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    public enum CorpusSourceFormat
    {
        Auto,
        Txt,
        Json,
        Jsonl
    }

    public sealed class CorpusExtractionResult
    {
        public string FileName { get; init; } = string.Empty;
        public CorpusSourceFormat ResolvedFormat { get; init; }
        public List<string> Documents { get; init; } = new();
        public List<string> Warnings { get; init; } = new();
    }

    /// <summary>
    /// Extracts plain-text documents from .txt, .json and .jsonl/.ndjson uploads.
    /// JSON shapes are deserialized (System.Text.Json also unescapes \n, \t,
    /// \uXXXX) so the training pipeline never sees container syntax.
    /// </summary>
    public static class TrainingCorpusExtractor
    {
        private static readonly string[] DefaultTextFields = { "text", "content", "body" };

        public static async Task<CorpusExtractionResult> ExtractAsync(
            Stream stream,
            string fileName,
            CorpusSourceFormat format = CorpusSourceFormat.Auto,
            string? textField = null,
            string? template = null,
            CancellationToken cancellationToken = default)
        {
            var resolved = format == CorpusSourceFormat.Auto
                ? ResolveFromExtension(fileName)
                : format;

            var result = new CorpusExtractionResult
            {
                FileName = fileName,
                ResolvedFormat = resolved
            };

            switch (resolved)
            {
                case CorpusSourceFormat.Txt:
                    using (var reader = new StreamReader(stream, leaveOpen: true))
                    {
                        result.Documents.Add(await reader.ReadToEndAsync(cancellationToken));
                    }
                    break;

                case CorpusSourceFormat.Json:
                    using (var reader = new StreamReader(stream, leaveOpen: true))
                    {
                        ExtractFromJsonDocument(
                            await reader.ReadToEndAsync(cancellationToken),
                            fileName, textField, template, result);
                    }
                    break;

                case CorpusSourceFormat.Jsonl:
                    using (var lineReader = new StreamReader(stream, leaveOpen: true))
                    {
                        string? line;
                        int lineNumber = 0;
                        while ((line = await lineReader.ReadLineAsync(cancellationToken)) != null)
                        {
                            lineNumber++;
                            if (string.IsNullOrWhiteSpace(line))
                            {
                                continue;
                            }

                            try
                            {
                                ExtractFromJsonLine(line, fileName, lineNumber, textField, template, result);
                            }
                            catch (JsonException ex)
                            {
                                result.Warnings.Add(
                                    $"{fileName}:{lineNumber}: skipped malformed JSON line ({ex.Message}).");
                            }
                        }
                    }
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(format), $"Unsupported corpus format '{format}'.");
            }

            return result;
        }

        public static CorpusSourceFormat ResolveFromExtension(string fileName)
        {
            return Path.GetExtension(fileName).ToLowerInvariant() switch
            {
                ".json" => CorpusSourceFormat.Json,
                ".jsonl" or ".ndjson" => CorpusSourceFormat.Jsonl,
                _ => CorpusSourceFormat.Txt,
            };
        }

        private static void ExtractFromJsonDocument(
            string content,
            string fileName,
            string? textField,
            string? template,
            CorpusExtractionResult result)
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                result.Warnings.Add($"{fileName}: file is empty, nothing extracted.");
                return;
            }

            using var doc = JsonDocument.Parse(content);
            var root = doc.RootElement;

            if (root.ValueKind == JsonValueKind.Array)
            {
                int index = 0;
                foreach (var element in root.EnumerateArray())
                {
                    ExtractElement(element, fileName, $"[{index}]", textField, template, result);
                    index++;
                }

                if (index == 0)
                {
                    result.Warnings.Add($"{fileName}: JSON array is empty, nothing extracted.");
                }

                return;
            }

            ExtractElement(root, fileName, string.Empty, textField, template, result);
        }

        private static void ExtractFromJsonLine(
            string line,
            string fileName,
            int lineNumber,
            string? textField,
            string? template,
            CorpusExtractionResult result)
        {
            using var doc = JsonDocument.Parse(line);
            ExtractElement(doc.RootElement, fileName, $":{lineNumber}", textField, template, result);
        }

        private static void ExtractElement(
            JsonElement element,
            string fileName,
            string location,
            string? textField,
            string? template,
            CorpusExtractionResult result)
        {
            // Plain string values (arrays of strings, bare string lines).
            if (element.ValueKind == JsonValueKind.String)
            {
                var value = element.GetString();
                if (!string.IsNullOrWhiteSpace(value))
                {
                    result.Documents.Add(value);
                }
                else
                {
                    result.Warnings.Add($"{fileName}{location}: skipped empty string value.");
                }

                return;
            }

            if (element.ValueKind != JsonValueKind.Object)
            {
                result.Warnings.Add(
                    $"{fileName}{location}: skipped {DescribeKind(element.ValueKind)} value; " +
                    "expected a string or an object with a text field.");
                return;
            }

            // Template rendering for multi-field records (e.g. prompt/completion pairs).
            if (!string.IsNullOrWhiteSpace(template))
            {
                result.Documents.Add(RenderTemplate(element, template));
                return;
            }

            var fields = ResolveTextFields(textField);
            foreach (var field in fields)
            {
                if (element.TryGetProperty(field, out var prop) &&
                    prop.ValueKind == JsonValueKind.String)
                {
                    var value = prop.GetString();
                    if (!string.IsNullOrWhiteSpace(value))
                    {
                        result.Documents.Add(value);
                        return;
                    }
                }
            }

            result.Warnings.Add(
                $"{fileName}{location}: object has no usable text field " +
                $"(looked for: {string.Join(", ", fields)}).");
        }
        private static string RenderTemplate(JsonElement element, string template)
        {
            var values = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
            foreach (var prop in element.EnumerateObject())
            {
                values[prop.Name] = prop.Value.ValueKind switch
                {
                    JsonValueKind.String => prop.Value.GetString(),
                    JsonValueKind.Number => prop.Value.GetRawText(),
                    JsonValueKind.True => "true",
                    JsonValueKind.False => "false",
                    JsonValueKind.Null => null,
                    _ => prop.Value.GetRawText()
                };
            }

            // "{prompt}\n\n{completion}" style placeholders; unknown keys render empty.
            return System.Text.RegularExpressions.Regex.Replace(
                template,
                @"\{([^{}]+)\}",
                m => values.TryGetValue(m.Groups[1].Value.Trim(), out var v) ? v ?? string.Empty : string.Empty);
        }

        private static IReadOnlyList<string> ResolveTextFields(string? textField)
        {
            if (!string.IsNullOrWhiteSpace(textField))
            {
                return new[] { textField.Trim() };
            }

            return DefaultTextFields;
        }

        private static string DescribeKind(JsonValueKind kind) => kind switch
        {
            JsonValueKind.Number => "numeric",
            JsonValueKind.True or JsonValueKind.False => "boolean",
            JsonValueKind.Null => "null",
            JsonValueKind.Array => "array",
            _ => kind.ToString().ToLowerInvariant()
        };
    }
}
