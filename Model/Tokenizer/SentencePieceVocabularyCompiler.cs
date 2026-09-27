using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class SentencePieceVocabularyCompiler : IVocabularyCompiler
    {
        private const char MetaSymbol = ' '; // U+2581 (Lower One Eighth Block)
        private const int MaxSubwordLength = 16; 
        private const int MinCandidateFrequency = 2; // Filter ultra-rare noise early

        private static readonly Dictionary<string, int> _specialTokens = new()
        {
            [SpecialTokens.Pad] = 0,
            [SpecialTokens.Unknown] = 1,
            [SpecialTokens.BeginningOfSequence] = 2,
            [SpecialTokens.EndOfSequence] = 3,
            [SpecialTokens.Mask] = 4,
        };

        public VocabularyCompilationResult BuildFromRawTextFile(
            string sourceDirectory,
            string filename,
            int targetVocabSize = 5000)
        {
            return BuildFromRawTextFiles(
                sourceDirectory,
                new[] { filename },
                targetVocabSize);
        }

        public VocabularyCompilationResult BuildFromRawTextFiles(
            string sourceDirectory,
            IEnumerable<string> filenames,
            int targetVocabSize = 5000)
        {
            List<string> texts = new();

            var filePaths = filenames.Select(filename => Path.Combine(sourceDirectory, filename)).ToList();
            foreach (var sourcePath in filePaths)
            {
                ValidateSourceFile(sourcePath);
            }

            foreach (var sourcePath in filePaths)
            {
                texts.Add(VocabularySourceReader.ReadAllText(sourcePath));
            }

            return TrainSentencePiece(texts, targetVocabSize);
        }

        private VocabularyCompilationResult TrainSentencePiece(IEnumerable<string> texts, int targetVocabSize)
        {
            var vocabulary = new Dictionary<string, int>(_specialTokens);
            int nextId = vocabulary.Count;

            // 1. Pre-process and normalize input blocks
            List<string> normalizedTokens = new();
            var baseCharacters = new HashSet<string>();

            foreach (var text in texts)
            {
                if (string.IsNullOrWhiteSpace(text)) continue;

                // Split by newlines first to keep control characters clean
                string[] lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);

                foreach (string line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    string cleanLine = line
                        .Replace('\t', ' ')
                        .Replace('\u00A0', ' ');

                    string normalized = MetaSymbol + cleanLine.Replace(' ', MetaSymbol);
                    normalizedTokens.Add(normalized);

                    // Collect single base characters
                    foreach (char c in normalized)
                    {
                        baseCharacters.Add(c.ToString());
                    }
                }
            }

            // Register base single characters first (guarantees 100% coverage)
            foreach (var ch in baseCharacters)
            {
                if (vocabulary.TryAdd(ch, nextId))
                {
                    nextId++;
                }
            }

            // 2. High-performance candidate counting using ReadOnlySpan<char>
            var candidateCounts = new Dictionary<string, int>(capacity: 100_000);

            foreach (var text in normalizedTokens)
            {
                ReadOnlySpan<char> span = text.AsSpan();
                int len = span.Length;

                for (int i = 0; i < len; i++)
                {
                    int maxLen = Math.Min(MaxSubwordLength, len - i);
                    for (int subLen = 1; subLen <= maxLen; subLen++)
                    {
                        // Slicing ReadOnlySpan is allocation-free
                        ReadOnlySpan<char> subSpan = span.Slice(i, subLen);

                        // Only allocate string when inserting into frequency map
                        string candidate = subSpan.ToString();
                        candidateCounts[candidate] = candidateCounts.GetValueOrDefault(candidate, 0) + 1;
                    }
                }
            }

            // 3. Filter noise and rank subwords by score (Frequency * Length multiplier)
            int remainingCapacity = targetVocabSize - vocabulary.Count;
            if (remainingCapacity > 0)
            {
                var topSubwords = candidateCounts
                    .Where(kvp => kvp.Value >= MinCandidateFrequency && !vocabulary.ContainsKey(kvp.Key))
                    // Score = Frequency * SubwordLength (favors longer meaningful tokens over raw character bigrams)
                    .OrderByDescending(kvp => (long)kvp.Value * kvp.Key.Length)
                    .ThenBy(kvp => kvp.Key)
                    .Take(remainingCapacity)
                    .Select(kvp => kvp.Key);

                foreach (string subword in topSubwords)
                {
                    if (vocabulary.TryAdd(subword, nextId))
                    {
                        nextId++;
                    }
                }
            }

            return new VocabularyCompilationResult(new Vocabulary(vocabulary), merges: null);
        }

        private static void ValidateSourceFile(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Vocabulary source file not found.", path);

            VocabularySourceReader.ValidateExtension(path);
        }
    }
}