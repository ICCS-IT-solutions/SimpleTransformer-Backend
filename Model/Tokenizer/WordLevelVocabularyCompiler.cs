using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class WordLevelVocabularyCompiler : IVocabularyCompiler
    {
        public VocabularyCompilationResult BuildFromRawTextFile(string sourceDir, string filename, int targetVocabSize = 5000)
        {
            return BuildFromRawTextFiles(sourceDir, new[] { filename }, targetVocabSize);
        }
        private static readonly Dictionary<string, int> _specialTokens = new()
        {
            [SpecialTokens.Pad] = 0,
            [SpecialTokens.Unknown] = 1,
            [SpecialTokens.BeginningOfSequence] = 2,
            [SpecialTokens.EndOfSequence] = 3,
            [SpecialTokens.Mask] = 4,
        };

        public VocabularyCompilationResult BuildFromRawTextFiles(string sourceDirectory, IEnumerable<string> filenames, int targetVocabSize = 5000)
        {
            var frequencies = new Dictionary<string, long>();
            long totalOccurrences = 0;

            foreach (string filename in filenames)
            {
                var sourcePath = Path.Combine(sourceDirectory, filename);
                
                ValidateSourceFile(sourcePath);

                string text = VocabularySourceReader.ReadAllText(sourcePath);

                CompileTokens(text, frequencies, ref totalOccurrences);
            }

            // Frequency-ranked truncation: keep the most frequent types so the
            // requested size is honored even on large corpora. Specials first,
            // then most frequent; ties broken alphabetically for determinism.
            var vocabulary = new Dictionary<string, int>(_specialTokens);
            int nextId = vocabulary.Count;
            int capacity = Math.Max(0, targetVocabSize - vocabulary.Count);

            long keptOccurrences = 0;
            foreach (var (word, count) in frequencies
                .OrderByDescending(kvp => kvp.Value)
                .ThenBy(kvp => kvp.Key, StringComparer.Ordinal)
                .Take(capacity))
            {
                vocabulary[word] = nextId++;
                keptOccurrences += count;
            }

            return new VocabularyCompilationResult(
                new Vocabulary(vocabulary),
                null,
                typesSeen: frequencies.Count,
                tokenOccurrences: totalOccurrences,
                keptOccurrences: keptOccurrences);
        }

        private static void CompileTokens(
            string text,
            Dictionary<string, long> frequencies,
            ref long totalOccurrences)
        {
            foreach (string word in TokenizationUtilities.TokenizeRawText(text))
            {
                frequencies[word] = frequencies.GetValueOrDefault(word, 0) + 1;
                totalOccurrences++;
            }
        }
        private static void ValidateSourceFile(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    "Vocabulary source file not found.",
                    path);

            VocabularySourceReader.ValidateExtension(path);
        }        
    }
}