using SimpleTransformer.AppDb;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    /// <summary>
    /// Reads vocabulary artifacts from disk and checks them against the model that
    /// is meant to use them.
    /// <para>
    /// The file is the source of truth for the token count:
    /// <see cref="VocabularyEntry.NumTokens"/> is compile-time metadata and can
    /// drift (a file replaced by hand, an interrupted copy), while the embedding
    /// matrix and every checkpoint are sized from the configuration.
    /// </para>
    /// </summary>
    public static class VocabularyArtifacts
    {
        /// <summary>Path of the compiled vocabulary artifact for an entry.</summary>
        public static string ResolvePath(VocabularyEntry entry) =>
            Path.Combine(entry.Filepath, entry.Filename);

        /// <summary>
        /// Loads the artifact for an entry, or null when it is missing, unreadable
        /// or not a valid vocabulary. Never throws.
        /// </summary>
        public static Vocabulary? TryLoad(VocabularyEntry entry)
        {
            try
            {
                var path = ResolvePath(entry);
                if (!File.Exists(path))
                {
                    return null;
                }

                var vocabulary = new JsonVocabularyLoader().LoadFromFile(path);
                return vocabulary.Count > 0 ? vocabulary : null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Token count read from the artifact, or null when the file is missing,
        /// unreadable or not a valid vocabulary. Never throws.
        /// </summary>
        public static int? TryReadTokenCount(VocabularyEntry entry) =>
            TryLoad(entry)?.Count;

        /// <summary>
        /// True when an artifact holds exactly the same token to id map as the
        /// vocabulary the process has loaded. A size check alone is not enough: two
        /// vocabularies can have the same token count and completely different ids,
        /// which is what turns output into garbage instead of throwing.
        /// </summary>
        public static bool MatchesLive(Vocabulary artifact, Vocabulary live)
        {
            if (artifact.Count != live.Count)
            {
                return false;
            }

            foreach (var (token, id) in artifact.TokenToId)
            {
                if (!live.TryGetId(token, out int liveId) || liveId != id)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// The reason a model cannot use the vocabulary pinned to it, or null when
        /// they line up or the model is not pinned yet.
        /// <para>
        /// Checkpoints record VocabSize and loading validates it, so acting on the
        /// problem here keeps it from surfacing later as an unloadable checkpoint.
        /// </para>
        /// </summary>
        public static string? DescribeModelVocabularyProblem(
            string modelName,
            int configVocabSize,
            VocabularyEntry? entry,
            int? artifactTokenCount)
        {
            //Not pinned yet: the model still runs on the vocabulary the server
            //loaded at startup, and a training job will pin one.
            if (entry == null)
            {
                return null;
            }

            if (artifactTokenCount == null)
            {
                return $"Vocabulary '{entry.Name}' is missing or unreadable on disk " +
                    $"({ResolvePath(entry)}). Recompile it, or point the model at another vocabulary.";
            }

            if (artifactTokenCount.Value != configVocabSize)
            {
                return $"Model '{modelName}' is configured for {configVocabSize} tokens but vocabulary " +
                    $"'{entry.Name}' contains {artifactTokenCount.Value}. Set the model's transformer config " +
                    $"Vocabulary Size to {artifactTokenCount.Value}, or choose a vocabulary with {configVocabSize} tokens.";
            }

            return null;
        }
    }
}
