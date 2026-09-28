using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Greedy longest-match subword fallback for BPE vocabularies whose merge
    /// table was not persisted (BpeVocabularyCompiler stores only the token
    /// map). Exact whole-word hits win first; otherwise the longest vocab
    /// entry at each position wins, single char -&gt; &lt;unk&gt;. Deterministic
    /// and stream-safe; slightly coarser than true BPE but id-space correct.
    /// </summary>
    public sealed class GreedySubwordTokenizerAdapter : ITokenizer
    {
        public TokenizerType Type => TokenizerType.Bpe;
        private readonly Dictionary<string, int> _toId;
        private readonly IReadOnlyDictionary<int, string> _toToken;
        private readonly int _pad, _unk, _bos, _eos, _maxLen;

        public int EosTokenId => _eos;
        public int VocabularySize => _toId.Count;

        public GreedySubwordTokenizerAdapter(Vocabulary vocabulary)
        {
            _toId = (Dictionary<string, int>)vocabulary.TokenToId;
            // Reverse map for Decode: scanning the vocab per token would be
            // O(vocab) on every token, which is fine for cache building but a
            // trap if this adapter is ever used for inference.
            _toToken = vocabulary.IdToToken;
            _pad = _toId[SpecialTokens.Pad];
            _unk = _toId[SpecialTokens.Unknown];
            _bos = _toId[SpecialTokens.BeginningOfSequence];
            _eos = _toId[SpecialTokens.EndOfSequence];
            _maxLen = 0;
            foreach (var k in _toId.Keys)
                if (k.Length > _maxLen) _maxLen = k.Length;
            if (_maxLen <= 0) _maxLen = 1;
        }

        public int[] Encode(string text)
        {
            var ids = new List<int> { _bos };
            // Same pre-tokenizer as BpeTokenizer so whole-word hits match.
            foreach (var word in TokenizationUtilities.TokenizeRawText(text))
            {
                if (_toId.TryGetValue(word, out int exact)) { ids.Add(exact); continue; }
                int i = 0;
                while (i < word.Length)
                {
                    int bestLen = 0; int bestId = _unk;
                    int max = Math.Min(_maxLen, word.Length - i);
                    for (int len = max; len > 0; len--)
                    {
                        var sub = word.Substring(i, len);
                        if (_toId.TryGetValue(sub, out int id))
                        { bestLen = len; bestId = id; break; }
                    }
                    ids.Add(bestId);
                    i += bestLen > 0 ? bestLen : 1;
                }
            }
            ids.Add(_eos);
            return ids.ToArray();
        }

        public string Decode(ReadOnlySpan<int> tokens)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var id in tokens)
            {
                if (id == _bos || id == _eos || id == _pad) continue;
                if (!_toToken.TryGetValue(id, out var token) || token == null) continue;
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(token);
            }
            return sb.ToString();
        }
    }
}
