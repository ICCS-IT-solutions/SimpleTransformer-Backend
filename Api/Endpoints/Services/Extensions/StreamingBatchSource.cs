using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Builds MiniBatches by paging windows off a .stbin token cache.
    /// Memory is O(batch*seq + shuffleIndex), never O(corpus).
    /// Window semantics match TrainingDataExtensions: non-overlapping stride
    /// of `window`, each sample = tokens[i..i+window) -&gt; tokens[i+1..i+window+1).
    /// Shuffle is over window START indices (Fisher-Yates on ints), so a 50M
    /// token corpus shuffles ~390k ints (~1.5MB), not tensors.
    /// </summary>
    public sealed class StreamingBatchSource : IDisposable
    {
        private readonly FileStream _fs;
        private readonly TokenCache.Header _header;
        private readonly int _window;
        private readonly int _batchSize;
        private readonly int _padTokenId;
        private readonly bool _dropLast;
        private readonly long _sampleCount;
        private readonly Tensor _inBuf;
        private readonly Tensor _tgtBuf;
        private readonly float[] _row;
        // Reused across every window of every epoch: the whole point of the
        // streaming path is that reading a batch allocates nothing.
        private readonly byte[] _scratch;

        public long SampleCount => _sampleCount;
        public int BatchCount { get; private set; }
        public TokenCache.Header Header => _header;

        public StreamingBatchSource(string binPath, int window, int batchSize, bool dropLast,
            int padTokenId = 0)
        {
            _header = TokenCache.ReadHeader(binPath);
            _fs = new FileStream(binPath, FileMode.Open, FileAccess.Read,
                FileShare.Read, bufferSize: 1 << 20, useAsync: false);
            _window = window;
            _batchSize = batchSize;
            _dropLast = dropLast;
            // Id written into padded INPUT positions. The matching padded TARGET
            // positions carry TransformerModel.IgnoreIndex instead, which is what
            // the loss skips and what the attention mask reads to find the valid
            // prefix. Defaults to 0, the id every vocabulary assigns to
            // SpecialTokens.Pad.
            _padTokenId = padTokenId;
            // Floor division gives the count of fully-populated windows; a remainder adds
            // one more, emitted PADDED rather than discarded. This mirrors
            // TrainingDataExtensions.CreateTrainingSamples exactly - the two
            // paths must agree, or a corpus that straddles the streaming
            // threshold (numBatches > 8) would train on a different number of
            // windows depending on which path served it.
            long fullWindows = _header.TokenCount > window
                ? (_header.TokenCount - 1) / window
                : 0;
            bool hasTail = _header.TokenCount > 0 && fullWindows * window < _header.TokenCount;
            _sampleCount = fullWindows + (hasTail ? 1 : 0);
            _inBuf = new Tensor(batchSize, window);
            _tgtBuf = new Tensor(batchSize, window);
            _row = new float[window + 1];
            _scratch = new byte[(window + 1) * TokenCache.BytesPerToken(_header.Dtype)];

            long full = _sampleCount / batchSize;
            long rem = _sampleCount % batchSize;
            bool drop = dropLast && rem != 0 && full > 0;
            BatchCount = (int)(full + (rem == 0 || drop ? 0 : 1));
        }

        public void Validate(Guid fingerprint, ITokenizer tokenizer) =>
            TokenCache.Validate(_header, fingerprint, tokenizer.VocabularySize, (int)tokenizer.Type);

        /// <summary>
        /// Shuffled epoch stream with ZERO-COPY batches: every yielded
        /// MiniBatch shares the same two reusable tensors, so a batch is only
        /// valid until the next MoveNext. Train on it before pulling the next
        /// one; never buffer the result (no .ToList()/.Chunk()) or every entry
        /// ends up aliasing the last batch. That restriction is what keeps a
        /// 50M-token corpus at O(batch*seq) memory.
        /// </summary>
        public IEnumerable<MiniBatch> StreamEpoch(int seed)
        {
            if (_sampleCount == 0 || BatchCount == 0) yield break;
            // Shuffle window indices, not data: int array only.
            var order = new int[_sampleCount];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            // Fully deterministic in the seed: the training loop derives a
            // per-epoch seed so a resume replays the same window order.
            var rng = new Random(seed);
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            long full = _sampleCount / _batchSize;
            long rem = _sampleCount % _batchSize;
            bool drop = _dropLast && rem != 0 && full > 0;
            long usable = drop ? full * _batchSize : _sampleCount;

            for (long b = 0; b < usable; b += _batchSize)
            {
                int cur = (int)Math.Min(_batchSize, usable - b);
                for (int r = 0; r < cur; r++)
                {
                    long tokenOff = (long)order[b + r] * _window;

                    // A tail window can run past the end of the stream. Read only
                    // the tokens that actually exist - ReadWindowAsFloat uses
                    // ReadExactly and would throw at EOF - then pad the rest,
                    // exactly as CreateTrainingSamples does for the buffered path.
                    long available = _header.TokenCount - tokenOff;
                    int toRead = (int)Math.Min(_window + 1L, Math.Max(0L, available));
                    if (toRead > 0)
                    {
                        TokenCacheIO.ReadWindowAsFloat(_fs, _header, tokenOff, toRead, _row, _scratch);
                    }

                    // Input row holds up to `window` tokens; the target row is
                    // shifted by one, so it is always the shorter of the two.
                    int validInput = (int)Math.Min(_window, Math.Max(0L, available));
                    int validTarget = (int)Math.Min(_window, Math.Max(0L, available - 1));

                    for (int c = 0; c < _window; c++)
                    {
                        _inBuf[r, c] = c < validInput ? _row[c] : _padTokenId;
                        _tgtBuf[r, c] = c < validTarget ? _row[c + 1] : TransformerModel.IgnoreIndex;
                    }
                }
                if (cur == _batchSize)
                {
                    yield return new MiniBatch { Inputs = _inBuf, Targets = _tgtBuf };
                }
                else
                {
                    // Trailing partial batch kept only when nothing else exists
                    // (mirrors CreateMiniBatches semantics).
                    var pi = new Tensor(cur, _window);
                    var pt = new Tensor(cur, _window);
                    for (int r = 0; r < cur; r++)
                        for (int c = 0; c < _window; c++)
                        {
                            pi[r, c] = _inBuf[r, c];
                            pt[r, c] = _tgtBuf[r, c];
                        }
                    yield return new MiniBatch { Inputs = pi, Targets = pt };
                }
            }
        }

        /// <summary>
        /// The same shuffled order as <see cref="StreamEpoch"/> but each batch
        /// owns freshly allocated tensors, so the caller may buffer, chunk or
        /// shuffle the list. Costs O(batches*batch*seq): only safe for the small
        /// (a handful of batches) path - never for a whole large corpus.
        /// </summary>
        public IReadOnlyList<MiniBatch> MaterializeEpoch(int seed)
        {
            var list = new List<MiniBatch>(BatchCount);
            foreach (var b in StreamEpoch(seed))
            {
                var inputs = new Tensor(b.BatchSize, _window);
                var targets = new Tensor(b.BatchSize, _window);
                Array.Copy(b.Inputs.Data, inputs.Data, b.Inputs.Length);
                Array.Copy(b.Targets.Data, targets.Data, b.Targets.Length);
                list.Add(new MiniBatch { Inputs = inputs, Targets = targets });
            }
            return list;
        }

        public void Dispose() => _fs.Dispose();
    }
}
