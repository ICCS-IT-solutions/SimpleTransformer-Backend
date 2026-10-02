using System;

namespace SimpleTransformer.Model
{
    //For any math, target the acceleration backend interface not any concrete versions.
    /// <summary>
    /// Inverted dropout: while enabled (training), elements are randomly zeroed
    /// and the survivors are scaled by 1/(1 - rate), so the expected value of
    /// the output matches the input and no rescaling is needed at inference.
    /// When disabled or when the rate is 0 the layer is an exact pass-through.
    /// <para>
    /// Always returns a freshly borrowed workspace tensor in both directions -
    /// never the input/gradient itself - so callers can release what they pass
    /// in and what they get back independently, matching the GeluLayer contract.
    /// </para>
    /// <para>
    /// The mask is owned by this layer (a plain float[]), not borrowed from the
    /// workspace: the pool recycles buffers between Forward and Backward, so a
    /// workspace-held mask would not survive until backpropagation. Same
    /// pattern as LayerNorm._lastInvStdBuffer.
    /// </para>
    /// <para>
    /// Randomness comes from <see cref="DropoutRng"/> rather than
    /// <see cref="System.Random"/>, and <see cref="Reseed"/> re-keys the stream
    /// from a caller-supplied seed. <see cref="DropoutSite"/> drives that: it
    /// keys this layer from (persisted salt, optimizer step, batch item), which
    /// is what makes a resumed run replay the identical mask sequence.
    /// </para>
    /// </summary>
    public class DropoutLayer : ILayer
    {
        public string Name { get; }
        public float DropoutRate { get; private set; }

        /// <summary>
        /// Train/eval switch. False by default so every dropout site is an
        /// exact pass-through until a training run explicitly enables it.
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>0, or 1/(1 - rate) for kept elements (inverted dropout).</summary>
        private float[] _mask = Array.Empty<float>();

        /// <summary>Number of valid mask entries written by the last Forward.</summary>
        private int _maskLength;

        /// <summary>Did the last Forward run as an identity pass? Backward mirrors it.</summary>
        private bool _passThrough = true;

        private DropoutRng _rng;

        public DropoutLayer(float dropoutRate, string name = "dropout", int? seed = null)
        {
            ValidateRate(dropoutRate);
            DropoutRate = dropoutRate;
            Name = name;
            // The explicit seed keeps self-tests reproducible; production leaves
            // it null and gets fresh entropy, then DropoutSite re-keys the stream
            // per step anyway.
            _rng = DropoutRng.FromSeed(seed.HasValue
                ? (ulong)(uint)seed.Value
                : DropoutRng.NextEntropy());
        }

        public void SetRate(float rate)
        {
            ValidateRate(rate);
            DropoutRate = rate;
        }

        /// <summary>
        /// Re-keys the mask stream. Called by <see cref="DropoutSite"/> at the
        /// start of each optimizer step so the masks are a pure function of
        /// (site salt, step, batch item) rather than of how many forwards have
        /// happened to run.
        /// </summary>
        public void Reseed(ulong seed) => _rng = DropoutRng.FromSeed(seed);

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Rank != 2 && input.Rank != 3)
                throw new ArgumentException($"Input must be Rank 2 or 3. Got Rank {input.Rank}.");

            // Fresh buffer even on the pass-through path: returning `input`
            // directly would make callers that release both the input and the
            // returned tensor double-release into the pool.
            TensorBase output = workspace.BorrowLike(input);

            _passThrough = !Enabled || DropoutRate == 0f;
            if (_passThrough)
            {
                workspace.Backend.CopyInto(input, output);
                return output;
            }

            ReadOnlySpan<float> src = input.ReadOnlySpan;
            Span<float> dst = output.Span;

            if (_mask.Length < src.Length)
                _mask = new float[src.Length];
            _maskLength = src.Length;

            float keepScale = 1f / (1f - DropoutRate);
            float rate = DropoutRate;

            for (int i = 0; i < src.Length; i++)
            {
                _mask[i] = _rng.NextSingle() < rate ? 0f : keepScale;
                dst[i] = src[i] * _mask[i];
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            TensorBase inputGradient = workspace.BorrowLike(gradient);

            if (_passThrough)
            {
                workspace.Backend.CopyInto(gradient, inputGradient);
                return inputGradient;
            }

            ReadOnlySpan<float> src = gradient.ReadOnlySpan;
            Span<float> dst = inputGradient.Span;

            if (src.Length != _maskLength)
                throw new InvalidOperationException(
                    $"Dropout gradient span ({src.Length}) does not match the mask from " +
                    $"Forward ({_maskLength}). Forward and Backward must see the same shape.");

            for (int i = 0; i < src.Length; i++)
                dst[i] = src[i] * _mask[i];

            return inputGradient;
        }

        private static void ValidateRate(float rate)
        {
            if (rate < 0f || rate >= 1f)
                throw new ArgumentOutOfRangeException(nameof(rate), rate,
                    "Dropout rate must be between [0.0, 1.0).");
        }
    }
}