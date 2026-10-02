using System;
using System.Collections.Generic;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// A logical dropout location - the unit of RNG identity that gets
    /// checkpointed.
    /// <para>
    /// It owns ONE durable <see cref="Salt"/> and lazily creates one
    /// mask-carrying <see cref="DropoutLayer"/> per batch item, each keyed from
    /// <c>Mix(Salt, Mix(step, batchIndex))</c>. That split is what makes the
    /// persisted state architecture-only: attention dropout needs a separate
    /// mask per batch item for backprop, but the number of items depends on the
    /// batch size, so the per-item layers must be derived rather than persisted.
    /// </para>
    /// <para>
    /// Because the mask is a pure function of (salt, step, batch item), a
    /// resume replays exactly, a changed batch size cannot perturb item 0's
    /// mask, and an interleaved inference Forward cannot advance the training
    /// stream - none of which a sequentially-advanced generator could offer.
    /// </para>
    /// </summary>
    public sealed class DropoutSite
    {
        /// <summary>
        /// Stable, architecture-derived identity, e.g. "layers.2.ffn_dropout" or
        /// "layers.0.attention.heads.1.attn_w". Checkpoint state is matched by
        /// this key, so it must never be derived from runtime ordering.
        /// </summary>
        public string Key { get; }

        /// <summary>
        /// Durable identity of this stream's mask sequence. Persisted in the
        /// checkpoint and randomly initialised per run (see
        /// <see cref="DropoutRng.NextEntropy"/>).
        /// </summary>
        public ulong Salt { get; set; }

        public float Rate => _rate;
        public bool Enabled => _enabled;

        private float _rate;
        private bool _enabled;

        /// <summary>Step the site is currently keyed for; 0 until first prepared.</summary>
        private long _step;

        // One mask-carrying layer per batch item, created lazily and reseeded on
        // creation so a newly built item is keyed immediately (Forward may build
        // items after PrepareForStep has already run).
        private readonly List<DropoutLayer> _perItem = new();

        public DropoutSite(string key, float rate, ulong? salt = null)
        {
            Key = key;
            _rate = rate;
            Salt = salt ?? DropoutRng.NextEntropy();
        }

        public void SetRate(float rate)
        {
            _rate = rate;
            for (int i = 0; i < _perItem.Count; i++)
                _perItem[i].SetRate(rate);
        }

        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
            for (int i = 0; i < _perItem.Count; i++)
                _perItem[i].Enabled = enabled;
        }

        /// <summary>
        /// The mask-carrying layer for a batch item. The single-sequence paths
        /// use item 0.
        /// </summary>
        public DropoutLayer ForItem(int batchIndex)
        {
            while (_perItem.Count <= batchIndex)
            {
                int index = _perItem.Count;
                var layer = new DropoutLayer(_rate, $"{Key}#{index}") { Enabled = _enabled };
                _perItem.Add(layer);
                ReseedItem(index);
            }

            return _perItem[batchIndex];
        }

        /// <summary>
        /// Re-keys every already-built item for a new optimizer step. Called once
        /// per TrainStep, before the forward pass.
        /// </summary>
        public void PrepareForStep(long step)
        {
            _step = step;
            for (int i = 0; i < _perItem.Count; i++)
                ReseedItem(i);
        }

        /// <summary>Number of per-item layers built so far (diagnostics/tests).</summary>
        public int BuiltLayerCount => _perItem.Count;

        private void ReseedItem(int index) =>
            _perItem[index].Reseed(DropoutRng.Mix(Salt, DropoutRng.Mix((ulong)_step, (ulong)index)));
    }
}