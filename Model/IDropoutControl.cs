using System.Collections.Generic;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Implemented by layer composites that own dropout somewhere inside their
    /// subtree, so the model can switch every dropout site (sub-layer outputs,
    /// attention weights) on/off and push the configured rate down from a
    /// single call site in <see cref="TransformerModel.BeginTraining"/> /
    /// <see cref="TransformerModel.EndTraining"/>.
    /// </summary>
    public interface IDropoutControl
    {
        /// <summary>Enables or disables every dropout site in the subtree.</summary>
        void SetDropoutEnabled(bool enabled);

        /// <summary>Sets the dropout rate on every site in the subtree. Must be in [0.0, 1.0).</summary>
        void SetDropoutRate(float rate);

        /// <summary>
        /// Re-keys every dropout site in the subtree for a new optimizer step, so
        /// masks are a pure function of (site salt, step, batch item). Called once
        /// per TrainStep before the forward pass.
        /// </summary>
        void PrepareDropoutForStep(long step);

        /// <summary>
        /// Appends this subtree's dropout sites to <paramref name="sites"/> for
        /// checkpoint save/load. Used only at those boundaries, so the append into
        /// a caller-owned list avoids per-step allocation.
        /// </summary>
        void CollectDropoutSites(List<DropoutSite> sites);
    }
}