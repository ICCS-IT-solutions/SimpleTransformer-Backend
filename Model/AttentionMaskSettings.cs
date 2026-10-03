namespace SimpleTransformer.Model
{
    /// <summary>
    /// Whether the decoder applies a causal (autoregressive) attention mask.
    /// Pushed in by the host process (Server / CLI) from config.ini before any
    /// model is constructed.
    /// <para>
    /// This model is a decoder-only language model: it embeds a token window,
    /// runs pre-norm transformer blocks, and projects to the vocabulary for
    /// next-token prediction. In that architecture the causal mask is not an
    /// optimisation but part of the definition - position t must not see t+1..S,
    /// or the model is trained to copy the future and generation is meaningless.
    /// It used to be absent: TransformerModel.Forward called
    /// block.Forward(x, workspace) with no mask, so every position attended to
    /// every other one.
    /// <para>
    /// The mask is architecture, not a training-time option, so it is applied in
    /// the shared Forward path and therefore affects inference and generation too.
    /// <b>That means checkpoints trained before this change were trained without
    /// it and will produce different - worse - output once it is on.</b> Set
    /// false only to reproduce or continue a legacy unmasked run.
    /// </para>
    /// </summary>
    public static class AttentionMaskSettings
    {
        /// <summary>Default true: a decoder must not attend to future tokens.</summary>
        public static bool UseCausalMask { get; set; } = true;
    }
}