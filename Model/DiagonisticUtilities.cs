namespace SimpleTransformer.Model
{
    public static class DiagonisticUtilities
    {
        /// <summary>
        /// Fails fast when a tensor contains NaN or Infinity.
        /// NaN-only checks let +Inf travel silently through matmuls/norms
        /// and turn into NaN one op later (e.g. SIMD GELU tanh approx),
        /// which misattributes the failure stage. Always check finiteness.
        /// </summary>
        public static void AssertNoNaN(TensorBase t, string stage)
        {
            AssertFinite(t, stage);
        }

        public static void AssertFinite(TensorBase t, string stage)
        {
            switch (t.Rank)
            {
                case 1:
                    for (int col = 0; col < t.Cols; col++)
                    {
                        float v = t[col];
                        if (!float.IsFinite(v))
                            throw new InvalidOperationException(
                                $"Non-finite value {v} detected first at stage: {stage} (col {col})");
                    }
                    break;

                case 2:
                    for (int row = 0; row < t.Rows; row++)
                    {
                        for (int col = 0; col < t.Cols; col++)
                        {
                            float v = t[row, col];
                            if (!float.IsFinite(v))
                                throw new InvalidOperationException(
                                    $"Non-finite value {v} detected first at stage: {stage} (row {row}, col {col})");
                        }
                    }
                    break;

                case 3:
                    for (int layer = 0; layer < t.Layers; layer++)
                    {
                        for (int row = 0; row < t.Rows; row++)
                        {
                            for (int col = 0; col < t.Cols; col++)
                            {
                                float v = t[layer, row, col];
                                if (!float.IsFinite(v))
                                    throw new InvalidOperationException(
                                        $"Non-finite value {v} detected first at stage: {stage} (layer {layer}, row {row}, col {col})");
                            }
                        }
                    }
                    break;

                default:
                    throw new NotSupportedException($"Rank {t.Rank} is not supported by AssertFinite.");
            }
        }

        /// <summary>Non-throwing scan: true when every element is finite.</summary>
        public static bool IsFinite(TensorBase t)
        {
            // NOTE: must use logical indexers, not t.Data.AsSpan(),
            // because pooled/strided views share buffers with slack.
            switch (t.Rank)
            {
                case 1:
                    for (int col = 0; col < t.Cols; col++)
                        if (!float.IsFinite(t[col])) return false;
                    return true;
                case 2:
                    for (int row = 0; row < t.Rows; row++)
                        for (int col = 0; col < t.Cols; col++)
                            if (!float.IsFinite(t[row, col])) return false;
                    return true;
                case 3:
                    for (int layer = 0; layer < t.Layers; layer++)
                        for (int row = 0; row < t.Rows; row++)
                            for (int col = 0; col < t.Cols; col++)
                                if (!float.IsFinite(t[layer, row, col])) return false;
                    return true;
                default:
                    throw new NotSupportedException($"Rank {t.Rank} is not supported by IsFinite.");
            }
        }
    }
}