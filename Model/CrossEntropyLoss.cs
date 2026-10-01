using System;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class CrossEntropyLoss : ILossFunction
    {
        private readonly int _ignoreIndex;

        public CrossEntropyLoss(int ignoreIndex = -100)
        {
            _ignoreIndex = ignoreIndex;
        }

        public float Forward(TensorBase prediction, TensorBase target)
        {
            return prediction.Rank switch
            {
                2 => ForwardSequence(prediction, target),
                3 => ForwardBatch(prediction, target),
                _ => throw new ArgumentException("Prediction must be rank 2 or rank 3.")
            };
        }

        public TensorBase Backward(TensorBase prediction, TensorBase target)
        {
            return prediction.Rank switch
            {
                2 => BackwardSequence(prediction, target),
                3 => BackwardBatch(prediction, target),
                _ => throw new ArgumentException("Prediction must be rank 2 or rank 3.")
            };
        }

        private float ForwardBatch(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);
            float totalLoss = 0f;
            int totalValidTokens = 0;

            for (int batch = 0; batch < prediction.Layers; batch++)
            {
                TensorBase predictionSlice = TensorUtilitiesSimd.GetLayer(prediction, batch);
                TensorBase targetSlice = TensorUtilitiesSimd.GetRow(target, batch);

                (float batchLoss, int validTokens) = ForwardSequenceWithCount(predictionSlice, targetSlice);
                totalLoss += batchLoss;
                totalValidTokens += validTokens;
            }

            return totalValidTokens > 0 ? totalLoss / totalValidTokens : 0f;
        }

        private float ForwardSequence(TensorBase prediction, TensorBase target)
        {
            (float totalLoss, int validTokens) = ForwardSequenceWithCount(prediction, target);
            return validTokens > 0 ? totalLoss / validTokens : 0f;
        }

        private (float TotalLoss, int ValidTokens) ForwardSequenceWithCount(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);

            ReadOnlySpan<float> predData = prediction.Data.AsSpan();
            ReadOnlySpan<float> targetData = target.Data.AsSpan();

            int rows = prediction.Rows;
            int cols = prediction.Cols;

            float totalLoss = 0f;
            int validTokens = 0;

            for (int r = 0; r < rows; r++)
            {
                int tokenId = (int)targetData[r];

                // Skip ignored tokens (e.g. padding)
                if (tokenId == _ignoreIndex)
                    continue;

                if ((uint)tokenId >= (uint)cols)
                    throw new ArgumentOutOfRangeException(nameof(target), $"Target token {tokenId} is outside vocabulary range [0, {cols - 1}].");

                ReadOnlySpan<float> rowLogits = predData.Slice(r * cols, cols);

                // Safe Max Calculation ignoring -Inf. +Inf/NaN poison the
                // exp/sum chain below (val - +Inf = NaN), so detect them here
                // where the row/token context is available for the message.
                float maxVal = float.MinValue;
                for (int c = 0; c < cols; c++)
                {
                    float v = rowLogits[c];
                    if (float.IsNaN(v))
                        throw new InvalidOperationException(
                            $"CrossEntropyLoss: NaN logit at row {r} (target {tokenId}). Failing fast instead of emitting NaN loss.");
                    if (float.IsPositiveInfinity(v))
                        throw new InvalidOperationException(
                            $"CrossEntropyLoss: +Inf logit at row {r} (target {tokenId}). Model diverged upstream (logits overflowed).");
                    if (!float.IsNegativeInfinity(v) && v > maxVal)
                        maxVal = v;
                }
                if (maxVal == float.MinValue) maxVal = 0f;

                float sumExp = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float val = rowLogits[c];
                    if (float.IsNegativeInfinity(val) || val <= -1e20f)
                        continue;

                    sumExp += MathF.Exp(val - maxVal);
                }

                if (sumExp <= 0f || float.IsNaN(sumExp)) sumExp = float.Epsilon;

                float targetLogit = rowLogits[tokenId];
                float logSoftmaxTarget = float.IsNegativeInfinity(targetLogit) 
                    ? -100f 
                    : (targetLogit - maxVal) - MathF.Log(sumExp);

                if (float.IsNaN(logSoftmaxTarget) || float.IsInfinity(logSoftmaxTarget))
                    logSoftmaxTarget = -100f;

                totalLoss -= logSoftmaxTarget;
                validTokens++;
            }

            return (totalLoss, validTokens);
        }

        private TensorBase BackwardBatch(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);
            TensorBase gradient = new Tensor(prediction.Layers, prediction.Rows, prediction.Cols);

            // Count total active tokens in batch to scale gradient accurately
            ReadOnlySpan<float> targetData = target.Data.AsSpan();
            int totalValidTokens = 0;
            for (int i = 0; i < targetData.Length; i++)
            {
                if ((int)targetData[i] != _ignoreIndex)
                    totalValidTokens++;
            }

            float scale = totalValidTokens > 0 ? 1f / totalValidTokens : 0f;

            for (int batch = 0; batch < prediction.Layers; batch++)
            {
                TensorBase predictionSlice = TensorUtilitiesSimd.GetLayer(prediction, batch);
                TensorBase targetSlice = TensorUtilitiesSimd.GetRow(target, batch);

                TensorBase gradSlice = BackwardSequenceInternal(predictionSlice, targetSlice, scale);
                TensorUtilitiesSimd.SetLayer(gradient, batch, gradSlice);
            }

            return gradient;
        }

        private TensorBase BackwardSequence(TensorBase prediction, TensorBase target)
        {
            ReadOnlySpan<float> targetData = target.Data.AsSpan();
            int validTokens = 0;
            for (int i = 0; i < targetData.Length; i++)
            {
                if ((int)targetData[i] != _ignoreIndex)
                    validTokens++;
            }

            float scale = validTokens > 0 ? 1f / validTokens : 0f;
            return BackwardSequenceInternal(prediction, target, scale);
        }

        private TensorBase BackwardSequenceInternal(TensorBase prediction, TensorBase target, float scale)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);

            int rows = prediction.Rows;
            int cols = prediction.Cols;

            TensorBase gradient = new Tensor(rows, cols);

            ReadOnlySpan<float> predData = prediction.Data.AsSpan();
            ReadOnlySpan<float> targetData = target.Data.AsSpan();
            Span<float> gradData = gradient.Data.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int tokenId = (int)targetData[r];
                Span<float> rowGrad = gradData.Slice(r * cols, cols);

                // If token is padding/ignored, leave gradient row as 0s
                if (tokenId == _ignoreIndex)
                    continue;

                ReadOnlySpan<float> rowLogits = predData.Slice(r * cols, cols);

                // Robust max-finding ignoring -Inf logits. +Inf/NaN would make
                // Exp(val - maxVal) = NaN downstream; fail fast instead.
                float maxVal = float.MinValue;
                for (int c = 0; c < cols; c++)
                {
                    float v = rowLogits[c];
                    if (float.IsNaN(v))
                        throw new InvalidOperationException(
                            $"CrossEntropyLoss backward: NaN logit at row {r} (target {tokenId}).");
                    if (float.IsPositiveInfinity(v))
                        throw new InvalidOperationException(
                            $"CrossEntropyLoss backward: +Inf logit at row {r} (target {tokenId}).");
                    if (!float.IsNegativeInfinity(v) && v > maxVal)
                        maxVal = v;
                }
                if (maxVal == float.MinValue) maxVal = 0f;

                float sumExp = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float val = rowLogits[c];
                    if (float.IsNegativeInfinity(val) || val <= -1e20f)
                    {
                        rowGrad[c] = 0f;
                        continue;
                    }

                    float p = MathF.Exp(val - maxVal);
                    rowGrad[c] = p;
                    sumExp += p;
                }

                float invSum = sumExp > 0f ? 1f / sumExp : 0f;

                for (int c = 0; c < cols; c++)
                {
                    if (rowGrad[c] != 0f)
                    {
                        rowGrad[c] *= invSum;
                    }
                }

                // dL/dz = (p_i - 1) for target class
                rowGrad[tokenId] -= 1f;

                // Scale by 1 / N_valid_tokens across batch
                for (int c = 0; c < cols; c++)
                {
                    rowGrad[c] *= scale;
                }
            }

            return gradient;
        }
    }
}