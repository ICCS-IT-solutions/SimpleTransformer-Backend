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
                2 => BackwardSequenceOffPool(prediction, target),
                3 => BackwardBatchOffPool(prediction, target),
                _ => throw new ArgumentException("Prediction must be rank 2 or rank 3.")
            };
        }

        /// <summary>
        /// Workspace-pooled backward: borrows the gradient from the pool instead
        /// of `new Tensor(...)`, so a training step allocates nothing here. The
        /// caller (TrainStep's finally-Reset) reclaims it with the rest of the
        /// step's activations.
        /// </summary>
        public TensorBase Backward(TensorBase prediction, TensorBase target, TensorWorkspace workspace)
        {
            ArgumentNullException.ThrowIfNull(workspace);
            return prediction.Rank switch
            {
                2 => BackwardSequence(prediction, target, workspace),
                3 => BackwardBatch(prediction, target, workspace),
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

        private TensorBase BackwardBatchOffPool(TensorBase prediction, TensorBase target)
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
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, batch);

                BackwardSequenceInto(predictionSlice, targetSlice, gradSlice, scale);
            }

            return gradient;
        }

        private TensorBase BackwardBatch(TensorBase prediction, TensorBase target, TensorWorkspace workspace)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);
            // Borrowed (already cleared by the pool): fully overwritten below.
            TensorBase gradient = workspace.BorrowLike(prediction);

            ReadOnlySpan<float> targetData = target.ReadOnlySpan;
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
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, batch);

                BackwardSequenceInto(predictionSlice, targetSlice, gradSlice, scale);
            }

            return gradient;
        }

        private TensorBase BackwardSequence(TensorBase prediction, TensorBase target, TensorWorkspace workspace)
        {
            ReadOnlySpan<float> targetData = target.ReadOnlySpan;
            int validTokens = 0;
            for (int i = 0; i < targetData.Length; i++)
            {
                if ((int)targetData[i] != _ignoreIndex)
                    validTokens++;
            }

            float scale = validTokens > 0 ? 1f / validTokens : 0f;
            // Borrowed (already cleared by the pool): fully overwritten below.
            TensorBase gradient = workspace.BorrowLike(prediction);
            BackwardSequenceInto(prediction, target, gradient, scale);
            return gradient;
        }

        /// <summary>
        /// Writes the softmax-cross-entropy gradient into a caller-provided
        /// buffer. Stride-aware: rows are addressed through Offset/Stride so
        /// pooled buffers and TensorView slices work, not just owned Tensors.
        /// Every row is fully written (ignored rows are zeroed), so callers may
        /// pass a borrowed buffer without pre-clearing.
        /// </summary>
        private void BackwardSequenceInto(TensorBase prediction, TensorBase target, TensorBase gradient, float scale)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);

            int rows = prediction.Rows;
            int cols = prediction.Cols;

            ReadOnlySpan<float> predBuffer = prediction.Buffer.AsSpan();
            ReadOnlySpan<float> targetBuffer = target.Buffer.AsSpan();
            Span<float> gradBuffer = gradient.Buffer.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int predRowOffset = prediction.Offset + r * prediction.Stride;
                int gradRowOffset = gradient.Offset + r * gradient.Stride;
                // Target is Rank-1 (sequence: Stride == 1, Offset == 0) or a
                // Rank-2 row view from GetRow (length Cols, Stride == Cols,
                // Offset == row * parent.Stride): logical index r maps to
                // Offset + r for both, NOT Offset + r * Stride.
                int tokenId = (int)targetBuffer[target.Offset + r];

                // Padding/ignored rows contribute nothing: zero the row.
                if (tokenId == _ignoreIndex)
                {
                    gradBuffer.Slice(gradRowOffset, cols).Clear();
                    continue;
                }

                float maxVal = float.MinValue;
                for (int c = 0; c < cols; c++)
                {
                    float v = predBuffer[predRowOffset + c];
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
                    float val = predBuffer[predRowOffset + c];
                    if (float.IsNegativeInfinity(val) || val <= -1e20f)
                    {
                        gradBuffer[gradRowOffset + c] = 0f;
                        continue;
                    }

                    float p = MathF.Exp(val - maxVal);
                    gradBuffer[gradRowOffset + c] = p;
                    sumExp += p;
                }

                float invSum = sumExp > 0f ? 1f / sumExp : 0f;

                for (int c = 0; c < cols; c++)
                    gradBuffer[gradRowOffset + c] *= invSum;

                // dL/dz = (p_i - 1) for target class, scaled by 1/N_valid.
                gradBuffer[gradRowOffset + tokenId] -= 1f;

                for (int c = 0; c < cols; c++)
                    gradBuffer[gradRowOffset + c] *= scale;
            }
        }

        private TensorBase BackwardSequenceOffPool(TensorBase prediction, TensorBase target)
        {
            ReadOnlySpan<float> targetData = target.Data.AsSpan();
            int validTokens = 0;
            for (int i = 0; i < targetData.Length; i++)
            {
                if ((int)targetData[i] != _ignoreIndex)
                    validTokens++;
            }

            float scale = validTokens > 0 ? 1f / validTokens : 0f;
            // Legacy off-pool path: kept for the interface default and any
            // non-training callers. Training uses the workspace overload.
            var gradient = new Tensor(prediction.Rows, prediction.Cols);
            BackwardSequenceInto(prediction, target, gradient, scale);
            return gradient;
        }
    }
}