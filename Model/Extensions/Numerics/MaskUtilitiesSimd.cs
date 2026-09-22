using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Numerics;
using System.Threading.Tasks;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class MaskUtilitiesSimd
    {
        #region Mask

        public static TensorBase ApplyMask(TensorBase src, TensorBase mask)
        {
            TensorBase res = src.Clone();
            ApplyMaskInPlace(res, mask);
            return res;
        }

        public static void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
        {        
            if (scores.Rows != scores.Cols)
                throw new ArgumentException("Attention score must be a square matrix.");

            TensorUtilitiesSimd.ValidateTensorShape(mask, scores.Rows, scores.Cols);

            int rows = scores.Rows;
            int cols = scores.Cols;
            int scoresStride = scores.Stride;
            int maskStride = mask.Stride;
            int width = Vector<float>.Count;

            float[] scoresBuffer = scores.Buffer; int scoresOffset = scores.Offset;
            float[] maskBuffer = mask.Buffer; int maskOffset = mask.Offset;

            // Use -1e9f for masking (matches MaskUtilities and the IAccelerationBackend
            // contract; softmax(exp(-1e9 - max)) underflows to exactly 0 either way,
            // but finite sentinels keep downstream logits finite for diagnostics).
            var vMaskValue = new Vector<float>(-1e9f);

            Parallel.For(0, rows, r =>
            {
                ReadOnlySpan<float> threadMaskSpan = maskBuffer.AsSpan(maskOffset);
                Span<float> threadScoresSpan = scoresBuffer.AsSpan(scoresOffset);

                ref float pMaskRow = ref MemoryMarshal.GetReference(threadMaskSpan.Slice(r * maskStride, cols));
                ref float pScoresRow = ref MemoryMarshal.GetReference(threadScoresSpan.Slice(r * scoresStride, cols));

                int c = 0;
                for (; c <= cols - width; c += width)
                {
                    Vector<float> vMask = Vector.LoadUnsafe(ref pMaskRow, (uint)c);
                    Vector<float> vScores = Vector.LoadUnsafe(ref pScoresRow, (uint)c);

                    Vector<float> result = Vector.ConditionalSelect(
                        Vector.Equals(vMask, Vector<float>.Zero),
                        vMaskValue,
                        vScores
                    );

                    Vector.StoreUnsafe(result, ref pScoresRow, (uint)c);
                }

                for (; c < cols; c++)
                {
                    if (Unsafe.Add(ref pMaskRow, c) == 0f)
                    {
                        Unsafe.Add(ref pScoresRow, c) = -1e9f;
                    }
                }
            });
        }

        public static TensorBase CreateCausalMask(int sequenceLength)
        {
            TensorUtilities.ValidateSequenceLength(sequenceLength);
            var mask = new Tensor(sequenceLength, sequenceLength);
            
            // Capture raw primitives for lambda safety
            float[] maskBuffer = mask.Buffer;
            int maskOffset = mask.Offset;
            int stride = mask.Stride;

            Parallel.For(0, sequenceLength, r =>
            {
                Span<float> threadMaskSpan = maskBuffer.AsSpan(maskOffset);
                ref float pMask = ref MemoryMarshal.GetReference(threadMaskSpan);

                int rowOffset = r * stride;
                int c = 0;

                for (; c <= r; c++)
                {
                    Unsafe.Add(ref pMask, (uint)(rowOffset + c)) = 1f;
                }
            });

            return mask;
        }

        public static TensorBase CreatePaddingMask(TensorBase tokens, int padToken = 0)
        {
            if (tokens.Rank != 1)
                throw new ArgumentException("Input must be a vector of token IDs.");

            int len = tokens.Length;
            var mask = new Tensor(len);
            
            ReadOnlySpan<float> tokensSpan = tokens.ReadOnlySpan;
            Span<float> maskSpan = mask.Span;
            int width = Vector<float>.Count;

            ref float pTok = ref MemoryMarshal.GetReference(tokensSpan);
            ref float pMask = ref MemoryMarshal.GetReference(maskSpan);

            var vPad = new Vector<float>(padToken);
            var vOne = Vector<float>.One;

            int i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> vTokens = Vector.LoadUnsafe(ref pTok, (uint)i);
                
                Vector<float> result = Vector.ConditionalSelect(
                    Vector.Equals(vTokens, vPad),
                    Vector<float>.Zero,
                    vOne
                );

                Vector.StoreUnsafe(result, ref pMask, (uint)i);
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref pMask, i) = (Unsafe.Add(ref pTok, i) == padToken) ? 0f : 1f;
            }

            return mask;
        }

        public static TensorBase ExpandPaddingMask(TensorBase paddingMask)
        {
            if (paddingMask.Rank != 1)
                throw new ArgumentException("Padding mask must be a vector.");

            int length = paddingMask.Length;
            var mask = new Tensor(length, length);

            // Capture raw arrays and offsets for parallel execution
            float[] padBuffer = paddingMask.Buffer; int padOffset = paddingMask.Offset;
            float[] dstBuffer = mask.Buffer; int dstOffset = mask.Offset;
            int stride = mask.Stride;

            Parallel.For(0, length, row =>
            {
                ReadOnlySpan<float> threadPadSpan = padBuffer.AsSpan(padOffset, length);
                Span<float> threadDstSpan = dstBuffer.AsSpan(dstOffset);

                Span<float> rowTarget = threadDstSpan.Slice(row * stride, length);
                threadPadSpan.CopyTo(rowTarget);
            });

            return mask;
        }        

        public static TensorBase CreateAllowAllMask(int rows, int cols)
        {
            if (rows <= 0 || cols <= 0)
                throw new ArgumentException("Both dimensions must be positive.");

            var mask = new Tensor(rows, cols);
            
            float[] maskBuffer = mask.Buffer;
            int maskOffset = mask.Offset;
            int stride = mask.Stride;

            Parallel.For(0, rows, r =>
            {
                Span<float> threadMaskSpan = maskBuffer.AsSpan(maskOffset);
                Span<float> rowSpan = threadMaskSpan.Slice(r * stride, cols);
                rowSpan.Fill(1f); 
            });

            return mask;
        }

        public static TensorBase CombineMasks(TensorBase a, TensorBase b)
        {
            TensorBase result = a.Clone();
            ElementWiseMultiplyInPlace(result, b);
            return result;
        }

        public static void CombineMasksInPlace(TensorBase destination, TensorBase other)
        {
            ElementWiseMultiplyInPlace(destination, other);
        }

        #endregion
        
        private static void ElementWiseMultiplyInPlace(TensorBase a, TensorBase b)
        {
            Span<float> aSpan = a.Span;
            ReadOnlySpan<float> bSpan = b.ReadOnlySpan;
            int len = aSpan.Length;
            int width = Vector<float>.Count;

            ref float pA = ref MemoryMarshal.GetReference(aSpan);
            ref float pB = ref MemoryMarshal.GetReference(bSpan);

            int i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> va = Vector.LoadUnsafe(ref pA, (uint)i);
                Vector<float> vb = Vector.LoadUnsafe(ref pB, (uint)i);
                Vector.StoreUnsafe(va * vb, ref pA, (uint)i);
            }
            for (; i < len; i++)
            {
                Unsafe.Add(ref pA, i) *= Unsafe.Add(ref pB, i);
            }
        }
    }
}
