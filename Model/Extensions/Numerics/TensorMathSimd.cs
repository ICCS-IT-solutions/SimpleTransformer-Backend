using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static partial class TensorMathSimd
    {
        // Pre-computed GELU constants
        private const float C0 = 0.044715f;
        private const float C1 = 0.7978845608f; // sqrt(2 / pi)
        private const float Sqrt2OverPi = 0.7978845608f;
        private const float GeluC = 0.044715f;
        private const float Gelu3C = 0.134145f; // 3 * 0.044715f
        public static void ValidateSameShape(TensorBase a, TensorBase b) => TensorUtilitiesSimd.ValidateSameShape(a, b);

        #region Multiple tensor operations
        public static void AddThreeTensors(TensorBase a, TensorBase b, TensorBase c, TensorBase target)
        {
            if (a.IsContiguous && b.IsContiguous && c.IsContiguous && target.IsContiguous)
            {
                ReadOnlySpan<float> spanA = a.ReadOnlySpan;
                ReadOnlySpan<float> spanB = b.ReadOnlySpan;
                ReadOnlySpan<float> spanC = c.ReadOnlySpan;
                Span<float> spanDst = target.Span;

                int length = spanDst.Length;
                int vectorSize = Vector<float>.Count;
                int i = 0;

                for (; i <= length - vectorSize; i += vectorSize)
                {
                    var vA = new Vector<float>(spanA.Slice(i, vectorSize));
                    var vB = new Vector<float>(spanB.Slice(i, vectorSize));
                    var vC = new Vector<float>(spanC.Slice(i, vectorSize));

                    (vA + vB + vC).CopyTo(spanDst.Slice(i, vectorSize));
                }

                for (; i < length; i++)
                {
                    spanDst[i] = spanA[i] + spanB[i] + spanC[i];
                }
            }
            else
            {
                // Fallback for non-contiguous views/slices
                int totalElements = a.Length;
                for (int i = 0; i < totalElements; i++)
                {
                    target[i] = a[i] + b[i] + c[i];
                }
            }
        }
        
        #endregion
        
        #region Special functions
        public static TensorBase Gelu(TensorBase src)
        {
            var result = src.Clone();
            GeluInPlace(result);
            return result;
        }

        public static void LayerNormInPlace(
            TensorBase tensor,
            TensorBase gamma,
            TensorBase beta,
            float epsilon = 1e-5f)
        {
            ValidateLayerNormArguments(
                tensor,
                gamma,
                beta,
                tensor);

            int normalizedSize = tensor.Cols;

            if (gamma.Length != normalizedSize ||
                beta.Length != normalizedSize)
            {
                throw new ArgumentException(
                    "Gamma and beta dimensions must match the normalized dimension.");
            }

            ReadOnlySpan<float> gammaSpan = gamma.ReadOnlySpan;
            ReadOnlySpan<float> betaSpan = beta.ReadOnlySpan;

            int rows = tensor.Length / normalizedSize;

            for (int row = 0; row < rows; row++)
            {
                Span<float> values = tensor
                    .GetRowSpan(row);                   

                LayerNormRowInPlace(
                    values,
                    gammaSpan,
                    betaSpan,
                    epsilon);
            }
        }
        public static void LayerNormInto(
            TensorBase input,
            TensorBase gamma,
            TensorBase beta,
            TensorBase result,
            float epsilon = 1e-5f)
        {
            ValidateLayerNormArguments(
                input,
                gamma,
                beta,
                result);

            int normalizedSize = input.Cols;
            int rows = input.Length / normalizedSize;

            ReadOnlySpan<float> gammaSpan = gamma.ReadOnlySpan;
            ReadOnlySpan<float> betaSpan = beta.ReadOnlySpan;
            ReadOnlySpan<float> inputSpan = input.ReadOnlySpan;
            Span<float> resultSpan = result.Span;

            for (int row = 0; row < rows; row++)
            {
                ReadOnlySpan<float> inputRow =
                    inputSpan.Slice(
                        row * normalizedSize,
                        normalizedSize);

                Span<float> resultRow =
                    resultSpan.Slice(
                        row * normalizedSize,
                        normalizedSize);

                LayerNormRowInto(
                    inputRow,
                    gammaSpan,
                    betaSpan,
                    resultRow,
                    epsilon);
            }
        }

        public static Tensor LayerNorm(
            TensorBase input,
            TensorBase gamma,
            TensorBase beta,
            float epsilon = 1e-5f)
        {
            var result = new Tensor(input.Shape);

            LayerNormInto(
                input,
                gamma,
                beta,
                result,
                epsilon);

            return result;
        }       
        private static void LayerNormRowInPlace(
            Span<float> values,
            ReadOnlySpan<float> gamma,
            ReadOnlySpan<float> beta,
            float epsilon)
        {
            int len = values.Length;

            if (len == 0)
                return;

            int width = Vector<float>.Count;

            ref float pValues =
                ref MemoryMarshal.GetReference(values);

            ref float pGamma =
                ref MemoryMarshal.GetReference(gamma);

            ref float pBeta =
                ref MemoryMarshal.GetReference(beta);

            // ------------------------------------------------------------
            // Phase 1: Calculate mean
            // ------------------------------------------------------------

            Vector<float> sumVector = Vector<float>.Zero;

            int i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> v =
                    Vector.LoadUnsafe(ref pValues, (uint)i);

                sumVector += v;
            }

            float sum = Vector.Dot(
                sumVector,
                Vector<float>.One);

            for (; i < len; i++)
            {
                sum += Unsafe.Add(ref pValues, i);
            }

            float mean = sum / len;

            // ------------------------------------------------------------
            // Phase 2: Calculate variance
            // ------------------------------------------------------------

            Vector<float> vMean =
                new(mean);

            Vector<float> varianceVector =
                Vector<float>.Zero;

            i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> v =
                    Vector.LoadUnsafe(ref pValues, (uint)i);

                Vector<float> diff =
                    v - vMean;

                varianceVector += diff * diff;
            }

            float variance = Vector.Dot(
                varianceVector,
                Vector<float>.One);

            for (; i < len; i++)
            {
                float diff =
                    Unsafe.Add(ref pValues, i) - mean;

                variance += diff * diff;
            }

            variance /= len;

            float invStd =
                1.0f / MathF.Sqrt(variance + epsilon);

            Vector<float> vInvStd =
                new(invStd);

            // ------------------------------------------------------------
            // Phase 3: Normalize + gamma + beta
            // ------------------------------------------------------------

            i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> v =
                    Vector.LoadUnsafe(ref pValues, (uint)i);

                Vector<float> g =
                    Vector.LoadUnsafe(ref pGamma, (uint)i);

                Vector<float> b =
                    Vector.LoadUnsafe(ref pBeta, (uint)i);

                Vector<float> normalized =
                    (v - vMean) * vInvStd;

                Vector<float> result =
                    normalized * g + b;

                Vector.StoreUnsafe(
                    result,
                    ref pValues,
                    (uint)i);
            }

            for (; i < len; i++)
            {
                float value =
                    Unsafe.Add(ref pValues, i);

                float normalized =
                    (value - mean) * invStd;

                Unsafe.Add(ref pValues, i) =
                    normalized *
                    Unsafe.Add(ref pGamma, i) +
                    Unsafe.Add(ref pBeta, i);
            }
        }

        private static void LayerNormRowInto(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gamma,
            ReadOnlySpan<float> beta,
            Span<float> result,
            float epsilon)
        {
            int len = input.Length;

            if (len == 0)
                return;

            if (gamma.Length != len ||
                beta.Length != len ||
                result.Length != len)
            {
                throw new ArgumentException(
                    "LayerNorm row dimensions must match.");
            }

            int width = Vector<float>.Count;

            ref float pInput =
                ref MemoryMarshal.GetReference(input);

            ref float pGamma =
                ref MemoryMarshal.GetReference(gamma);

            ref float pBeta =
                ref MemoryMarshal.GetReference(beta);

            ref float pResult =
                ref MemoryMarshal.GetReference(result);

            // ------------------------------------------------------------
            // Phase 1: Calculate mean
            // ------------------------------------------------------------

            Vector<float> sumVector = Vector<float>.Zero;

            int i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> values =
                    Vector.LoadUnsafe(ref pInput, (uint)i);

                sumVector += values;
            }

            float sum = Vector.Dot(
                sumVector,
                Vector<float>.One);

            for (; i < len; i++)
            {
                sum += Unsafe.Add(ref pInput, i);
            }

            float mean = sum / len;

            // ------------------------------------------------------------
            // Phase 2: Calculate variance
            // ------------------------------------------------------------

            Vector<float> vMean = new(mean);

            Vector<float> varianceVector =
                Vector<float>.Zero;

            i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> values =
                    Vector.LoadUnsafe(ref pInput, (uint)i);

                Vector<float> difference =
                    values - vMean;

                varianceVector +=
                    difference * difference;
            }

            float variance = Vector.Dot(
                varianceVector,
                Vector<float>.One);

            for (; i < len; i++)
            {
                float difference =
                    Unsafe.Add(ref pInput, i) - mean;

                variance +=
                    difference * difference;
            }

            variance /= len;

            float inverseStdDev =
                1.0f / MathF.Sqrt(variance + epsilon);

            Vector<float> vInverseStdDev =
                new(inverseStdDev);

            // ------------------------------------------------------------
            // Phase 3: Normalize + gamma + beta
            // ------------------------------------------------------------

            i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> values =
                    Vector.LoadUnsafe(ref pInput, (uint)i);

                Vector<float> vGamma =
                    Vector.LoadUnsafe(ref pGamma, (uint)i);

                Vector<float> vBeta =
                    Vector.LoadUnsafe(ref pBeta, (uint)i);

                Vector<float> normalized =
                    (values - vMean) * vInverseStdDev;

                Vector<float> output =
                    normalized * vGamma + vBeta;

                Vector.StoreUnsafe(
                    output,
                    ref pResult,
                    (uint)i);
            }

            // ------------------------------------------------------------
            // Scalar remainder
            // ------------------------------------------------------------

            for (; i < len; i++)
            {
                float value =
                    Unsafe.Add(ref pInput, i);

                float normalized =
                    (value - mean) * inverseStdDev;

                Unsafe.Add(ref pResult, i) =
                    normalized *
                    Unsafe.Add(ref pGamma, i) +
                    Unsafe.Add(ref pBeta, i);
            }
        }        
        private static void ValidateLayerNormArguments(
            TensorBase input,
            TensorBase gamma,
            TensorBase beta,
            TensorBase result)
        {
            if (input.Rank < 1)
                throw new ArgumentException(
                    "Input tensor must have at least one dimension.",
                    nameof(input));

            if (result.Rank != input.Rank)
                throw new ArgumentException(
                    "Result tensor must have the same rank as the input.",
                    nameof(result));

            if (!input.Shape.SequenceEqual(result.Shape))
                throw new ArgumentException(
                    "Result tensor must have the same shape as the input.",
                    nameof(result));

            int normalizedSize = input.Shape[^1];

            if (gamma.Rank != 1 ||
                gamma.Length != normalizedSize)
            {
                throw new ArgumentException(
                    "Gamma must be a vector matching the last input dimension.",
                    nameof(gamma));
            }

            if (beta.Rank != 1 ||
                beta.Length != normalizedSize)
            {
                throw new ArgumentException(
                    "Beta must be a vector matching the last input dimension.",
                    nameof(beta));
            }
        }

        public static void GeluInto(TensorBase src, TensorBase dst)
        {
            ReadOnlySpan<float> srcSpan = src.AsSpan(); // Fast flat span accessor
            Span<float> dstSpan = dst.AsWritableSpan();

            if (srcSpan.Length != dstSpan.Length)
                throw new ArgumentException("Source and destination spans must be the same length.");

            int vectorSize = Vector<float>.Count;
            int i = 0;

            // Vectorized SIMD constants
            Vector<float> vHalf = new Vector<float>(0.5f);
            Vector<float> vOne = new Vector<float>(1.0f);
            Vector<float> vC0 = new Vector<float>(C0);
            Vector<float> vC1 = new Vector<float>(C1);
            // Saturation threshold: for |x| >= 12 the tanh term is +/-1 to
            // float precision, so GELU(x) = x (x>0) or 0 (x<0). Branching
            // avoids x^3 overflow (Inf) which poisons the tanh approximation
            // into Inf/Inf = NaN (scalar MathF.Tanh(Inf) = 1 stays finite,
            // hiding the divergence until this vector path emits NaN).
            Vector<float> vLim = new Vector<float>(12.0f);
            Vector<float> vNegLim = new Vector<float>(-12.0f);
            Vector<float> vZero = Vector<float>.Zero;

            // 1. Vectorized Loop (Processes 8/16 floats at a time)
            int simdLim = srcSpan.Length - vectorSize;
            for (; i <= simdLim; i += vectorSize)
            {
                Vector<float> x = new Vector<float>(srcSpan.Slice(i, vectorSize));

                // NaN mask first: Vector.Max seen below follows x86 maxps
                // semantics (returns the second operand when either is NaN),
                // so NaN would be clamped to -12 and silently become 0.
                // Capture it up front and re-inject NaN at the end.
                Vector<int> nanMask = Vector.OnesComplement(Vector.Equals(x, x));

                // Clamp into the approximation's valid range before cubing.
                Vector<float> xc = Vector.Min(Vector.Max(x, vNegLim), vLim);

                // inner = C1 * (xc + C0 * xc^3)
                Vector<float> x3 = xc * xc * xc;
                Vector<float> inner = vC1 * (xc + vC0 * x3);

                // Vectorized Tanh approximation or component evaluation
                Vector<float> tanhVal = VectorTanh(inner);

                // res = 0.5 * xc * (1.0 + tanhVal)
                Vector<float> res = vHalf * xc * (vOne + tanhVal);

                // Saturate: x >= 12 -> x (also +Inf), x <= -12 -> 0 (-Inf -> 0).
                res = Vector.ConditionalSelect(
                    Vector.GreaterThanOrEqual(x, vLim), x,
                    Vector.ConditionalSelect(
                        Vector.LessThanOrEqual(x, vNegLim), vZero, res));

                // Re-inject NaN lanes so non-finite inputs stay fail-fast.
                res = Vector.ConditionalSelect(nanMask, x, res);

                res.CopyTo(dstSpan.Slice(i, vectorSize));
            }

            // 2. Scalar Fallback Loop for remainder elements
            for (; i < srcSpan.Length; i++)
            {
                float x = srcSpan[i];
                if (float.IsNaN(x)) { dstSpan[i] = float.NaN; continue; }
                if (x >= 12f) { dstSpan[i] = x; continue; }
                if (x <= -12f) { dstSpan[i] = 0f; continue; }
                float inner = C1 * (x + C0 * x * x * x);
                dstSpan[i] = 0.5f * x * (1.0f + MathF.Tanh(inner));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> VectorTanh(Vector<float> x)
        {
            // Fast SIMD Tanh approximation: tanh(x) ~ x / (1 + |x| + 0.15 * x^2)
            // Highly accurate within [-4, 4] for GELU and drastically outperforms MathF.Tanh
            Vector<float> absX = Vector.Abs(x);
            Vector<float> x2 = x * x;
            Vector<float> denom = Vector<float>.One + absX + (new Vector<float>(0.15f) * x2);
            
            // Clamp bounds to [-1, 1]
            Vector<float> approx = x / denom;
            return Vector.Min(Vector<float>.One, Vector.Max(new Vector<float>(-1.0f), approx));
        }

        public static void GeluInPlace(TensorBase tensor)
        {
            Span<float> data = tensor.Span;
            int len = data.Length;
            int width = Vector<float>.Count;

            // Cache vector constants entirely in registers
            var half = new Vector<float>(0.5f);
            var one = new Vector<float>(1f);
            var c = new Vector<float>(0.044715f);
            var s = new Vector<float>(0.7978845608f);
            // Saturation threshold (see GeluInto): avoids x^3 overflow -> NaN.
            var lim = new Vector<float>(12.0f);
            var negLim = new Vector<float>(-12.0f);
            var zero = Vector<float>.Zero;

            // Pin memory reference to completely eliminate bounds checking and slicing overhead
            ref float pData = ref MemoryMarshal.GetReference(data);
            int i = 0;

            // 1. Vectorized SIMD Hot Path
            for (; i <= len - width; i += width)
            {
                // Vectorized load straight from the memory address
                Vector<float> x = Vector.LoadUnsafe(ref pData, (uint)i);

                // NaN mask first (x86 maxps returns the 2nd operand on NaN,
                // so the clamp below would silently turn NaN into 0).
                Vector<int> nanMask = Vector.OnesComplement(Vector.Equals(x, x));

                Vector<float> xc = Vector.Min(Vector.Max(x, negLim), lim);
                Vector<float> x3 = xc * xc * xc;
                Vector<float> u = s * (xc + c * x3);

                // --- ZERO-ALLOCATION SIMD TANH APPROXIMATION ---
                // Tanh(u) ≈ sgn(u) * (1 - 1 / (1 + |u| + u^2 + 0.5857 * |u|^3))
                // This keeps the entire math calculation inside CPU vector registers.
                Vector<float> absU = Vector.Abs(u);
                Vector<float> u2 = u * u;
                Vector<float> absU3 = u2 * absU;
                
                Vector<float> denom = one + absU + u2 + (new Vector<float>(0.5857f) * absU3);
                Vector<float> approxTanh = Vector.ConditionalSelect(
                    Vector.LessThan(u, Vector<float>.Zero),
                    -one + (one / denom),
                    one - (one / denom)
                );
                // ------------------------------------------------

                // Final GELU combination step
                Vector<float> result = half * xc * (one + approxTanh);

                // Saturate extreme lanes: x >= 12 -> x, x <= -12 -> 0.
                result = Vector.ConditionalSelect(
                    Vector.GreaterThanOrEqual(x, lim), x,
                    Vector.ConditionalSelect(
                        Vector.LessThanOrEqual(x, negLim), zero, result));

                // Re-inject NaN lanes so non-finite inputs stay fail-fast.
                result = Vector.ConditionalSelect(nanMask, x, result);

                // Direct memory store back into the tensor buffer
                Vector.StoreUnsafe(result, ref pData, (uint)i);
            }

            // 2. Scalar Cleanup Path for remainder elements
            for (; i < len; i++)
            {
                ref float xRef = ref Unsafe.Add(ref pData, i);
                float x = xRef;
                if (float.IsNaN(x)) { xRef = float.NaN; continue; }
                if (x >= 12f) { xRef = x; continue; }
                if (x <= -12f) { xRef = 0f; continue; }
                float u = 0.7978845608f * (x + 0.044715f * x * x * x);
                xRef = 0.5f * x * (1f + MathF.Tanh(u));
            }
        }

        public static void GeluBackwardInto(
            TensorBase input,
            TensorBase outputGradient,
            TensorBase inputGradient)
        {
            ValidateSameShape(input, outputGradient);
            ValidateSameShape(input, inputGradient);

            ReadOnlySpan<float> xSpan = input.ReadOnlySpan;
            ReadOnlySpan<float> dySpan = outputGradient.ReadOnlySpan;
            Span<float> dxSpan = inputGradient.Span;

            int len = xSpan.Length;
            int width = Vector<float>.Count;
            int unrollWidth = width * 2; // Process 2 vector chunks simultaneously

            ref float pX = ref MemoryMarshal.GetReference(xSpan);
            ref float pDy = ref MemoryMarshal.GetReference(dySpan);
            ref float pDx = ref MemoryMarshal.GetReference(dxSpan);

            // Pre-hoisted vector constants
            var vSqrt2OverPi = new Vector<float>(Sqrt2OverPi);
            var vConstC = new Vector<float>(GeluC);
            var vConst3C = new Vector<float>(Gelu3C);
            var vOne = Vector<float>.One;
            var vNegOne = new Vector<float>(-1.0f);
            var vHalf = new Vector<float>(0.5f);
            var vPolyCoeff = new Vector<float>(0.5857f);

            int i = 0;

            // 1. Unrolled Vectorized Hot Path (2x SIMD blocks per iteration)
            for (; i <= len - unrollWidth; i += unrollWidth)
            {
                // Block A
                Vector<float> valA = Vector.LoadUnsafe(ref pX, (uint)i);
                Vector<float> dyA = Vector.LoadUnsafe(ref pDy, (uint)i);

                Vector<float> x2A = valA * valA;
                Vector<float> x3A = x2A * valA;
                Vector<float> uA = vSqrt2OverPi * (valA + vConstC * x3A);

                Vector<float> absUA = Vector.Abs(uA);
                Vector<float> u2A = uA * uA;
                Vector<float> denomA = vOne + absUA + u2A + (vPolyCoeff * u2A * absUA);
                Vector<float> invDenomA = vOne / denomA;
                Vector<float> tA = Vector.ConditionalSelect(
                    Vector.LessThan(uA, Vector<float>.Zero),
                    vNegOne + invDenomA,
                    vOne - invDenomA
                );

                Vector<float> term1A = vHalf * (vOne + tA);
                Vector<float> term2A = vHalf * valA * (vOne - (tA * tA)) * vSqrt2OverPi * (vOne + vConst3C * x2A);
                Vector<float> dxA = dyA * (term1A + term2A);
                Vector.StoreUnsafe(dxA, ref pDx, (uint)i);

                // Block B
                int iB = i + width;
                Vector<float> valB = Vector.LoadUnsafe(ref pX, (uint)iB);
                Vector<float> dyB = Vector.LoadUnsafe(ref pDy, (uint)iB);

                Vector<float> x2B = valB * valB;
                Vector<float> x3B = x2B * valB;
                Vector<float> uB = vSqrt2OverPi * (valB + vConstC * x3B);

                Vector<float> absUB = Vector.Abs(uB);
                Vector<float> u2B = uB * uB;
                Vector<float> denomB = vOne + absUB + u2B + (vPolyCoeff * u2B * absUB);
                Vector<float> invDenomB = vOne / denomB;
                Vector<float> tB = Vector.ConditionalSelect(
                    Vector.LessThan(uB, Vector<float>.Zero),
                    vNegOne + invDenomB,
                    vOne - invDenomB
                );

                Vector<float> term1B = vHalf * (vOne + tB);
                Vector<float> term2B = vHalf * valB * (vOne - (tB * tB)) * vSqrt2OverPi * (vOne + vConst3C * x2B);
                Vector<float> dxB = dyB * (term1B + term2B);
                Vector.StoreUnsafe(dxB, ref pDx, (uint)iB);
            }

            // 2. Standard Single Vector Loop for remaining aligned elements
            for (; i <= len - width; i += width)
            {
                Vector<float> value = Vector.LoadUnsafe(ref pX, (uint)i);
                Vector<float> dy = Vector.LoadUnsafe(ref pDy, (uint)i);

                Vector<float> x2 = value * value;
                Vector<float> x3 = x2 * value;
                Vector<float> u = vSqrt2OverPi * (value + vConstC * x3);

                Vector<float> absU = Vector.Abs(u);
                Vector<float> u2 = u * u;
                Vector<float> denom = vOne + absU + u2 + (vPolyCoeff * u2 * absU);
                Vector<float> invDenom = vOne / denom;

                Vector<float> t = Vector.ConditionalSelect(
                    Vector.LessThan(u, Vector<float>.Zero),
                    vNegOne + invDenom,
                    vOne - invDenom
                );

                Vector<float> term1 = vHalf * (vOne + t);
                Vector<float> term2 = vHalf * value * (vOne - (t * t)) * vSqrt2OverPi * (vOne + vConst3C * x2);
                Vector<float> dx = dy * (term1 + term2);

                Vector.StoreUnsafe(dx, ref pDx, (uint)i);
            }

            // 3. Scalar Cleanup Loop for tail elements
            for (; i < len; i++)
            {
                float val = Unsafe.Add(ref pX, i);
                float dy = Unsafe.Add(ref pDy, i);

                float x2 = val * val;
                float x3 = x2 * val;
                float u = Sqrt2OverPi * (val + GeluC * x3);
                float t = MathF.Tanh(u);

                float deriv = 0.5f * (1f + t) + 0.5f * val * (1f - t * t) * Sqrt2OverPi * (1f + Gelu3C * x2);
                Unsafe.Add(ref pDx, i) = dy * deriv;
            }
        }
        /// <summary>
        /// Provides a direct ReadOnlySpan over the underlying contiguous tensor data.
        /// </summary>
        public static ReadOnlySpan<float> AsSpan(this TensorBase tensor)
        {
            // If TensorBase already provides a ReadOnlySpan property
            return tensor.ReadOnlySpan;
        }

        /// <summary>
        /// Provides a mutable Span over the underlying contiguous tensor data.
        /// </summary>
        public static Span<float> AsWritableSpan(this TensorBase tensor)
        {
            // If TensorBase already provides a Span property
            return tensor.Span;
        }        
        #endregion 
    }
}