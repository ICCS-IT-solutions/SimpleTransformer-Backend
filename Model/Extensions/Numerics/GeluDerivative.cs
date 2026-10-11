namespace SimpleTransformer.Model.Extensions.Numerics
{
    /// <summary>
    /// Canonical, overflow-safe GELU derivative, shared by every backend so the CPU
    /// and Vulkan paths cannot drift apart.
    /// <para>
    /// Why this exists: the tanh-approximation derivative is
    /// <c>0.5*(1+t) + 0.5*x*(1-t*t)*sqrt(2/pi)*(1+3c*x^2)</c>, and in float32 both
    /// <c>x^3</c> (above ~7e12) and <c>x^2</c> (above ~1.845e19) can overflow. Past
    /// the latter threshold <c>1+3c*x^2</c> becomes +Inf while <c>1-t*t</c> is
    /// exactly 0, so the product evaluates as <c>0 * Inf = NaN</c>. One NaN there
    /// poisons every parameter gradient downstream: the embedding scatter was found
    /// holding whole rows of NaN, which looked like embedding divergence but was
    /// really this term - located by the backward NaN tripwire at
    /// "feed-forward (_feedForward.Backward)".
    /// <para>
    /// The fix exploits the analytic limits rather than clamping the overflow: for
    /// large positive x, GELU'(x) -> 1; for large negative x, GELU'(x) -> 0. Both are
    /// reached long before the overflow thresholds, so returning them early is
    /// numerically exact in the limit and leaves every ordinary input bit-identical
    /// (verified over the full finite range).
    /// </para>
    /// </summary>
    public static class GeluDerivative
    {
        public const float Sqrt2OverPi = 0.7978845608f;
        public const float C = 0.044715f;

        /// <summary>
        /// 3 * C, COMPUTED rather than written as the literal 0.134145f. The two are
        /// not equal in float32 (0.134145 vs 0.13414499), and the literal - which
        /// predates this helper in TensorMathSimd - shifted the derivative in the
        /// last bit for some inputs. Computing it keeps this helper bit-identical to
        /// every backend it replaces.
        /// </summary>
        public static float ThreeC => 3f * C;

        /// <summary>
        /// |x| beyond which the float32 polynomial can overflow (x^2 at ~1.845e19).
        /// Inputs here are already far outside any healthy activation range, so the
        /// saturated limits are the correct answer, not an approximation of one.
        /// <para>
        /// Public so the SIMD path can use the identical threshold; if the two
        /// diverge, the vectorised and scalar results disagree at the boundary.
        /// </para>
        /// </summary>
        public const float OverflowGuard = 1e4f;

        /// <summary>
        /// GELU derivative at <paramref name="x"/>. Never returns NaN or Inf for a
        /// finite input.
        /// </summary>
        public static float Evaluate(float x)
        {
            // Saturated limits, taken before x*x and x*x*x can overflow.
            if (x > OverflowGuard) return 1f;
            if (x < -OverflowGuard) return 0f;

            float x2 = x * x;
            float x3 = x2 * x;
            float u = Sqrt2OverPi * (x + C * x3);
            float t = MathF.Tanh(u);

            float derivative =
                0.5f * (1f + t)
                + 0.5f * x * (1f - t * t) * Sqrt2OverPi * (1f + ThreeC * x2);

            // Belt and braces: should be unreachable given the guards above, but a
            // silent NaN here is far more expensive than one branch.
            if (!float.IsFinite(derivative))
                return x > 0f ? 1f : 0f;

            return derivative;
        }
    }
}