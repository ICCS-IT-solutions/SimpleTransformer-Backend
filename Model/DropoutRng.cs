using System;
using System.Buffers.Binary;
using System.Security.Cryptography;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// xoshiro256** - a 4x ulong state PRNG owned by this project rather than
    /// <see cref="System.Random"/>.
    /// <para>
    /// Why not <see cref="System.Random"/>: it exposes no reseed API, cannot be
    /// serialized, and .NET explicitly reserves the right to change its
    /// algorithm between runtime versions - so it cannot back a reproducibility
    /// guarantee. This type can: the state is four ulongs we write to the
    /// checkpoint ourselves.
    /// </para>
    /// <para>
    /// This is a struct so it can live inline in a dropout layer with no
    /// allocation and no GC pressure on the per-step path. It is NOT
    /// thread-safe by design - each dropout site keeps its own instance, which
    /// is required because the training path fans out to Parallel.For.
    /// </para>
    /// </summary>
    public struct DropoutRng
    {
        private ulong _s0;
        private ulong _s1;
        private ulong _s2;
        private ulong _s3;

        /// <summary>splitmix64 increment (the 64-bit golden gamma).</summary>
        private const ulong GoldenGamma = 0x9E3779B97F4A7C15UL;

        /// <summary>
        /// Expands a single 64-bit seed into the four-word state via splitmix64,
        /// which is the seeding routine recommended for xoshiro.
        /// </summary>
        public static DropoutRng FromSeed(ulong seed)
        {
            ulong x = seed;
            ulong s0 = NextSplitMix(ref x);
            ulong s1 = NextSplitMix(ref x);
            ulong s2 = NextSplitMix(ref x);
            ulong s3 = NextSplitMix(ref x);

            return FromState(s0, s1, s2, s3);
        }

        /// <summary>
        /// Restores a generator from a state previously read out of a
        /// checkpoint. xoshiro is undefined for an all-zero state, so that one
        /// pathological value is replaced rather than allowed to collapse the
        /// generator to a constant stream.
        /// </summary>
        public static DropoutRng FromState(ulong s0, ulong s1, ulong s2, ulong s3)
        {
            if ((s0 | s1 | s2 | s3) == 0UL)
                s0 = GoldenGamma;

            return new DropoutRng { _s0 = s0, _s1 = s1, _s2 = s2, _s3 = s3 };
        }

        /// <summary>Reads the four state words for checkpoint serialization.</summary>
        public readonly void GetState(out ulong s0, out ulong s1, out ulong s2, out ulong s3)
        {
            s0 = _s0;
            s1 = _s1;
            s2 = _s2;
            s3 = _s3;
        }

        /// <summary>Next 64-bit output (xoshiro256**).</summary>
        public ulong NextUInt64()
        {
            unchecked
            {
                ulong result = RotateLeft(_s1 * 5UL, 7) * 9UL;
                ulong t = _s1 << 17;

                _s2 ^= _s0;
                _s3 ^= _s1;
                _s1 ^= _s2;
                _s0 ^= _s3;
                _s2 ^= t;
                _s3 = RotateLeft(_s3, 45);

                return result;
            }
        }

        /// <summary>
        /// Uniform float in [0, 1), built from the top 24 bits so every value is
        /// exactly representable as a float. Same construction (and therefore
        /// same statistical quality) as Random.NextSingle.
        /// </summary>
        public float NextSingle() => (NextUInt64() >> 40) * (1.0f / 16777216.0f);

        /// <summary>
        /// splitmix64 finalizer. Bijective (xorshifts plus multiplication by an
        /// odd constant), which is what makes <see cref="Mix(ulong, ulong)"/>
        /// injective in each argument.
        /// </summary>
        public static ulong Finalize(ulong z)
        {
            unchecked
            {
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                return z ^ (z >> 31);
            }
        }

        /// <summary>
        /// Combines two independent values into one seed. Because
        /// <see cref="Finalize"/> is a bijection, distinct inputs give distinct
        /// outputs, so two different (step, batch item) pairs can never share a
        /// mask. Used to derive per-step / per-item substreams from a site's
        /// persisted salt.
        /// </summary>
        public static ulong Mix(ulong a, ulong b) => Finalize(a ^ Finalize(b));

        /// <summary>
        /// Fresh cryptographic entropy for a site's initial salt. Each training
        /// run gets a new salt, so two independent runs of the same model see
        /// different masks, while a resume restores the original sequence.
        /// </summary>
        public static ulong NextEntropy()
        {
            Span<byte> bytes = stackalloc byte[8];
            RandomNumberGenerator.Fill(bytes);
            return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
        }

        private static ulong RotateLeft(ulong x, int k) => (x << k) | (x >> (64 - k));

        private static ulong NextSplitMix(ref ulong state)
        {
            unchecked
            {
                state += GoldenGamma;
                return Finalize(state);
            }
        }
    }
}