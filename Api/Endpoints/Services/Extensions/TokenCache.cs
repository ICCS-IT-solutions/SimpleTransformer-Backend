using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Versioned pre-tokenized corpus cache. Raw text is encoded ONCE (at job
    /// creation) into a flat token-id stream so every epoch / resume / retry
    /// streams ids instead of re-running the tokenizer over a giant string.
    /// Layout (little-endian): header (fixed 64 bytes) + token body.
    /// Dtype is u16 when vocabSize &lt;= 65535, u32 otherwise.
    /// The ids come from the tokenizer the app ACTUALLY trains and infers with
    /// (the process-wide singleton), so the header pins that tokenizer's
    /// fingerprint, size and type rather than any job-specific vocabulary.
    /// </summary>
    public static class TokenCache
    {
        // Encoded little-endian, so the first four bytes on disk are the ASCII
        // bytes 53 54 4B 4E = "STKN" exactly as a hex dump shows it.
        public const uint Magic = 0x4E4B5453; // "STKN"
        public const int Version = 1;
        public const int HeaderBytes = 64;

        public enum DType : byte { U16 = 0, U32 = 1 }

        public sealed class Header
        {
            public DType Dtype;
            public int TokenizerType;
            public int VocabSize;
            public long TokenCount;
            public long DocCount;
            public Guid TokenizerFingerprint;
            public uint BodyCrc32;
        }

        public static DType DtypeFor(int vocabSize) =>
            vocabSize <= ushort.MaxValue ? DType.U16 : DType.U32;

        public static int BytesPerToken(DType dtype) => dtype == DType.U16 ? 2 : 4;

        /// <summary>
        /// Fixed 64-byte header layout (little-endian throughout):
        /// 0..3 magic "STKN" | 4..7 version | 8 dtype | 12..15 tokenizer type |
        /// 16..19 vocab size | 20..27 token count | 28..35 doc count |
        /// 36..51 tokenizer fingerprint | 52..55 body CRC32 | 56..63 reserved 0.
        /// </summary>
        public static byte[] EncodeHeader(Header h)
        {
            var b = new byte[HeaderBytes];
            BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(0), Magic);
            BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(4), Version);
            b[8] = (byte)h.Dtype;
            BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(12), h.TokenizerType);
            BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(16), h.VocabSize);
            BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(20), h.TokenCount);
            BinaryPrimitives.WriteInt64LittleEndian(b.AsSpan(28), h.DocCount);
            // 36..51: tokenizer fingerprint (16 bytes)
            h.TokenizerFingerprint.TryWriteBytes(b.AsSpan(36, 16));
            BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(52), h.BodyCrc32);
            return b;
        }

        public static Header ReadHeader(string binPath)
        {
            using var fs = new FileStream(binPath, FileMode.Open, FileAccess.Read,
                FileShare.Read, bufferSize: 4096, useAsync: false);
            if (fs.Length < HeaderBytes)
                throw new InvalidDataException($"Token cache '{binPath}' is truncated.");
            var b = new byte[HeaderBytes];
            if (fs.Read(b, 0, HeaderBytes) != HeaderBytes)
                throw new InvalidDataException($"Token cache '{binPath}' header unreadable.");
            if (BinaryPrimitives.ReadUInt32LittleEndian(b.AsSpan(0)) != Magic)
                throw new InvalidDataException($"Token cache '{binPath}' bad magic (not .stbin).");
            if (BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(4)) != Version)
                throw new InvalidDataException($"Token cache '{binPath}' unsupported version.");
            var h = new Header
            {
                Dtype = (DType)b[8],
                TokenizerType = BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(12)),
                VocabSize = BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(16)),
                TokenCount = BinaryPrimitives.ReadInt64LittleEndian(b.AsSpan(20)),
                DocCount = BinaryPrimitives.ReadInt64LittleEndian(b.AsSpan(28)),
                TokenizerFingerprint = new Guid(b.AsSpan(36, 16)),
                BodyCrc32 = BinaryPrimitives.ReadUInt32LittleEndian(b.AsSpan(52)),
            };
            long expectLen = HeaderBytes + h.TokenCount * BytesPerToken(h.Dtype);
            if (fs.Length < expectLen)
                throw new InvalidDataException($"Token cache '{binPath}' body truncated.");
            return h;
        }

        /// <summary>
        /// Staleness guard. Throws if this cache was not produced by the exact
        /// tokenizer id space the training loop is about to use, or if it is
        /// empty. A caller that catches (as RunTrainingLoop does) simply falls
        /// back to tokenizing the text, so a mismatch costs time, never correctness.
        /// </summary>
        public static void Validate(Header h, Guid expectedFingerprint,
            int expectedVocabSize, int expectedTokenizerType)
        {
            if (h.TokenizerFingerprint != expectedFingerprint)
                throw new InvalidOperationException(
                    $"Token cache tokenizer fingerprint {h.TokenizerFingerprint} != live {expectedFingerprint}. Rebuild.");
            if (h.VocabSize != expectedVocabSize)
                throw new InvalidOperationException(
                    $"Token cache vocab size {h.VocabSize} != live {expectedVocabSize}. Rebuild.");
            if (h.TokenizerType != expectedTokenizerType)
                throw new InvalidOperationException("Token cache tokenizer mismatch. Rebuild.");
            if (h.TokenCount <= 0)
                throw new InvalidOperationException("Token cache is empty.");
        }

        /// <summary>
        /// Stable identity of a vocabulary's id space: a content hash over every
        /// token/id pair. Dictionary iteration order varies with the loader, so
        /// entries are sorted ordinally first - the same vocabulary always yields
        /// the same Guid, and ANY change to the id space (an edited token, a
        /// re-mapped id, a different size) yields a different one, which is what
        /// forces a cache rebuild. This is a cache key, not a security control.
        /// </summary>
        public static Guid Fingerprint(Vocabulary vocabulary)
        {
            var ordered = vocabulary.TokenToId
                .OrderBy(kv => kv.Key, StringComparer.Ordinal)
                .ThenBy(kv => kv.Value);

            var sb = new StringBuilder(vocabulary.TokenToId.Count * 24);
            foreach (var kv in ordered)
            {
                sb.Append(kv.Key).Append('\u001F').Append(kv.Value).Append('\u001E');
            }

            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()));
            return new Guid(hash.AsSpan(0, 16));
        }
    }
}
