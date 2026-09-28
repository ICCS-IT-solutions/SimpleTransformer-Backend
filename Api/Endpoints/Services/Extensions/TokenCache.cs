using System.Buffers.Binary;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Versioned pre-tokenized corpus cache. Raw text is encoded ONCE (at job
    /// creation) into a flat token-id stream so every epoch / resume / retry
    /// streams ids instead of re-running BPE over a giant string.
    /// Layout (little-endian): header (fixed 64 bytes) + token body.
    /// Dtype is u16 when vocabSize &lt;= 65535, u32 otherwise.
    /// </summary>
    public static class TokenCache
    {
        public const uint Magic = 0x534B4E54; // "STKN" LE
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
            public Guid VocabId;
            public uint BodyCrc32;
        }

        public static DType DtypeFor(int vocabSize) =>
            vocabSize <= ushort.MaxValue ? DType.U16 : DType.U32;

        public static int BytesPerToken(DType dtype) => dtype == DType.U16 ? 2 : 4;

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
            h.VocabId.TryWriteBytes(b.AsSpan(36, 16));
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
                VocabId = new Guid(b.AsSpan(36, 16)),
                BodyCrc32 = BinaryPrimitives.ReadUInt32LittleEndian(b.AsSpan(52)),
            };
            long expectLen = HeaderBytes + h.TokenCount * BytesPerToken(h.Dtype);
            if (fs.Length < expectLen)
                throw new InvalidDataException($"Token cache '{binPath}' body truncated.");
            return h;
        }

        public static void Validate(Header h, Guid expectedVocabId, int expectedVocabSize, int expectedTokenizerType)
        {
            if (h.VocabId != expectedVocabId)
                throw new InvalidOperationException(
                    $"Token cache vocab {h.VocabId} != job vocab {expectedVocabId}. Rebuild.");
            if (h.VocabSize != expectedVocabSize)
                throw new InvalidOperationException(
                    $"Token cache vocab size {h.VocabSize} != live {expectedVocabSize}. Rebuild.");
            if (h.TokenizerType != expectedTokenizerType)
                throw new InvalidOperationException("Token cache tokenizer mismatch. Rebuild.");
            if (h.TokenCount <= 0)
                throw new InvalidOperationException("Token cache is empty.");
        }
    }
}
