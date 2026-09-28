using System.Buffers.Binary;
using System.Text;

namespace SimpleTransformer.Api.Endpoints.Services.Extensions
{
    /// <summary>
    /// Streaming writer/reader for the TokenCache body. The writer encodes one
    /// document at a time (peak = one doc, never whole corpus); the reader
    /// pages windows straight off disk into reusable float buffers.
    /// </summary>
    public static class TokenCacheIO
    {
        public static TokenCache.Header Write(
            string binPath,
            IEnumerable<string> documents,
            Func<string, int[]> encode,
            int tokenizerType,
            int vocabSize,
            Guid vocabId)
        {
            var dtype = TokenCache.DtypeFor(vocabSize);
            int bpt = TokenCache.BytesPerToken(dtype);
            long docCount = 0;
            uint crc = 0;

            Directory.CreateDirectory(Path.GetDirectoryName(binPath) ?? ".");
            using var fs = new FileStream(binPath, FileMode.Create, FileAccess.Write,
                FileShare.None, bufferSize: 1 << 20, useAsync: false);
            fs.Write(new byte[TokenCache.HeaderBytes]); // reserve, patch at end

            var buf = new byte[Math.Max(4096, bpt * 2048)];
            int used = 0;
            void Flush()
            {
                if (used == 0) return;
                fs.Write(buf, 0, used);
                crc = Crc(crc, buf.AsSpan(0, used));
                used = 0;
            }
            void Emit(int id)
            {
                if (id < 0) throw new ArgumentOutOfRangeException(nameof(id));
                if (dtype == TokenCache.DType.U16 && id > ushort.MaxValue)
                    throw new InvalidOperationException(
                        $"Token id {id} exceeds uint16; use u32 cache.");
                if (used + bpt > buf.Length) Flush();
                if (dtype == TokenCache.DType.U16)
                    BinaryPrimitives.WriteUInt16LittleEndian(buf.AsSpan(used), (ushort)id);
                else
                    BinaryPrimitives.WriteUInt32LittleEndian(buf.AsSpan(used), (uint)id);
                used += bpt;
            }

            foreach (var doc in documents)
            {
                // Whitespace-only docs would emit a bogus bos/eos pair and
                // inflate DocCount, so skip them the same way ReadDocuments
                // never produces them.
                if (string.IsNullOrWhiteSpace(doc)) continue;
                foreach (var id in encode(doc)) Emit(id);
                docCount++;
            }
            Flush();
            long tokenCount = (fs.Position - TokenCache.HeaderBytes) / bpt;

            var header = new TokenCache.Header
            {
                Dtype = dtype,
                TokenizerType = tokenizerType,
                VocabSize = vocabSize,
                TokenCount = tokenCount,
                DocCount = docCount,
                VocabId = vocabId,
                BodyCrc32 = crc,
            };
            fs.Seek(0, SeekOrigin.Begin);
            fs.Write(TokenCache.EncodeHeader(header));
            return header;
        }

        /// <summary>
        /// Page one window [offset, offset+length) as floats into dst.
        /// Convenience overload for one-shot / whole-body reads; allocates a
        /// scratch buffer. Use the scratch overload in per-window loops.
        /// </summary>
        public static void ReadWindowAsFloat(
            FileStream fs, TokenCache.Header h, long tokenOffset, int length, Span<float> dst)
            => ReadWindowAsFloat(fs, h, tokenOffset, length, dst,
                new byte[length * TokenCache.BytesPerToken(h.Dtype)]);

        /// <summary>
        /// Zero-allocation variant: the caller reuses ONE scratch byte buffer
        /// (at least length * bytesPerToken) across every window of an epoch, so
        /// streaming a 50M-token corpus adds no per-window garbage.
        /// </summary>
        public static void ReadWindowAsFloat(
            FileStream fs, TokenCache.Header h, long tokenOffset, int length,
            Span<float> dst, byte[] scratch)
        {
            int bpt = TokenCache.BytesPerToken(h.Dtype);
            int needed = length * bpt;
            if (scratch.Length < needed)
                throw new ArgumentException(
                    $"Scratch buffer must hold at least {needed} bytes.", nameof(scratch));
            fs.Seek(TokenCache.HeaderBytes + tokenOffset * bpt, SeekOrigin.Begin);
            if (h.Dtype == TokenCache.DType.U16)
            {
                ReadExactly(fs, scratch, length * 2);
                for (int i = 0; i < length; i++)
                    dst[i] = BinaryPrimitives.ReadUInt16LittleEndian(scratch.AsSpan(i * 2));
            }
            else
            {
                ReadExactly(fs, scratch, length * 4);
                for (int i = 0; i < length; i++)
                    dst[i] = BinaryPrimitives.ReadUInt32LittleEndian(scratch.AsSpan(i * 4));
            }
        }

        /// <summary>
        /// Lazily split a cleaned corpus.txt back into documents. The writer
        /// separates docs with a blank line and single-line docs stay single
        /// lines, so: accumulate non-blank lines until a blank line. Peak
        /// memory is one document, never the whole file.
        /// </summary>
        public static IEnumerable<string> ReadDocuments(string path)
        {
            using var reader = new StreamReader(path, Encoding.UTF8,
                detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 16);
            var sb = new StringBuilder();
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    if (sb.Length > 0) { yield return sb.ToString(); sb.Clear(); }
                }
                else
                {
                    if (sb.Length > 0) sb.Append(' ');
                    sb.Append(line.Trim());
                }
            }
            if (sb.Length > 0) yield return sb.ToString();
        }

        private static void ReadExactly(FileStream fs, byte[] buf, int count)
        {
            int off = 0;
            while (off < count)
            {
                int n = fs.Read(buf, off, count - off);
                if (n == 0) throw new EndOfStreamException("Token cache truncated mid-window.");
                off += n;
            }
        }

        private static uint Crc(uint crc, ReadOnlySpan<byte> data)
        {
            crc ^= 0xFFFFFFFFu;
            foreach (var by in data)
            {
                crc ^= by;
                for (int k = 0; k < 8; k++)
                    crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
            }
            return crc ^ 0xFFFFFFFFu;
        }
    }
}
