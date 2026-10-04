using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore.Internal;

namespace SimpleTransformer.Model.Tokenizer
{
    public interface ITokenizer
    {
        TokenizerType Type { get; }
        int EosTokenId { get; }

        /// <summary>
        /// Id used to fill padded input positions. Padding is expressed with
        /// this id in the INPUT row and with TransformerModel.IgnoreIndex in the
        /// TARGET row, so the loss can skip it and the attention mask can block
        /// across it. Defined to be 0 by every tokenizer's vocabulary build
        /// (SpecialTokens.Pad is assigned first).
        /// </summary>
        int PadTokenId { get; }

        int[] Encode(string text);

        string Decode(ReadOnlySpan<int> tokens);

        int VocabularySize { get; }
    }
    public static class SpecialTokens
    {
        public const string Pad = "<pad>";
        public const string Unknown = "<unk>";
        public const string BeginningOfSequence = "<bos>";
        public const string EndOfSequence = "<eos>";
        public const string Mask = "<mask>";
    }
}