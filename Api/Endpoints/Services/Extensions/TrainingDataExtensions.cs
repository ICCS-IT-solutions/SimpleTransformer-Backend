using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

public static class TrainingDataExtensions
{
    /// <summary>
    /// Build the next-token training samples over a token stream.
    /// <para>
    /// Windows are non-overlapping with stride <c>window</c> (sample w covers
    /// tokens w*window .. w*window+window), and each sample is
    /// tokens[w..w+window) -> tokens[w+1..w+window+1). The final window is
    /// emitted PADDED rather than discarded when the stream does not divide
    /// evenly: its real tokens are kept, the input tail is filled with the pad
    /// id, and the target tail is marked with
    /// <see cref="TransformerModel.IgnoreIndex"/> so the loss ignores it and the
    /// attention mask refuses to attend across it. Without this, up to
    /// <c>window-1</c> tokens are dropped every epoch - negligible on a large
    /// corpus, but a large fraction of a small one.
    /// </para>
    /// </summary>
    public static IReadOnlyList<TrainingSample> CreateTrainingSamples(TransformerModel model, ITokenizer tokenizer, string src)
    {
        int[] tokens = tokenizer.Encode(src);
        List<TrainingSample> data = new();

        int window = model.Config.MaxSequenceLength;
        int padId = tokenizer.PadTokenId;
        int ignoreIndex = TransformerModel.IgnoreIndex;

        // Floor division gives the count of fully-populated windows; add one more
        // when a remainder exists so the tail is padded instead of discarded.
        int fullWindows = tokens.Length > window ? (tokens.Length - 1) / window : 0;
        bool hasTail = tokens.Length > 0 && fullWindows * window < tokens.Length;
        int windows = fullWindows + (hasTail ? 1 : 0);

        for (int w = 0; w < windows; w++)
        {
            int start = w * window;

            // Real tokens available for the input row, and for the (shifted by
            // one) target row. The target row is always the shorter of the two.
            int validInput = Math.Clamp(tokens.Length - start, 0, window);
            int validTarget = Math.Clamp(tokens.Length - start - 1, 0, window);

            Tensor input = new(window);
            Tensor target = new(window);

            for (int j = 0; j < window; j++)
            {
                input[j] = j < validInput ? tokens[start + j] : padId;
                target[j] = j < validTarget ? tokens[start + j + 1] : ignoreIndex;
            }

            data.Add(new TrainingSample { Input = input, Target = target });
        }

        return data;
    }

    public static IReadOnlyList<MiniBatch> CreateMiniBatches(TransformerModel model, IReadOnlyList<TrainingSample> samples)
    {
        List<MiniBatch> miniBatches = new();
        
        // Read directly from the model's TrainingConfig
        int batchSize = model.TrainingConfig.BatchSize;
        bool dropLast = model.TrainingConfig.DropLast;

        //A short trailing batch is trained, but it is a poor gradient estimate
        //taken from far fewer samples than the rest of the epoch. Dropping it
        //keeps every optimizer step based on a full batch, at the cost of
        //leaving up to BatchSize-1 samples unused.
        //
        //That trade is only worth making when a full batch actually exists:
        //if the dataset is smaller than one batch, dropping the remainder would
        //leave nothing to train on, so the single partial batch is kept.
        int fullBatches = samples.Count / batchSize;
        bool willDropRemainder =
            dropLast && samples.Count % batchSize != 0 && fullBatches > 0;

        int limit = willDropRemainder
            ? samples.Count - (samples.Count % batchSize)
            : samples.Count;

        for (int start = 0; start < limit; start += batchSize)
        {
            int currentBatchSize = Math.Min(batchSize, limit - start);
            int sequenceLength = samples[start].Input.Shape[0];

            Tensor inputBatch = new Tensor(currentBatchSize, sequenceLength);
            Tensor targetBatch = new Tensor(currentBatchSize, sequenceLength);

            for (int r = 0; r < currentBatchSize; r++)
            {
                var sample = samples[start + r];
                for (int c = 0; c < sequenceLength; c++)
                {
                    inputBatch[r, c] = sample.Input[c];
                    targetBatch[r, c] = sample.Target[c];
                }
            }

            miniBatches.Add(new MiniBatch
            {
                Inputs = inputBatch, 
                Targets = targetBatch
            });
        }
        return miniBatches;
    }
}