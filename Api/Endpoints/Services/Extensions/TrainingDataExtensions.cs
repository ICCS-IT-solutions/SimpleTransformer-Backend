using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

public static class TrainingDataExtensions
{
    public static IReadOnlyList<TrainingSample> CreateTrainingSamples(TransformerModel model, ITokenizer tokenizer, string src)
    {
        int[] tokens = tokenizer.Encode(src);
        List<TrainingSample> data = new();

        int window = model.Config.MaxSequenceLength;
        
        for (int i = 0; i <= tokens.Length - window - 1; i += window)
        {
            Tensor input = new(window);
            Tensor target = new(window);
            
            for (int j = 0; j < window; j++)
            {
                input[j] = tokens[i + j];
                target[j] = tokens[i + j + 1];
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