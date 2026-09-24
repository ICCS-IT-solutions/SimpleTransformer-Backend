using System.Text;
using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Api.ManagementEngine;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Config;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class InferenceService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly ConfigManager _configManager;
        private readonly ITokenizer _tokenizer;
        private readonly ModelManager _modelManager;

        public InferenceService(
            ITokenizer tokenizer, 
            ConfigManager configManager, 
            IDbContextFactory<AppDbContext> dbFactory, 
            ModelManager modelManager)
        {
            _tokenizer = tokenizer;
            _configManager = configManager;
            _dbFactory = dbFactory;
            _modelManager = modelManager;
        }

        public async Task<ApiResponse<InferenceResponse>> Infer(InferenceRequest req)
        {
            using var db = await _dbFactory.CreateDbContextAsync();
            var modelEntry = db.TransformerModels.FirstOrDefault(x => x.EntryId == req.TransformerModelId);

            if (modelEntry == null)
            {
                return new ApiResponse<InferenceResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Error,
                    StatusCode = 404
                };
            }

            if (modelEntry.IsLoaded == false)
            {
                return new ApiResponse<InferenceResponse>
                {
                    Message = "Model not loaded.",
                    Status = ResponseStatus.Error,
                    StatusCode = 404
                };
            }

            //Only once the model entry has been found, can I create the model

            var model = await _modelManager.LoadModelAsync(modelEntry.EntryId);

            if(model == null)
            {
                return new ApiResponse<InferenceResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Error,
                    StatusCode = 404
                };
            }
            //Now I can block inference if training is underway.
            if(model.IsTraining == true)
            {
                return new ApiResponse<InferenceResponse>
                {
                    Message = "Model is currently training. Please try again later.",
                    Status = ResponseStatus.Error,
                    StatusCode = 500
                };
            }
            //Load and hydrate checkpoint
            // -------------------------------------------------------------
            // Checkpoint Hydration Logic
            // -------------------------------------------------------------
            if (req.TrainingCheckpointId.HasValue)
            {
                // Only load if the requested checkpoint differs from what is currently in memory
                if (model.LoadedCheckpointId != req.TrainingCheckpointId.Value)
                {
                    var checkpointEntry = await db.TrainingCheckpoints
                        .FirstOrDefaultAsync(x => x.EntryId == req.TrainingCheckpointId.Value);

                    if (checkpointEntry == null)
                    {
                        return new ApiResponse<InferenceResponse>
                        {
                            Message = $"Checkpoint {req.TrainingCheckpointId.Value} not found.",
                            Status = ResponseStatus.Error,
                            StatusCode = 404
                        };
                    }

                    // Ensure checkpoint matches model
                    if (checkpointEntry.TransformerModelId != Guid.Empty &&
                        checkpointEntry.TransformerModelId != model.TransformerModelId)
                    {
                        return new ApiResponse<InferenceResponse>
                        {
                            Message = "Checkpoint does not belong to the selected model.",
                            Status = ResponseStatus.Error,
                            StatusCode = 400
                        };
                    }

                    string fullPath = Path.Combine(checkpointEntry.Filepath, checkpointEntry.Filename);
                    if (!File.Exists(fullPath))
                    {
                        return new ApiResponse<InferenceResponse>
                        {
                            Message = $"Checkpoint binary file not found at '{fullPath}'.",
                            Status = ResponseStatus.Error,
                            StatusCode = 404
                        };
                    }

                    try
                    {
                        await using var stream = File.OpenRead(fullPath);
                        TransformerModel.LoadCheckpoint(stream, model);
                        
                        model.LoadedCheckpointId = checkpointEntry.EntryId;
                    }
                    catch (Exception ex)
                    {
                        return new ApiResponse<InferenceResponse>
                        {
                            Message = $"Failed to load checkpoint '{checkpointEntry.Filename}': {ex.Message}",
                            Status = ResponseStatus.Error,
                            StatusCode = 500
                        };
                    }
                }
            }            


            //Need a way to block inference if training is underway.
            if (string.IsNullOrEmpty(req.InputText)) 
                throw new ArgumentException("Input must not be empty or null.");

            // 1. Tokenize initial prompt
            var inputTokens = _tokenizer.Encode(req.InputText);

            // 2. Set generation limit (default to req.MaxTokens or fallback, e.g., 20)
            int maxNewTokens = req.GenerationParameters.MaxTokens > 0 ? req.GenerationParameters.MaxTokens : 20;

            // 3. Run Auto-Regressive Generation
            // Change line 33:
            var (generatedTokens, lastLogits, lastProbabilities, lastHiddenState) = GenerateSequence(model,
                inputTokens.ToList(), maxNewTokens, req.GenerationParameters.Penalty, req.GenerationParameters.TopK, 
                req.GenerationParameters.TopP, req.GenerationParameters.Temperature);

            // 4. Decode full generated output
            var outputText = _tokenizer.Decode(generatedTokens.ToArray());

            var response = new ApiResponse<InferenceResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new InferenceResponse
                {
                    OutputText = string.IsNullOrEmpty(outputText) 
                        ? "Could not generate usable output from the tokens." 
                        : outputText
                }
            };

#if DEBUG
            // Dump debug file based on the final prediction pass
            var debugOutputSb = new StringBuilder();
            debugOutputSb.AppendLine("Input: " + req.InputText);
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Model vocabulary size: " + model.Config.VocabSize);
            debugOutputSb.AppendLine("Tokenizer vocabulary size: " + _tokenizer.VocabularySize);
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Initial Prompt Tokens: " + string.Join(", ", inputTokens.Select(x => x.ToString())));
            debugOutputSb.AppendLine("Full Generated Tokens: " + string.Join(", ", generatedTokens.Select(x => x.ToString())));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Last Pass Probabilities per Row:\n" + DumpRows(lastProbabilities));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Logits: " + DumpFirstRowLogits(lastLogits));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Top probability per row:\n" + DumpArgMax(lastProbabilities));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Hidden state:\n" + DumpHiddenState(lastHiddenState));
            File.WriteAllText("debug_prediction_output.txt", debugOutputSb.ToString());
#endif

            return response;
        }

        public async Task<ApiResponse<List<TrainingCheckpointEntry>>> GetCheckpointsForModel(Guid transformerModelId)
        {
            using var db = await _dbFactory.CreateDbContextAsync();
            var modelEntry = db.TransformerModels.FirstOrDefault(x => x.EntryId == transformerModelId);

            if (modelEntry == null)
            {
                return new ApiResponse<List<TrainingCheckpointEntry>>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Error,
                    StatusCode = 404
                };
            }

            if (modelEntry.IsLoaded == false)
            {
                return new ApiResponse<List<TrainingCheckpointEntry>>
                {
                    Message = "Model not loaded.",
                    Status = ResponseStatus.Error,
                    StatusCode = 404
                };
            }

            var foundCheckpoints = await db.TrainingCheckpoints
            .Where(cp => cp.TransformerModelId == transformerModelId)
            .ToListAsync();
            
            if(foundCheckpoints == null || foundCheckpoints.Count == 0)
            {
                return new ApiResponse<List<TrainingCheckpointEntry>>
                {
                    Message = "No checkpoints found.",
                    Status = ResponseStatus.Error,
                    StatusCode = 404
                };
            }

            return new ApiResponse<List<TrainingCheckpointEntry>>
            {
                Message = "Checkpoints retrieved successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = foundCheckpoints
            };  
        }

        /// <summary>
        /// Auto-regressive generation loop
        /// </summary>
        private (List<int> tokens, TensorBase lastLogits, TensorBase lastProbs, TensorBase lastHiddenState) 
            GenerateSequence(
                TransformerModel model,
                List<int> promptTokens, 
                int maxNewTokens, 
                float penalty = 1.2f, 
                int topK = 10, 
                float topP = 0.9f, 
                float temperature = 0.8f
                )
        {
            var currentSequence = new List<int>(promptTokens);

            TensorBase lastLogits = null!;
            TensorBase lastProbs = null!;
            TensorBase lastHiddenState = null!;

            for (int i = 0; i < maxNewTokens; i++)
            {
                int contextSize = Math.Min(currentSequence.Count, model.Config.MaxSequenceLength);
                var context = currentSequence
                    .GetRange(currentSequence.Count - contextSize, contextSize)
                    .ToArray();

                var inputTensor = TokenizationUtilities.FromTokenIds(context);

                var (_, _, logits, probabilities, hiddenState) = model.Predict(inputTensor);

                lastLogits = logits;
                lastProbs = probabilities;
                lastHiddenState = hiddenState;

                // 1. Apply repetition penalty directly to logits
                if (penalty != 1.0f)
                {
                    ApplyRepetitionPenalty(logits, currentSequence, penalty: penalty);
                }

                // 2. Sample next token using Top-K, Top-P, and Temperature combined
                int nextTokenId = SampleNextTokenFromLogits(
                    logits, 
                    topK: topK, 
                    topP: topP, 
                    temperature: temperature);

                currentSequence.Add(nextTokenId);

                if (nextTokenId == _tokenizer.EosTokenId)
                    break;
            }

            return (currentSequence, lastLogits, lastProbs, lastHiddenState);
        }

        private void ApplyRepetitionPenalty(TensorBase logits, List<int> generatedTokens, float penalty = 1.2f)
        {
            int lastRow = logits.Rows - 1;

            foreach (var tokenId in generatedTokens.Distinct())
            {
                if (tokenId >= logits.Cols) continue; // Boundary guard

                float score = logits[lastRow, tokenId];
                logits[lastRow, tokenId] = score < 0 ? score * penalty : score / penalty;
            }
        }

        private int SampleNextTokenFromLogits(
            TensorBase logits, 
            int topK = 10, 
            float topP = 0.9f, 
            float temperature = 0.8f)
        {
            int lastRow = logits.Rows - 1;
            int vocabSize = logits.Cols;

            // 1. Apply Temperature directly to raw logits first (Logits / Temp)
            float temp = Math.Max(temperature, 0.01f);
            var scaledLogits = new (int Index, float Logit)[vocabSize];
            for (int col = 0; col < vocabSize; col++)
            {
                scaledLogits[col] = (col, logits[lastRow, col] / temp);
            }

            // 2. Sort descending
            var sorted = scaledLogits.OrderByDescending(x => x.Logit).ToList();

            // 3. Truncate to Top-K
            var topKList = (topK > 0 ? sorted.Take(topK) : sorted).ToList();

            // 4. Compute Softmax over the Top-K candidates to get valid probabilities
            float maxLogit = topKList.Max(x => x.Logit); // Numeric stability normalization
            var expSum = topKList.Sum(x => MathF.Exp(x.Logit - maxLogit));

            var candidates = topKList.Select(x => (
                x.Index, 
                Prob: MathF.Exp(x.Logit - maxLogit) / expSum
            )).ToList();

            // 5. Apply Top-P (Nucleus) filter on candidate probabilities
            float cumulativeProb = 0f;
            var filtered = new List<(int Index, float Prob)>();
            foreach (var c in candidates)
            {
                filtered.Add(c);
                cumulativeProb += c.Prob;
                if (topP < 1.0f && cumulativeProb >= topP) break;
            }

            // 6. Weighted Random Draw
            float totalMass = filtered.Sum(x => x.Prob);
            float randomVal = Random.Shared.NextSingle() * totalMass;
            float currentSum = 0f;

            for (int i = 0; i < filtered.Count; i++)
            {
                currentSum += filtered[i].Prob;
                if (randomVal <= currentSum)
                    return filtered[i].Index;
            }

            return filtered[0].Index;
        }              

        private string DumpFirstRowLogits(TensorBase logits)
        {
            var sb = new StringBuilder();
            var logitsCols = logits.Cols;
            for (int i = 0; i < logitsCols; i++)
            {
                sb.AppendLine($"Column {i}: First row: {logits[0, i]}");
            }
            return sb.ToString();
        }

        private string DumpArgMax(TensorBase probabilities)
        {
            var sb = new StringBuilder();

            for (int row = 0; row < probabilities.Rows; row++)
            {
                float max = float.MinValue;
                int maxIndex = -1;

                for (int col = 0; col < probabilities.Cols; col++)
                {
                    float value = probabilities[row, col];

                    if (value > max)
                    {
                        max = value;
                        maxIndex = col;
                    }
                }

                sb.AppendLine($"Row {row}: token={maxIndex}, probability={max:E6}");
            }

            return sb.ToString();
        }

        private string DumpRows(TensorBase logits, int columns = 10)
        {
            var sb = new StringBuilder();

            for (int row = 0; row < logits.Rows; row++)
            {
                sb.Append($"Row {row}: ");

                for (int col = 0; col < Math.Min(columns, logits.Cols); col++)
                {
                    sb.Append($"{logits[row, col]:F4} ");
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }

        private string DumpHiddenState(TensorBase hiddenState)
        {
            var sb = new StringBuilder();

            sb.AppendLine($"Hidden state shape: {hiddenState.Rows} × {hiddenState.Cols}");
            sb.AppendLine();

            for (int row = 0; row < hiddenState.Rows; row++)
            {
                sb.Append($"Token {row}: ");

                for (int col = 0; col < hiddenState.Cols; col++)
                {
                    sb.Append(hiddenState[row, col].ToString("F4"));

                    if (col < hiddenState.Cols - 1)
                        sb.Append(", ");
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
}