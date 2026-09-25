using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Serilog;
using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    //Future work: Implement batched training and gradient clipping by norm.
    public class TransformerModel : IDisposable
    {
        public Guid TransformerModelId { get; private set; }
        public Guid? LoadedCheckpointId { get; internal set; }

        //These don't yet exist, but are created when the constructor calls BuildModel(). 
        private EmbeddingLayer _embedding = null!;
        private ILinearLayer _outputProjection = null!;
        private PositionalEncodingLayer _position = null!;
        private TensorWorkspace _workspace = null!;
        //The acceleration backend that drives all tensor math for this model instance
        //(SIMD, pure managed reference, or GPU backends plugged into IAccelerationBackend).
        private IAccelerationBackend _backend = null!;
        //Set this to true whenever training is underway. It can be defaulted to false if necessary.
        private bool _isTraining;
        public bool IsTraining => _isTraining;
        public bool CanInfer => !_isTraining;
        /// <summary>
        /// The acceleration backend currently driving this model's training and prediction math.
        /// </summary>
        public IAccelerationBackend Backend => _workspace.Backend;
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                // 1. Input Embeddings
                foreach (var p in _embedding.Parameters)
                    yield return p;

                // 2. Positional Encodings (if trainable/learned)
                if (_position is ITrainableLayer trainablePos)
                {
                    foreach (var p in trainablePos.Parameters)
                        yield return p;
                }

                // 3. Transformer Block Stack (Sequential execution order)
                foreach (var layer in _layers)
                {
                    if (layer is ITrainableLayer trainableLayer)
                    {
                        foreach (var p in trainableLayer.Parameters)
                            yield return p;
                    }
                }

                // 4. Output Head (Final Linear projection)
                foreach (var p in _outputProjection.Parameters)
                    yield return p;
            }
        }      
        private ILossFunction _loss = null!;
        private IOptimizer _optimizer = null!;
        
        public static TransformerConfig DefaultConfig => new()
        {
            VocabSize = 30522, // Common vocabulary size for BERT-like models
            EmbeddingSize = 768, // Common embedding size for BERT-like models
            NumLayers = 12, // Common number of layers for BERT-like models
            NumHeads = 12, // Common number of attention heads for BERT-like models
            FeedForwardSize = 3072, // Common feed-forward size for BERT-like models
            MaxSequenceLength = 512, // Common maximum sequence length for BERT-like models
        };
        

        //For around 8GB of memory usage.
        public static TransformerConfig MediumConfig => new()
        {
            VocabSize = 30522,           
            EmbeddingSize = 512,         // Increased model capacity while remaining well under 8 GB
            NumLayers = 8,               // 8 Transformer layers
            NumHeads = 8,                // 512 / 8 = 64 head dim (Perfect alignment for AVX2 SIMD)
            FeedForwardSize = 2048,      // 4x EmbeddingSize standard ratio
            MaxSequenceLength = 256,     // 256 tokens gives a strong context window
        };
        //For around 4GB of memory usage.
        public static TransformerConfig SmallConfig => new()
        {
            VocabSize = 30522,           
            EmbeddingSize = 256,         // Reduced embedding size for smaller memory footprint
            NumLayers = 4,               // 4 Transformer layers
            NumHeads = 4,                // 256 / 4 = 64 head dim (Perfect alignment for AVX2 SIMD)
            FeedForwardSize = 1024,      // 4x EmbeddingSize standard ratio
            MaxSequenceLength = 128,     // Shorter context window for smaller models
        };


        private readonly List<ILayer> _layers = new();
        public TransformerConfig Config { get; }
        public TrainingConfig TrainingConfig { get; }

        /// <summary>
        /// True when the model was built as quantised LoRA (frozen 4-bit base
        /// weights plus trainable adapters), false for a raw model where every
        /// weight is a dense fp32 trainable parameter. Recorded in the checkpoint
        /// so the two can never be confused on resume.
        /// </summary>
        public bool UsesQLora { get; private set; }
        public TransformerModel(Guid modelId, TransformerConfig? config = null, TrainingConfig? trainingConfig = null, bool useQLora = true, BackendSelector.BackendType backendType = BackendSelector.BackendType.Auto)
        {
            TransformerModelId = modelId;
            Config = config ?? DefaultConfig;
            TrainingConfig = trainingConfig ?? new TrainingConfig();
            //Select the acceleration backend (Auto probes hardware; defaults to CpuSimd/CpuReference).
            _backend = BackendSelector.SelectBackend(backendType);
            Log.Information("Acceleration backend selected: {BackendName}.", _backend.Name);
            //Second pass configuration validation to ensure nothing accidentally slips through first-pass validation in the config class.
            ValidateConfig();
            Log.Information("Configuration is valid. Proceeding...");

            UsesQLora = useQLora;
            BuildModel(useQLora);
            Log.Information($"Transformer model {modelId} ready to be loaded. Use Quantised LoRA: {useQLora}.");
        }

        public void BeginTraining() => _isTraining = true;
        public void EndTraining() => _isTraining = false;
        
        public (TensorBase logits, TensorBase hiddenState) Forward(TensorBase input)
        {
            Log.Information("Starting forward pass through the transformer model...");
            var forwardWatch = Stopwatch.StartNew();

            DiagonisticUtilities.AssertNoNaN(input, "Input contains NaN.");
            TensorBase x = _embedding.Forward(input, _workspace);

            DiagonisticUtilities.AssertNoNaN(x, "Embedding contains NaN.");
            x = _position.Forward(x, _workspace);

            DiagonisticUtilities.AssertNoNaN(x, "Positional encoding contains NaN.");
            foreach (var layer in _layers)
            {
                var layerWatch = Stopwatch.StartNew();
                var layerIndex = _layers.IndexOf(layer);
                x = layer.Forward(x, _workspace);
                Log.Information($"Forward pass through layer {layerIndex} ({layer.GetType().Name}) completed in {layerWatch.ElapsedMilliseconds} ms.");
                layerWatch.Stop();
                DiagonisticUtilities.AssertNoNaN(x, $"Transformer layer {layerIndex} contains NaN.");
            }

            TensorBase hiddenState = x;
            DiagonisticUtilities.AssertNoNaN(hiddenState, "Hidden state contains NaN.");

            TensorBase logits = _outputProjection.Forward(hiddenState, _workspace);
            DiagonisticUtilities.AssertNoNaN(logits, "Logits contains NaN.");

            forwardWatch.Stop();
            Log.Information($"Forward pass completed in {forwardWatch.ElapsedMilliseconds} ms.");

            return (logits, hiddenState);
        }

        public void Backward(TensorBase gradient)
        {
            Log.Information("Starting backward pass through the transformer model...");
            var backwardWatch = Stopwatch.StartNew();
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient contains NaN.");

            gradient = _outputProjection.Backward(gradient, _workspace);
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through output projection contains NaN.");

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                var thislayer = _layers[i];
                var layerWatch = Stopwatch.StartNew();
                
                gradient = _layers[i].Backward(gradient, _workspace);
                layerWatch.Stop();
                Log.Information($"Backward pass through layer {i} ({thislayer.GetType().Name}) completed in {layerWatch.ElapsedMilliseconds} ms.");
                DiagonisticUtilities.AssertNoNaN(gradient, $"Gradient after backward pass through layer {i} contains NaN.");
            }

            gradient = _position.Backward(gradient, _workspace);
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through positional encoding contains NaN.");

            gradient = _embedding.Backward(gradient, _workspace);
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through embedding contains NaN.");

            // REMOVED: _embedding.ClipGradients(1.0f); -> Handled globally in TrainStep!
            backwardWatch.Stop();
            Log.Information($"Backward pass completed in {backwardWatch.ElapsedMilliseconds} ms.");
        }

        public (int nextTokenId, int[] allTokenIds, TensorBase logits, TensorBase probabilities, TensorBase hiddenState) Predict(TensorBase input)
        {
            // 1. Run forward pass
            var (logits, hiddenState) = Forward(input);

            // 2. Softmax logits to get probabilities (routed through the acceleration backend)
            TensorBase probabilities = logits.Clone();
            _workspace.Backend.SoftmaxInPlace(probabilities);

            // 3. Get predicted token IDs for all positions
            int[] allTokenIds = TokenizationUtilities.ToTokenIds((Tensor)probabilities);

            // 4. The actual NEXT token is the ArgMax of the LAST row
            int nextTokenId = allTokenIds[allTokenIds.Length - 1];

            return (nextTokenId, allTokenIds, logits, probabilities, hiddenState);
        }
        
        public void ZeroGradients()
        {
            _outputProjection.ZeroGradients();
            //Reset embedding
            _embedding.ZeroGradients();

            foreach (var layer in _layers)
            {
                if (layer is ITrainableLayer trainableLayer)
                {
                    trainableLayer.ZeroGradients();
                }
            }
        }

        /// <summary>
        /// Stream-based serialization: decoupling from direct File API dependencies
        /// </summary>
        public void SaveCheckpoint(Stream destinationStream, int currentEpoch, float currentLoss)
        {
            using var writer = new BinaryWriter(destinationStream, Encoding.UTF8, leaveOpen: true);
            
            // Fixed 4-byte header (no length prefix)
            writer.Write("STCK"u8); 
            writer.Write(3); // Schema version (v3 adds the useQLora flag)

            //Add the transformer model id to the checkpoint file 
            writer.Write(TransformerModelId.ToByteArray());

            //v3: record whether this checkpoint came from a QLoRA or a raw model.
            //A QLoRA model exposes LoRA adapters as trainable parameters while a
            //raw model exposes dense weights, so the two checkpoint families are
            //not interchangeable and must be rejected explicitly on load.
            writer.Write(UsesQLora);

            writer.Write(currentEpoch);
            writer.Write(currentLoss);

            CheckpointConfigExtensions.WriteConfig(writer, Config);

            var paramList = Parameters as IReadOnlyCollection<TrainableParameter> ?? Parameters.ToList();
            writer.Write(paramList.Count);

            foreach (var param in paramList)
            {
                // 1. Parameter Name
                writer.Write(param.Name);

                // 2. Value Tensor
                var valueData = new TensorData { Shape = param.Value.Shape, Data = param.Value.Data };
                WriteTensor(writer, valueData);

                // 3. Gradient Tensor (optional)
                bool hasGrad = param.Gradient != null;
                writer.Write(hasGrad);

                if (hasGrad)
                {
                    var gradData = new TensorData { Shape = param.Gradient!.Shape, Data = param.Gradient!.Data };
                    WriteTensor(writer, gradData);
                }
            }
        }

        private static void WriteTensor(BinaryWriter writer, TensorData tensor)
        {
            if (!tensor.IsValid)
            {
                throw new InvalidOperationException(
                    $"Cannot serialize invalid TensorData. Data length ({tensor.Data.Length}) " +
                    $"does not match shape product ({tensor.TotalElements}).");
            }

            writer.Write(tensor.Shape.Length);

            for (int i = 0; i < tensor.Shape.Length; i++)
            {
                writer.Write(tensor.Shape[i]);
            }

            ReadOnlySpan<byte> byteBuffer = MemoryMarshal.AsBytes(tensor.Data.AsSpan());
            writer.BaseStream.Write(byteBuffer);
        }

        /// <summary>
        /// Static factory method to instantiate and hydrate a TransformerModel directly from a checkpoint stream.
        /// </summary>
        public static (int Epoch, float Loss) LoadCheckpoint(
            Stream sourceStream,
            TransformerModel model)
        {
            using var reader = new BinaryReader(
                sourceStream,
                Encoding.UTF8,
                leaveOpen: true);

            // ------------------------------------------------------------
            // 1. Validate checkpoint header
            // ------------------------------------------------------------

            byte[] magicBytes = reader.ReadBytes(4);
            string magic = Encoding.UTF8.GetString(magicBytes);

            if (magic != "STCK")
            {
                throw new InvalidDataException(
                    $"Invalid checkpoint file magic header: '{magic}'. Expected 'STCK'.");
            }

            int version = reader.ReadInt32();

            //v2 predates the useQLora flag. Every v2 checkpoint was written by a
            //QLoRA model (raw training was not reachable), so they are read as
            //QLoRA=true and remain loadable.
            if (version != 2 && version != 3)
            {
                throw new InvalidDataException(
                    $"Unsupported checkpoint schema version: {version}. Expected 2 or 3.");
            }
            //Validate the checkpoint model id against the loaded model

            Guid checkpointModelId =
                new Guid(reader.ReadBytes(16));

            if (checkpointModelId != model.TransformerModelId)
            {
                throw new InvalidDataException(
                    $"Checkpoint belongs to model {checkpointModelId}, " +
                    $"but was loaded into model {model.TransformerModelId}.");
            }

            bool checkpointUseQLora = true;
            if (version >= 3)
            {
                checkpointUseQLora = reader.ReadBoolean();

                if (checkpointUseQLora != model.UsesQLora)
                {
                    throw new InvalidDataException(
                        $"Checkpoint was saved from a {(checkpointUseQLora ? "QLoRA" : "raw")} model, " +
                        $"but this model is {(model.UsesQLora ? "QLoRA" : "raw")}. " +
                        "The trainable parameters differ, so the checkpoint cannot be loaded. " +
                        "Create a new model with the matching training mode.");
                }
            }

            int epoch = reader.ReadInt32();
            float loss = reader.ReadSingle();

            // ------------------------------------------------------------
            // 2. Read checkpoint configuration
            // ------------------------------------------------------------

            var checkpointConfig =
                CheckpointConfigExtensions.ReadConfig(reader);

            // ------------------------------------------------------------
            // 3. Validate checkpoint against the existing model
            // ------------------------------------------------------------

            ValidateCheckpointConfig(model.Config, checkpointConfig);

            // ------------------------------------------------------------
            // 4. Read parameter data
            // ------------------------------------------------------------

            int paramCount = reader.ReadInt32();

            if (paramCount <= 0)
            {
                throw new InvalidDataException(
                    $"Checkpoint contains an invalid parameter count: {paramCount}.");
            }

            var loadedParameters =
                new List<TrainableParameterCheckpoint>(paramCount);

            for (int i = 0; i < paramCount; i++)
            {
                string name = reader.ReadString();

                var value = ReadTensorOptimized(reader);

                bool hasGradient = reader.ReadBoolean();

                TensorData? gradient =
                    hasGradient
                        ? ReadTensorOptimized(reader)
                        : null;

                loadedParameters.Add(
                    new TrainableParameterCheckpoint
                    {
                        Name = name,
                        Value = value,
                        Gradient = gradient
                    });
            }

            // ------------------------------------------------------------
            // 5. Hydrate the existing runtime model
            // ------------------------------------------------------------

            model.LoadCheckpointData(loadedParameters);

            Log.Information(
                "Checkpoint successfully loaded into model {ModelId}. Epoch: {Epoch}, Loss: {Loss}.",
                model.TransformerModelId,
                epoch,
                loss);

            return (epoch, loss);
        }       

        private static TensorData ReadTensorOptimized(BinaryReader reader)
        {
            int rank = reader.ReadInt32();
            if (rank <= 0 || rank > 8)
            {
                throw new InvalidDataException($"Corrupt checkpoint format: invalid tensor rank ({rank}).");
            }

            int[] shape = new int[rank];
            long totalElements = 1;

            for (int i = 0; i < rank; i++)
            {
                shape[i] = reader.ReadInt32();
                
                if (shape[i] <= 0 || shape[i] > 100_000_000) 
                {
                    throw new InvalidDataException($"Corrupt checkpoint format: dimension [{i}] has invalid size ({shape[i]}).");
                }

                if (totalElements > (long.MaxValue / shape[i]))
                {
                    throw new InvalidDataException("Corrupt checkpoint format: calculated tensor size exceeds maximum allowable limits.");
                }

                totalElements *= shape[i];
            }

            if (totalElements > int.MaxValue)
            {
                throw new InvalidDataException($"Tensor size ({totalElements} elements) exceeds standard C# array limit.");
            }

            float[] data = new float[(int)totalElements];
            Span<byte> byteBuffer = MemoryMarshal.AsBytes(data.AsSpan());
            
            reader.BaseStream.ReadExactly(byteBuffer);

            return new TensorData
            {
                Shape = shape,
                Data = data
            };
        }      

        /// <summary>
        /// Hydrates model weight (and optional gradient) tensors from a loaded checkpoint using parameter names.
        /// </summary>
        public void LoadCheckpointData(IReadOnlyList<TrainableParameterCheckpoint> checkpointParameters)
        {
            var checkpointMap = checkpointParameters.ToDictionary(p => p.Name, p => p);
            var modelParameters = Parameters.ToList();

            using var debugFileWriter = new StreamWriter("checkpoint-debug.log", append: true);

            if (modelParameters.Count != checkpointParameters.Count)
            {
                throw new InvalidOperationException(
                    $"Checkpoint parameter count ({checkpointParameters.Count}) does not match model parameter count ({modelParameters.Count}).");
            }

            foreach (var targetParam in modelParameters)
            {
                //Todo: Implement diagnostic logging to file for checkpoint save/load.
                //This should not be live logged, just dumped to a file for debugging purposes and to see that names are properly loaded and saved.
                if (!checkpointMap.TryGetValue(targetParam.Name, out var loadedParam))
                {
                    throw new InvalidOperationException(
                        $"Parameter '{targetParam.Name}' missing from checkpoint data.");
                }

                if (!Enumerable.SequenceEqual(targetParam.Value.Shape, loadedParam.Value.Shape))
                {
                    throw new InvalidOperationException(
                        $"Parameter shape mismatch for '{targetParam.Name}'. Expected [{string.Join(',', targetParam.Value.Shape)}], but checkpoint has [{string.Join(',', loadedParam.Value.Shape)}].");
                }

                Array.Copy(loadedParam.Value.Data, targetParam.Value.Data, targetParam.Value.Data.Length);

                if (loadedParam.Gradient != null && targetParam.Gradient != null)
                {
                    if (!Enumerable.SequenceEqual(targetParam.Gradient.Shape, loadedParam.Gradient.Shape))
                    {
                        throw new InvalidOperationException(
                            $"Gradient shape mismatch for '{targetParam.Name}'.");
                    }

                    Array.Copy(loadedParam.Gradient.Data, targetParam.Gradient.Data, targetParam.Gradient.Data.Length);
                }
                debugFileWriter.WriteLine($"Loaded parameter '{targetParam.Name}' with value {targetParam.Value.Data.Length} and {(targetParam.Gradient != null ? $"gradient {targetParam.Gradient.Data.Length}" : "no gradient")} from checkpoint.");
            }

            //Array.Copy above mutated the weight arrays in place, so any VRAM
            //resident copies now hold the pre-load weights; refresh them before
            //the first MatMul can bind them.
            RefreshResidentWeights();

            Log.Information("Successfully hydrated {Count} model weight tensors from checkpoint by parameter name.", modelParameters.Count);
        }

        public void Train(
            IReadOnlyList<(TensorBase Input, TensorBase Target)> dataset,
            int startEpoch = 0,
            IProgress<TrainingProgressReport>? progress = null,
            CancellationToken cancellationToken = default)
        {
            var stopwatch = Stopwatch.StartNew();

            for (int epoch = startEpoch; epoch < TrainingConfig.Epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                float epochLoss = 0f;
                foreach (var sample in dataset)
                {
                    epochLoss += TrainStep(sample.Input, sample.Target);
                }

                epochLoss /= dataset.Count;

                // Report progress back to caller (TrainingService or CLI)
                progress?.Report(new TrainingProgressReport(
                    CurrentEpoch: epoch + 1,
                    TotalEpochs: TrainingConfig.Epochs,
                    Loss: epochLoss,
                    ElapsedTime: stopwatch.Elapsed
                ));
            }
        }

        public float TrainStep(TensorBase inputs, TensorBase expectedOutputs)
        {
            ZeroGradients();

            TensorBase? prediction = null;
            TensorBase? auxOutput = null;
            TensorBase? gradient = null;

            try
            {
                // 1. Forward Pass
                (prediction, auxOutput) = Forward(inputs);

                // 2. Compute Loss & Initial Backprop Gradient
                float loss = _loss.Forward(prediction, expectedOutputs);
                gradient = _loss.Backward(prediction, expectedOutputs);

                // 3. Backpropagate through Model
                Backward(gradient);

                // 4. Clip ALL parameter gradients globally
                ClipGradients(1.0f);

                // 5. Verify no gradients contain NaN before updating weights
                foreach (var p in Parameters)
                {
                    if (p.Gradient != null)
                    {
                        DiagonisticUtilities.AssertNoNaN(p.Gradient, "Gradient contains NaN prior to optimizer step.");
                    }
                }

                // 6. Step optimizer
                _optimizer.Step(Parameters);

                // 7. Verify no weights became NaN after optimizer step
                foreach (var p in Parameters)
                {
                    DiagonisticUtilities.AssertNoNaN(p.Value, "Model weight matrix poisoned by optimizer step.");
                }

                // 8. Push the updated weights into their VRAM-resident copies (if
                // any): one staged upload per tensor per step keeps the Vulkan
                // MatMul fast path binding fresh device memory. CPU backends and
                // non-resident tensors no-op immediately.
                RefreshResidentWeights();

                return loss;
            }
            finally
            {
                // Clean up transient forward & loss tensors for this step
                DisposeIfDisposable(prediction);
                DisposeIfDisposable(auxOutput);
                _workspace.Reset();
                // If _loss.Backward returns a cached tensor managed internally by _loss, 
                // DO NOT dispose it here. Otherwise, dispose if it's transient:
            }
            
        }

        private static void DisposeIfDisposable(object? obj)
        {
            if (obj is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        /// <summary>
        /// Re-uploads every trainable parameter into its existing VRAM-resident
        /// device copy. Called after in-place weight updates (optimizer step,
        /// checkpoint hydration) so the resident MatMul path never binds stale
        /// weights. Safe no-op on CPU backends and for parameters that were never
        /// promoted (they simply keep using the ordinary upload path).
        /// </summary>
        private void RefreshResidentWeights()
        {
            var backend = _workspace.Backend;
            foreach (var p in Parameters)
                backend.TryRefreshResidentWeights(p.Value);
        }

        public async Task<float> TrainStepAsync(TensorBase inputs, TensorBase expectedOutputs)
        {
            return await Task.Run(() => TrainStep(inputs, expectedOutputs));
        }

        public async Task<(TensorBase logits, TensorBase hiddenState)> ForwardAsync(TensorBase input)
        {
            return await Task.Run(() => Forward(input));
        }


        private void ValidateConfig()
        {
            if (Config.VocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.VocabSize));

            if (Config.EmbeddingSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.EmbeddingSize));

            if (Config.NumLayers <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.NumLayers));

            if (Config.NumHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.NumHeads));

            if (Config.FeedForwardSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.FeedForwardSize));

            if (Config.MaxSequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.MaxSequenceLength));

            if (Config.EmbeddingSize % Config.NumHeads != 0)
                throw new ArgumentException(
                    "Embedding size must be divisible by the number of heads.");
        }

        private static void ValidateCheckpointConfig(
            TransformerConfig modelConfig,
            TransformerConfig checkpointConfig)
        {
            if (modelConfig.VocabSize != checkpointConfig.VocabSize)
            {
                throw new InvalidDataException(
                    $"Checkpoint vocabulary size ({checkpointConfig.VocabSize}) " +
                    $"does not match model vocabulary size ({modelConfig.VocabSize}).");
            }

            if (modelConfig.EmbeddingSize != checkpointConfig.EmbeddingSize)
            {
                throw new InvalidDataException(
                    $"Checkpoint embedding size ({checkpointConfig.EmbeddingSize}) " +
                    $"does not match model embedding size ({modelConfig.EmbeddingSize}).");
            }

            if (modelConfig.NumLayers != checkpointConfig.NumLayers)
            {
                throw new InvalidDataException(
                    $"Checkpoint layer count ({checkpointConfig.NumLayers}) " +
                    $"does not match model layer count ({modelConfig.NumLayers}).");
            }

            if (modelConfig.NumHeads != checkpointConfig.NumHeads)
            {
                throw new InvalidDataException(
                    $"Checkpoint head count ({checkpointConfig.NumHeads}) " +
                    $"does not match model head count ({modelConfig.NumHeads}).");
            }

            if (modelConfig.FeedForwardSize != checkpointConfig.FeedForwardSize)
            {
                throw new InvalidDataException(
                    $"Checkpoint feed-forward size ({checkpointConfig.FeedForwardSize}) " +
                    $"does not match model feed-forward size ({modelConfig.FeedForwardSize}).");
            }

            if (modelConfig.MaxSequenceLength != checkpointConfig.MaxSequenceLength)
            {
                throw new InvalidDataException(
                    $"Checkpoint maximum sequence length ({checkpointConfig.MaxSequenceLength}) " +
                    $"does not match model maximum sequence length ({modelConfig.MaxSequenceLength}).");
            }
        }        
        private void BuildModel(bool useQLora = false)
        {
            // 1. Root-level layers with clean, standard names
            _embedding = new EmbeddingLayer(Config.VocabSize, Config.EmbeddingSize, name: "token_embeddings");
            _position = new PositionalEncodingLayer(Config.EmbeddingSize, Config.MaxSequenceLength, name: "position_embeddings");
            _outputProjection = useQLora 
            ? new QLoraLinearLayer(Config.EmbeddingSize, Config.VocabSize, useBias: false, name: "lm_head")
            : new LinearLayer(Config.EmbeddingSize, Config.VocabSize, useBias: false, name: "lm_head");
            _workspace = new TensorWorkspace(_backend);

            _loss = new CrossEntropyLoss();
            _optimizer = TrainingConfig.Optimizer switch
            {
                OptimizerType.AdamW => new AdamWOptimizer(
                    TrainingConfig.LearningRate,
                    TrainingConfig.Beta1,
                    TrainingConfig.Beta2,
                    TrainingConfig.Epsilon,
                    TrainingConfig.WeightDecay),

                OptimizerType.Sgd => new SgdOptimizer(
                    TrainingConfig.LearningRate,
                    TrainingConfig.SgdMomentum,
                    TrainingConfig.WeightDecay,
                    TrainingConfig.UseNesterov),

                _ => throw new ArgumentOutOfRangeException(nameof(TrainingConfig.Optimizer), "Unsupported optimizer type.")
            };

            Log.Information($@"Current configuration:
        Vocabulary size: {Config.VocabSize}
        Embedding size: {Config.EmbeddingSize}
        Number of layers: {Config.NumLayers}
        Number of heads: {Config.NumHeads}
        Maximum sequence length: {Config.MaxSequenceLength}
        Feed forward size: {Config.FeedForwardSize}");

            // Clear any old layers.
            _layers.Clear();

            var watch = System.Diagnostics.Stopwatch.StartNew();
            var layerWatch = System.Diagnostics.Stopwatch.StartNew();
            Log.Information("Constructing transformer architecture. Please stand by...");

            // 2. Loop through blocks, scoping names by layer index "layers.{i}"
            for (int i = 0; i < Config.NumLayers; i++)
            {
                string blockName = $"layers.{i}";
                var componentWatch = System.Diagnostics.Stopwatch.StartNew();

                var attention = new MultiHeadAttention(
                    Config.EmbeddingSize,
                    Config.NumHeads,
                    name: $"{blockName}.attention",
                    useQLora
                );
                Log.Information($"Layer {i}: Attention layer constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var feedForward = new FeedForwardLayer(
                    Config.EmbeddingSize,
                    Config.FeedForwardSize,
                    name: $"{blockName}.feed_forward",
                    useQLora
                );
                Log.Information($"Layer {i}: Feed forward layer constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var norm1 = new LayerNorm(Config.EmbeddingSize, name: $"{blockName}.attn_norm");
                Log.Information($"Layer {i}: Layer norm 1 constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var norm2 = new LayerNorm(Config.EmbeddingSize, name: $"{blockName}.ffn_norm");
                Log.Information($"Layer {i}: Layer norm 2 constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                _layers.Add(
                    new TransformerBlock(
                        attention,
                        feedForward,
                        norm1,
                        norm2,
                        name: blockName));

                Log.Information($"Layer {i} constructed in {layerWatch.ElapsedMilliseconds}ms. Total elapsed: {watch.ElapsedMilliseconds}ms.");
                layerWatch.Restart();
            }

            Log.Information($"Transformer architecture initialisation completed in {watch.ElapsedMilliseconds}ms."); 
            watch.Stop();
        }
        public float ClipGradients(float maxNorm = 1.0f)
        {
            double sumSquaredNorm = 0.0;

            // 1. Accumulate squared gradients across ALL trainable parameters
            foreach (var param in Parameters)
            {
                if (param.Gradient == null) continue;
                
                ReadOnlySpan<float> gData = param.Gradient.Data.AsSpan();
                for (int i = 0; i < gData.Length; i++)
                {
                    float g = gData[i];
                    sumSquaredNorm += g * g;
                }
            }

            float totalNorm = MathF.Sqrt((float)sumSquaredNorm);

            // 2. Scale gradients if global norm exceeds maxNorm
            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / (totalNorm + 1e-6f);

                foreach (var param in Parameters)
                {
                    if (param.Gradient == null) continue;

                    Span<float> gData = param.Gradient.Data.AsSpan();
                    for (int i = 0; i < gData.Length; i++)
                    {
                        gData[i] *= scale;
                    }
                }
            }

            return totalNorm;
        }

        public void Dispose()
        {
            // Dispose layers/resources that own unmanaged or pooled resources.
            _embedding?.Dispose();
            _position?.Dispose();
            _outputProjection?.Dispose();

            foreach (var layer in _layers)
            {
                if (layer is IDisposable disposable)
                    disposable.Dispose();
            }

            _workspace?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
