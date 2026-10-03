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
        //Embedding dropout (Vaswani §5.4): applied to the embedding + positional
        //output. Disabled by default; toggled by BeginTraining/EndTraining.
        private DropoutSite _embeddingDropout = null!;
        //Every dropout site in the model, flattened once by BuildModel so the
        //per-step re-key and checkpoint save/load are a flat walk rather than a
        //recursive traversal.
        private readonly List<DropoutSite> _dropoutSites = new();
        private TensorWorkspace _workspace = null!;
        //The acceleration backend that drives all tensor math for this model instance
        //(SIMD, pure managed reference, or GPU backends plugged into IAccelerationBackend).
        private IAccelerationBackend _backend = null!;
        //Set this to true whenever training is underway. It can be defaulted to false if necessary.
        private bool _isTraining;
        public bool IsTraining => _isTraining;
        public bool CanInfer => !_isTraining;
        /// <summary>
        /// Decides when this model's memory pressure needs relieving. One per
        /// model, so the latch and cooldown apply per training run rather than
        /// per step.
        /// </summary>
        private readonly MemoryPressureValve _memoryValve = new();

        /// <summary>
        /// Global optimizer step counter driving the LR schedule. Checkpointed
        /// (schema v4) so pause/resume reproduces the identical LR curve.
        /// </summary>
        private int _globalStep;
        public int GlobalStep => _globalStep;

        /// <summary>
        /// Snapshot of the optimizer's training state (moments/velocity plus
        /// bias-correction step). Used by the v5 checkpoint writer and by the
        /// resume-exactness self-test for direct before/after comparison.
        /// </summary>
        internal OptimizerState ExportOptimizerState() => _optimizer.ExportState();

        /// <summary>
        /// The model's training workspace (checkpoint tests exercise the same
        /// pooled loss-backward path TrainStep uses).
        /// </summary>
        internal TensorWorkspace Workspace => _workspace;

        /// <summary>Checkpoint/determinism tests: reach a transformer block.</summary>
        internal TransformerBlock BlockForTest(int index) => (TransformerBlock)_layers[index];

        /// <summary>Checkpoint/determinism tests: number of transformer blocks.</summary>
        internal int LayerCount => _layers.Count;

        /// <summary>
        /// Replaces the live optimizer state (checkpoint tests: wipe control).
        /// </summary>
        internal int ImportOptimizerState(OptimizerState state) =>
            _optimizer.ImportState(
                state, Parameters as IReadOnlyList<TrainableParameter> ?? Parameters.ToList());

        /// <summary>
        /// The model's durable dropout sites: architecture-keyed identity plus
        /// salt/step. Internal so checkpoint round-trip tests can compare salts.
        /// </summary>
        internal IReadOnlyList<DropoutSite> DropoutSites => _dropoutSites;

        /// <summary>
        /// Per-run LR policy. Null until <see cref="ConfigureScheduler"/> is
        /// called by the training loop; when null the optimizer's configured
        /// base rate is used unchanged (legacy flat-LR path).
        /// </summary>
        private LRScheduler? _scheduler;
        public float CurrentLearningRate => _scheduler != null
            ? _scheduler.GetLR(Math.Max(0, _globalStep - 1))
            : _optimizer?.LearningRate ?? TrainingConfig.LearningRate;

        /// <summary>
        /// Post-resume LR ramp: number of steps blending 0 -&gt; scheduled LR
        /// (absorbs the fresh-Adam-moments bump), and how many are consumed.
        /// Transient per run - deliberately NOT checkpointed.
        /// </summary>
        private int _resumeRampTotal;
        private int _resumeRampDone;

        /// <summary>
        /// Effective LR for the current step: scheduled value, blended from
        /// ~0 when a post-resume ramp is active.
        /// </summary>
        public float GetStepLearningRate()
        {
            if (_scheduler == null)
                return _optimizer?.LearningRate ?? TrainingConfig.LearningRate;

            float scheduled = _scheduler.GetLR(_globalStep);
            if (_resumeRampDone < _resumeRampTotal)
            {
                float blend = (float)(_resumeRampDone + 1) / (_resumeRampTotal + 1);
                return scheduled * blend;
            }
            return scheduled;
        }

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

        public void BeginTraining()
        {
            _isTraining = true;
            ApplyDropout(enabled: true);
        }

        public void EndTraining()
        {
            _isTraining = false;
            ApplyDropout(enabled: false);
        }

        /// <summary>
        /// Pushes the configured rate and the train/eval switch to every dropout
        /// site: the embedding output, each block's two residual sub-layer
        /// outputs, and the attention weights inside every head. The rate is
        /// re-read here (not just at build time) so a rate changed through
        /// <see cref="TrainingConfig"/> after construction still takes effect,
        /// and so a run left with the default rate of 0 is a byte-exact
        /// pass-through.
        /// </summary>
        private void ApplyDropout(bool enabled)
        {
            float rate = TrainingConfig.DropoutRate;

            _embeddingDropout.SetRate(rate);
            _embeddingDropout.SetEnabled(enabled);

            foreach (var layer in _layers)
            {
                if (layer is IDropoutControl dropoutControl)
                {
                    dropoutControl.SetDropoutRate(rate);
                    dropoutControl.SetDropoutEnabled(enabled);
                }
            }
        }

        /// <summary>
        /// Re-keys every dropout site for the current optimizer step, so the masks
        /// a step sees are a pure function of (site salt, step, batch item).
        /// <para>
        /// This is what makes the sequence reproducible: a resumed run restores
        /// the salts and <see cref="_globalStep"/>, so the identical masks replay;
        /// a changed batch size cannot perturb item 0's mask; and an interleaved
        /// inference Forward cannot advance the training stream, because the next
        /// step re-keys from the step counter rather than continuing a sequence.
        /// </para>
        /// Sites that are disabled or at rate 0 are skipped, so the default
        /// (no dropout) path costs nothing.
        /// </summary>
        private void PrepareDropoutForStep()
        {
            for (int i = 0; i < _dropoutSites.Count; i++)
            {
                DropoutSite site = _dropoutSites[i];
                if (site.Enabled && site.Rate > 0f)
                    site.PrepareForStep(_globalStep);
            }
        }

        /// <summary>
        /// Flattens every dropout site in the model (embedding + all blocks) into
        /// <see cref="_dropoutSites"/>. Called once from BuildModel; site keys are
        /// architecture-derived and stable, so they survive checkpoint round-trips.
        /// </summary>
        private void CollectDropoutSites()
        {
            _dropoutSites.Clear();
            _dropoutSites.Add(_embeddingDropout);

            foreach (var layer in _layers)
            {
                if (layer is IDropoutControl dropoutControl)
                    dropoutControl.CollectDropoutSites(_dropoutSites);
            }
        }

        /// <summary>
        /// Restores the durable per-site dropout state (salt + keyed step) from
        /// a v5 checkpoint, matched by <see cref="DropoutSite.Key"/>. Per-item
        /// mask layers stay derived: a freshly built item reseeds from the
        /// restored salt, so batch size is irrelevant here.
        /// <para>
        /// A site missing from the checkpoint, or a checkpoint key unknown to
        /// this model, is warned about rather than fatal: config validation has
        /// already rejected architecture changes before load, so either case
        /// indicates a keying defect, not a legitimately different model.
        /// </para>
        /// </summary>
        /// <returns>Number of live sites matched and restored.</returns>
        private int RestoreDropoutSites(IReadOnlyList<CheckpointStateExtensions.DropoutSiteState> saved)
        {
            var byKey = new Dictionary<string, CheckpointStateExtensions.DropoutSiteState>(saved.Count);
            for (int i = 0; i < saved.Count; i++)
                byKey[saved[i].Key] = saved[i];

            int restored = 0;
            foreach (DropoutSite site in _dropoutSites)
            {
                if (byKey.TryGetValue(site.Key, out var state))
                {
                    site.Salt = state.Salt;
                    site.PrepareForStep(state.Step);
                    byKey.Remove(site.Key);
                    restored++;
                }
                else
                {
                    Log.Warning(
                        "Dropout site {Key} is absent from the checkpoint; keeping its fresh random salt (mask replay will differ for this site).",
                        site.Key);
                }
            }

            foreach (string unknownKey in byKey.Keys)
            {
                Log.Warning(
                    "Checkpoint carries dropout site {Key} unknown to this model; ignored.",
                    unknownKey);
            }

            return restored;
        }

        /// <summary>
        /// Embedding-output dropout split per batch item so each mask is a pure
        /// function of (salt, step, item). Rank-2 (single sequence) uses item 0;
        /// Rank-3 applies ForItem(b) per slice like the attention path.
        /// </summary>
        private TensorBase ApplyEmbeddingDropout(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Rank == 2)
                return _embeddingDropout.ForItem(0).Forward(input, workspace);

            if (input.Rank != 3)
                throw new ArgumentException($"Embedding dropout expects rank 2 or rank 3. Got Rank {input.Rank}.");

            TensorBase output = workspace.BorrowLike(input);
            for (int b = 0; b < input.Layers; b++)
            {
                TensorBase inSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase droppedSlice = _embeddingDropout.ForItem(b).Forward(inSlice, workspace);
                TensorUtilitiesSimd.SetLayer(output, b, droppedSlice);
                workspace.Release(droppedSlice);
            }

            return output;
        }

        /// <summary>
        /// Pushes a gradient back through the matching per-item embedding masks.
        /// </summary>
        private TensorBase BackwardEmbeddingDropout(TensorBase gradient, TensorWorkspace workspace)
        {
            if (gradient.Rank == 2)
                return _embeddingDropout.ForItem(0).Backward(gradient, workspace);

            if (gradient.Rank != 3)
                throw new ArgumentException($"Embedding dropout expects rank 2 or rank 3. Got Rank {gradient.Rank}.");

            TensorBase output = workspace.BorrowLike(gradient);
            for (int b = 0; b < gradient.Layers; b++)
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase droppedSlice = _embeddingDropout.ForItem(b).Backward(gradSlice, workspace);
                TensorUtilitiesSimd.SetLayer(output, b, droppedSlice);
                workspace.Release(droppedSlice);
            }

            return output;
        }
        
        public (TensorBase logits, TensorBase hiddenState) Forward(TensorBase input)
        {
            MultiHeadAttention.WatchedConcatsForTest.Clear();
            bool verbose = TrainingStepProfile.Enabled;
            if (verbose)
                Log.Information("Starting forward pass through the transformer model...");
            var forwardWatch = Stopwatch.StartNew();

            if (verbose)
                DiagonisticUtilities.AssertNoNaN(input, "Input contains NaN.");
            TensorBase x = _embedding.Forward(input, _workspace);

            if (verbose)
                DiagonisticUtilities.AssertNoNaN(x, "Embedding contains NaN.");
            x = _position.Forward(x, _workspace);

            if (verbose)
                DiagonisticUtilities.AssertNoNaN(x, "Positional encoding contains NaN.");
            x = ApplyEmbeddingDropout(x, _workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(x, "Embedding dropout contains NaN.");

            // Causal (autoregressive) mask. This is a decoder-only language
            // model, so position t must not attend to t+1..S: without it every
            // position sees the whole window and next-token training degenerates
            // into copying. Built once per sequence length and reused - it is
            // constant, so rebuilding it per forward would be pure waste.
            TensorBase? mask = GetCausalMask(x.Rows);

            for (int layerIndex = 0; layerIndex < _layers.Count; layerIndex++)
            {
                var layer = _layers[layerIndex];
                var layerWatch = Stopwatch.StartNew();
                x = layer.Forward(x, _workspace, mask);
                layerWatch.Stop();
                if (verbose)
                {
                    Log.Information($"Forward pass through layer {layerIndex} ({layer.GetType().Name}) completed in {layerWatch.ElapsedMilliseconds} ms.");
                    DiagonisticUtilities.AssertNoNaN(x, $"Transformer layer {layerIndex} contains NaN.");
                }
            }

            TensorBase hiddenState = x;
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(hiddenState, "Hidden state contains NaN.");

            TensorBase logits = _outputProjection.Forward(hiddenState, _workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(logits, "Logits contains NaN.");

            forwardWatch.Stop();
            if (verbose)
                Log.Information($"Forward pass completed in {forwardWatch.ElapsedMilliseconds} ms.");

            return (logits, hiddenState);
        }

        public void Backward(TensorBase gradient)
        {
            bool verbose = TrainingStepProfile.Enabled;
            if (verbose)
                Log.Information("Starting backward pass through the transformer model...");
            var backwardWatch = Stopwatch.StartNew();
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(gradient, "Gradient contains NaN.");

            gradient = _outputProjection.Backward(gradient, _workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through output projection contains NaN.");

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                var thislayer = _layers[i];
                var layerWatch = Stopwatch.StartNew();

                gradient = _layers[i].Backward(gradient, _workspace);
                layerWatch.Stop();
                if (verbose)
                {
                    Log.Information($"Backward pass through layer {i} ({thislayer.GetType().Name}) completed in {layerWatch.ElapsedMilliseconds} ms.");
                    DiagonisticUtilities.AssertNoNaN(gradient, $"Gradient after backward pass through layer {i} contains NaN.");
                }
            }

            gradient = BackwardEmbeddingDropout(gradient, _workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through embedding dropout contains NaN.");

            gradient = _position.Backward(gradient, _workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through positional encoding contains NaN.");

            gradient = _embedding.Backward(gradient, _workspace);
            if (verbose)
                DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through embedding contains NaN.");

            // REMOVED: _embedding.ClipGradients(1.0f); -> Handled globally in TrainStep!
            backwardWatch.Stop();
            if (verbose)
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
        /// <summary>
        /// Installs the per-run LR schedule. Must be called once by the
        /// training loop before the first step, with the FULL-RUN step budget
        /// (total epochs x steps-per-epoch, including already-trained epochs
        /// on resume); the model's checkpointed <see cref="GlobalStep"/>
        /// lands mid-curve so pause/resume continues the identical schedule.
        /// Guessing a remaining-only horizon would leave the restored step
        /// past the end and pin LR at the floor.
        /// </summary>
        /// <param name="totalSteps">Total optimizer steps for the whole run.</param>
        /// <param name="resumeRewarmupSteps">
        /// Brief re-warmup applied when resuming mid-schedule (fresh Adam
        /// moments on trained weights otherwise spike the loss). Implemented
        /// as a ramp blending 0 -&gt; scheduled LR over the first post-resume
        /// steps. Pass 0 for genuinely fresh runs.
        /// </param>
        public void ConfigureScheduler(int totalSteps, int resumeRewarmupSteps = 0)
        {
            if (totalSteps <= 0)
                throw new ArgumentOutOfRangeException(nameof(totalSteps), "Total steps must be positive.");
            if (resumeRewarmupSteps < 0)
                throw new ArgumentOutOfRangeException(nameof(resumeRewarmupSteps), "Resume re-warmup must be non-negative.");

            int warmup = Math.Max(0, TrainingConfig.WarmupSteps);

            _scheduler = new LRScheduler(
                TrainingConfig.LearningRate,
                TrainingConfig.MinLearningRate,
                warmup,
                totalSteps);

            // Fresh run: no ramp. Resume: ramp the next N steps from ~0 back
            // to the scheduled value. Not checkpointed: a second resume
            // simply starts a fresh ramp, which is the safe direction.
            _resumeRampTotal = (_globalStep > 0) ? resumeRewarmupSteps : 0;
            _resumeRampDone = 0;
        }

        /// <summary>
        /// Clears scheduler state (used by tests / fresh model reuse).
        /// </summary>
        public void ClearScheduler()
        {
            _scheduler = null;
            _globalStep = 0;
            _resumeRampTotal = 0;
            _resumeRampDone = 0;
        }

        public void SaveCheckpoint(Stream destinationStream, int currentEpoch, float currentLoss)
        {
            using var writer = new BinaryWriter(destinationStream, Encoding.UTF8, leaveOpen: true);
            
            // Fixed 4-byte header (no length prefix)
            writer.Write("STCK"u8); 
            // v5 appends the optimizer + dropout-site training-state trailer
            // after the parameter block; v2-v4 have no trailer and still load
            // (with logged restart warnings).
            writer.Write(5);

            //Add the transformer model id to the checkpoint file 
            writer.Write(TransformerModelId.ToByteArray());

            //v3: record whether this checkpoint came from a QLoRA or a raw model.
            //A QLoRA model exposes LoRA adapters as trainable parameters while a
            //raw model exposes dense weights, so the two checkpoint families are
            //not interchangeable and must be rejected explicitly on load.
            writer.Write(UsesQLora);

            writer.Write(currentEpoch);
            writer.Write(currentLoss);

            // v4: global LR-schedule step so resume continues the identical
            // cosine curve instead of restarting at peak LR.
            writer.Write(_globalStep);

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

            // v5: training-state trailer. Without it a resume restores weights
            // but restarts AdamW moments/bias correction at zero and re-salts
            // every dropout site, so the run leaves the original trajectory
            // even though the LR schedule picks up correctly.
            CheckpointStateExtensions.WriteOptimizerState(writer, _optimizer.ExportState());
            CheckpointStateExtensions.WriteDropoutSiteStates(writer, _dropoutSites);
        }

        // Shared with CheckpointStateExtensions (v5 trailer tensor I/O).
        internal static void WriteTensor(BinaryWriter writer, TensorData tensor)
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
            //v4 appends the LR scheduler step after (epoch, loss); v2/v3 load
            //with step 0 (schedule restarts - logged below).
            //v5 appends the optimizer + dropout-site training-state trailer
            //after the parameter block.
            if (version < 2 || version > 5)
            {
                throw new InvalidDataException(
                    $"Unsupported checkpoint schema version: {version}. Expected 2, 3, 4 or 5.");
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

            // v4: restore the LR schedule position. v2/v3 have no step field;
            // the schedule restarts from 0 (warn - resuming old checkpoints
            // re-warms rather than continuing the cosine curve).
            int savedGlobalStep = 0;
            if (version >= 4)
            {
                savedGlobalStep = reader.ReadInt32();
                if (savedGlobalStep < 0)
                    throw new InvalidDataException(
                        $"Corrupt checkpoint: negative scheduler step ({savedGlobalStep}).");
            }

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
            model._globalStep = savedGlobalStep;

            // v5: training-state trailer follows the parameter block on the
            // stream. Restoring it after hydration keeps the v2-v4 path untouched.
            OptimizerState? savedOptimizerState = null;
            List<CheckpointStateExtensions.DropoutSiteState>? savedDropoutSites = null;
            if (version >= 5)
            {
                savedOptimizerState = CheckpointStateExtensions.ReadOptimizerState(reader);
                savedDropoutSites = CheckpointStateExtensions.ReadDropoutSiteStates(reader);
            }

            if (savedOptimizerState != null)
            {
                var parameterList = model.Parameters as IReadOnlyList<TrainableParameter>
                    ?? model.Parameters.ToList();

                // A kind mismatch (AdamW snapshot into SGD or vice versa) throws
                // here: the layouts are not interchangeable, so failing loudly
                // beats silently restarting the moments.
                int skippedEntries =
                    model._optimizer.ImportState(savedOptimizerState, parameterList);
                if (skippedEntries > 0)
                {
                    Log.Warning(
                        "Optimizer state import skipped {Skipped} of {Total} entries (unknown or shape-mismatched parameters).",
                        skippedEntries, savedOptimizerState.Parameters.Count);
                }

                int restoredSites = model.RestoreDropoutSites(savedDropoutSites!);
                Log.Information(
                    "v5 training state restored: optimizer kind {Kind}, bias-correction step {StepCount}, dropout sites {Restored}/{Total}.",
                    savedOptimizerState.Kind,
                    savedOptimizerState.StepCount,
                    restoredSites,
                    savedDropoutSites!.Count);
            }

            if (version < 4 && savedGlobalStep == 0)
            {
                Log.Warning(
                    "Checkpoint schema v{Version} has no scheduler step; LR schedule restarts from step 0 on resume.",
                    version);
            }

            Log.Information(
                "Checkpoint successfully loaded into model {ModelId}. Epoch: {Epoch}, Loss: {Loss}, SchedulerStep: {Step}.",
                model.TransformerModelId,
                epoch,
                loss,
                savedGlobalStep);

            return (epoch, loss);
        }       

        // Shared with CheckpointStateExtensions (v5 trailer tensor I/O).
        internal static TensorData ReadTensorOptimized(BinaryReader reader)
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

            // Re-key every dropout site for this step before the forward pass, so
            // the masks are a pure function of (site salt, step, batch item) and a
            // resume replays the identical sequence.
            PrepareDropoutForStep();

            TensorBase? prediction = null;
            TensorBase? auxOutput = null;
            TensorBase? gradient = null;

            // Per-step profiler: one bool check when disabled (default), one
            // log line per step when enabled via --train-profile.
            bool profiling = TrainingStepProfile.Enabled;
            using var stepProfile = profiling
                ? TrainingStepProfile.Begin(_workspace.Backend)
                : default(TrainingStepProfile.StepScope);

            try
            {
                // 1. Forward Pass
                long tPhase = profiling ? Stopwatch.GetTimestamp() : 0;
                (prediction, auxOutput) = Forward(inputs);
                if (profiling) stepProfile.MarkForward(Stopwatch.GetTimestamp() - tPhase);

                // 2. Compute Loss & Initial Backprop Gradient
                tPhase = profiling ? Stopwatch.GetTimestamp() : 0;
                float loss = _loss.Forward(prediction, expectedOutputs);
                // A non-finite loss means activations already diverged (logits
                // overflowed to Inf/NaN upstream). Failing here preserves the
                // last good optimizer state instead of backpropping NaN into
                // every parameter.
                if (!float.IsFinite(loss))
                    throw new InvalidOperationException(
                        $"Non-finite loss {loss} detected during training. Model diverged upstream (logits overflowed); check learning rate and gradient clipping.");
                gradient = _loss.Backward(prediction, expectedOutputs, _workspace);
                if (profiling) stepProfile.MarkLossBackward(Stopwatch.GetTimestamp() - tPhase);

                // 3. Backpropagate through Model.
                // Deliberately NOT wrapped in BeginBatchScope: Backward fans out
                // to Parallel.For workers inside LinearLayer.BackwardBatch, and a
                // scope is thread-affine, so holding one across the fan-out makes
                // workers wait on a batch only the (blocked) owner can submit.
                // That deadlocked until the dispatch timeout killed the job. This
                // path relies on the launcher's automatic wave coalescing
                // (OpStart/OpFinish) instead. CPU backends are unaffected either
                // way.
                tPhase = profiling ? Stopwatch.GetTimestamp() : 0;
                Backward(gradient);
                if (profiling) stepProfile.MarkModelBackward(Stopwatch.GetTimestamp() - tPhase);

                // 4. Clip ALL parameter gradients globally + NaN validation.
                // Honor the configured norm (default 1.0); the previous
                // hardcoded constant ignored TrainingConfig.MaxGradientNorm.
                // A non-positive setting disables clipping for this step.
                tPhase = profiling ? Stopwatch.GetTimestamp() : 0;
                float gradNorm = TrainingConfig.MaxGradientNorm > 0f
                    ? ClipGradients(TrainingConfig.MaxGradientNorm)
                    : ClipGradients(float.PositiveInfinity);

                // Non-finite norm (NaN/Inf grads from an overflowed step) makes
                // every scale factor NaN; stepping would poison all weights.
                // Skip the step and keep last good weights instead.
                if (!float.IsFinite(gradNorm))
                {
                    Log.Warning("[TrainStep] Non-finite gradient norm detected - skipping optimizer step to preserve last good weights.");
                    ZeroGradients();
                    return float.NaN;
                }

                // 5. Verify no gradients contain NaN before updating weights
                foreach (var p in Parameters)
                {
                    if (p.Gradient != null)
                    {
                        DiagonisticUtilities.AssertNoNaN(p.Gradient, "Gradient contains NaN prior to optimizer step.");
                    }
                }
                if (profiling) stepProfile.MarkClipValidate(Stopwatch.GetTimestamp() - tPhase);

                // 6. Step optimizer. The LR schedule owns the rate: apply this
                // step's value before the optimizer consumes it. Without a
                // configured schedule the base rate stands (legacy flat LR).
                // The schedule advances on every completed step - LR is a
                // function of step index, not of gradient magnitude.
                tPhase = profiling ? Stopwatch.GetTimestamp() : 0;
                if (_scheduler != null)
                {
                    _optimizer.LearningRate = GetStepLearningRate();
                    if (_resumeRampDone < _resumeRampTotal)
                        _resumeRampDone++;
                }
                _optimizer.Step(Parameters);
                _globalStep++;
                if (profiling) stepProfile.MarkOptimizer(Stopwatch.GetTimestamp() - tPhase);

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

                // 9. Return idle pooled device state toward budget: variable
                // sequence lengths and batch sizes churn power-of-two buckets
                // every step, so trim here rather than pinning them forever.
                // CPU backends no-op.
                _workspace.Backend.TrimIdleMemory();

                // 10. Host memory pressure valve. Runs after the cheap unmanaged
                // trim above, so the first rung has already had its effect. The
                // workspace pool at this point holds the tensors returned by
                // *earlier* steps (each step's finally-Reset), which are idle and
                // therefore eligible for trimming; this step's own activations
                // are still borrowed and are never touched.
                ApplyMemoryRelief();

                return loss;
            }
            finally
            {
                // Clean up transient forward & loss tensors for this step.
                // The loss gradient is workspace-borrowed (pooled with the
                // step's activations), so Reset() reclaims it: do NOT dispose.
                DisposeIfDisposable(prediction);
                DisposeIfDisposable(auxOutput);
                _workspace.Reset();
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
        /// Executes whatever relief the memory valve has asked for. The valve
        /// itself only decides; the actions live here because they need the
        /// workspace and the backend, which the policy deliberately does not
        /// hold references to.
        /// </summary>
        private void ApplyMemoryRelief()
        {
            MemoryReliefLevel level = _memoryValve.Tick();

            if (level == MemoryReliefLevel.None)
                return;

            const double mib = 1024.0 * 1024.0;

            switch (level)
            {
                case MemoryReliefLevel.TrimDevicePools:
                    // Step 9 already ran TrimIdleMemory this step, so the cheap
                    // rung is satisfied by construction. Logged rather than
                    // repeated so the escalation history stays visible. Reads
                    // the sample the valve already took - no second syscall.
                    Log.Debug(
                        "Memory valve: trimmed idle device pools at {Pct:F0}% of system memory.",
                        _memoryValve.LastSample.EffectiveFraction * 100.0);
                    break;

                case MemoryReliefLevel.TrimWorkspace:
                {
                    long before = _workspace.RetainedBytes;
                    int dropped = _workspace.TrimRetainedTo(RetainedTarget(before));

                    // Gen-0, non-blocking: enough to hand the released arrays
                    // back without stalling the training thread.
                    GC.Collect(0, GCCollectionMode.Forced, blocking: false, compacting: false);

                    Log.Warning(
                        "Memory valve: released {Dropped} pooled activations ({Before:F0} -> {After:F0} MiB) under pressure. {Status}",
                        dropped, before / mib, _workspace.RetainedBytes / mib, _memoryValve.Describe());
                    break;
                }

                case MemoryReliefLevel.Compact:
                {
                    // Hard floor: keep a quarter of what the pool holds rather
                    // than the configured retain percent, because at this point
                    // the cheap rungs have already failed to bring usage down.
                    long before = _workspace.RetainedBytes;
                    int dropped = _workspace.TrimRetainedTo(before / 4);

                    // The only rung that reclaims a fragmented multi-gigabyte
                    // heap, and the only one that blocks. Rate-limited by the
                    // valve's latch and cooldown, so it cannot run per step.
                    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);

                    Log.Warning(
                        "Memory valve: COMPACT at {Pct:F0}% of system memory; released {Dropped} pooled activations ({Before:F0} -> {After:F0} MiB). Raise [Memory] max_quota_mb or lower batch_size if this repeats. {Status}",
                        _memoryValve.LastSample.EffectiveFraction * 100.0,
                        dropped, before / mib, _workspace.RetainedBytes / mib, _memoryValve.Describe());
                    break;
                }
            }
        }

        /// <summary>Bytes the workspace pool should be trimmed down to.</summary>
        private static long RetainedTarget(long retained)
        {
            double percent = Math.Clamp(MemoryPressureSettings.WorkspaceRetainPercent, 0.0, 100.0);
            return (long)(retained * percent / 100.0);
        }

        /// <summary>
        /// One-line host-memory telemetry, for the per-epoch log and the
        /// frontend job message. Complements
        /// <see cref="IAccelerationBackend.DescribeMemoryUsage"/>, which reports
        /// device-side memory only.
        /// </summary>
        public string DescribeMemoryPressure()
        {
            if (!MemoryPressureSettings.Enabled)
                return "memory valve off";

            //A freshly loaded model's valve has never evaluated, so its last
            //sample would be the default (unusable) struct and Describe would
            //read "no memory container reported"; refresh so status endpoints
            //and logs describe the machine as it is now.
            _memoryValve.RefreshSample();

            const double mib = 1024.0 * 1024.0;
            return $"{_memoryValve.Describe()}; " +
                   $"activation pool {_workspace.RetainedBytes / mib:F0} MiB " +
                   $"across {_workspace.PooledTensorCount} tensors " +
                   $"({_workspace.PooledShapeCount} shapes)";
        }

        /// <summary>
        /// Frontend-triggered recovery for the memory pressure valve (POST
        /// api/v1/memory/reset). Two halves:
        /// <list type="bullet">
        /// <item>
        /// The valve's latch, cooldown and counters are cleared, so a retry
        /// gets the full escalation ladder instead of history from the run
        /// that failed under pressure.
        /// </item>
        /// <item>
        /// When <paramref name="relieveNow"/> is true, the same actions the
        /// Compact rung takes happen immediately: drop idle device pools, trim
        /// the activation pool to a quarter of what it holds, and force a
        /// blocking, compacting gen-2 collection. The collection can stall a
        /// running training job, which is why callers may turn it off.
        /// </item>
        /// </list>
        /// Returns the refreshed <see cref="DescribeMemoryPressure"/> telemetry.
        /// </summary>
        public string ResetMemoryPressure(bool relieveNow = true)
        {
            _memoryValve.Reset();

            if (!relieveNow)
                return DescribeMemoryPressure();

            const double mib = 1024.0 * 1024.0;

            _workspace.Backend.TrimIdleMemory();

            long before = _workspace.RetainedBytes;
            int dropped = _workspace.TrimRetainedTo(before / 4);

            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);

            Log.Warning(
                "Memory valve: manual RESET released {Dropped} pooled activations ({Before:F0} -> {After:F0} MiB) and compacted on request. {Status}",
                dropped, before / mib, _workspace.RetainedBytes / mib, _memoryValve.Describe());

            return DescribeMemoryPressure();
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
            _embeddingDropout = new DropoutSite("embedding_dropout", TrainingConfig.DropoutRate);
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
                        dropoutRate: TrainingConfig.DropoutRate,
                        name: blockName));

                Log.Information($"Layer {i} constructed in {layerWatch.ElapsedMilliseconds}ms. Total elapsed: {watch.ElapsedMilliseconds}ms.");
                layerWatch.Restart();
            }

            // Flatten the dropout sites once the architecture is complete, so the
            // per-step re-key and checkpoint save/load can walk a flat list.
            CollectDropoutSites();


            Log.Information($"Transformer architecture initialisation completed in {watch.ElapsedMilliseconds}ms."); 
            watch.Stop();
        }
        public float ClipGradients(float maxNorm = 1.0f)
        {
            double sumSquaredNorm = 0.0;

            // 1. Accumulate squared gradients across ALL trainable parameters.
            // A single NaN/Inf gradient makes (g*g) NaN/Inf; without an explicit
            // finite check below, `NaN > maxNorm` is false so clipping is
            // silently skipped and the optimizer poisons every weight.
            foreach (var param in Parameters)
            {
                if (param.Gradient == null) continue;

                ReadOnlySpan<float> gData = param.Gradient.Data.AsSpan();
                for (int i = 0; i < gData.Length; i++)
                {
                    float g = gData[i];
                    if (!float.IsFinite(g))
                        return float.NaN;
                    sumSquaredNorm += (double)g * g;
                }
            }

            float totalNorm = (float)Math.Sqrt(sumSquaredNorm);

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

        //Causal (autoregressive) attention mask for a window of this length, or null
        //when AttentionMaskSettings.UseCausalMask is off.
        //
        //Owned by the model rather than borrowed from the workspace: it must stay
        //valid for the whole step, because the recompute path in
        //ScaledDotProductAttention re-applies it during Backward. Rebuilt only
        //when the window length changes, which in practice means once per model.
        //At seq 2048 this is a single 16 MiB tensor - constant, not per-step.
        private TensorBase? _causalMask;
        private int _causalMaskLength;

        private TensorBase? GetCausalMask(int sequenceLength)
        {
            if (!AttentionMaskSettings.UseCausalMask)
                return null;

            if (_causalMask != null && _causalMaskLength == sequenceLength)
                return _causalMask;

            _causalMask?.Dispose();
            _causalMask = MaskUtilitiesSimd.CreateCausalMask(sequenceLength);
            _causalMaskLength = sequenceLength;
            return _causalMask;
        }

        public void Dispose()
        {
            // Dispose layers/resources that own unmanaged or pooled resources.
            _embedding?.Dispose();
            _position?.Dispose();
            _outputProjection?.Dispose();

            //Model-owned, not workspace-pooled, so Reset() never reclaims it.
            _causalMask?.Dispose();
            _causalMask = null;

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
