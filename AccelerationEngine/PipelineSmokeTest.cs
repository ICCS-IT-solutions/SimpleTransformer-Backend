using System;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine
{
    /// <summary>
    /// End-to-end smoke test that drives an actual tiny TransformerModel through
    /// TrainStep/Predict on two different backends (the auto-selected SIMD backend
    /// and the pure managed reference backend) with identical weights, verifying:
    ///  1. A full training step runs on the migrated layer stack without errors.
    ///  2. Both backends produce matching loss for the same weights and inputs
    ///     (small tolerance: the SIMD GELU fast-tanh approximation).
    ///  3. The loss trajectory stays finite across several optimizer steps.
    ///  4. Predict returns a valid token id.
    ///
    /// Run with: dotnet run -- --pipeline-smoketest
    /// </summary>
    public static class PipelineSmokeTest
    {
        private const float StepParityTolerance = 0.05f; // GELU fast-tanh approximation differences

        public static bool RunAndPrint()
        {
            Console.WriteLine("=== Pipeline Smoke Test (full TrainStep/Predict through IAccelerationBackend) ===");
            Console.WriteLine();

            var config = new TransformerConfig
            {
                VocabSize = 64,
                EmbeddingSize = 16,
                NumLayers = 1,
                NumHeads = 2,
                FeedForwardSize = 32,
                MaxSequenceLength = 8
            };

            var trainingConfig = new TrainingConfig
            {
                Optimizer = OptimizerType.AdamW,
                LearningRate = 0.001f
            };

            const int sequenceLength = 8;
            var random = new Random(7);

            // 1. Two structurally identical models on different backends
            using var simdModel = new TransformerModel(
                Guid.NewGuid(), config, trainingConfig, useQLora: true);

            using var refModel = new TransformerModel(
                Guid.NewGuid(), config, trainingConfig, useQLora: true,
                BackendSelector.BackendType.CpuReference);

            Console.WriteLine($"SIMD pipeline backend      : {simdModel.Backend.Name}");
            Console.WriteLine($"Reference pipeline backend : {refModel.Backend.Name}");
            Console.WriteLine($"Parameters                 : {simdModel.Parameters.Count()}");
            Console.WriteLine();

            // 2. Force identical weights on both models
            CopyParameters(simdModel, refModel);
            Console.WriteLine("Weights synchronised between models.");
            Console.WriteLine();

            // 3. Random sequence sample (token ids as floats, matching tokenizer output)
            (TensorBase Input, TensorBase Target) MakeSample()
            {
                var input = new Tensor(sequenceLength);
                var target = new Tensor(sequenceLength);
                for (int i = 0; i < sequenceLength; i++)
                {
                    input[i] = random.Next(config.VocabSize);
                    target[i] = random.Next(config.VocabSize);
                }
                return (input, target);
            }

            var (inputA, targetA) = MakeSample();

            // 4. One-step loss parity between backends
            float simdLoss = simdModel.TrainStep(inputA, targetA);
            float refLoss = refModel.TrainStep(inputA, targetA);
            float lossDelta = MathF.Abs(simdLoss - refLoss);

            Console.WriteLine($"Step 1 loss (SIMD)      : {simdLoss:G6}");
            Console.WriteLine($"Step 1 loss (Reference) : {refLoss:G6}");
            Console.WriteLine($"Loss delta              : {lossDelta:G3} (tol {StepParityTolerance})");
            bool parityOk = lossDelta <= StepParityTolerance && IsFinite(simdLoss) && IsFinite(refLoss);

            // 5. Training trajectory on the SIMD pipeline (finite losses, rough descent)
            float firstLoss = simdLoss;
            float lastLoss = simdLoss;
            bool trajectoryOk = true;

            for (int step = 2; step <= 10; step++)
            {
                var (input, target) = MakeSample();
                float loss = simdModel.TrainStep(input, target);

                if (!IsFinite(loss))
                    trajectoryOk = false;

                lastLoss = loss;
                Console.WriteLine($"Step {step,2} loss (SIMD)      : {loss:G6}");
            }

            trajectoryOk &= IsFinite(lastLoss);
            Console.WriteLine($"Trajectory: first {firstLoss:G4} -> last {lastLoss:G4} (all finite: {trajectoryOk}).");

            // 6. Prediction through the backend
            var (inputB, _) = MakeSample();
            var (nextTokenId, allTokenIds, logits, probabilities, hiddenState) = simdModel.Predict(inputB);

            bool predictOk =
                nextTokenId >= 0 && nextTokenId < config.VocabSize &&
                allTokenIds.Length == sequenceLength &&
                IsFinite(hiddenState[0, 0]);

            Console.WriteLine();
            Console.WriteLine($"Predict: {allTokenIds.Length} token ids, next = {nextTokenId} (valid range 0..{config.VocabSize - 1}): {predictOk}.");

            // 7. Mini-batch assembly, including the DropLast remainder rule.
            Console.WriteLine();
            Console.WriteLine("--- Mini-batch assembly (DropLast) ---");
            bool batchingOk = CheckMiniBatching(config, trainingConfig);

            // 8. Checkpoint round-trip and QLoRA/raw mode isolation.
            //A QLoRA model and a raw model expose different trainable parameters,
            //so a checkpoint from one must never load into the other. Schema v3
            //records the mode so that is caught with a clear message instead of a
            //confusing parameter-name failure.
            Console.WriteLine();
            Console.WriteLine("--- Checkpoint mode isolation ---");
            bool checkpointOk = CheckpointModeIsolation(config, trainingConfig);

            Console.WriteLine();
            Console.WriteLine($"Results: {(parityOk && trajectoryOk && predictOk && batchingOk && checkpointOk ? "PASS" : "FAIL")}.");
            return parityOk && trajectoryOk && predictOk && batchingOk && checkpointOk;
        }

        /// <summary>
        /// Verifies the DropLast remainder rule: a short trailing batch is dropped
        /// when at least one full batch exists, and kept when dropping would leave
        /// nothing to train on.
        /// </summary>
        private static bool CheckMiniBatching(TransformerConfig config, TrainingConfig trainingConfig)
        {
            //Case 1: 9 samples, batch 8 -> one full batch, remainder of 1 dropped.
            //Case 2: 5 samples, batch 8 -> no full batch exists, so the single
            //partial batch must be kept or nothing would train.
            //Case 3: 16 samples, batch 8 -> exactly two full batches, nothing dropped.
            var cases = new (int samples, bool dropLast, int expectedBatches, int expectedSmallest)[]
            {
                (9, true, 1, 8),
                (9, false, 2, 1),
                (5, true, 1, 5),
                (5, false, 1, 5),
                (16, true, 2, 8),
                (16, false, 2, 8),
            };

            bool allOk = true;
            foreach (var (sampleCount, dropLast, expectedBatches, expectedSmallest) in cases)
            {
                var cfg = new TrainingConfig
                {
                    BatchSize = 8,
                    DropLast = dropLast,
                    Optimizer = OptimizerType.AdamW
                };

                using var model = new TransformerModel(
                    Guid.NewGuid(), config, cfg, useQLora: true,
                    BackendSelector.BackendType.CpuReference);

                var samples = new List<TrainingSample>(sampleCount);
                for (int i = 0; i < sampleCount; i++)
                {
                    samples.Add(new TrainingSample
                    {
                        Input = new Tensor(config.MaxSequenceLength),
                        Target = new Tensor(config.MaxSequenceLength)
                    });
                }

                var batches = TrainingDataExtensions.CreateMiniBatches(model, samples);
                int smallest = batches.Count > 0 ? batches.Min(b => b.BatchSize) : 0;
                int used = batches.Sum(b => b.BatchSize);

                //DropLast on: the short trailing batch is discarded, so every
                //kept batch is full. The single exception is a dataset smaller
                //than one batch, where the only partial batch is kept so that
                //something still trains.
                //DropLast off: the remainder is kept, so a partial batch is correct.
                bool datasetSmallerThanOneBatch = sampleCount < 8;
                bool partialsAreLegal = !dropLast || datasetSmallerThanOneBatch;
                bool everyBatchFull = batches.All(b => b.BatchSize == 8);

                bool ok = batches.Count == expectedBatches &&
                          smallest == expectedSmallest &&
                          (partialsAreLegal || everyBatchFull);

                Console.WriteLine(
                    $"  samples={sampleCount,2} dropLast={dropLast,-5} -> " +
                    $"{batches.Count} batch(es), smallest {smallest}, {used} used : {ok}");

                allOk &= ok;
            }

            return allOk;
        }

        /// <summary>
        /// Verifies that a saved checkpoint round-trips into a model of the same
        /// training mode, and is rejected by a model of the opposite mode.
        /// </summary>
        private static bool CheckpointModeIsolation(TransformerConfig config, TrainingConfig trainingConfig)
        {
            var modelId = Guid.NewGuid();

            using var qloraModel = new TransformerModel(modelId, config, trainingConfig, useQLora: true);
            using var rawModel = new TransformerModel(modelId, config, trainingConfig, useQLora: false);

            Console.WriteLine($"QLoRA model parameters : {qloraModel.Parameters.Count()}");
            Console.WriteLine($"Raw model parameters   : {rawModel.Parameters.Count()}");

            if (qloraModel.Parameters.Count() == rawModel.Parameters.Count())
            {
                Console.WriteLine("[FAIL] Expected the two modes to expose different parameter counts.");
                return false;
            }

            // Round-trip: save from the QLoRA model, load into a fresh QLoRA model.
            using var roundTrip = new TransformerModel(modelId, config, trainingConfig, useQLora: true);
            using var buffer = new MemoryStream();
            qloraModel.SaveCheckpoint(buffer, currentEpoch: 3, currentLoss: 1.25f);

            buffer.Position = 0;
            var (epoch, loss) = TransformerModel.LoadCheckpoint(buffer, roundTrip);
            bool sameModeOk = epoch == 3 && MathF.Abs(loss - 1.25f) < 1e-6f;
            Console.WriteLine($"QLoRA -> QLoRA load   : epoch {epoch}, loss {loss:G4} (expected 3, 1.25): {sameModeOk}");

            // Raw round-trip.
            using var rawRoundTrip = new TransformerModel(modelId, config, trainingConfig, useQLora: false);
            using var rawBuffer = new MemoryStream();
            rawModel.SaveCheckpoint(rawBuffer, currentEpoch: 2, currentLoss: 0.5f);

            rawBuffer.Position = 0;
            var (rawEpoch, rawLoss) = TransformerModel.LoadCheckpoint(rawBuffer, rawRoundTrip);
            bool rawSameModeOk = rawEpoch == 2 && MathF.Abs(rawLoss - 0.5f) < 1e-6f;
            Console.WriteLine($"Raw -> Raw load       : epoch {rawEpoch}, loss {rawLoss:G4} (expected 2, 0.5): {rawSameModeOk}");

            // Cross-mode: a QLoRA checkpoint must be refused by a raw model.
            bool crossRejected = false;
            string? crossMessage = null;
            try
            {
                buffer.Position = 0;
                TransformerModel.LoadCheckpoint(buffer, rawRoundTrip);
            }
            catch (InvalidDataException ex)
            {
                crossRejected = true;
                crossMessage = ex.Message;
            }

            Console.WriteLine($"QLoRA ckpt -> raw model: rejected = {crossRejected}");
            if (crossMessage != null)
                Console.WriteLine($"  message: {crossMessage}");

            return sameModeOk && rawSameModeOk && crossRejected;
        }

        private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        /// <summary>
        /// Copies all trainable parameter values from the source model to the target model.
        /// </summary>
        private static void CopyParameters(TransformerModel source, TransformerModel target)
        {
            var checkpoints = source.Parameters
                .Select(p => new TrainableParameterCheckpoint
                {
                    Name = p.Name,
                    Value = new TensorData
                    {
                        Shape = (int[])p.Value.Shape.Clone(),
                        Data = (float[])p.Value.Data.Clone()
                    }
                })
                .ToList();

            target.LoadCheckpointData(checkpoints);
        }
    }
}