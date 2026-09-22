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

            Console.WriteLine();
            Console.WriteLine($"Results: {(parityOk && trajectoryOk && predictOk ? "PASS" : "FAIL")}.");
            return parityOk && trajectoryOk && predictOk;
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