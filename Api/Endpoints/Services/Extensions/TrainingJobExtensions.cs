using Microsoft.EntityFrameworkCore;
using Serilog;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.ManagementEngine;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

public static class TrainingJobExtensions
{
    public static async Task RunTrainingLoop(
        TrainingJobEntry job,
        TransformerModel model,
        int startEpoch,
        TrainingConfig config,
        TransformerModelEntry modelEntry,
        TrainingJobControl control,
        IDbContextFactory<AppDbContext> dbFactory,
        ITokenizer tokenizer,
        TrainingJobManager jobManager     
    )
    {
        using var db = await dbFactory.CreateDbContextAsync();
        string inputText;

        if(job.InputText != null)
        {
            inputText = job.InputText;
        }
        else if(job.InputFilePath != null)
        {
            inputText = await File.ReadAllTextAsync(job.InputFilePath);
        }
        else
        {
            throw new ArgumentException("Input text or file path must be provided.");
        }

        var samples = TrainingDataExtensions.CreateTrainingSamples(model, tokenizer, inputText);
        var miniBatches = TrainingDataExtensions.CreateMiniBatches(model, samples);
        var numBatches = miniBatches.Count;
        Log.Information($"{numBatches} batches created from {samples.Count} samples.");
        // Determine chunkSize dynamically based on miniBatches count
        int chunkSize = miniBatches.Count switch
        {
            >= 8 => 8,
            >= 4 => 4,
            >= 2 => 2,
            _    => 1 // A single batch of size N (or 1) wrapped cleanly
        };

        // Zero-allocation chunking using .NET 6+ Chunk()
        var subBatches = miniBatches.Chunk(chunkSize);
        
        int totalOuterBatches = subBatches.Count();

        var totalEpochs = startEpoch + (config?.Epochs ?? 10); // Default to 10 epochs if not specified

        //Update the job status
        await UpdateJob(dbFactory, job.EntryId, job =>
        {
            job.Status = TrainingJobStatus.Started;
            job.CurrentEpoch = startEpoch;
            job.TotalEpochs = totalEpochs;

            job.CurrentBatch = 0;
            job.TotalBatches = totalOuterBatches;

            job.TotalSubBatches = chunkSize;
            job.CurrentSubBatch = 0;

            job.Message =
                $"Training prepared: {samples.Count} samples in " +
                $"{numBatches} mini-batches ({totalOuterBatches} outer batches).";
        });

        // 1. Begin training. This notifies other endpoint services that the model is busy training and should not be used.
        model.BeginTraining();
        try
        {
            // 2. Training loop
            for (int epoch = startEpoch; epoch < totalEpochs; epoch++)
            {
                float epochLoss = 0f;
                // Single Random instance reused across the engine
                var rng = new Random();

                await UpdateJob(dbFactory, job.EntryId, job =>
                {
                    job.Status = TrainingJobStatus.Running;
                    job.CurrentEpoch = epoch + 1;
                    job.CurrentBatch = 0;
                    job.CurrentLoss = 0;
                    job.Message = $"Training epoch {epoch + 1} of {totalEpochs}.";
                });

                if (numBatches > 8)
                {
                    // Shuffle the outer sub-batch groups ONCE per epoch
                    var shuffledSubBatches = Shuffle(subBatches.ToList(), rng);
                    await UpdateJob (dbFactory, job.EntryId, job => 
                    {
                        job.Message = $"Shuffling batches (epoch {epoch + 1}/{totalEpochs}).";
                        job.TotalSubBatches = chunkSize;
                    });

                    for (int batch = 0; batch < shuffledSubBatches.Count; batch++)
                    {
                        await control.WaitIfPausedAsync();

                        control.Cancellation.Token.ThrowIfCancellationRequested();

                        if ((batch + 1) % 5 == 0 ||
                            batch == 0 ||
                            batch == numBatches - 1)
                        {
                            await UpdateJob(dbFactory, job.EntryId, job =>
                            {
                                job.CurrentBatch = batch + 1;
                                job.CurrentLoss = epochLoss;

                                job.Message =
                                    $"Epoch {epoch + 1}/{totalEpochs}, " +
                                    $"batch {batch + 1}/{numBatches}.";
                            });
                        }
                        var currentSubBatch = shuffledSubBatches[batch];

                        for (int subBatch = 0; subBatch < currentSubBatch.Count(); subBatch++)
                        {
                            await control.WaitIfPausedAsync();

                            control.Cancellation.Token.ThrowIfCancellationRequested();

                            var item = currentSubBatch[subBatch];
                            epochLoss += model.TrainStep(item.Inputs, item.Targets);

                            // Correct parenthesis grouping for modulus
                            if ((subBatch + 1) % 4 == 0 || subBatch == 0)
                            {
                                await UpdateJob(dbFactory, job.EntryId, job => job.CurrentSubBatch = subBatch + 1);
                                Log.Information($"Epoch {epoch + 1}, Batch {batch + 1}, Sub-Batch {subBatch + 1} training loss: {epochLoss:F6}");
                            }
                        }
                        //It may be helpful to save temporary checkpoints here, and simply overwrite them. Possible name could be checkpoint-epoch-{epoch}-temp.bin
                        //This can help prevent loss of progress should a training run fail or be stopped.
                        //Get the transformer config name from the model entry
                        var transformerModelName = await GetModelNameFromId(dbFactory, modelEntry.EntryId);
                        var checkpointFilename = $"checkpoint-epoch-{epoch + 1}-temp.bin";
                        var checkpointDirname = $"checkpoints/{transformerModelName}";
                        var checkpointEntry = new TrainingCheckpointEntry
                        {
                            Epoch = epoch + 1,
                            Loss = epochLoss,
                            Filename = checkpointFilename,
                            Filepath = $"checkpoints/{transformerModelName}/"
                        };

                        await db.TrainingCheckpoints.AddAsync(checkpointEntry);
                        await db.SaveChangesAsync();
                    
                        string checkpointFilepath = $"{checkpointDirname}/{checkpointFilename}";

                        if (!Directory.Exists(checkpointDirname))
                        {
                            Directory.CreateDirectory(checkpointDirname);
                        }

                        await using var stream = File.Create(checkpointFilepath);

                        model.SaveCheckpoint(stream, epoch, epochLoss);
                        await UpdateJob(dbFactory, job.EntryId, job => {job.CheckpointFilename = checkpointFilepath; job.TrainingCheckpointId = checkpointEntry.EntryId;});
                    }
                }
                else
                {   // Number of sub batches here is 1.
                    // Shuffle standard mini-batches ONCE per epoch
                    var shuffledMiniBatches = Shuffle(miniBatches.ToList(), rng);
                    await UpdateJob(dbFactory, job.EntryId, job =>
                    {
                        job.Message = $"Shuffling batches (epoch {epoch + 1}/{totalEpochs}).";
                        job.TotalBatches = shuffledMiniBatches.Count;
                        job.TotalSubBatches = 1;
                    });
                    for (int batch = 0; batch < numBatches; batch++)
                    {
                        var item = shuffledMiniBatches[batch];
                        epochLoss += model.TrainStep(item.Inputs, item.Targets);

                        // Correct parenthesis grouping for modulus
                        if ((batch + 1) % 10 == 0 || batch == 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            Log.Information($"Epoch {epoch + 1}, Batch {batch + 1} training loss: {epochLoss:F6}");
                            Console.ResetColor();
                        }
                    }
                }

                epochLoss /= miniBatches.Count;

                Log.Information($"Epoch {epoch + 1}: Loss={epochLoss:F6}");
                await UpdateJob(dbFactory, job.EntryId, job =>
                {
                    job.CurrentEpoch = epoch + 1;
                    job.CurrentBatch = numBatches;
                    job.CurrentLoss = epochLoss;
                    job.Message =
                        $"Epoch {epoch + 1} of {totalEpochs} completed.";
                });

                // Save checkpoint periodically via TransformerModel. Every 5 epochs or on the last epoch, save a checkpoint
                if ((epoch + 1) % 5 == 0 || epoch == totalEpochs - 1)
                {
                    var transformerModelName = await GetModelNameFromId(dbFactory, modelEntry.EntryId);
                    var checkpointFilename = $"checkpoint-epoch-{epoch + 1}-loss-{epochLoss:F6}.bin";
                    var checkpointDirname = $"checkpoints/{transformerModelName}";
                    var checkpointEntry = new TrainingCheckpointEntry
                    {
                        Epoch = epoch + 1,
                        Loss = epochLoss,
                        Filename = checkpointFilename,
                        Filepath = $"checkpoints/{transformerModelName}/"
                    };

                    await db.TrainingCheckpoints.AddAsync(checkpointEntry);
                    await db.SaveChangesAsync();
                    
                    string checkpointFilepath = $"{checkpointDirname}/{checkpointFilename}";
                    
                    await using var stream = File.Create(checkpointFilepath);
                    model.SaveCheckpoint(stream, epoch, epochLoss);
                    
                    await UpdateJob(dbFactory, job.EntryId, job =>
                    {
                        job.CheckpointFilename = checkpointFilepath;
                        job.CurrentLoss = epochLoss;
                        job.Message =
                            $"Epoch {epoch + 1} completed. Checkpoint saved.";
                        job.TrainingCheckpointId = checkpointEntry.EntryId;
                    });
                    Log.Information("Saved checkpoint to {FileName}", checkpointFilepath);
                }
            }

            //End training normally
            model.EndTraining();
        }
        catch (OperationCanceledException)
        {
            Log.Information(
                "Training job {JobId} was stopped.",
                job.EntryId);

            await UpdateJob(dbFactory, job.EntryId, job =>
            {
                job.Status = TrainingJobStatus.Stopped;
                job.Message = "Training job stopped.";
                job.DateCompleted = DateTime.UtcNow;
            });
        }            
        finally
        {
            //If something goes wrong, end training.
            model.EndTraining();
            await UpdateJob(dbFactory, job.EntryId, job =>
            {
                job.Status = TrainingJobStatus.Completed;
                job.CurrentEpoch = totalEpochs;
                job.CurrentBatch = job.TotalBatches;
                job.Message = "Model training completed successfully.";
                job.DateCompleted = DateTime.UtcNow;
            });
            jobManager.Remove(job.EntryId);
        }

        //Update the job status
        await UpdateJob(dbFactory, job.EntryId, job =>
        {
            job.Status = TrainingJobStatus.Completed;
            job.CurrentEpoch = totalEpochs;
            job.CurrentBatch = job.TotalBatches;
            job.Message = "Model training completed successfully.";
            job.DateCompleted = DateTime.UtcNow;
        });           
    }
    public static async Task RegisterJob(IDbContextFactory<AppDbContext> dbFactory, TrainingJobEntry jobEntry)
    {
        await using var db = await dbFactory.CreateDbContextAsync();

        await db.TrainingJobs.AddAsync(jobEntry);
        await db.SaveChangesAsync();
    }

    public static async Task UpdateJob(
        IDbContextFactory<AppDbContext> dbFactory,
        Guid jobId,
        Action<TrainingJobEntry> update)
    {
        await using var db = await dbFactory.CreateDbContextAsync();

        var jobEntry = await db.TrainingJobs
            .FirstOrDefaultAsync(x => x.EntryId == jobId);

        if (jobEntry == null)
            return;

        update(jobEntry);

        jobEntry.DateUpdated = DateTime.UtcNow;

        await db.SaveChangesAsync();
    }

    public static async Task<string?> GetModelNameFromId(IDbContextFactory<AppDbContext> dbFactory, Guid modelId)
    {
        await using var db = await dbFactory.CreateDbContextAsync();

        return await db.TransformerModels
            .Where(x => x.EntryId == modelId)
            .Select(x => x.Name)
            .FirstOrDefaultAsync();
    }   

    public static List<T> Shuffle<T>(IList<T> source, Random random)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        
        TrainingServiceExtensions.ShuffleInPlace<T>(source, random);
        return source.ToList();
    }         
}