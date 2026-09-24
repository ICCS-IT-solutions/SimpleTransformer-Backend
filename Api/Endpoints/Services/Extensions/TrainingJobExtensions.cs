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
    public static async Task ReconcileDbStateOnStartupAsync(
        IDbContextFactory<AppDbContext> dbFactory)
    {
        Log.Information("Reconciling training runtime state after startup...");
        await using var db = await dbFactory.CreateDbContextAsync();

        //No runtime model survives a process restart, so any "loaded" flags
        //left in the database are stale and would block a fresh load.
        var loadedModels = await db.TransformerModels
            .Where(x => x.IsLoaded)
            .ToListAsync();
        foreach (var modelEntry in loadedModels)
        {
            Log.Information(
                "Clearing stale loaded flag for model {ModelId}.",
                modelEntry.EntryId);
            modelEntry.IsLoaded = false;
            modelEntry.DateUpdated = DateTime.UtcNow;
        }

        //A job whose background loop is gone from memory is not training. Since
        //TrainingJobManager is empty on startup, any live-looking state means the
        //server was interrupted mid-run: return those jobs to Paused so they can
        //be resumed from their latest checkpoint instead of rotting as
        //"Running".
        var interrupted = await db.TrainingJobs
            .Where(x =>
                x.Status == TrainingJobStatus.Started ||
                x.Status == TrainingJobStatus.Running ||
                x.Status == TrainingJobStatus.Pending)
            .ToListAsync();
        foreach (var jobEntry in interrupted)
        {
            Log.Information(
                "Returning interrupted job {JobId} ({Status}) to Paused; no live training task survives a restart.",
                jobEntry.EntryId, jobEntry.Status);
            jobEntry.Status = TrainingJobStatus.Paused;
            jobEntry.Message = "Server restarted while this job was active. It has been returned to Paused; resume it to continue from the latest checkpoint.";
            jobEntry.DateUpdated = DateTime.UtcNow;
        }

        await db.SaveChangesAsync();
    }

    public static bool IsTerminalStatus(TrainingJobStatus status)
    {
        return status == TrainingJobStatus.Completed ||
               status == TrainingJobStatus.Failed ||
               status == TrainingJobStatus.Cancelled ||
               status == TrainingJobStatus.Stopped;
    }

    public static async Task<TrainingCheckpointEntry?> ResolveResumeCheckpointAsync(
        AppDbContext db,
        Guid jobId,
        Guid transformerModelId)
    {
        var jobEntry = await db.TrainingJobs
            .FirstOrDefaultAsync(x => x.EntryId == jobId);

        if (jobEntry == null)
        {
            return null;
        }

        //Prefer the checkpoint the job already points at; if that row is gone,
        //fall back to the newest record whose file still exists on disk.
        if (jobEntry.TrainingCheckpointId.HasValue)
        {
            var stored = await db.TrainingCheckpoints
                .FirstOrDefaultAsync(x => x.EntryId == jobEntry.TrainingCheckpointId.Value);

            if (stored != null && File.Exists(Path.Combine(stored.Filepath, stored.Filename)))
            {
                return stored;
            }

            Log.Warning(
                "Training job {JobId} pointed at checkpoint {CheckpointId}, but the row or file is gone; falling back to the latest surviving checkpoint.",
                jobId, jobEntry.TrainingCheckpointId);
        }

        var modelName = await db.TransformerModels
            .Where(x => x.EntryId == transformerModelId)
            .Select(x => x.Name)
            .FirstOrDefaultAsync();

        if (string.IsNullOrWhiteSpace(modelName))
        {
            return null;
        }

        var candidates = await db.TrainingCheckpoints
            .Where(x => x.Filepath == $"checkpoints/{modelName}/")
            .OrderByDescending(x => x.DateCreated)
            .ToListAsync();

        return candidates.FirstOrDefault(x =>
            File.Exists(Path.Combine(x.Filepath, x.Filename)));
    }
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

        //Report any shortfall so a short batch is never a silent surprise: either
        //DropLast discarded a remainder, or the dataset was too small to fill one
        //full batch and the partial batch was kept deliberately.
        int batchSize = config?.BatchSize ?? model.TrainingConfig.BatchSize;
        int usedSampleCount = miniBatches.Sum(b => b.BatchSize);
        int unusedSamples = samples.Count - usedSampleCount;
        int smallestBatch = miniBatches.Count > 0 ? miniBatches.Min(b => b.BatchSize) : 0;
        string batchingNote;
        if (unusedSamples > 0 && model.TrainingConfig.DropLast)
        {
            batchingNote =
                $" DropLast discarded the final partial batch ({unusedSamples} of " +
                $"{samples.Count} samples unused this epoch).";
            Log.Information(batchingNote);
        }
        else if (smallestBatch < batchSize)
        {
            //DropLast could not apply: the dataset never fills a full batch, so
            //the partial batch is kept rather than leaving nothing to train on.
            batchingNote =
                $" Dataset does not fill a full batch of {batchSize} " +
                $"({samples.Count} samples, smallest batch {smallestBatch}), " +
                "so the partial batch is kept.";
            Log.Information(batchingNote);
        }
        else
        {
            batchingNote = string.Empty;
        }
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
                $"{numBatches} mini-batches ({totalOuterBatches} outer batches)." +
                batchingNote;
        });

        //Cancellation is scoped to this run. Stop and Cancel signal the job's
        //control, and the loop observes it through this token; the linked source
        //keeps a relaunch (which gets a fresh control) unaffected by older runs.
        using var controlTokenSource =
            CancellationTokenSource.CreateLinkedTokenSource(control.Cancellation.Token);
        var controlToken = controlTokenSource.Token;

        // 1. Begin training. This notifies other endpoint services that the model is busy training and should not be used.
        model.BeginTraining();
        try
        {
            // 2. Training loop
            for (int epoch = startEpoch; epoch < totalEpochs; epoch++)
            {
                float epochLoss = 0f;
                int stepsCompleted = 0;
                // Single Random instance reused across the engine
                var rng = new Random();

                await UpdateJob(dbFactory, job.EntryId, job =>
                {
                    job.Status = TrainingJobStatus.Running;
                    job.CurrentEpoch = epoch + 1;
                    job.CurrentBatch = 0;
                    job.CurrentSubBatch = 0;
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
                        //Stop and cancel must break out of the batch loops, so
                        //each waits/polls the same dedicated token the Stop and
                        //Cancel endpoints signal.
                        await control.WaitIfPausedAsync(controlTokenSource.Token);

                        controlTokenSource.Token.ThrowIfCancellationRequested();

                        var currentSubBatch = shuffledSubBatches[batch];

                        for (int subBatch = 0; subBatch < currentSubBatch.Count(); subBatch++)
                        {
                            await control.WaitIfPausedAsync(controlTokenSource.Token);

                            controlTokenSource.Token.ThrowIfCancellationRequested();

                            var item = currentSubBatch[subBatch];
                            epochLoss += model.TrainStep(item.Inputs, item.Targets);
                            stepsCompleted++;

                            // Persist after every sub-batch (one TrainStep) so the
                            // frontend sees progress as fast as training actually moves.
                            // CurrentLoss is the running mean of per-step loss so it
                            // matches the scale of the epoch-end average.
                            var runningMean = epochLoss / stepsCompleted;
                            await UpdateJob(dbFactory, job.EntryId, job =>
                            {
                                job.CurrentBatch = batch + 1;
                                job.CurrentSubBatch = subBatch + 1;
                                job.CurrentLoss = runningMean;
                                job.Message =
                                    $"Epoch {epoch + 1}/{totalEpochs}, batch {batch + 1}/{shuffledSubBatches.Count}, " +
                                    $"sub-batch {subBatch + 1}/{currentSubBatch.Count()} - loss {runningMean:F6}";
                            });

                            // Correct parenthesis grouping for modulus
                            if ((subBatch + 1) % 4 == 0 || subBatch == 0)
                            {
                                Log.Information($"Epoch {epoch + 1}, Batch {batch + 1}, Sub-Batch {subBatch + 1} training loss: {epochLoss:F6}");
                            }
                        }
                        //It may be helpful to save temporary checkpoints here, and simply overwrite them. Possible name could be checkpoint-epoch-{epoch}-temp.bin
                        //This can help prevent loss of progress should a training run fail or be stopped.
                        //Get the transformer config name from the model entry
                        var transformerModelName = await GetModelNameFromId(dbFactory, modelEntry.EntryId);
                        var checkpointFilename = $"checkpoint-epoch-{epoch + 1}-temp.bin";
                        var checkpointDirname = $"checkpoints/{transformerModelName}";
                        Directory.CreateDirectory(checkpointDirname);

                        string checkpointFilepath = $"{checkpointDirname}/{checkpointFilename}";

                        await using (var stream = File.Create(checkpointFilepath))
                        {
                            model.SaveCheckpoint(stream, epoch, epochLoss);
                        }

                        //The temp checkpoint file is overwritten in place, so upsert a single
                        //DB record per file instead of inserting a new row after every batch.
                        var checkpointEntry = await UpsertCheckpointAsync(
                            db,
                            model.TransformerModelId,
                            $"checkpoints/{transformerModelName}/",
                            checkpointFilename,
                            epoch + 1,
                            epochLoss,
                            new FileInfo(checkpointFilepath).Length);

                        await UpdateJob(dbFactory, job.EntryId, job =>
                        {
                            job.CheckpointFilename = checkpointFilepath;
                            job.TrainingCheckpointId = checkpointEntry.EntryId;
                        });
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
                        //Pause and stop have to be honoured on this path too, or a
                        //small dataset would ignore both.
                        await control.WaitIfPausedAsync(controlTokenSource.Token);

                        controlTokenSource.Token.ThrowIfCancellationRequested();

                        var item = shuffledMiniBatches[batch];
                        epochLoss += model.TrainStep(item.Inputs, item.Targets);
                        stepsCompleted++;

                        // Persist after every batch so the frontend stays fresh here too
                        // (CurrentLoss = running mean, same scale as the epoch-end value).
                        var runningMean = epochLoss / stepsCompleted;
                        await UpdateJob(dbFactory, job.EntryId, job =>
                        {
                            job.CurrentBatch = batch + 1;
                            job.CurrentSubBatch = 1;
                            job.CurrentLoss = runningMean;
                            job.Message =
                                $"Epoch {epoch + 1}/{totalEpochs}, batch {batch + 1}/{numBatches} - loss {runningMean:F6}";
                        });

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
                    //TotalBatches is the outer-batch count in the >8 path and the
                    //mini-batch count in the small path - never the raw mini-batch
                    //count here, or the frontend renders e.g. "229 / 29".
                    job.CurrentBatch = job.TotalBatches;
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
                    Directory.CreateDirectory(checkpointDirname);

                    string checkpointFilepath = $"{checkpointDirname}/{checkpointFilename}";

                    await using (var stream = File.Create(checkpointFilepath))
                    {
                        model.SaveCheckpoint(stream, epoch, epochLoss);
                    }

                    //Filenames are deterministic, so upsert to guarantee one record per file
                    //(a resumed run can recreate the same epoch/loss filename).
                    var checkpointEntry = await UpsertCheckpointAsync(
                        db,
                        model.TransformerModelId,
                        $"checkpoints/{transformerModelName}/",
                        checkpointFilename,
                        epoch + 1,
                        epochLoss,
                        new FileInfo(checkpointFilepath).Length);

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

            //Every epoch ran to completion: record success. Stopped or cancelled
            //runs never reach this line because they throw out of the loops above.
            model.EndTraining();

            await UpdateJob(dbFactory, job.EntryId, job =>
            {
                job.Status = TrainingJobStatus.Completed;
                job.CurrentEpoch = totalEpochs;
                job.CurrentBatch = job.TotalBatches;
                job.Message = "Model training completed successfully.";
                job.DateCompleted = DateTime.UtcNow;
            });
        }
        catch (OperationCanceledException)
        {
            Log.Information(
                "Training job {JobId} was stopped or cancelled.",
                job.EntryId);

            //Cancel records its own terminal status; a plain stop is recorded as
            //Stopped. Either way a cancelled run must not be reported as completed.
            await UpdateJob(dbFactory, job.EntryId, job =>
            {
                if (job.Status == TrainingJobStatus.Cancelled)
                {
                    job.Message = "Training job cancelled.";
                    job.DateCompleted ??= DateTime.UtcNow;
                    return;
                }

                job.Status = TrainingJobStatus.Stopped;
                job.Message = "Training job stopped. It can be resumed from its latest checkpoint.";
                job.DateCompleted = DateTime.UtcNow;
            });
        }
        catch (Exception ex)
        {
            //Without this a faulting loop left the job reading as Running forever.
            Log.Error(ex, "Training job {JobId} failed.", job.EntryId);

            await UpdateJob(dbFactory, job.EntryId, job =>
            {
                job.Status = TrainingJobStatus.Failed;
                job.Message = "Model training failed.";
                job.Error = ex.Message;
                job.DateCompleted = DateTime.UtcNow;
            });
        }
        finally
        {
            //Training always ends, whatever the outcome, and the in-memory guard is
            //released so the job can later be resumed (rebuilt from its checkpoint).
            model.EndTraining();
            jobManager.Remove(job.EntryId);
        }
    }
    public static async Task RegisterJob(IDbContextFactory<AppDbContext> dbFactory, TrainingJobEntry jobEntry)
    {
        await using var db = await dbFactory.CreateDbContextAsync();

        await db.TrainingJobs.AddAsync(jobEntry);
        await db.SaveChangesAsync();
    }

    /// <summary>
    /// Ensures exactly one TrainingCheckpointEntry exists per checkpoint file:
    /// the record is looked up by (Filepath, Filename) and updated in place,
    /// or created when the file is seen for the first time.
    /// </summary>
    private static async Task<TrainingCheckpointEntry> UpsertCheckpointAsync(
        AppDbContext db,
        Guid transformerModelId,
        string filepath,
        string filename,
        int epoch,
        float loss,
        long fileSize)
    {
        var entry = await db.TrainingCheckpoints
            .FirstOrDefaultAsync(x => x.Filepath == filepath && x.Filename == filename);

        if (entry == null)
        {
            entry = new TrainingCheckpointEntry
            {
                TransformerModelId = transformerModelId,
                Filename = filename,
                Filepath = filepath
            };
            await db.TrainingCheckpoints.AddAsync(entry);
        }
        else
        {
            entry.TransformerModelId = transformerModelId;
        }

        entry.Epoch = epoch;
        entry.Loss = loss;
        entry.FileSize = fileSize;

        await db.SaveChangesAsync();
        return entry;
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