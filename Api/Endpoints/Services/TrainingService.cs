using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Serilog;
using SimpleTransformer.Api.ManagementEngine;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Config;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TrainingService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly ConfigManager _configManager;
        private readonly ITokenizer _tokenizer;
        private ModelManager _modelManager;
        private readonly TrainingJobManager _jobManager;

        public TrainingService(
            ITokenizer tokenizer, 
            ConfigManager configManager, 
            IDbContextFactory<AppDbContext> dbFactory,  
            ModelManager modelManager,
            TrainingJobManager jobManager)
        {
            _tokenizer = tokenizer;
            _configManager = configManager;
            _dbFactory = dbFactory;
            _modelManager = modelManager;
            _jobManager = jobManager;
        }

        public async Task<ApiResponse<TrainingResponse>> CreateJob(TrainingRequest req)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var modelEntry = db.TransformerModels.FirstOrDefault(x => x.EntryId == req.TransformerModelId);

            if(modelEntry == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //If the model is not loaded, training can't be done
            if (!modelEntry.IsLoaded)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not loaded.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            var model = await _modelManager.LoadModelAsync(modelEntry.EntryId);

            if (model == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (string.IsNullOrEmpty(req.InputText))
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Source text must not be empty or null.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }
            var job = new TrainingJobEntry
            {
                Name = $"Training job {DateTime.UtcNow}",
                EntryId = Guid.NewGuid(),                
                Status = TrainingJobStatus.Pending,
                TrainingConfigId = modelEntry.TrainingConfigId,
                TransformerConfigId = modelEntry.TransformerConfigId,
                TransformerModelId = modelEntry.EntryId,
                Message = "Training job created.",
                DateUpdated = DateTime.UtcNow,
                InputText = req.InputText,
                PreviousCheckpointId = req.PreviousCheckpointId,
              
                //Vocabulary related
                VocabularyId = req.VocabularyId,
            };
            

            //Register the job.
            await RegisterJob(job);

            return new ApiResponse<TrainingResponse>
            {
                Message = "Training job created successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TrainingResponse
                {
                    Message = "Training job created successfully.",
                    Status = InteractionStatus.Success,
                }
            };
        }

        public async Task<ApiResponse<TrainingResponse>> CreateJobFromFile(
            TrainingFileRequest req)
        {
            if (req.TextFile == null || req.TextFile.Length == 0)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Training file must not be empty.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            await using var db = await _dbFactory.CreateDbContextAsync();

            var modelEntry = await db.TransformerModels
                .FirstOrDefaultAsync(x => x.EntryId == req.TransformerModelId);

            if (modelEntry == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (!modelEntry.IsLoaded)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not loaded.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            var model = await _modelManager.LoadModelAsync(modelEntry.EntryId);

            if (model == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            var jobId = Guid.NewGuid();

            var extension = Path.GetExtension(req.TextFile.FileName);
            var fileName = $"training-data{extension}";

            var jobDirectory = Path.Combine(
                "training-data",
                jobId.ToString());

            Directory.CreateDirectory(jobDirectory);

            var filePath = Path.Combine(
                jobDirectory,
                fileName);

            await using (var fileStream = File.Create(filePath))
            {
                await req.TextFile.CopyToAsync(fileStream);
            }

            var job = new TrainingJobEntry
            {
                EntryId = jobId,
                Name = $"Training job {DateTime.UtcNow}",
                Status = TrainingJobStatus.Pending,

                TransformerConfigId = modelEntry.TransformerConfigId,
                TrainingConfigId = modelEntry.TrainingConfigId,
                TransformerModelId = modelEntry.EntryId,
                VocabularyId = req.VocabularyId,

                InputText = null,
                InputFilePath = filePath,

                PreviousCheckpointId = req.PreviousCheckpointId,

                Message = "Training job created.",
                DateUpdated = DateTime.UtcNow
            };

            await db.TrainingJobs.AddAsync(job);
            await db.SaveChangesAsync();

            return new ApiResponse<TrainingResponse>
            {
                Message = "Training job created successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> PauseTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (TrainingJobExtensions.IsTerminalStatus(job.Status))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = $"Training job is {job.Status} and cannot be paused.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            if (!_jobManager.TryGet(jobId, out var control) ||
                control is null ||
                !control.HasLiveLoop)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running in this process.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control.Pause();

            job.Status = TrainingJobStatus.Paused;
            job.Message = "Training job paused.";
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job paused successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> ResumeTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //Fast path: the loop is still alive in this process (an ordinary pause),
            //so resuming is just releasing the pause latch.
            if (_jobManager.TryGet(jobId, out var control) &&
                control is { IsStopped: false } &&
                !control.Cancellation.IsCancellationRequested &&
                control.HasLiveLoop)
            {
                control.Resume();

                job.Status = TrainingJobStatus.Running;
                job.Message = "Training job resumed.";
                job.DateUpdated = DateTime.UtcNow;

                await db.SaveChangesAsync();

                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job resumed successfully.",
                    Status = ResponseStatus.Success,
                    StatusCode = 200
                };
            }

            //No loop survives a process restart (nor a stop or cancel), so the run
            //has to be rebuilt: that is exactly what starting the job does, and it
            //reloads the latest checkpoint, so both paths share one implementation.
            if (TrainingJobExtensions.IsTerminalStatus(job.Status))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = $"Training job is {job.Status} and cannot be resumed. Use Start to run it again from its latest checkpoint.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            return await StartTrainingJob(jobId);
        }

        public async Task<ApiResponse<TrainingProgressResponse>> StopTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //Stop in memory only stops a live loop. When the backend restarted (or
            //the loop already wound down) there is nothing to signal, so the row is
            //still marked Stopped instead of 409ing a no-op.
            if (_jobManager.TryGet(jobId, out var control) &&
                control is { } &&
                control.HasLiveLoop)
            {
                control.Stop();
            }

            job.Status = TrainingJobStatus.Stopped;
            job.Message = "Training job stopped. It can be resumed from its latest checkpoint.";
            job.DateCompleted = DateTime.UtcNow;
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job stopped.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> CancelTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };

            if (_jobManager.TryGet(jobId, out var control) && control != null)
            {
                control.Stop();
                _jobManager.Remove(jobId);
            }

            //Stop in memory only stops the loop; the status in the database has
            //to follow or the job keeps reading as running forever.
            job.Status = TrainingJobStatus.Cancelled;
            job.Message = "Training job cancelled.";
            job.DateCompleted = DateTime.UtcNow;
            job.DateUpdated = DateTime.UtcNow;
            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job cancelled successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> StartTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //Don't start a job whose loop is still alive in this process. A stale
            //database row (e.g. Running left behind by a restart) must not block the
            //relaunch because the reconciliation moves such rows to Paused.
            if (_jobManager.TryGet(job.EntryId, out var existingControl) &&
                existingControl is { IsStopped: false } &&
                !existingControl.Cancellation.IsCancellationRequested &&
                existingControl.HasLiveLoop)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is already running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            //Preparing a run takes tens of seconds (model load + checkpoint read),
            //and no loop is registered until it finishes. Without this reservation a
            //duplicate click would start a second preparation and relaunch the job.
            if (!_jobManager.TryBeginLaunch(job.EntryId))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is already starting up. Wait for it to finish preparing.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            try
            {
                return await LaunchTrainingRunAsync(db, job);
            }
            finally
            {
                _jobManager.EndLaunch(job.EntryId);
            }
        }

        /// <summary>
        /// Prepares (model + optional checkpoint) and launches a training run for a
        /// job that has already been checked and reserved by <see cref="StartTrainingJob"/>.
        /// </summary>
        private async Task<ApiResponse<TrainingProgressResponse>> LaunchTrainingRunAsync(
            AppDbContext db,
            TrainingJobEntry job)
        {
            // Resolve the exact model this job targets (by id, not config id —
            // multiple models can share a transformer config).
            var modelEntry = await db.TransformerModels
                .FirstOrDefaultAsync(x => x.EntryId == job.TransformerModelId);

            if (modelEntry == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Model definition not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //Tell the UI what is happening before the slow part: loading the model
            //and reading a checkpoint takes tens of seconds. The status is only
            //flipped to Started once the run is genuinely about to launch, so a
            //failed preparation leaves the job's previous status intact.
            await UpdateJob(job.EntryId, j =>
            {
                j.DateStarted ??= DateTime.UtcNow;
                j.Message = "Preparing training run (loading the model and checkpoint)...";
                j.Error = string.Empty;
                j.DateUpdated = DateTime.UtcNow;
            });

            if (!modelEntry.IsLoaded && _modelManager.LoadedModelId != modelEntry.EntryId)
            {
                try
                {
                    await _modelManager.LoadModelAsync(modelEntry.EntryId);
                }
                catch (Exception ex)
                {
                    return new ApiResponse<TrainingProgressResponse>
                    {
                        Message = $"Failed to load model {modelEntry.Name}: {ex.Message}",
                        Status = ResponseStatus.Failure,
                        StatusCode = 500
                    };
                }

                await using var loadDb = await _dbFactory.CreateDbContextAsync();
                var loadedEntry = await loadDb.TransformerModels
                    .FirstOrDefaultAsync(x => x.EntryId == modelEntry.EntryId);

                if (loadedEntry != null)
                {
                    var stale = await loadDb.TransformerModels
                        .Where(x => x.IsLoaded && x.EntryId != loadedEntry.EntryId)
                        .ToListAsync();

                    foreach (var entry in stale)
                    {
                        entry.IsLoaded = false;
                    }

                    loadedEntry.IsLoaded = true;
                    loadedEntry.DateUpdated = DateTime.UtcNow;
                    await loadDb.SaveChangesAsync();
                }

                //Refresh the tracked row above so the guards below see the load.
                await db.Entry(modelEntry).ReloadAsync();
            }

            if (!modelEntry.IsLoaded && _modelManager.LoadedModelId != modelEntry.EntryId)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Model is not loaded.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            var model = await _modelManager.LoadModelAsync(modelEntry.EntryId);

            if (model == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Could not load model.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //One model instance cannot drive two runs: the tensors and workspace are
            //shared state, so concurrent TrainStep calls would corrupt each other.
            //After a stop the previous loop may still be winding down for a moment.
            if (model.IsTraining)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = $"Model {modelEntry.Name} is still tied up with another training run. If you just stopped one, wait a few seconds and start again.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            // Validate the training source.
            if (string.IsNullOrWhiteSpace(job.InputText) &&
                string.IsNullOrWhiteSpace(job.InputFilePath))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job has no training source.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            if (!string.IsNullOrWhiteSpace(job.InputFilePath) &&
                !File.Exists(job.InputFilePath))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training source file could not be found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            // Load the training configuration.
            var configEntry = await db.TrainingConfigs
                .FirstOrDefaultAsync(x => x.EntryId == job.TrainingConfigId);

            if (configEntry == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training config not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            var config = configEntry.Config;

            int startEpoch = 0;

            //Resume shares the launch with Start. A guard left over from a stopped
            //run carries a cancelled token, so the new run gets a clean one; a live
            //loop is never replaced because two loops would fight over the model.
            if (!_jobManager.TryGetLaunchControl(job.EntryId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is already running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            //Resume from the job's known checkpoint, falling back to the newest
            //surviving checkpoint file when the old pointer no longer resolves.
            var resumeCheckpoint =
                await TrainingJobExtensions.ResolveResumeCheckpointAsync(
                    db, job.EntryId, job.TransformerModelId);

            if (resumeCheckpoint != null)
            {
                var checkpointPath =
                    Path.Combine(resumeCheckpoint.Filepath, resumeCheckpoint.Filename);

                await UpdateJob(job.EntryId, job =>
                {
                    job.Message = "Loading previous checkpoint...";
                    job.PreviousCheckpointId = resumeCheckpoint.EntryId;
                    job.DateUpdated = DateTime.UtcNow;
                });

                Log.Information(
                    "Loading training state from checkpoint: {Path}",
                    checkpointPath);

                try
                {
                    await using var stream = File.OpenRead(checkpointPath);

                    var (savedEpoch, savedLoss) =
                        TransformerModel.LoadCheckpoint(stream, model);

                    startEpoch = savedEpoch + 1;

                    await UpdateJob(job.EntryId, job =>
                    {
                        job.Message =
                            $"Resuming training from epoch {startEpoch}.";
                        job.CurrentEpoch = startEpoch;
                        job.CurrentLoss = savedLoss;
                        job.DateUpdated = DateTime.UtcNow;
                    });

                    Log.Information(
                        "Resuming training from Epoch {Epoch} (Last Loss: {Loss:F6})",
                        startEpoch,
                        savedLoss);
                }
                catch (Exception ex)
                {
                    return new ApiResponse<TrainingProgressResponse>
                    {
                        Message = $"Could not load checkpoint {resumeCheckpoint.Filename}: {ex.Message}",
                        Status = ResponseStatus.Failure,
                        StatusCode = 500
                    };
                }
            }
            else if (job.PreviousCheckpointId.HasValue)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Previous training checkpoint could not be found. Pick a different checkpoint or start the job from scratch.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //A duplicate start click can otherwise orphan the current control while
            //a live loop is still out there, so a real second run always wins with
            //a 409 and the just-created spare guard is dropped again.
            if (_jobManager.TryGet(job.EntryId, out var liveControl) &&
                liveControl is { IsStopped: false } &&
                !liveControl.Cancellation.IsCancellationRequested &&
                liveControl.HasLiveLoop &&
                !ReferenceEquals(liveControl, control))
            {
                _jobManager.Remove(job.EntryId);

                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is already running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            await UpdateJob(job.EntryId, job =>
            {
                job.Status = TrainingJobStatus.Started;
                job.DateStarted ??= DateTime.UtcNow;
                job.Message = "Training job starting...";
                job.DateUpdated = DateTime.UtcNow;
            });

            //The loop runs detached so the POST returns immediately; unhandled
            //faults are logged and the guard is released through the loop itself.
            var trainingTask = Task.Run(() =>
                TrainingJobExtensions.RunTrainingLoop(
                    job: job,
                    model: model,
                    startEpoch: startEpoch,
                    config: config,
                    control: control,
                    modelEntry: modelEntry,
                    dbFactory: _dbFactory,
                    tokenizer: _tokenizer,
                    jobManager: _jobManager));

            //Register the loop immediately: the pause/resume/start guards all rely on
            //control.RunningTask, and awaiting it here would block this POST for the
            //whole training run.
            control.RunningTask = trainingTask;

            //Fire-and-forget fault logging only (the loop records its own errors).
            _ = trainingTask.ContinueWith(
                task =>
                {
                    if (task.IsFaulted)
                    {
                        Log.Error(
                            task.Exception,
                            "Training job {JobId} faulted.",
                            job.EntryId);
                        _jobManager.Remove(job.EntryId);
                    }
                },
                TaskScheduler.Default);

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job started successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> DeleteTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };

            //A deleted job must not keep a live loop (or a pause latch) behind,
            //or its guard would leak into any job created with the same id.
            if (_jobManager.TryGet(jobId, out var control) && control != null)
            {
                control.Stop();
                _jobManager.Remove(jobId);
            }

            db.TrainingJobs.Remove(job);
            await db.SaveChangesAsync();
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job deleted successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> ResetTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };

            if (_jobManager.TryGet(jobId, out var control) && control != null)
            {
                control.Stop();
                _jobManager.Remove(jobId);
            }

            //Reset returns the row to a pristine Pending job. Without the save and
            //the cleared counters the frontend kept showing the old progress.
            job.Status = TrainingJobStatus.Pending;
            job.Message = "Training job reset.";
            job.Error = string.Empty;

            job.CurrentEpoch = 0;
            job.EpochsCompleted = 0;
            job.TotalEpochs = 0;

            job.CurrentBatch = 0;
            job.BatchesCompleted = 0;
            job.TotalBatches = 0;

            job.CurrentSubBatch = 0;
            job.SubBatchesCompleted = 0;
            job.TotalSubBatches = 0;

            job.CurrentLoss = 0;
            job.CheckpointFilename = string.Empty;
            job.TrainingCheckpointId = null;
            job.PreviousCheckpointId = null;

            job.DateStarted = null;
            job.DateCompleted = null;
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job reset successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

    



        private async Task RegisterJob(TrainingJobEntry jobEntry)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            await db.TrainingJobs.AddAsync(jobEntry);
            await db.SaveChangesAsync();
        }

        private async Task UpdateJob(
            Guid jobId,
            Action<TrainingJobEntry> update)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var jobEntry = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (jobEntry == null)
                return;

            update(jobEntry);

            jobEntry.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();
        }

        public async Task<ApiResponse<TrainingProgressResponse>> GetTrainingProgress(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
            {
                return await Task.FromResult(new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                });
            }

            //Resolve the model name for the active response payload.
            var model = await db.TransformerModels
                .AsNoTracking()
                .FirstOrDefaultAsync(x => x.EntryId == job.TransformerModelId);

            return await Task.FromResult(new ApiResponse<TrainingProgressResponse>
            {
                Message = job.Message,
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TrainingProgressResponse
                {
                    JobId = job.EntryId.ToString(),
                    Name =  job.Name,
                    TransformerModelId = job.TransformerModelId.ToString(),
                    TransformerModelName = model?.Name ?? string.Empty,
                    Status = job.Status,
                    CurrentEpoch = job.CurrentEpoch,
                    TotalEpochs = job.TotalEpochs,
                    CurrentBatch = job.CurrentBatch,
                    TotalBatches = job.TotalBatches,
                    CurrentLoss = job.CurrentLoss,
                    Loss = job.CurrentLoss,
                    CurrentSubBatch = job.CurrentSubBatch,
                    NumSubBatches = job.TotalSubBatches,
                    Message = job.Message,
                    Checkpoint = job.CheckpointFilename,
                    StartedAt = job.DateStarted,
                    CompletedAt = job.DateCompleted,
                    LastUpdatedAt = job.DateUpdated,
                    Error = job.Error ?? "none"
                }
            });
        }

        //Checkpoints for the frontend "previous checkpoint" list box. Only
        //entries whose file still exists on disk are returned, newest first.
        public async Task<ApiResponse<List<TrainingCheckpointEntry>>> GetCheckpoints()
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var entries = await db.TrainingCheckpoints
                .AsNoTracking()
                .OrderByDescending(x => x.DateCreated)
                .ToListAsync();

            entries = entries
                .Where(x => File.Exists(Path.Combine(x.Filepath, x.Filename)))
                .ToList();

            return new ApiResponse<List<TrainingCheckpointEntry>>
            {
                Message = "Checkpoints found.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = entries
            };
        }

        public async Task<ApiResponse<List<TrainingProgressResponse>>> GetTrainingJobs()
        {
            await using var _db = await _dbFactory.CreateDbContextAsync();
            var modelNames = _db.TransformerModels
                .AsNoTracking()
                .ToDictionary(x => x.EntryId, x => x.Name);

            return new ApiResponse<List<TrainingProgressResponse>>
            {
                Message = "Training jobs found.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = _db.TrainingJobs
                    .ToList()
                    .Select(x => new TrainingProgressResponse
                    {
                        JobId = x.EntryId.ToString(),
                        Name = x.Name,
                        TransformerModelId = x.TransformerModelId.ToString(),
                        TransformerModelName = modelNames.GetValueOrDefault(x.TransformerModelId, string.Empty),
                        Status = x.Status,
                    CurrentEpoch = x.CurrentEpoch,
                    TotalEpochs = x.TotalEpochs,
                    CurrentBatch = x.CurrentBatch,
                    TotalBatches = x.TotalBatches,
                    CurrentLoss = x.CurrentLoss,
                    Loss = x.CurrentLoss,
                    CurrentSubBatch = x.CurrentSubBatch,
                    NumSubBatches = x.TotalSubBatches,
                    Message = x.Message,
                    Checkpoint = x.CheckpointFilename,
                    StartedAt = x.DateStarted,
                    CompletedAt = x.DateCompleted,
                    LastUpdatedAt = x.DateUpdated,
                    Error = x.Error ?? "none"
                }).ToList()
            };
        }

        //Debug purposes only. Write the checkpoint data to a json file. Is there a different way to do this without creating a massive file?
        private async Task WriteCheckpointDataToFile(IReadOnlyList<TrainableParameterCheckpoint> checkpointParameters)
        {
            string json = JsonSerializer.Serialize(checkpointParameters, new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
            await File.WriteAllTextAsync("checkpoint-debugdata.json", json);
        }


    }
}