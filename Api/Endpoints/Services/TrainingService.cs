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
                VocabularyId = req.VocabularyId,

                InputText = null,
                InputFilePath = filePath,

                PreviousCheckpointId = req.PreviousCheckpointId,

                Message = "Training job created.",
                DateUpdated = DateTime.UtcNow
            };

            await db.TrainingJobs.AddAsync(job);
            await db.SaveChangesAsync();

            var control = _jobManager.GetOrCreate(job.EntryId);

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

            if (!_jobManager.TryGet(jobId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control!.Pause();

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

            if (!_jobManager.TryGet(jobId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control!.Resume();

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

            if (!_jobManager.TryGet(jobId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control!.Stop();

            job.Status = TrainingJobStatus.Stopped;
            job.Message = "Training job stopping...";
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job stop requested.",
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

            job.Status = TrainingJobStatus.Cancelled;
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

            // Don't start a job that is already running.
            if (job.Status == TrainingJobStatus.Running ||
                job.Status == TrainingJobStatus.Started)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is already running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            // A model must be associated with the job.
            var modelEntry = await db.TransformerModels
                .FirstOrDefaultAsync(x =>
                    x.TransformerConfigId == job.TransformerConfigId);

            if (modelEntry == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Model definition not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (!modelEntry.IsLoaded)
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

            var control = _jobManager.GetOrCreate(job.EntryId);

            // Resume from the job's previous checkpoint.
            if (job.PreviousCheckpointId.HasValue)
            {
                var checkpoint = await db.TrainingCheckpoints
                    .FirstOrDefaultAsync(x =>
                        x.EntryId == job.PreviousCheckpointId.Value);

                if (checkpoint == null)
                {
                    return new ApiResponse<TrainingProgressResponse>
                    {
                        Message = "Previous training checkpoint not found.",
                        Status = ResponseStatus.Failure,
                        StatusCode = 404
                    };
                }

                if (!File.Exists(
                        Path.Combine(checkpoint.Filepath, checkpoint.Filename)))
                {
                    return new ApiResponse<TrainingProgressResponse>
                    {
                        Message = "Previous training checkpoint file could not be found.",
                        Status = ResponseStatus.Failure,
                        StatusCode = 404
                    };
                }

                var checkpointPath =
                    Path.Combine(checkpoint.Filepath, checkpoint.Filename);

                await UpdateJob(job.EntryId, job =>
                {
                    job.Message = "Loading previous checkpoint...";
                    job.PreviousCheckpointId = checkpoint.EntryId;
                });

                Log.Information(
                    "Loading training state from checkpoint: {Path}",
                    checkpointPath);

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
                });

                Log.Information(
                    "Resuming training from Epoch {Epoch} (Last Loss: {Loss:F6})",
                    startEpoch,
                    savedLoss);
            }

            await UpdateJob(job.EntryId, job =>
            {
                job.Status = TrainingJobStatus.Started;
                job.DateStarted ??= DateTime.UtcNow;
                job.Message = "Training job starting...";
            });

            await TrainingJobExtensions.RunTrainingLoop(
                job: job,
                model: model,
                startEpoch: startEpoch,
                config: config,
                control: control,
                modelEntry: modelEntry,
                dbFactory: _dbFactory,
                tokenizer: _tokenizer,
                jobManager: _jobManager
            );

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

            job.Status = TrainingJobStatus.Pending;
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

            return await Task.FromResult(new ApiResponse<TrainingProgressResponse>
            {
                Message = job.Message,
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TrainingProgressResponse
                {
                    JobId = job.EntryId.ToString(),
                    Name =  job.Name,
                    Status = job.Status,
                    CurrentEpoch = job.CurrentEpoch,
                    TotalEpochs = job.TotalEpochs,
                    CurrentBatch = job.CurrentBatch,
                    TotalBatches = job.TotalBatches,
                    CurrentLoss = job.CurrentLoss,
                    Checkpoint = job.CheckpointFilename,
                    StartedAt = job.DateStarted,
                    CompletedAt = job.DateCompleted,
                    LastUpdatedAt = job.DateUpdated,
                    Error = job.Error ?? "none"
                }
            });
        }

        public async Task<ApiResponse<List<TrainingProgressResponse>>> GetTrainingJobs()
        {
            await using var _db = await _dbFactory.CreateDbContextAsync();
            return new ApiResponse<List<TrainingProgressResponse>>
            {
                Message = "Training jobs found.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = _db.TrainingJobs.Select(x => new TrainingProgressResponse
                {
                    JobId = x.EntryId.ToString(),
                    Name = x.Name,
                    Status = x.Status,
                    CurrentEpoch = x.CurrentEpoch,
                    TotalEpochs = x.TotalEpochs,
                    CurrentBatch = x.CurrentBatch,
                    TotalBatches = x.TotalBatches,
                    CurrentLoss = x.CurrentLoss,
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