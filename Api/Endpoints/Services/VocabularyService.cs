using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Api.Endpoints.Controllers;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Config;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class VocabularyService
    {
        private readonly ITokenizer _tokenizer;
        private readonly Vocabulary _vocabulary;
        private readonly ConfigManager _configManager;
        private readonly IVocabularyCompiler _vocabularyCompiler;
        private readonly IDbContextFactory<AppDbContext> _dbFactory;

        public VocabularyService(ITokenizer tokenizer, IVocabularyCompiler vocabularyCompiler, Vocabulary vocabulary, IDbContextFactory<AppDbContext> dbFactory, ConfigManager configManager)
        {
            _tokenizer = tokenizer;
            _vocabularyCompiler = vocabularyCompiler;
            _vocabulary = vocabulary;
            _configManager = configManager;
            _dbFactory = dbFactory;
        }

        public async Task<ApiResponse<VocabularyLoaderResponse>> LoadFromFile(LoadVocabularyRequest req)
        {
            //Load vocabulary from a json file.
            if(string.IsNullOrEmpty(req.File)) throw new ArgumentNullException(nameof(req.File));

            var loader = new JsonVocabularyLoader();

            loader.LoadFromFile(req.File);

            var response = new VocabularyLoaderResponse
            {
                Status = InteractionStatus.Success,
            };

            return new ApiResponse<VocabularyLoaderResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = response
            };

        }

        public async Task<ApiResponse<AvailableVocabulariesResponse>> GetAvailableVocabulariesAsync()
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var vocabularies = db.Vocabularies.ToList();
            return new ApiResponse<AvailableVocabulariesResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new AvailableVocabulariesResponse
                {
                    Vocabularies = vocabularies
                }
            };
        }

        public async Task<ApiResponse<VocabularyCompilationResponse>> Compile(CompileVocabularyRequest req)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            //Store in vocabularies\compiled
            var sourceDir = _configManager.GetValue("vocabulary_Source_Directory", "Paths");
            var compiledFolder = _configManager.GetValue("vocabulary_Compiled_Directory", "Paths");
            // Size: explicit request wins, otherwise the configured medium default.
            int defaultSize = int.TryParse(
                _configManager.GetValue("num_tokens_medium", "Vocabulary"), out var parsedDefault)
                ? parsedDefault
                : 10000;

            if (req.Files == null || req.Files.Count == 0)
            {
                return new ApiResponse<VocabularyCompilationResponse>
                {
                    Message = "No files were provided.",
                    Status = ResponseStatus.Error,
                    StatusCode = 500,
                };
            }

            int vocabSize = req.VocabSize ?? defaultSize;
            if (vocabSize < 500 || vocabSize > 100000)
            {
                return CompileError(
                    $"Vocabulary size must be between 500 and 100000 (requested {req.VocabSize}).");
            }

            // Tokenizer: explicit request wins, otherwise the server default.
            TokenizerType tokenizerType = _tokenizer.Type;
            if (!string.IsNullOrWhiteSpace(req.TokenizerType))
            {
                if (!Enum.TryParse<TokenizerType>(req.TokenizerType, ignoreCase: true, out var requested))
                {
                    return CompileError(
                        $"Unknown tokenizer type '{req.TokenizerType}'. Expected WordLevel, Bpe or SentencePiece.");
                }

                tokenizerType = requested;
            }

            IVocabularyCompiler compiler = tokenizerType switch
            {
                TokenizerType.WordLevel => new WordLevelVocabularyCompiler(),
                TokenizerType.Bpe => new BpeVocabularyCompiler(),
                TokenizerType.SentencePiece => new SentencePieceVocabularyCompiler(),
                _ => _vocabularyCompiler
            };

            string name = string.IsNullOrWhiteSpace(req.Name)
                ? $"vocabulary-{tokenizerType.ToString().ToLowerInvariant()}-{vocabSize}-{DateTime.UtcNow:yyyyMMddHHmmss}"
                : req.Name.Trim();

            if (await db.Vocabularies.AnyAsync(x => x.Name == name))
            {
                return CompileError($"A vocabulary named '{name}' already exists. Choose a different name.", 409);
            }

            if (req.Files.Count == 1)
            {
                VocabularyCompilationResult singleResult;
                try
                {
                    singleResult = compiler.BuildFromRawTextFile(sourceDir, req.Files[0], vocabSize);
                }
                catch (FileNotFoundException ex)
                {
                    return CompileError(ex.Message, 404);
                }
                catch (ArgumentException ex)
                {
                    return CompileError(ex.Message, 400);
                }

                return await StoreCompilationAsync(
                    db, compiledFolder, name, tokenizerType, vocabSize, req.Files, singleResult);
            }

            if (req.Files.Count > 1)
            {
                VocabularyCompilationResult multiResult;
                try
                {
                    multiResult = compiler.BuildFromRawTextFiles(sourceDir, req.Files, vocabSize);
                }
                catch (FileNotFoundException ex)
                {
                    return CompileError(ex.Message, 404);
                }
                catch (ArgumentException ex)
                {
                    return CompileError(ex.Message, 400);
                }

                return await StoreCompilationAsync(
                    db, compiledFolder, name, tokenizerType, vocabSize, req.Files, multiResult);
            }

            return CompileError("No files were provided.");
        }

        /// <summary>
        /// Writes vocabulary.json under {compiled}/{tokenizer}/{count}/{name}/,
        /// records the exact location on the entry, and returns stats.
        /// </summary>
        private static async Task<ApiResponse<VocabularyCompilationResponse>> StoreCompilationAsync(
            AppDbContext db,
            string compiledFolder,
            string name,
            TokenizerType tokenizerType,
            int requestedSize,
            List<string> files,
            VocabularyCompilationResult result)
        {
            var vocabJson = JsonSerializer.Serialize(
                result.Vocabulary.TokenToId,
                new JsonSerializerOptions
                {
                    WriteIndented = true,
                    Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                });

            // Include the entry name so two compiles at the same size cannot
            // collide on disk; Filepath points at the real directory.
            var dest = Path.Combine(
                compiledFolder,
                tokenizerType.ToString().ToLowerInvariant(),
                result.Vocabulary.Count.ToString(),
                name);
            if (!Directory.Exists(dest)) Directory.CreateDirectory(dest);
            await File.WriteAllTextAsync(Path.Combine(dest, "vocabulary.json"), vocabJson);

            var vocab = new VocabularyEntry
            {
                Name = name,
                TokenizerType = tokenizerType,
                NumTokens = result.Vocabulary.Count,
                Filename = "vocabulary.json",
                Filepath = dest,
                DateCreated = DateTime.UtcNow,
                //Provenance so the details tab can show how this vocabulary was
                //produced without re-reading the source files (BPE and SentencePiece
                //leave TypesSeen/Coverage at their "not measured" defaults).
                SourceFileNames = string.Join(", ", files.Select(Path.GetFileName)),
                RequestedSize = requestedSize,
                TypesSeen = result.TypesSeen,
                Coverage = result.Coverage
            };
            await db.Vocabularies.AddAsync(vocab);
            await db.SaveChangesAsync();

            var samples = result.Vocabulary.IdToToken
                .OrderBy(kvp => kvp.Key)
                .Take(20)
                .Select(kvp => kvp.Value)
                .ToList();

            return new ApiResponse<VocabularyCompilationResponse>
            {
                Message = result.TokenOccurrences > 0
                    ? $"Vocabulary '{name}' compiled: {result.Vocabulary.Count} tokens " +
                      $"(requested {requestedSize}) from {files.Count} file(s); " +
                      $"coverage {result.Coverage:P1} over {result.TokenOccurrences} occurrences " +
                      $"and {result.TypesSeen} distinct types."
                    : $"Vocabulary '{name}' compiled: {result.Vocabulary.Count} tokens " +
                      $"(requested {requestedSize}) from {files.Count} file(s). " +
                      $"{tokenizerType} does not report coverage statistics.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new VocabularyCompilationResponse
                {
                    Vocabulary = result.Vocabulary,
                    RequestedVocabSize = requestedSize,
                    ActualVocabSize = result.Vocabulary.Count,
                    TokenizerType = tokenizerType.ToString(),
                    TypesSeen = result.TypesSeen,
                    Coverage = result.Coverage,
                    SampleTokens = samples
                }
            };
        }

        private static ApiResponse<VocabularyCompilationResponse> CompileError(
            string message, int statusCode = 400)
        {
            return new ApiResponse<VocabularyCompilationResponse>
            {
                Message = message,
                Status = statusCode == 409 ? ResponseStatus.Failure : ResponseStatus.Error,
                StatusCode = statusCode,
                Data = null
            };
        }

        /// <summary>
        /// Vocabulary details for a model: what the process actually tokenises with,
        /// the database row the model is pinned to, and every mismatch found between
        /// the transformer config, the artifact on disk and the live vocabulary.
        /// Pass null to use the model the server has loaded.
        /// </summary>
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetActiveVocabularyProperties(Guid? modelId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var data = new VocabularyPropertiesResponse
            {
                VocabSize = _vocabulary.Count,
                TokenizerType = _tokenizer.Type.ToString(),
                TokenizerSource = "vocabulary.json loaded at startup",
                UnknownToken = TokenIdOr(SpecialTokens.Unknown, "<unk>"),
                PaddingToken = TokenIdOr(SpecialTokens.Pad, "<pad>"),
                MaskToken = TokenIdOr(SpecialTokens.Mask, "<mask>"),
                BosToken = TokenIdOr(SpecialTokens.BeginningOfSequence, "<bos>"),
                EosToken = TokenIdOr(SpecialTokens.EndOfSequence, "<eos>")
            };

            var model = modelId.HasValue
                ? await db.TransformerModels.AsNoTracking()
                    .FirstOrDefaultAsync(x => x.EntryId == modelId.Value)
                : await db.TransformerModels.AsNoTracking()
                    .FirstOrDefaultAsync(x => x.IsLoaded);

            if (modelId.HasValue && model == null)
            {
                data.Issues.Add($"Model {modelId.Value} was not found.");
                return ActiveResponse(data);
            }

            if (model == null)
            {
                data.Issues.Add("No model is loaded, so there is no pinned vocabulary to report.");
                return ActiveResponse(data);
            }

            data.ModelId = model.EntryId;
            data.ModelName = model.Name;

            var config = await db.TransformerConfigs.AsNoTracking()
                .FirstOrDefaultAsync(x => x.EntryId == model.TransformerConfigId);
            data.ModelVocabSize = config?.Config.VocabSize;

            if (!model.VocabularyId.HasValue)
            {
                data.Issues.Add(
                    $"Model '{model.Name}' has no vocabulary pinned yet, so it tokenises with the server's " +
                    "startup vocabulary. Training it once pins the vocabulary that matched.");
            }
            else
            {
                var entry = await db.Vocabularies.AsNoTracking()
                    .FirstOrDefaultAsync(x => x.EntryId == model.VocabularyId.Value);

                if (entry == null)
                {
                    data.Issues.Add(
                        $"Model '{model.Name}' references vocabulary {model.VocabularyId.Value}, which no " +
                        "longer exists in the database.");
                }
                else
                {
                    data.Vocabulary = VocabularyEntryInfo.FromEntry(entry);
                    data.VocabularyPath = VocabularyArtifacts.ResolvePath(entry);
                    FillArtifactChecks(data, model.Name, entry);

                    if (entry.TokenizerType != _tokenizer.Type)
                    {
                        data.Issues.Add(
                            $"Vocabulary '{entry.Name}' was compiled with {entry.TokenizerType} but the " +
                            $"server is running {_tokenizer.Type}, so encoded ids will differ.");
                    }
                }
            }

            //Only when the live vocabulary is not the model's: otherwise a VocabSize
            //difference was already reported against the artifact itself.
            if (!data.LiveVocabularyMatchesModel
                && data.ModelVocabSize.HasValue
                && data.ModelVocabSize.Value != _vocabulary.Count)
            {
                data.Issues.Add(
                    $"Model '{model.Name}' expects {data.ModelVocabSize.Value} tokens but the live " +
                    $"tokenizer has {_vocabulary.Count}.");
            }

            return ActiveResponse(data);
        }

        /// <summary>
        /// Artifact level checks for a pinned vocabulary: the file readable, the
        /// database metadata agreeing with it, the config VocabSize matching it, and
        /// the live vocabulary holding the same token to id map.
        /// </summary>
        private void FillArtifactChecks(
            VocabularyPropertiesResponse data,
            string modelName,
            VocabularyEntry entry)
        {
            var artifact = VocabularyArtifacts.TryLoad(entry);
            if (artifact == null)
            {
                data.Issues.Add(
                    $"Vocabulary '{entry.Name}' has no readable artifact at {data.VocabularyPath}. Recompile it.");
                return;
            }

            data.VocabularyFileExists = true;
            data.VocabularyFileTokenCount = artifact.Count;

            if (entry.NumTokens != artifact.Count)
            {
                data.Issues.Add(
                    $"Vocabulary '{entry.Name}' records {entry.NumTokens} tokens but its file holds " +
                    $"{artifact.Count}. The file is the source of truth.");
            }

            if (data.ModelVocabSize.HasValue)
            {
                var problem = VocabularyArtifacts.DescribeModelVocabularyProblem(
                    modelName, data.ModelVocabSize.Value, entry, artifact.Count);
                if (problem != null)
                {
                    data.Issues.Add(problem);
                }
            }

            //Exact token to id comparison: equal sizes with different ids still
            //produce garbage output instead of an error.
            data.LiveVocabularyMatchesModel = VocabularyArtifacts.MatchesLive(artifact, _vocabulary);
            if (!data.LiveVocabularyMatchesModel)
            {
                data.Issues.Add(
                    $"The server tokenises with its startup vocabulary ({_vocabulary.Count} tokens), " +
                    $"not '{entry.Name}' ({artifact.Count} tokens). Output will use the wrong ids until " +
                    "that vocabulary is the one loaded.");
            }
        }

        private string TokenIdOr(string token, string fallback) =>
            _vocabulary.TokenToId.TryGetValue(token, out int id) ? id.ToString() : fallback;

        private static ApiResponse<VocabularyPropertiesResponse> ActiveResponse(
            VocabularyPropertiesResponse data) => new()
        {
            Status = ResponseStatus.Success,
            StatusCode = 200,
            Data = data
        };

        //The idea here is to query the model for the current vocabulary properties and return them
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetCurrentVocabularyProperties()
        {
            return new ApiResponse<VocabularyPropertiesResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new VocabularyPropertiesResponse
                {
                    VocabSize = _vocabulary.TokenToId.Count,
                    //This will get me the id's of these tokens, which should be ok.
                    UnknownToken = _vocabulary.TokenToId[SpecialTokens.Unknown].ToString(),
                    PaddingToken = _vocabulary.TokenToId[SpecialTokens.Pad].ToString(),
                    MaskToken = _vocabulary.TokenToId[SpecialTokens.Mask].ToString(),
                    BosToken = _vocabulary.TokenToId[SpecialTokens.BeginningOfSequence].ToString(),
                    EosToken = _vocabulary.TokenToId[SpecialTokens.EndOfSequence].ToString(),
                }
            };
        }
        public async Task<ApiResponse<VocabularyLoaderResponse>> UploadFiles(List<IFormFile> files)
        {
            var sourceFolder = _configManager.GetValue("vocabulary_Source_Directory", "Paths");
            foreach (var file in files)
            {
                //Copy them to the vocabularies\src folder
                var filePath = Path.Combine(sourceFolder, file.FileName);
                if (string.IsNullOrEmpty(filePath))
                {
                    return new ApiResponse<VocabularyLoaderResponse>
                    {
                        Status = ResponseStatus.Error,
                        StatusCode = 500,
                        Data = null
                    };
                }
                //If the folder does not exist, create it
                if (!Directory.Exists(Path.GetDirectoryName(filePath)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
                }
                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }
            }

            return new ApiResponse<VocabularyLoaderResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new VocabularyLoaderResponse
                {
                    Status = InteractionStatus.Success,
                }
            };
        }
        public async Task<ApiResponse<List<VocabularySourceFile>>> GetLoadVocabularySources()
        {
            var sourceFolder = _configManager.GetValue("vocabulary_Source_Directory", "Paths");
            var srcFiles = Directory.GetFiles(sourceFolder);
            return new ApiResponse<List<VocabularySourceFile>>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = srcFiles.Select(f => new VocabularySourceFile 
                { 
                    Name = Path.GetFileName(f),
                    FileSize = new FileInfo(f).Length
                }).ToList()
            };
        }        
    }

    public class VocabularyPropertiesResponse
    {
        public int VocabSize { get; set; }
        //Question now is: how to get these to populate from the model config? Do I store them in it as well?
        //(The legacy fields below carry the token's id as a string, kept as-is for existing consumers.)
        public string UnknownToken { get; set; } = "<unk>";
        public string PaddingToken { get; set; } = "<pad>";
        public string MaskToken { get; set; } = "<mask>";
        public string BosToken { get; set; } = "<bos>";
        public string EosToken { get; set; } = "<eos>";

        /// <summary>Tokenizer algorithm the live vocabulary was compiled with.</summary>
        public string TokenizerType { get; set; } = string.Empty;

        /// <summary>Where the live vocabulary came from (always the startup file).</summary>
        public string TokenizerSource { get; set; } = string.Empty;

        public Guid? ModelId { get; set; }
        public string? ModelName { get; set; }

        /// <summary>VocabSize the model's persisted transformer config expects.</summary>
        public int? ModelVocabSize { get; set; }

        /// <summary>Database row for the vocabulary the model is pinned to.</summary>
        public VocabularyEntryInfo? Vocabulary { get; set; }

        /// <summary>Absolute path of the pinned vocabulary's artifact on disk.</summary>
        public string VocabularyPath { get; set; } = string.Empty;

        public bool VocabularyFileExists { get; set; }

        /// <summary>Token count read from the artifact file (source of truth).</summary>
        public int? VocabularyFileTokenCount { get; set; }

        /// <summary>
        /// True only when the pinned artifact holds exactly the token to id map the
        /// process is using. False when the model is unpinned or points at another
        /// vocabulary, meaning text would be encoded with different ids.
        /// </summary>
        public bool LiveVocabularyMatchesModel { get; set; }

        /// <summary>Everything found to be inconsistent; empty when it all lines up.</summary>
        public List<string> Issues { get; set; } = new();

        public bool IsConsistent => Issues.Count == 0;
    }

    /// <summary>
    /// Database fields of a vocabulary entry for detail views. Flat by design: no
    /// navigation objects, so serialising it cannot walk into the model graph.
    /// </summary>
    public class VocabularyEntryInfo
    {
        public Guid EntryId { get; set; }
        public string Name { get; set; } = string.Empty;
        public string TokenizerType { get; set; } = string.Empty;
        public DateTime DateCreated { get; set; }
        public int NumTokens { get; set; }
        public string Filename { get; set; } = string.Empty;
        public string Filepath { get; set; } = string.Empty;

        /// <summary>Source files this was compiled from; empty for older entries.</summary>
        public string SourceFileNames { get; set; } = string.Empty;

        /// <summary>Size requested at compile time; 0 for older entries.</summary>
        public int RequestedSize { get; set; }

        /// <summary>Distinct types observed; 0 when the compiler reports none.</summary>
        public long TypesSeen { get; set; }

        /// <summary>Fraction of running tokens covered; 1.0 means not measured.</summary>
        public double Coverage { get; set; }

        public static VocabularyEntryInfo FromEntry(VocabularyEntry entry) => new()
        {
            EntryId = entry.EntryId,
            Name = entry.Name,
            TokenizerType = entry.TokenizerType.ToString(),
            DateCreated = entry.DateCreated,
            NumTokens = entry.NumTokens,
            Filename = entry.Filename,
            Filepath = entry.Filepath,
            SourceFileNames = entry.SourceFileNames,
            RequestedSize = entry.RequestedSize,
            TypesSeen = entry.TypesSeen,
            Coverage = entry.Coverage
        };
    }
}
