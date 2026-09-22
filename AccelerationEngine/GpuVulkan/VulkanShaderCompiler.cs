using System;
using Silk.NET.Shaderc;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Compiles GLSL compute shaders to SPIR-V at startup via shaderc.
    /// One shared compiler; throws with the shaderc error log on failure.
    /// </summary>
    internal sealed unsafe class VulkanShaderCompiler : IDisposable
    {
        private readonly Shaderc _shaderc = Shaderc.GetApi();
        private Compiler* _compiler;
        private bool _disposed;

        public VulkanShaderCompiler()
        {
            _compiler = _shaderc.CompilerInitialize();
            if (_compiler == null)
                throw new InvalidOperationException("shaderc CompilerInitialize failed.");
        }

        public byte[] CompileCompute(string glslSource, string name)
        {
            var options = _shaderc.CompileOptionsInitialize();
            try
            {
                _shaderc.CompileOptionsSetSourceLanguage(options, SourceLanguage.Glsl);
                _shaderc.CompileOptionsSetTargetEnv(options, TargetEnv.Vulkan, Vk10Version());
                _shaderc.CompileOptionsSetOptimizationLevel(options, OptimizationLevel.Performance);

                CompilationResult* result = _shaderc.CompileIntoSpv(
                    _compiler, glslSource, (UIntPtr)glslSource.Length,
                    ShaderKind.GlslDefaultComputeShader, name, "main", options);
                try
                {
                    var status = _shaderc.ResultGetCompilationStatus(result);
                    if (status != CompilationStatus.Success)
                    {
                        string log = _shaderc.ResultGetErrorMessageS(result) ?? string.Empty;
                        throw new InvalidOperationException($"GLSL {name}: {status}: {log}");
                    }

                    nuint length = (nuint)_shaderc.ResultGetLength(result);
                    byte* bytes = _shaderc.ResultGetBytes(result);
                    var spirv = new byte[length];
                    new ReadOnlySpan<byte>(bytes, (int)length).CopyTo(spirv);
                    return spirv;
                }
                finally
                {
                    _shaderc.ResultRelease(result);
                }
            }
            finally
            {
                _shaderc.CompileOptionsRelease(options);
            }
        }

        private static uint Vk10Version()
        {
            return ((uint)1 << 22) | ((uint)0 << 12);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_compiler != null)
            {
                _shaderc.CompilerRelease(_compiler);
                _compiler = null;
            }
        }
    }
}
