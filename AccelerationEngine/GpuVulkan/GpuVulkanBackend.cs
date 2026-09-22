using System;
using SimpleTransformer.AccelerationEngine.CpuReference;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Vulkan compute backend. Phase 1: Scale/Fill/Add/Mul run on GPU;
    /// all other ops delegate to the pure managed reference backend.
    /// Tensors are densified on upload (stride-aware pack) so
    /// TensorView inputs work correctly from day one.
    /// </summary>
    public sealed class GpuVulkanBackend : IAccelerationBackend
    {
        private readonly CpuReferenceBackend _fallback = new();
        private readonly VulkanContext? _ctx;
        private readonly VulkanShaderCompiler? _compiler;
        private readonly VulkanKernelLauncher? _launcher;

        private bool _disposed;

        public GpuVulkanBackend()
        {
            var ctx = new VulkanContext();
            if (!ctx.TryInitialize())
            {
                ctx.Dispose();
                return;
            }

            try
            {
                var compiler = new VulkanShaderCompiler();
                var launcher = new VulkanKernelLauncher(ctx, compiler);
                _ctx = ctx;
                _compiler = compiler;
                _launcher = launcher;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GpuVulkan] init failed, CPU fallback: {ex.Message}");
                ctx.Dispose();
            }
        }

        public string Name => _launcher != null
            ? $"GpuVulkan ({_ctx!.DeviceName})"
            : "GpuVulkan (unavailable - CPU reference fallback)";

        public bool IsGpuAccelerated => _launcher != null;

        public bool IsAvailable => _launcher != null;

        private static int RowCount(TensorBase t) => t.Rank switch
        {
            1 => 1,
            2 => t.Rows,
            3 => t.Layers * t.Rows,
            _ => throw new ArgumentException($"Rank {t.Rank} unsupported.")
        };

        private static Span<float> Row(TensorBase t, int r)
        {
            switch (t.Rank)
            {
                case 1: return t.Buffer.AsSpan(t.Offset, t.Cols);
                case 2: return t.Buffer.AsSpan(t.Offset + r * t.Stride, t.Cols);
                case 3:
                    int layer = r / t.Rows;
                    int row = r % t.Rows;
                    return t.Buffer.AsSpan(
                        t.Offset + layer * t.LayerStride + row * t.Stride, t.Cols);
                default: throw new ArgumentException($"Rank {t.Rank} unsupported.");
            }
        }

        private static float[] Pack(TensorBase t)
        {
            int rows = RowCount(t);
            var flat = new float[rows * t.Cols];
            for (int r = 0; r < rows; r++)
                Row(t, r).CopyTo(flat.AsSpan(r * t.Cols, t.Cols));
            return flat;
        }

        private static void Unpack(ReadOnlySpan<float> flat, TensorBase t)
        {
            int rows = RowCount(t);
            for (int r = 0; r < rows; r++)
                flat.Slice(r * t.Cols, t.Cols).CopyTo(Row(t, r));
        }

        private void RunUnary(VulkanKernel kernel, TensorBase tensor, float alpha)
        {
            if (_launcher == null)
            {
                if (kernel == VulkanKernel.Scale) _fallback.ScaleInPlace(tensor, alpha);
                else _fallback.Fill(tensor, alpha);
                return;
            }

            float[] flat = Pack(tensor);
            using var buf = new VulkanBuffer(_ctx!, (ulong)(flat.Length * 4));
            buf.Upload(flat);
            _launcher.Dispatch(kernel, new[] { buf }, (uint)flat.Length, alpha);
            buf.Download(flat);
            Unpack(flat, tensor);
        }

        private void RunBinaryInPlace(VulkanKernel kernel, TensorBase target, TensorBase source)
        {
            if (target.Rank != source.Rank || target.Layers != source.Layers ||
                target.Rows != source.Rows || target.Cols != source.Cols)
                throw new ArgumentException("Shape mismatch.");

            if (_launcher == null)
            {
                if (kernel == VulkanKernel.AddInPlace) _fallback.ElementWiseAddInPlace(target, source);
                else _fallback.ElementWiseMultiplyInPlace(target, source);
                return;
            }

            float[] a = Pack(target);
            float[] b = Pack(source);
            using var ba = new VulkanBuffer(_ctx!, (ulong)(a.Length * 4));
            using var bb = new VulkanBuffer(_ctx!, (ulong)(b.Length * 4));
            ba.Upload(a);
            bb.Upload(b);
            _launcher.Dispatch(kernel, new[] { ba, bb }, (uint)a.Length, 0f);
            ba.Download(a);
            Unpack(a, target);
        }

        private void RunBinaryInto(VulkanKernel kernel, TensorBase a, TensorBase b, TensorBase result)
        {
            if (a.Rank != b.Rank || a.Layers != b.Layers || a.Rows != b.Rows || a.Cols != b.Cols ||
                a.Rank != result.Rank || a.Layers != result.Layers ||
                a.Rows != result.Rows || a.Cols != result.Cols)
                throw new ArgumentException("Shape mismatch.");

            if (_launcher == null)
            {
                if (kernel == VulkanKernel.AddInto) _fallback.ElementWiseAddInto(a, b, result);
                else _fallback.ElementWiseMultiplyInto(a, b, result);
                return;
            }

            float[] fa = Pack(a);
            float[] fb = Pack(b);
            var fr = new float[fa.Length];
            using var ba = new VulkanBuffer(_ctx!, (ulong)(fa.Length * 4));
            using var bb = new VulkanBuffer(_ctx!, (ulong)(fb.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            ba.Upload(fa);
            bb.Upload(fb);
            _launcher.Dispatch(kernel, new[] { ba, bb, br }, (uint)fa.Length, 0f);
            br.Download(fr);
            Unpack(fr, result);
        }

        private void RunMatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA, bool transposeB, bool accumulate)
        {
            int batch = a.Rank == 3 ? a.Layers : 1;
            int m = transposeA ? a.Cols : a.Rows;
            int kA = transposeA ? a.Rows : a.Cols;
            int kB = transposeB ? b.Cols : b.Rows;
            int n = transposeB ? b.Rows : b.Cols;
            if (kA != kB)
                throw new ArgumentException($"MatMul inner dims mismatch ({kA} vs {kB}).");
            if (result.Rank == 3)
            {
                if (result.Layers != batch || result.Rows != m || result.Cols != n)
                    throw new ArgumentException("MatMul result shape mismatch.");
            }
            else if (result.Rows != m || result.Cols != n)
            {
                throw new ArgumentException("MatMul result shape mismatch.");
            }

            if (_launcher == null)
            {
                if (accumulate) _fallback.MatMulAccumulate(a, b, result, transposeA, transposeB);
                else _fallback.MatMul(a, b, result, transposeA, transposeB);
                return;
            }

            float[] fa = Pack(a);
            float[] fb = Pack(b);
            int k = kA;
            float[] fr = new float[batch * m * n];
            if (accumulate)
                Pack(result).CopyTo(fr, 0);

            using var ba = new VulkanBuffer(_ctx!, (ulong)(fa.Length * 4));
            using var bb = new VulkanBuffer(_ctx!, (ulong)(fb.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            ba.Upload(fa);
            bb.Upload(fb);
            br.Upload(fr);
            _launcher.DispatchMatMul(
                accumulate ? VulkanKernel.MatMulAccumulate : VulkanKernel.MatMul,
                ba, bb, br, (uint)m, (uint)n, (uint)k, (uint)batch,
                transposeA, transposeB, accumulate);
            br.Download(fr);
            Unpack(fr, result);
        }

        public void ScaleInPlace(TensorBase tensor, float scalar)
            => RunUnary(VulkanKernel.Scale, tensor, scalar);

        public void Fill(TensorBase tensor, float value)
            => RunUnary(VulkanKernel.Fill, tensor, value);

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
            => RunBinaryInPlace(VulkanKernel.AddInPlace, target, source);

        public void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
            => RunBinaryInto(VulkanKernel.AddInto, a, b, result);

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
            => RunBinaryInPlace(VulkanKernel.MulInPlace, target, source);

        public void ElementWiseMultiplyInto(TensorBase a, TensorBase b, TensorBase result)
            => RunBinaryInto(VulkanKernel.MulInto, a, b, result);

        public void GeluInPlace(TensorBase tensor)
            => _fallback.GeluInPlace(tensor);

        public void GeluInto(TensorBase input, TensorBase result)
            => _fallback.GeluInto(input, result);

        public void GeluBackwardInto(TensorBase input, TensorBase outputGradient, TensorBase inputGradient)
            => _fallback.GeluBackwardInto(input, outputGradient, inputGradient);

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f)
            => _fallback.LayerNormInPlace(tensor, gamma, beta, epsilon);

        public void LayerNormInto(TensorBase input, TensorBase gamma, TensorBase beta, TensorBase result, float epsilon = 1e-5f)
            => _fallback.LayerNormInto(input, gamma, beta, result, epsilon);

        public void SoftmaxInPlace(TensorBase tensor)
            => _fallback.SoftmaxInPlace(tensor);

        public void SoftmaxBackwardInto(TensorBase softmaxOutput, TensorBase outputGradient, TensorBase inputGradient)
            => _fallback.SoftmaxBackwardInto(softmaxOutput, outputGradient, inputGradient);

        public void MatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => RunMatMul(a, b, result, transposeA, transposeB, false);

        public void MatMulAccumulate(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => RunMatMul(a, b, result, transposeA, transposeB, true);

        public void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
            => _fallback.ApplyMaskInPlace(scores, mask);

        public void TransposeInto(TensorBase source, TensorBase destination)
            => _fallback.TransposeInto(source, destination);

        public void CopyInto(TensorBase source, TensorBase destination)
            => _fallback.CopyInto(source, destination);

        public void Synchronize()
        {
            _ctx?.Synchronize();
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _launcher?.Dispose();
            _compiler?.Dispose();
            _fallback.Dispose();
            _ctx?.Dispose();
        }
    }
}
