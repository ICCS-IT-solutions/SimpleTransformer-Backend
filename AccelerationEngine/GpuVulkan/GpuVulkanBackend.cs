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
        {
            if (_launcher == null) { _fallback.GeluInPlace(tensor); return; }
            float[] f = Pack(tensor);
            using var b = new VulkanBuffer(_ctx!, (ulong)(f.Length * 4));
            b.Upload(f);
            _launcher.DispatchRowwise(VulkanKernel.GeluInPlace, new[] { b }, (uint)RowCount(tensor), (uint)tensor.Cols);
            b.Download(f);
            Unpack(f, tensor);
        }

        public void GeluInto(TensorBase input, TensorBase result)
        {
            ValidateSame(input, result, "GeluInto");
            if (_launcher == null) { _fallback.GeluInto(input, result); return; }
            float[] fa = Pack(input);
            var fr = new float[fa.Length];
            using var ba = new VulkanBuffer(_ctx!, (ulong)(fa.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            ba.Upload(fa);
            _launcher.DispatchRowwise(VulkanKernel.GeluInto, new[] { ba, br }, (uint)RowCount(input), (uint)input.Cols);
            br.Download(fr);
            Unpack(fr, result);
        }

        public void GeluBackwardInto(TensorBase input, TensorBase outputGradient, TensorBase inputGradient)
        {
            ValidateSame(input, outputGradient, "GeluBackwardInto");
            ValidateSame(input, inputGradient, "GeluBackwardInto");
            if (_launcher == null) { _fallback.GeluBackwardInto(input, outputGradient, inputGradient); return; }
            float[] fx = Pack(input);
            float[] fy = Pack(outputGradient);
            var fr = new float[fx.Length];
            using var bx = new VulkanBuffer(_ctx!, (ulong)(fx.Length * 4));
            using var by = new VulkanBuffer(_ctx!, (ulong)(fy.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            bx.Upload(fx);
            by.Upload(fy);
            _launcher.DispatchRowwise(VulkanKernel.GeluBackward, new[] { bx, by, br }, (uint)RowCount(input), (uint)input.Cols);
            br.Download(fr);
            Unpack(fr, inputGradient);
        }

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f)
        {
            ValidateNorm(tensor, gamma, beta, "LayerNormInPlace");
            if (_launcher == null) { _fallback.LayerNormInPlace(tensor, gamma, beta, epsilon); return; }
            float[] f = Pack(tensor);
            float[] g = Pack1D(gamma);
            float[] bb = Pack1D(beta);
            using var bx = new VulkanBuffer(_ctx!, (ulong)(f.Length * 4));
            using var bg = new VulkanBuffer(_ctx!, (ulong)(g.Length * 4));
            using var bbb = new VulkanBuffer(_ctx!, (ulong)(bb.Length * 4));
            bx.Upload(f);
            bg.Upload(g);
            bbb.Upload(bb);
            _launcher.DispatchRowwise(VulkanKernel.LayerNormInPlace, new[] { bx, bg, bbb }, (uint)RowCount(tensor), (uint)tensor.Cols, epsilon);
            bx.Download(f);
            Unpack(f, tensor);
        }

        public void LayerNormInto(TensorBase input, TensorBase gamma, TensorBase beta, TensorBase result, float epsilon = 1e-5f)
        {
            ValidateSame(input, result, "LayerNormInto");
            ValidateNorm(input, gamma, beta, "LayerNormInto");
            if (_launcher == null) { _fallback.LayerNormInto(input, gamma, beta, result, epsilon); return; }
            float[] fa = Pack(input);
            float[] g = Pack1D(gamma);
            float[] bb = Pack1D(beta);
            var fr = new float[fa.Length];
            using var ba = new VulkanBuffer(_ctx!, (ulong)(fa.Length * 4));
            using var bg = new VulkanBuffer(_ctx!, (ulong)(g.Length * 4));
            using var bbb = new VulkanBuffer(_ctx!, (ulong)(bb.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            ba.Upload(fa);
            bg.Upload(g);
            bbb.Upload(bb);
            _launcher.DispatchRowwise(VulkanKernel.LayerNormInto, new[] { ba, bg, bbb, br }, (uint)RowCount(input), (uint)input.Cols, epsilon);
            br.Download(fr);
            Unpack(fr, result);
        }

        public void SoftmaxInPlace(TensorBase tensor)
        {
            if (_launcher == null) { _fallback.SoftmaxInPlace(tensor); return; }
            float[] f = Pack(tensor);
            using var b = new VulkanBuffer(_ctx!, (ulong)(f.Length * 4));
            b.Upload(f);
            _launcher.DispatchRowwise(VulkanKernel.SoftmaxInPlace, new[] { b }, (uint)RowCount(tensor), (uint)tensor.Cols);
            b.Download(f);
            Unpack(f, tensor);
        }

        public void SoftmaxBackwardInto(TensorBase softmaxOutput, TensorBase outputGradient, TensorBase inputGradient)
        {
            ValidateSame(softmaxOutput, outputGradient, "SoftmaxBackwardInto");
            ValidateSame(softmaxOutput, inputGradient, "SoftmaxBackwardInto");
            if (_launcher == null) { _fallback.SoftmaxBackwardInto(softmaxOutput, outputGradient, inputGradient); return; }
            float[] fs = Pack(softmaxOutput);
            float[] fy = Pack(outputGradient);
            var fr = new float[fs.Length];
            using var bs = new VulkanBuffer(_ctx!, (ulong)(fs.Length * 4));
            using var by = new VulkanBuffer(_ctx!, (ulong)(fy.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            bs.Upload(fs);
            by.Upload(fy);
            _launcher.DispatchRowwise(VulkanKernel.SoftmaxBackward, new[] { bs, by, br }, (uint)RowCount(softmaxOutput), (uint)softmaxOutput.Cols);
            br.Download(fr);
            Unpack(fr, inputGradient);
        }

        public void MatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => RunMatMul(a, b, result, transposeA, transposeB, false);

        public void MatMulAccumulate(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
            => RunMatMul(a, b, result, transposeA, transposeB, true);

        public void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
        {
            if (scores.Rank != 2 || mask.Rank != 2 || scores.Rows != mask.Rows || scores.Cols != mask.Cols)
                throw new ArgumentException("Mask shape mismatch.");
            if (_launcher == null) { _fallback.ApplyMaskInPlace(scores, mask); return; }
            float[] fs = Pack(scores);
            float[] fm = Pack(mask);
            using var bs = new VulkanBuffer(_ctx!, (ulong)(fs.Length * 4));
            using var bm = new VulkanBuffer(_ctx!, (ulong)(fm.Length * 4));
            bs.Upload(fs);
            bm.Upload(fm);
            _launcher.DispatchRowwise(VulkanKernel.ApplyMask, new[] { bs, bm }, (uint)fs.Length, 1);
            bs.Download(fs);
            Unpack(fs, scores);
        }

        public void TransposeInto(TensorBase source, TensorBase destination)
        {
            if (source.Rank != 2 || destination.Rank != 2 ||
                destination.Rows != source.Cols || destination.Cols != source.Rows)
                throw new ArgumentException("Transpose shape mismatch.");
            if (_launcher == null) { _fallback.TransposeInto(source, destination); return; }
            float[] fs = Pack(source);
            var fr = new float[fs.Length];
            using var bs = new VulkanBuffer(_ctx!, (ulong)(fs.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            bs.Upload(fs);
            _launcher.DispatchTranspose(bs, br, (uint)source.Rows, (uint)source.Cols);
            br.Download(fr);
            Unpack(fr, destination);
        }

        public void CopyInto(TensorBase source, TensorBase destination)
        {
            ValidateSame(source, destination, "CopyInto");
            if (_launcher == null) { _fallback.CopyInto(source, destination); return; }
            float[] fs = Pack(source);
            var fr = new float[fs.Length];
            using var bs = new VulkanBuffer(_ctx!, (ulong)(fs.Length * 4));
            using var br = new VulkanBuffer(_ctx!, (ulong)(fr.Length * 4));
            bs.Upload(fs);
            _launcher.DispatchRowwise(VulkanKernel.Copy, new[] { bs, br }, (uint)fs.Length, 1);
            br.Download(fr);
            Unpack(fr, destination);
        }

        private static void ValidateSame(TensorBase a, TensorBase b, string op)
        {
            if (a.Rank != b.Rank || a.Layers != b.Layers || a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException($"{op}: shape mismatch.");
        }

        private static void ValidateNorm(TensorBase t, TensorBase gamma, TensorBase beta, string op)
        {
            if (gamma.Rank != 1 || beta.Rank != 1 || gamma.Cols != t.Cols || beta.Cols != t.Cols)
                throw new ArgumentException($"{op}: gamma/beta must be rank-1 length Cols.");
        }

        private static float[] Pack1D(TensorBase t)
        {
            var flat = new float[t.Cols];
            t.Buffer.AsSpan(t.Offset, t.Cols).CopyTo(flat);
            return flat;
        }

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
