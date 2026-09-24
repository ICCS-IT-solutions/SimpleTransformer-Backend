using System;
using System.Collections.Generic;
using Silk.NET.Vulkan;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    internal enum VulkanKernel
    {
        Scale,
        Fill,
        AddInPlace,
        AddInto,
        MulInPlace,
        MulInto,
        MatMul,
        MatMulAccumulate,
        GeluInPlace,
        GeluInto,
        GeluBackward,
        LayerNormInPlace,
        LayerNormInto,
        SoftmaxInPlace,
        SoftmaxBackward,
        ApplyMask,
        Transpose,
        Copy
    }

    internal struct PushConstants
    {
        public uint N;
        public float Alpha;
    }

    internal struct RowPushConstants
    {
        public uint Rows;
        public uint Cols;
        public float Epsilon;
        public uint Flags;
    }

    internal struct MatMulPushConstants
    {
        public uint M;
        public uint N;
        public uint K;
        public uint Batch;
        public uint TransposeA;
        public uint TransposeB;
        public uint Accumulate;
        public uint Pad;
    }

    internal static class VulkanShaders
    {
        public const string Scale = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InBuf { float x[]; };
layout(push_constant) uniform Push { uint n; float alpha; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    x[i] = x[i] * p.alpha;
}";

        public const string Fill = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InBuf { float x[]; };
layout(push_constant) uniform Push { uint n; float value; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    x[i] = p.value;
}";

        public const string AddInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    a[i] = a[i] + b[i];
}";

        public const string AddInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(set = 0, binding = 2) buffer RBuf { float r[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    r[i] = a[i] + b[i];
}";

        public const string MulInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    a[i] = a[i] * b[i];
}";

        public const string MulInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(set = 0, binding = 2) buffer RBuf { float r[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    r[i] = a[i] * b[i];
}";

        // Tiled GEMM: 16x16 workgroup, 32-byte push constants.
        // Inputs must be packed dense row-major per batch layer.
        // A: batch * aRows x aCols, B: batch * bRows x bCols, R: batch * m x n.
        public const string MatMul = @"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(set = 0, binding = 2) buffer RBuf { float r[]; };
layout(push_constant) uniform Push {
    uint m; uint n; uint k; uint batch;
    uint transposeA; uint transposeB; uint accumulate; uint _pad;
} p;
shared float tileA[16][16];
shared float tileB[16][16];
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint lidY = gl_LocalInvocationID.y;
    uint lidX = gl_LocalInvocationID.x;
    uint batch = gl_WorkGroupID.z;
    if (batch >= p.batch) return;

    uint aRows = p.transposeA != 0 ? p.k : p.m;
    uint aCols = p.transposeA != 0 ? p.m : p.k;
    uint bRows = p.transposeB != 0 ? p.n : p.k;
    uint bCols = p.transposeB != 0 ? p.k : p.n;

    uint aBase = batch * aRows * aCols;
    uint bBase = batch * bRows * bCols;
    uint rBase = batch * p.m * p.n;

    float sum = 0.0;
    uint tiles = (p.k + 15u) / 16u;
    for (uint t = 0u; t < tiles; t++) {
        uint aRow = row;
        uint aCol = t * 16u + lidX;
        float av = 0.0;
        if (aRow < p.m && aCol < p.k) {
            uint ai = p.transposeA != 0 ? aCol * aCols + aRow : aRow * aCols + aCol;
            av = a[aBase + ai];
        }
        tileA[lidY][lidX] = av;

        uint bRow = t * 16u + lidY;
        uint bCol = col;
        float bv = 0.0;
        if (bRow < p.k && bCol < p.n) {
            uint bi = p.transposeB != 0 ? bCol * bCols + bRow : bRow * bCols + bCol;
            bv = b[bBase + bi];
        }
        tileB[lidY][lidX] = bv;

        barrier();

        for (uint e = 0u; e < 16u; e++) {
            sum += tileA[lidY][e] * tileB[e][lidX];
        }

        barrier();
    }

    if (row < p.m && col < p.n) {
        uint ri = rBase + row * p.n + col;
        r[ri] = p.accumulate != 0 ? r[ri] + sum : sum;
    }
}";

        public const string GeluInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer T { float x[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    for (uint c = 0u; c < p.cols; c++) {
        uint i = r * p.cols + c;
        float v = x[i];
        float t = tanh(0.7978845608 * (v + 0.044715 * v * v * v));
        x[i] = 0.5 * v * (1.0 + t);
    }
}";

        public const string GeluInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer R { float r[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= p.rows) return;
    for (uint c = 0u; c < p.cols; c++) {
        uint i = row * p.cols + c;
        float v = a[i];
        float t = tanh(0.7978845608 * (v + 0.044715 * v * v * v));
        r[i] = 0.5 * v * (1.0 + t);
    }
}";

        public const string GeluBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer X { float x[]; };
layout(set = 0, binding = 1) buffer DY { float dy[]; };
layout(set = 0, binding = 2) buffer DX { float dx[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= p.rows) return;
    for (uint c = 0u; c < p.cols; c++) {
        uint i = row * p.cols + c;
        float v = x[i];
        float t = tanh(0.7978845608 * (v + 0.044715 * v * v * v));
        float dt = 0.7978845608 * (1.0 + 3.0 * 0.044715 * v * v);
        dx[i] = dy[i] * (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * dt);
    }
}";

        public const string LayerNormInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer T { float x[]; };
layout(set = 0, binding = 1) buffer G { float g[]; };
layout(set = 0, binding = 2) buffer B { float b[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float eps; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    uint base_ = r * p.cols;
    float sum = 0.0;
    for (uint c = 0u; c < p.cols; c++) sum += x[base_ + c];
    float mean = sum / float(p.cols);
    float var_ = 0.0;
    for (uint c = 0u; c < p.cols; c++) {
        float d = x[base_ + c] - mean;
        var_ += d * d;
    }
    var_ /= float(p.cols);
    float inv = inversesqrt(var_ + p.eps);
    for (uint c = 0u; c < p.cols; c++) {
        uint i = base_ + c;
        x[i] = (x[i] - mean) * inv * g[c] + b[c];
    }
}";

        public const string LayerNormInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer G { float g[]; };
layout(set = 0, binding = 2) buffer B { float b[]; };
layout(set = 0, binding = 3) buffer R { float r[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float eps; uint _f; } p;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= p.rows) return;
    uint base_ = row * p.cols;
    float sum = 0.0;
    for (uint c = 0u; c < p.cols; c++) sum += a[base_ + c];
    float mean = sum / float(p.cols);
    float var_ = 0.0;
    for (uint c = 0u; c < p.cols; c++) {
        float d = a[base_ + c] - mean;
        var_ += d * d;
    }
    var_ /= float(p.cols);
    float inv = inversesqrt(var_ + p.eps);
    for (uint c = 0u; c < p.cols; c++) {
        uint i = base_ + c;
        r[i] = (a[i] - mean) * inv * g[c] + b[c];
    }
}";

        public const string SoftmaxInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer T { float x[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    uint base_ = r * p.cols;
    float m = x[base_];
    for (uint c = 1u; c < p.cols; c++) m = max(m, x[base_ + c]);
    float s = 0.0;
    for (uint c = 0u; c < p.cols; c++) {
        float e = exp(x[base_ + c] - m);
        x[base_ + c] = e;
        s += e;
    }
    for (uint c = 0u; c < p.cols; c++) x[base_ + c] /= s;
}";

        public const string SoftmaxBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer DY { float dy[]; };
layout(set = 0, binding = 2) buffer DX { float dx[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    uint base_ = r * p.cols;
    float dot = 0.0;
    for (uint c = 0u; c < p.cols; c++) dot += dy[base_ + c] * s[base_ + c];
    for (uint c = 0u; c < p.cols; c++) {
        uint i = base_ + c;
        dx[i] = s[i] * (dy[i] - dot);
    }
}";

        public const string ApplyMask = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer M { float m[]; };
layout(push_constant) uniform Push { uint n; float _v; uint _a; uint _b; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    if (m[i] == 0.0) s[i] = -1e9;
}";

        public const string Transpose = @"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer D { float d[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.y;
    uint c = gl_GlobalInvocationID.x;
    if (r >= p.rows || c >= p.cols) return;
    d[c * p.rows + r] = s[r * p.cols + c];
}";

        public const string Copy = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer D { float d[]; };
layout(push_constant) uniform Push { uint n; float _v; uint _a; uint _b; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    d[i] = s[i];
}";
    }

    internal sealed unsafe class VulkanKernelLauncher : IDisposable
    {
        private readonly VulkanContext _ctx;
        private readonly Dictionary<VulkanKernel, Pipeline> _pipelines = new();
        private readonly Dictionary<VulkanKernel, PipelineLayout> _layouts = new();
        private readonly Dictionary<VulkanKernel, DescriptorSetLayout> _setLayouts = new();
        private readonly Dictionary<ulong, Dictionary<DescriptorSetCacheKey, DescriptorSet>> _setCache = new();
        private DescriptorPool _pool;
        private Fence _fence;
        private CommandBuffer _cmd;
        private bool _disposed;
        // The launcher owns process-wide Vulkan state that must not be touched
        // concurrently: a single reusable command buffer + fence, the command
        // pool, the descriptor pool/cache and the device queue. Callers like
        // QLoRA's Parallel.For batches dispatch from multiple threads, so every
        // dispatch path is serialized on this gate. SubmitAndWait blocks on a
        // fence anyway, so no GPU-side parallelism is lost by this.
        private readonly object _dispatchGate = new();

        /// <summary>Distinct descriptor sets currently held by the cache.</summary>
        public int CachedDescriptorSetCount
        {
            get
            {
                int total = 0;
                foreach (var perLayout in _setCache.Values)
                    total += perLayout.Count;
                return total;
            }
        }

        /// <summary>Total compute dispatches issued (one per backend op).</summary>
        public long DispatchCount { get; private set; }

        /// <summary>Times the descriptor pool had to be reset and refilled.</summary>
        public long DescriptorPoolRecycles { get; private set; }

        private readonly struct DescriptorSetCacheKey : IEquatable<DescriptorSetCacheKey>
        {
            private readonly ulong _h0;
            private readonly ulong _h1;
            private readonly ulong _h2;
            private readonly ulong _h3;
            private readonly int _count;

            public DescriptorSetCacheKey(VulkanBuffer[] buffers)
            {
                _count = buffers.Length;
                _h0 = buffers.Length > 0 ? buffers[0].Handle.Handle : 0;
                _h1 = buffers.Length > 1 ? buffers[1].Handle.Handle : 0;
                _h2 = buffers.Length > 2 ? buffers[2].Handle.Handle : 0;
                _h3 = buffers.Length > 3 ? buffers[3].Handle.Handle : 0;
            }

            public bool Equals(DescriptorSetCacheKey other)
                => _count == other._count && _h0 == other._h0 && _h1 == other._h1 &&
                   _h2 == other._h2 && _h3 == other._h3;

            public override bool Equals(object? obj)
                => obj is DescriptorSetCacheKey other && Equals(other);

            public override int GetHashCode()
                => HashCode.Combine(_h0, _h1, _h2, _h3, _count);
        }

        public VulkanKernelLauncher(VulkanContext ctx, VulkanShaderCompiler compiler)
        {
            _ctx = ctx;
            Register(VulkanKernel.Scale, compiler, VulkanShaders.Scale, 1, 8);
            Register(VulkanKernel.Fill, compiler, VulkanShaders.Fill, 1, 8);
            Register(VulkanKernel.AddInPlace, compiler, VulkanShaders.AddInPlace, 2, 8);
            Register(VulkanKernel.AddInto, compiler, VulkanShaders.AddInto, 3, 8);
            Register(VulkanKernel.MulInPlace, compiler, VulkanShaders.MulInPlace, 2, 8);
            Register(VulkanKernel.MulInto, compiler, VulkanShaders.MulInto, 3, 8);
            Register(VulkanKernel.MatMul, compiler, VulkanShaders.MatMul, 3, 32);
            Register(VulkanKernel.MatMulAccumulate, compiler, VulkanShaders.MatMul, 3, 32);
            Register(VulkanKernel.GeluInPlace, compiler, VulkanShaders.GeluInPlace, 1, 16);
            Register(VulkanKernel.GeluInto, compiler, VulkanShaders.GeluInto, 2, 16);
            Register(VulkanKernel.GeluBackward, compiler, VulkanShaders.GeluBackward, 3, 16);
            Register(VulkanKernel.LayerNormInPlace, compiler, VulkanShaders.LayerNormInPlace, 3, 16);
            Register(VulkanKernel.LayerNormInto, compiler, VulkanShaders.LayerNormInto, 4, 16);
            Register(VulkanKernel.SoftmaxInPlace, compiler, VulkanShaders.SoftmaxInPlace, 1, 16);
            Register(VulkanKernel.SoftmaxBackward, compiler, VulkanShaders.SoftmaxBackward, 3, 16);
            Register(VulkanKernel.ApplyMask, compiler, VulkanShaders.ApplyMask, 2, 16);
            Register(VulkanKernel.Transpose, compiler, VulkanShaders.Transpose, 2, 16);
            Register(VulkanKernel.Copy, compiler, VulkanShaders.Copy, 2, 16);

            var poolSize = new DescriptorPoolSize
            {
                Type = DescriptorType.StorageBuffer,
                DescriptorCount = 256
            };
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = 64,
                PoolSizeCount = 1,
                PPoolSizes = &poolSize
            };
            Result r = _ctx.Vk.CreateDescriptorPool(_ctx.Device,in poolInfo, null, out _pool);
            if (r != Result.Success)
                throw new InvalidOperationException($"pool failed: {r}");
        }

        private void Register(
            VulkanKernel kernel,
            VulkanShaderCompiler compiler,
            string glsl,
            int bindings,
            uint pushSize)
        {
            var vk = _ctx.Vk;
            byte[] spirv = compiler.CompileCompute(glsl, kernel.ToString());

            fixed (byte* code = spirv)
            {
                var moduleInfo = new ShaderModuleCreateInfo
                {
                    SType = StructureType.ShaderModuleCreateInfo,
                    CodeSize = (nuint)spirv.Length,
                    PCode = (uint*)code
                };
                Result r = vk.CreateShaderModule(
                    _ctx.Device, in moduleInfo, null, out ShaderModule module);
                if (r != Result.Success)
                    throw new InvalidOperationException($"{kernel} module: {r}");

                try
                {
                    RegisterPipeline(kernel, bindings, module, pushSize);
                }
                finally
                {
                    vk.DestroyShaderModule(_ctx.Device, module, null);
                }
            }
        }

        private void RegisterPipeline(VulkanKernel kernel, int bindings, ShaderModule module, uint pushSize)
        {
            var vk = _ctx.Vk;
            using var entryMem = Silk.NET.Core.Native.SilkMarshal.StringToMemory("main", Silk.NET.Core.Native.NativeStringEncoding.UTF8);
            byte* entry = (byte*)entryMem.AsPtr<byte>();
            {
                var stage = new PipelineShaderStageCreateInfo
                {
                    SType = StructureType.PipelineShaderStageCreateInfo,
                    Stage = ShaderStageFlags.ComputeBit,
                    Module = module,
                    PName = entry
                };

                var layoutBindings = new DescriptorSetLayoutBinding[bindings];
                for (int i = 0; i < bindings; i++)
                {
                    layoutBindings[i] = new DescriptorSetLayoutBinding
                    {
                        Binding = (uint)i,
                        DescriptorType = DescriptorType.StorageBuffer,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlags.ComputeBit
                    };
                }

                DescriptorSetLayout setLayout;
                fixed (DescriptorSetLayoutBinding* lb = layoutBindings)
                {
                    var setInfo = new DescriptorSetLayoutCreateInfo
                    {
                        SType = StructureType.DescriptorSetLayoutCreateInfo,
                        BindingCount = (uint)bindings,
                        PBindings = lb
                    };
                    Result r = vk.CreateDescriptorSetLayout(
                        _ctx.Device, in setInfo, null, out setLayout);
                    if (r != Result.Success)
                        throw new InvalidOperationException($"{kernel} setlayout: {r}");
                }

                var pushRange = new PushConstantRange
                {
                    StageFlags = ShaderStageFlags.ComputeBit,
                    Offset = 0,
                    Size = pushSize
                };
                var pipeLayoutInfo = new PipelineLayoutCreateInfo
                {
                    SType = StructureType.PipelineLayoutCreateInfo,
                    SetLayoutCount = 1,
                    PSetLayouts = &setLayout,
                    PushConstantRangeCount = 1,
                    PPushConstantRanges = &pushRange
                };
                Result rl = vk.CreatePipelineLayout(
                    _ctx.Device, in pipeLayoutInfo, null, out PipelineLayout layout);
                if (rl != Result.Success)
                    throw new InvalidOperationException($"{kernel} pipelayout: {rl}");

                var pipeInfo = new ComputePipelineCreateInfo
                {
                    SType = StructureType.ComputePipelineCreateInfo,
                    Stage = stage,
                    Layout = layout
                };
                Result rp = vk.CreateComputePipelines(
                    _ctx.Device, default, 1, in pipeInfo, null, out Pipeline pipeline);
                if (rp != Result.Success)
                    throw new InvalidOperationException($"{kernel} pipeline: {rp}");

                _pipelines[kernel] = pipeline;
                _layouts[kernel] = layout;
                _setLayouts[kernel] = setLayout;
            }
        }

        public void Dispatch(VulkanKernel kernel, VulkanBuffer[] buffers, uint n, float alpha)
        {
            lock (_dispatchGate)
            {
                DispatchCount++;
                var vk = _ctx.Vk;
                var layout = _layouts[kernel];
                var setLayout = _setLayouts[kernel];
                var pipeline = _pipelines[kernel];

                DescriptorSet set = AllocateAndBind(buffers, setLayout);
                // Phase 4: one persistent command buffer, reset+re-recorded per
                // dispatch instead of vkAllocate/vkFreeCommandBuffers every op.
                CommandBuffer cmd = BeginRecording();
                vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);
                vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);
                var push = new PushConstants { N = n, Alpha = alpha };
                vk.CmdPushConstants(cmd, layout, ShaderStageFlags.ComputeBit, 0, 8, &push);
                uint groups = (n + 255) / 256;
                vk.CmdDispatch(cmd, groups, 1, 1);
                vk.EndCommandBuffer(cmd);
                SubmitAndWait(cmd);
            }
        }

        public void DispatchMatMul(
            VulkanKernel kernel,
            VulkanBuffer a,
            VulkanBuffer b,
            VulkanBuffer r,
            uint m,
            uint n,
            uint k,
            uint batch,
            bool transposeA,
            bool transposeB,
            bool accumulate)
        {
            lock (_dispatchGate)
            {
                DispatchCount++;
                var vk = _ctx.Vk;
                var layout = _layouts[kernel];
                var setLayout = _setLayouts[kernel];
                var pipeline = _pipelines[kernel];

                DescriptorSet set = AllocateAndBind(new[] { a, b, r }, setLayout);
                CommandBuffer cmd = BeginRecording();
                vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);
                vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);
                var push = new MatMulPushConstants
                {
                    M = m, N = n, K = k, Batch = batch,
                    TransposeA = transposeA ? 1u : 0u,
                    TransposeB = transposeB ? 1u : 0u,
                    Accumulate = accumulate ? 1u : 0u,
                    Pad = 0
                };
                vk.CmdPushConstants(cmd, layout, ShaderStageFlags.ComputeBit, 0, 32, &push);
                uint gx = (n + 15) / 16;
                uint gy = (m + 15) / 16;
                vk.CmdDispatch(cmd, gx, gy, batch);
                vk.EndCommandBuffer(cmd);
                SubmitAndWait(cmd);
            }
        }

        public void DispatchRowwise(
            VulkanKernel kernel,
            VulkanBuffer[] buffers,
            uint rows,
            uint cols,
            float epsilon = 0f,
            uint flags = 0)
        {
            lock (_dispatchGate)
            {
                DispatchCount++;
                var vk = _ctx.Vk;
                var layout = _layouts[kernel];
                var setLayout = _setLayouts[kernel];
                var pipeline = _pipelines[kernel];

                DescriptorSet set = AllocateAndBind(buffers, setLayout);
                CommandBuffer cmd = BeginRecording();
                vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);
                vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);
                var push = new RowPushConstants { Rows = rows, Cols = cols, Epsilon = epsilon, Flags = flags };
                vk.CmdPushConstants(cmd, layout, ShaderStageFlags.ComputeBit, 0, 16, &push);
                uint groups = (rows + 255) / 256;
                vk.CmdDispatch(cmd, groups, 1, 1);
                vk.EndCommandBuffer(cmd);
                SubmitAndWait(cmd);
            }
        }

        public void DispatchTranspose(
            VulkanBuffer src,
            VulkanBuffer dst,
            uint rows,
            uint cols)
        {
            lock (_dispatchGate)
            {
                DispatchCount++;
                var vk = _ctx.Vk;
                var layout = _layouts[VulkanKernel.Transpose];
                var setLayout = _setLayouts[VulkanKernel.Transpose];
                var pipeline = _pipelines[VulkanKernel.Transpose];

                DescriptorSet set = AllocateAndBind(new[] { src, dst }, setLayout);
                CommandBuffer cmd = BeginRecording();
                vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);
                vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);
                var push = new RowPushConstants { Rows = rows, Cols = cols, Epsilon = 0f, Flags = 0 };
                vk.CmdPushConstants(cmd, layout, ShaderStageFlags.ComputeBit, 0, 16, &push);
                vk.CmdDispatch(cmd, (cols + 15) / 16, (rows + 15) / 16, 1);
                vk.EndCommandBuffer(cmd);
                SubmitAndWait(cmd);
            }
        }

        /// <summary>
        /// Copies bytes from a host-visible staging buffer into a device-resident
        /// one (VRAM) and makes the write visible to the compute stage. Used once
        /// when a weight tensor is promoted into the resident cache: a one-time
        /// transfer instead of re-uploading the tensor for every op.
        /// </summary>
        public void CopyBuffer(VulkanBuffer source, VulkanBuffer destination, ulong sizeBytes)
        {
            if (sizeBytes == 0)
                return;

            lock (_dispatchGate)
            {
                var vk = _ctx.Vk;
                CommandBuffer cmd = BeginRecording();

                var region = new BufferCopy { SrcOffset = 0, DstOffset = 0, Size = sizeBytes };
                vk.CmdCopyBuffer(cmd, source.Handle, destination.Handle, 1, &region);

                //The transfer write must be visible to the compute reads that
                //follow. Both are on the same queue, so no ownership transfer.
                var barrier = new BufferMemoryBarrier
                {
                    SType = StructureType.BufferMemoryBarrier,
                    SrcAccessMask = AccessFlags.TransferWriteBit,
                    DstAccessMask = AccessFlags.ShaderReadBit,
                    SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                    DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                    Buffer = destination.Handle,
                    Offset = 0,
                    Size = sizeBytes
                };

                vk.CmdPipelineBarrier(
                    cmd,
                    PipelineStageFlags.TransferBit,
                    PipelineStageFlags.ComputeShaderBit,
                    0,
                    0, null,
                    1, &barrier,
                    0, null);

                vk.EndCommandBuffer(cmd);
                SubmitAndWait(cmd);
            }
        }

        private DescriptorSet AllocateAndBind(VulkanBuffer[] buffers, DescriptorSetLayout setLayout)
        {
            var vk = _ctx.Vk;

            // Fast path: reuse a cached descriptor set when the exact same
            // buffer handles were bound last time for this layout. Handle
            // identity implies identical sizes (pool buckets are power-of-two
            // and buffers are never destroyed), so no re-write is needed -
            // this is what removes vkUpdateDescriptorSets from the hot path.
            DescriptorSetCacheKey key = new(buffers);
            if (_setCache.TryGetValue(setLayout.Handle, out var perLayout) &&
                perLayout.TryGetValue(key, out DescriptorSet cached))
            {
                return cached;
            }

            var allocInfo = new DescriptorSetAllocateInfo
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = _pool,
                DescriptorSetCount = 1,
                PSetLayouts = &setLayout
            };
            Result r = vk.AllocateDescriptorSets(_ctx.Device, in allocInfo, out DescriptorSet set);
            if (r != Result.Success)
            {
                // Pool exhausted (more distinct binding tuples than MaxSets).
                // Every dispatch waits on its fence, so no set is in flight and
                // the whole pool can be recycled; the cache must go with it.
                vk.ResetDescriptorPool(_ctx.Device, _pool, 0);
                DescriptorPoolRecycles++;
                _setCache.Clear();
                r = vk.AllocateDescriptorSets(_ctx.Device, in allocInfo, out set);
                if (r != Result.Success)
                    throw new InvalidOperationException($"alloc set: {r}");
                UpdateBindings(set, buffers);
                _setCache[setLayout.Handle] = new Dictionary<DescriptorSetCacheKey, DescriptorSet>
                {
                    [key] = set
                };
                return set;
            }

            UpdateBindings(set, buffers);

            if (!_setCache.TryGetValue(setLayout.Handle, out perLayout))
            {
                perLayout = new Dictionary<DescriptorSetCacheKey, DescriptorSet>();
                _setCache[setLayout.Handle] = perLayout;
            }
            perLayout[key] = set;
            return set;
        }

        private void UpdateBindings(DescriptorSet set, VulkanBuffer[] buffers)
        {
            var vk = _ctx.Vk;
            var infos = new DescriptorBufferInfo[buffers.Length];
            var writes = new WriteDescriptorSet[buffers.Length];
            for (int i = 0; i < buffers.Length; i++)
            {
                infos[i] = new DescriptorBufferInfo
                {
                    Buffer = buffers[i].Handle,
                    Offset = 0,
                    Range = buffers[i].SizeBytes
                };
                writes[i] = new WriteDescriptorSet
                {
                    SType = StructureType.WriteDescriptorSet,
                    DstSet = set,
                    DstBinding = (uint)i,
                    DescriptorCount = 1,
                    DescriptorType = DescriptorType.StorageBuffer
                };
            }
            fixed (DescriptorBufferInfo* ip = infos)
            {
                for (int i = 0; i < writes.Length; i++)
                    writes[i].PBufferInfo = &ip[i];
                fixed (WriteDescriptorSet* w = writes)
                {
                    vk.UpdateDescriptorSets(_ctx.Device, (uint)writes.Length, w, 0, null);
                }
            }
        }

        /// <summary>
        /// Phase 4: reuses a single primary command buffer. It is reset and
        /// re-recorded for every dispatch, so no vkAllocateCommandBuffers /
        /// vkFreeCommandBuffers traffic per op. Safe because SubmitAndWait
        /// blocks on the fence, leaving the buffer never pending here.
        /// </summary>
        private CommandBuffer BeginRecording()
        {
            var vk = _ctx.Vk;
            if (_cmd.Handle == 0)
            {
                var cmdAlloc = new CommandBufferAllocateInfo
                {
                    SType = StructureType.CommandBufferAllocateInfo,
                    CommandPool = _ctx.CommandPool,
                    Level = CommandBufferLevel.Primary,
                    CommandBufferCount = 1
                };
                Result r = vk.AllocateCommandBuffers(_ctx.Device, in cmdAlloc, out _cmd);
                if (r != Result.Success)
                    throw new InvalidOperationException($"alloc cmd: {r}");
            }
            else
            {
                vk.ResetCommandBuffer(_cmd, CommandBufferResetFlags.None);
            }

            var begin = new CommandBufferBeginInfo
            {
                SType = StructureType.CommandBufferBeginInfo,
                Flags = CommandBufferUsageFlags.OneTimeSubmitBit
            };
            Result rb = vk.BeginCommandBuffer(_cmd, in begin);
            if (rb != Result.Success)
                throw new InvalidOperationException($"begin cmd: {rb}");
            return _cmd;
        }

        private unsafe void SubmitAndWait(CommandBuffer cmd)
        {
            var vk = _ctx.Vk;

            //Address issue with unsafe pointer here.
            var localCmd = cmd;

            var submit = new SubmitInfo
            {
                SType = StructureType.SubmitInfo,
                CommandBufferCount = 1,
                PCommandBuffers = &localCmd
            };
            if (_fence.Handle == 0)
            {
                var fenceInfo = new FenceCreateInfo { SType = StructureType.FenceCreateInfo };
                vk.CreateFence(_ctx.Device, in fenceInfo, null, out _fence);
            }
            else
            {
                vk.ResetFences(_ctx.Device, 1, in _fence);
            }
            vk.QueueSubmit(_ctx.Queue, 1, in submit, _fence);
            vk.WaitForFences(_ctx.Device, 1, in _fence, true, 10_000_000_000);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_cmd.Handle != 0)
            {
                _ctx.Vk.FreeCommandBuffers(_ctx.Device, _ctx.CommandPool, 1, in _cmd);
                _cmd = default;
            }
            if (_fence.Handle != 0)
            {
                _ctx.Vk.DestroyFence(_ctx.Device, _fence, null);
                _fence = default;
            }
            foreach (var p in _pipelines.Values)
                _ctx.Vk.DestroyPipeline(_ctx.Device, p, null);
            foreach (var l in _layouts.Values)
                _ctx.Vk.DestroyPipelineLayout(_ctx.Device, l, null);
            foreach (var s in _setLayouts.Values)
                _ctx.Vk.DestroyDescriptorSetLayout(_ctx.Device, s, null);
            if (_pool.Handle != 0)
                _ctx.Vk.DestroyDescriptorPool(_ctx.Device, _pool, null);
        }
    }
}
