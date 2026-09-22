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
        MatMulAccumulate
    }

    internal struct PushConstants
    {
        public uint N;
        public float Alpha;
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
    }

    internal sealed unsafe class VulkanKernelLauncher : IDisposable
    {
        private readonly VulkanContext _ctx;
        private readonly Dictionary<VulkanKernel, Pipeline> _pipelines = new();
        private readonly Dictionary<VulkanKernel, PipelineLayout> _layouts = new();
        private readonly Dictionary<VulkanKernel, DescriptorSetLayout> _setLayouts = new();
        private DescriptorPool _pool;
        private bool _disposed;

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

            var poolSize = new DescriptorPoolSize
            {
                Type = DescriptorType.StorageBuffer,
                DescriptorCount = 64
            };
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = 32,
                PoolSizeCount = 1,
                PPoolSizes = &poolSize
            };
            Result r = _ctx.Vk.CreateDescriptorPool(_ctx.Device, poolInfo, null, out _pool);
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
                    _ctx.Device, moduleInfo, null, out ShaderModule module);
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
                        _ctx.Device, setInfo, null, out setLayout);
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
                    _ctx.Device, pipeLayoutInfo, null, out PipelineLayout layout);
                if (rl != Result.Success)
                    throw new InvalidOperationException($"{kernel} pipelayout: {rl}");

                var pipeInfo = new ComputePipelineCreateInfo
                {
                    SType = StructureType.ComputePipelineCreateInfo,
                    Stage = stage,
                    Layout = layout
                };
                Result rp = vk.CreateComputePipelines(
                    _ctx.Device, default, 1, pipeInfo, null, out Pipeline pipeline);
                if (rp != Result.Success)
                    throw new InvalidOperationException($"{kernel} pipeline: {rp}");

                _pipelines[kernel] = pipeline;
                _layouts[kernel] = layout;
                _setLayouts[kernel] = setLayout;
            }
        }

        public void Dispatch(VulkanKernel kernel, VulkanBuffer[] buffers, uint n, float alpha)
        {
            var vk = _ctx.Vk;
            var layout = _layouts[kernel];
            var setLayout = _setLayouts[kernel];
            var pipeline = _pipelines[kernel];

            DescriptorSet set = AllocateAndBind(buffers, setLayout);
            CommandBuffer cmd = BeginOneTime();
            try
            {
                vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);
                vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, set, 0, null);
                var push = new PushConstants { N = n, Alpha = alpha };
                vk.CmdPushConstants(cmd, layout, ShaderStageFlags.ComputeBit, 0, 8, &push);
                uint groups = (n + 255) / 256;
                vk.CmdDispatch(cmd, groups, 1, 1);
                vk.EndCommandBuffer(cmd);
                SubmitAndWait(cmd);
            }
            finally
            {
                vk.FreeCommandBuffers(_ctx.Device, _ctx.CommandPool, 1, cmd);
            }

            vk.FreeDescriptorSets(_ctx.Device, _pool, 1, set);
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
            var vk = _ctx.Vk;
            var layout = _layouts[kernel];
            var setLayout = _setLayouts[kernel];
            var pipeline = _pipelines[kernel];

            DescriptorSet set = AllocateAndBind(new[] { a, b, r }, setLayout);
            CommandBuffer cmd = BeginOneTime();
            try
            {
                vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);
                vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, set, 0, null);
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
            finally
            {
                vk.FreeCommandBuffers(_ctx.Device, _ctx.CommandPool, 1, cmd);
            }
            vk.FreeDescriptorSets(_ctx.Device, _pool, 1, set);
        }

        private DescriptorSet AllocateAndBind(VulkanBuffer[] buffers, DescriptorSetLayout setLayout)
        {
            var vk = _ctx.Vk;
            var allocInfo = new DescriptorSetAllocateInfo
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = _pool,
                DescriptorSetCount = 1,
                PSetLayouts = &setLayout
            };
            Result r = vk.AllocateDescriptorSets(_ctx.Device, allocInfo, out DescriptorSet set);
            if (r != Result.Success)
                throw new InvalidOperationException($"alloc set: {r}");

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
            return set;
        }

        private CommandBuffer BeginOneTime()
        {
            var cmdAlloc = new CommandBufferAllocateInfo
            {
                SType = StructureType.CommandBufferAllocateInfo,
                CommandPool = _ctx.CommandPool,
                Level = CommandBufferLevel.Primary,
                CommandBufferCount = 1
            };
            _ctx.Vk.AllocateCommandBuffers(_ctx.Device, cmdAlloc, out CommandBuffer cmd);
            var begin = new CommandBufferBeginInfo
            {
                SType = StructureType.CommandBufferBeginInfo,
                Flags = CommandBufferUsageFlags.OneTimeSubmitBit
            };
            _ctx.Vk.BeginCommandBuffer(cmd, begin);
            return cmd;
        }

        private void SubmitAndWait(CommandBuffer cmd)
        {
            var vk = _ctx.Vk;
            var submit = new SubmitInfo
            {
                SType = StructureType.SubmitInfo,
                CommandBufferCount = 1,
                PCommandBuffers = &cmd
            };
            var fenceInfo = new FenceCreateInfo { SType = StructureType.FenceCreateInfo };
            vk.CreateFence(_ctx.Device, fenceInfo, null, out Fence fence);
            try
            {
                vk.QueueSubmit(_ctx.Queue, 1, submit, fence);
                vk.WaitForFences(_ctx.Device, 1, fence, true, 10_000_000_000);
                vk.ResetFences(_ctx.Device, 1, fence);
            }
            finally
            {
                vk.DestroyFence(_ctx.Device, fence, null);
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
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
