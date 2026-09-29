using System;
using System.Collections.Generic;
using System.Diagnostics;

using Silk.NET.Vulkan;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    public enum VulkanKernel
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

    /// <summary>
    /// Records compute work into reusable command buffers and hands it to the
    /// queue in BATCHES: many dispatches are recorded back to back, ordered
    /// against each other by GPU-side barriers inside one command buffer, then
    /// submitted with a single <c>vkQueueSubmit</c> and awaited with a single
    /// fence wait.
    /// <para>
    /// Automatic batching: a caller announces an op with <see cref="OpStart"/>
    /// before preparing it (pack + upload). Every announced-but-not-yet-recorded
    /// op keeps the current batch open, so a whole Parallel.For wave lands in one
    /// submission. A single-threaded caller closes its own batch immediately -
    /// one submit per op, as before, with no added latency.
    /// </para>
    /// <para>
    /// Explicit batching: <see cref="BeginBatch"/> / <see cref="EndBatch"/>
    /// record everything in the scope into one submission. Thread-affine; the
    /// owner only blocks at <see cref="EndBatch"/>.
    /// </para>
    /// <para>
    /// Batching removes the round trips BETWEEN ops, not the one correctness
    /// needs: a dispatch still completes before its caller may read the result,
    /// because the tensors flowing through the model are ordinary managed arrays.
    /// </para>
    /// </summary>
    public sealed unsafe class VulkanKernelLauncher : IDisposable
    {
        // ---- dispatch geometry (mirrors local_size_x/y in VulkanShaders) -----
        private const uint ScalarWorkgroupSize = 256;   // element-wise kernels
        private const uint RowwiseWorkgroupSize = 256;  // GELU / norm / softmax / mask / copy
        private const uint MatMulTileSize = 16;         // MatMul local_size_x/y
        private const uint TransposeTileSize = 16;      // Transpose local_size_x/y

        // Two rotating (command buffer, fence) pairs: recording of the next
        // batch overlaps execution of the batch already in flight.
        private const int SlotCount = 2;
        private const ulong FenceTimeoutNanos = 30_000_000_000UL;  // 30 s
        private const int DispatchTimeoutMillis = 120_000;

        /// <summary>
        /// What the last command recorded into the open batch was; the next
        /// record uses it to emit the barrier ordering the two.
        /// </summary>
        private enum RecordKind : byte { None = 0, Dispatch = 1, Transfer = 2 }

        // Per-op announcement state. OpStart claims a place in the open batch
        // and RunOp releases it; OpFinish releases it on the exception path, so
        // a thread that prepared an op but never dispatched cannot wedge the
        // batch it was coalescing into.
        [ThreadStatic] private static bool t_announced;
        [ThreadStatic] private static bool t_recorded;

        private readonly VulkanContext _ctx;
        private readonly Dictionary<VulkanKernel, Pipeline> _pipelines = new();
        private readonly Dictionary<VulkanKernel, PipelineLayout> _layouts = new();
        private readonly Dictionary<VulkanKernel, DescriptorSetLayout> _setLayouts = new();
        private readonly Dictionary<ulong, Dictionary<DescriptorSetCacheKey, CachedDescriptorSet>> _setCache = new();
        private DescriptorPool _pool;
        private DescriptorPool? _spool;
        private bool _disposed;

        private readonly CommandBuffer[] _cmds = new CommandBuffer[SlotCount];
        private readonly Fence[] _fences = new Fence[SlotCount];
        private readonly bool[] _slotPending = new bool[SlotCount];

        // ---- batch state, all guarded by _dispatchGate ----------------------
        private bool _cmdOpen;              // _cmds[_activeSlot] has begun recording
        private int _activeSlot;
        private long _nextSeq;              // sequence number of the next batch to open
        private long _openSeq = -1;         // sequence of the batch being recorded
        private long _completedSeq = -1;    // highest sequence known finished
        private int _openRecords;           // commands recorded into the open batch
        private RecordKind _lastRecord = RecordKind.None;
        private int _pendingPrep;           // ops announced, not yet recorded
        private int _explicitDepth;         // BeginBatch nesting
        private int _explicitOwner;         // managed thread id owning the explicit batch
        private int _inflightBatches;       // submitted, fence not yet waited on

        // The launcher owns process-wide Vulkan state that must not be touched
        // concurrently: the reusable command buffers + fences, the command pool,
        // the descriptor pool/cache and the device queue. Callers like QLoRA's
        // Parallel.For batches dispatch from multiple threads, so recording is
        // serialized on this gate - but the gate is deliberately released while
        // a fence is waited on. That is what lets concurrent callers record into
        // ONE batch (one submit, one fence wait) instead of queueing up behind a
        // stall per op.
        private readonly object _dispatchGate = new();

        /// <summary>Distinct descriptor sets currently held by the cache.</summary>
        public int CachedDescriptorSetCount
        {
            get
            {
                lock (_dispatchGate)
                {
                    int total = 0;
                    foreach (var perLayout in _setCache.Values)
                        total += perLayout.Count;
                    return total;
                }
            }
        }

        /// <summary>Total compute dispatches recorded (one per backend op).</summary>
        public long DispatchCount { get; private set; }

        /// <summary>Queue submissions issued - one per batch, not one per op.</summary>
        public long SubmitCount { get; private set; }

        /// <summary>Submissions that carried more than one dispatch.</summary>
        public long BatchedSubmits { get; private set; }

        /// <summary>Largest number of dispatches coalesced into a single submit.</summary>
        public int MaxBatchSize { get; private set; }

        /// <summary>Mean dispatches per submit; 1.0 means no coalescing happened.</summary>
        public double AvgDispatchesPerSubmit =>
            SubmitCount == 0 ? 0.0 : (double)DispatchCount / SubmitCount;

        /// <summary>Times the descriptor pool had to be reset and refilled.</summary>
        public long DescriptorPoolRecycles { get; private set; }

        /// <summary>A cached descriptor set plus the pool that owns it.</summary>
        private readonly struct CachedDescriptorSet
        {
            public CachedDescriptorSet(DescriptorSet set, bool fromSpool)
            {
                Set = set;
                FromSpool = fromSpool;
            }

            public DescriptorSet Set { get; }

            /// <summary>True when the set lives in the overflow pool.</summary>
            public bool FromSpool { get; }
        }

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

            // Sized for a batch, not for one op: a coalesced batch can carry one
            // distinct binding tuple per dispatch (every Parallel.For worker in a
            // training step brings its own buffers), and the pool must not run dry
            // while batches are in flight - exhausting it would force a reset,
            // which is illegal while a submitted batch still references its sets.
            const uint poolSets = 1024;
            const uint poolDescriptors = 4 * poolSets;
            var poolSize = new DescriptorPoolSize
            {
                Type = DescriptorType.StorageBuffer,
                DescriptorCount = poolDescriptors
            };
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = poolSets,
                PoolSizeCount = 1,
                PPoolSizes = &poolSize
            };
            Result r = _ctx.Vk.CreateDescriptorPool(_ctx.Device, in poolInfo, null, out _pool);
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

        // ---- batch scopes ---------------------------------------------------

        /// <summary>
        /// Announces that this thread is about to prepare and dispatch one op.
        /// The backend calls it before it packs/uploads, so every op of a parallel
        /// wave holds the current batch open and the whole wave ends up in one
        /// submission. A single-threaded caller pays only a counter update.
        /// </summary>
        public void OpStart()
        {
            if (_disposed)
                return;
            t_announced = true;
            t_recorded = false;
            lock (_dispatchGate)
            {
                _pendingPrep++;
            }
        }

        /// <summary>
        /// Releases an op that was announced but never dispatched (an exception
        /// during pack/upload). Without this the batch it was coalescing into
        /// would never look complete and every waiter would eventually time out.
        /// </summary>
        public void OpFinish()
        {
            if (!t_announced)
                return;
            t_announced = false;
            if (t_recorded)
                return; // RunOp already released the claim

            Fence? owned = null;
            long seq = -1;
            lock (_dispatchGate)
            {
                if (_pendingPrep > 0)
                    _pendingPrep--;
                if (ShouldCloseLocked())
                    (seq, owned) = CloseAndSubmitLocked();
            }
            if (owned != null)
                Complete(owned, seq);
        }

        /// <summary>
        /// Opens an explicit batch: dispatches recorded by this thread until
        /// <see cref="EndBatch"/> all land in one submission. Thread-affine.
        /// </summary>
        public void BeginBatch()
        {
            if (_disposed)
                return;

            lock (_dispatchGate)
            {
                int me = Environment.CurrentManagedThreadId;
                if (_explicitDepth > 0 && _explicitOwner != me)
                    throw new InvalidOperationException(
                        "A Vulkan batch is already open on another thread; batches are thread-affine.");

                _explicitDepth++;
                _explicitOwner = me;
                EnsureOpenLocked();
            }
        }

        /// <summary>
        /// Closes the batch opened by <see cref="BeginBatch"/> and blocks until
        /// it has completed. Safe to call unbalanced (it then just flushes).
        /// </summary>
        public void EndBatch()
        {
            if (_disposed)
                return;

            Fence? owned = null;
            long seq = -1;
            bool wait = false;
            lock (_dispatchGate)
            {
                if (_explicitDepth == 0)
                    return;
                if (_explicitOwner != Environment.CurrentManagedThreadId)
                    throw new InvalidOperationException(
                        "EndBatch must run on the thread that called BeginBatch.");

                _explicitDepth--;
                if (_explicitDepth == 0 && _openRecords > 0)
                {
                    seq = _openSeq;
                    if (_pendingPrep == 0)
                        (seq, owned) = CloseAndSubmitLocked();
                    else
                        wait = true; // a concurrent op is still preparing; it will close
                }
            }

            if (owned != null)
                Complete(owned, seq);
            else if (wait)
                WaitForCompletion(seq);
        }

        /// <summary>
        /// Submits anything recorded-but-unsent and waits for the device to
        /// drain. Used by <c>Synchronize</c> and before teardown so no recorded
        /// work is silently dropped.
        /// </summary>
        public void Flush()
        {
            if (_disposed)
                return;
            FlushCore();
        }

        /// <summary>
        /// Submits and waits for anything recorded but not yet sent, so the caller
        /// may safely read results that were produced by recorded work. Unlike
        /// <see cref="Flush"/>, this also runs inside an explicitly open batch: the
        /// records recorded so far are submitted early and later ops reopen a fresh
        /// command buffer, so <c>EndBatch</c> is unaffected.
        /// <para>
        /// This is required by every op that hands its result straight back to the
        /// caller: inside an explicit batch <see cref="RunOp"/> deliberately does
        /// not wait (EndBatch is the completer), so without this the read-back
        /// would return the pooled buffer's previous contents - a silent wrong
        /// answer rather than an error.
        /// </para>
        /// </summary>
        public void EnsureRecordedWorkComplete()
        {
            if (_disposed)
                return;

            Fence? owned = null;
            long seq = -1;
            lock (_dispatchGate)
            {
                if (_openRecords == 0)
                    return; // nothing recorded since the last submit
                (seq, owned) = CloseAndSubmitLocked();
            }
            Complete(owned, seq);
        }


        public void Dispatch(VulkanKernel kernel, VulkanBuffer[] buffers, uint n, float alpha)
            => RunOp(() => RecordScalar(kernel, buffers, n, alpha));

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
            var buffers = new[] { a, b, r };
            RunOp(() => RecordMatMul(
                kernel, buffers, m, n, k, batch, transposeA, transposeB, accumulate));
        }

        public void DispatchRowwise(
            VulkanKernel kernel,
            VulkanBuffer[] buffers,
            uint rows,
            uint cols,
            float epsilon = 0f,
            uint flags = 0)
        {
            RunOp(() => RecordRowwise(kernel, buffers, rows, cols, epsilon, flags));
        }

        public void DispatchTranspose(
            VulkanBuffer src,
            VulkanBuffer dst,
            uint rows,
            uint cols)
        {
            var buffers = new[] { src, dst };
            RunOp(() => RecordTranspose(buffers, rows, cols));
        }

        /// <summary>
        /// Copies bytes from a host-visible staging buffer into a device-resident
        /// one (VRAM) and makes the write visible to the compute stage. Used once
        /// when a weight tensor is promoted into the resident cache: a one-time
        /// transfer instead of re-uploading the tensor for every op.
        /// </summary>
        public void CopyBuffer(VulkanBuffer source, VulkanBuffer destination, ulong sizeBytes)
        {
            if (sizeBytes == 0 || _disposed)
                return;
            RunOp(() => RecordCopyBuffer(source, destination, sizeBytes));
        }

        /// <summary>
        /// Records the copy and blocks until the device has executed it, so the
        /// caller may immediately recycle or overwrite <paramref name="source"/>.
        /// <see cref="CopyBuffer"/> alone is not sufficient for that: inside an
        /// explicitly open batch (<see cref="BeginBatch"/>) <see cref="RunOp"/>
        /// only records and returns while the transfer is still queued, so a
        /// source handed back to the staging pool can be re-rented and
        /// overwritten before the copy runs - the destination would then capture
        /// another op's payload (or untouched allocator memory) instead of the
        /// bytes that were uploaded. Submitting here splits the open batch in
        /// two: the caller's scope stays open, later ops reopen a fresh command
        /// buffer, and <c>EndBatch</c> is unaffected. Use this for one-off
        /// transfers whose source buffer is recycled straight afterwards.
        /// </summary>
        public void CopyBufferSync(VulkanBuffer source, VulkanBuffer destination, ulong sizeBytes)
        {
            if (sizeBytes == 0 || _disposed)
                return;

            Fence? owned = null;
            long seq = -1;
            lock (_dispatchGate)
            {
                EnsureOpenLocked();
                RecordCopyBuffer(source, destination, sizeBytes);
                DispatchCount++;
                (seq, owned) = CloseAndSubmitLocked();
            }
            Complete(owned, seq);
        }

        // ---- recording (runs with _dispatchGate held, batch open) ------------

        private void RecordScalar(VulkanKernel kernel, VulkanBuffer[] buffers, uint n, float alpha)
        {
            var vk = _ctx.Vk;
            var cmd = _cmds[_activeSlot];
            var layout = _layouts[kernel];
            DescriptorSet set = AllocateAndBind(buffers, _setLayouts[kernel]);

            RecordTransfersIn(buffers);
            SyncWithPrevious(RecordKind.Dispatch);

            vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _pipelines[kernel]);
            vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);

            var push = new PushConstants { N = n, Alpha = alpha };
            vk.CmdPushConstants(
                cmd, layout, ShaderStageFlags.ComputeBit, 0,
                (uint)sizeof(PushConstants), &push);

            vk.CmdDispatch(cmd, Groups(n, ScalarWorkgroupSize), 1, 1);
            FinishDispatch(buffers);
        }

        private void RecordMatMul(
            VulkanKernel kernel,
            VulkanBuffer[] buffers,
            uint m,
            uint n,
            uint k,
            uint batch,
            bool transposeA,
            bool transposeB,
            bool accumulate)
        {
            var vk = _ctx.Vk;
            var cmd = _cmds[_activeSlot];
            var layout = _layouts[kernel];
            DescriptorSet set = AllocateAndBind(buffers, _setLayouts[kernel]);

            RecordTransfersIn(buffers);
            SyncWithPrevious(RecordKind.Dispatch);

            vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _pipelines[kernel]);
            vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);

            var push = new MatMulPushConstants
            {
                M = m, N = n, K = k, Batch = batch,
                TransposeA = transposeA ? 1u : 0u,
                TransposeB = transposeB ? 1u : 0u,
                Accumulate = accumulate ? 1u : 0u,
                Pad = 0
            };
            vk.CmdPushConstants(
                cmd, layout, ShaderStageFlags.ComputeBit, 0,
                (uint)sizeof(MatMulPushConstants), &push);

            vk.CmdDispatch(cmd, Groups(n, MatMulTileSize), Groups(m, MatMulTileSize), batch);
            FinishDispatch(buffers);
        }

        private void RecordRowwise(
            VulkanKernel kernel,
            VulkanBuffer[] buffers,
            uint rows,
            uint cols,
            float epsilon,
            uint flags)
        {
            var vk = _ctx.Vk;
            var cmd = _cmds[_activeSlot];
            var layout = _layouts[kernel];
            DescriptorSet set = AllocateAndBind(buffers, _setLayouts[kernel]);

            RecordTransfersIn(buffers);
            SyncWithPrevious(RecordKind.Dispatch);

            vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _pipelines[kernel]);
            vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);

            var push = new RowPushConstants
            {
                Rows = rows, Cols = cols, Epsilon = epsilon, Flags = flags
            };
            vk.CmdPushConstants(
                cmd, layout, ShaderStageFlags.ComputeBit, 0,
                (uint)sizeof(RowPushConstants), &push);

            vk.CmdDispatch(cmd, Groups(rows, RowwiseWorkgroupSize), 1, 1);
            FinishDispatch(buffers);
        }

        private void RecordTranspose(VulkanBuffer[] buffers, uint rows, uint cols)
        {
            var vk = _ctx.Vk;
            var cmd = _cmds[_activeSlot];
            var layout = _layouts[VulkanKernel.Transpose];
            DescriptorSet set = AllocateAndBind(buffers, _setLayouts[VulkanKernel.Transpose]);

            RecordTransfersIn(buffers);
            SyncWithPrevious(RecordKind.Dispatch);

            vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _pipelines[VulkanKernel.Transpose]);
            vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, layout, 0, 1, in set, 0, null);

            var push = new RowPushConstants
            {
                Rows = rows, Cols = cols, Epsilon = 0f, Flags = 0
            };
            vk.CmdPushConstants(
                cmd, layout, ShaderStageFlags.ComputeBit, 0,
                (uint)sizeof(RowPushConstants), &push);

            vk.CmdDispatch(
                cmd, Groups(cols, TransposeTileSize), Groups(rows, TransposeTileSize), 1);
            FinishDispatch(buffers);
        }

        private void RecordCopyBuffer(VulkanBuffer source, VulkanBuffer destination, ulong sizeBytes)
        {
            // A pooled device-tier buffer keeps its payload in staging memory,
            // so the copy has to start there rather than at the VRAM handle.
            Silk.NET.Vulkan.Buffer from = source.UsesStaging
                ? source.StagingHandle
                : source.Handle;

            RecordCopy(from, destination.Handle, sizeBytes);
            source.PendingUpload = false;
            _openRecords++;
        }

        /// <summary>Workgroup count for <paramref name="count"/> items.</summary>
        private static uint Groups(uint count, uint workgroupSize) =>
            (count + workgroupSize - 1) / workgroupSize;

        // ---- staged transfers + barriers -------------------------------------

        /// <summary>
        /// Records the host->device copies for buffers that were uploaded since
        /// the last dispatch, so a dispatch in this batch reads the data the
        /// caller just packed rather than the previous op's contents.
        /// </summary>
        private void RecordTransfersIn(VulkanBuffer[] buffers)
        {
            foreach (VulkanBuffer buffer in buffers)
            {
                if (!buffer.UsesStaging || !buffer.PendingUpload)
                    continue;

                RecordCopy(buffer.StagingHandle, buffer.Handle, buffer.SizeBytes);
                buffer.PendingUpload = false;
            }
        }

        /// <summary>
        /// Records the device->host copies for buffers whose caller asked to read
        /// the result, so by the time the batch's fence signals the staging
        /// memory already holds the result and the host read is a plain memcpy.
        /// </summary>
        private void RecordTransfersOut(VulkanBuffer[] buffers)
        {
            foreach (VulkanBuffer buffer in buffers)
            {
                if (!buffer.UsesStaging || !buffer.ReadbackWanted)
                    continue;

                RecordCopy(buffer.Handle, buffer.StagingHandle, buffer.SizeBytes);
                buffer.ReadbackWanted = false;
            }
        }

        private void RecordCopy(Silk.NET.Vulkan.Buffer source, Silk.NET.Vulkan.Buffer destination, ulong sizeBytes)
        {
            var region = new BufferCopy { SrcOffset = 0, DstOffset = 0, Size = sizeBytes };
            SyncWithPrevious(RecordKind.Transfer);
            _ctx.Vk.CmdCopyBuffer(_cmds[_activeSlot], source, destination, 1, &region);
            _lastRecord = RecordKind.Transfer;
        }

        /// <summary>Closes out a dispatch: counts it, then emits its read-back copies.</summary>
        private void FinishDispatch(VulkanBuffer[] buffers)
        {
            _lastRecord = RecordKind.Dispatch;
            _openRecords++;
            RecordTransfersOut(buffers);
        }

        /// <summary>
        /// Emits the GPU-side barrier that orders the next record after whatever
        /// was recorded last. This is what lets several dispatches share one
        /// command buffer: without it, consecutive dispatches in the same
        /// submission have no defined relationship and a producer/consumer pair
        /// (or two accumulate writes to one buffer) could race.
        /// </summary>
        private void SyncWithPrevious(RecordKind kind)
        {
            if (_lastRecord == RecordKind.None)
                return;

            var barrier = new MemoryBarrier
            {
                SType = StructureType.MemoryBarrier,
                SrcAccessMask = AccessFor(_lastRecord),
                DstAccessMask = AccessFor(kind)
            };

            _ctx.Vk.CmdPipelineBarrier(
                _cmds[_activeSlot],
                StageFor(_lastRecord),
                StageFor(kind),
                DependencyFlags.None,
                1, &barrier,
                0, null,
                0, null);
        }

        private static AccessFlags AccessFor(RecordKind kind) => kind == RecordKind.Dispatch
            ? AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit
            : AccessFlags.TransferReadBit | AccessFlags.TransferWriteBit;

        private static PipelineStageFlags StageFor(RecordKind kind) => kind == RecordKind.Dispatch
            ? PipelineStageFlags.ComputeShaderBit
            : PipelineStageFlags.TransferBit;

        // ---- batch state machine --------------------------------------------

        /// <summary>
        /// Records one operation into the open batch, submits the batch when no
        /// other op is still preparing, and finally blocks until the batch this
        /// dispatch landed in has completed - so the caller may read its result.
        /// </summary>
        private void RunOp(Action record)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(VulkanKernelLauncher));

            Fence? owned = null;
            long seq = -1;
            bool wait = true;

            long t0 = VulkanPhaseProfile.Enabled ? Stopwatch.GetTimestamp() : 0;
            lock (_dispatchGate)
            {
                EnsureOpenLocked();
                record();

                t_recorded = true;
                DispatchCount++;
                if (t_announced)
                {
                    t_announced = false;
                    _pendingPrep--;
                }

                seq = _openSeq;
                if (ShouldCloseLocked())
                {
                    // Nobody else is preparing work: this is the whole batch, so
                    // this thread submits it and waits for the one fence.
                    (seq, owned) = CloseAndSubmitLocked();
                }
                else if (_explicitDepth > 0 && _explicitOwner == Environment.CurrentManagedThreadId)
                {
                    // Our own explicit batch: EndBatch is what submits it, so
                    // waiting here would deadlock against ourselves.
                    wait = false;
                }
            }
            if (VulkanPhaseProfile.Enabled)
                VulkanPhaseProfile.RecordDispatch(Stopwatch.GetTimestamp() - t0);

            long tw0 = VulkanPhaseProfile.Enabled ? Stopwatch.GetTimestamp() : 0;
            if (owned != null)
                Complete(owned, seq);
            else if (wait)
                WaitForCompletion(seq);
            if (VulkanPhaseProfile.Enabled)
                VulkanPhaseProfile.RecordDispatchWait(Stopwatch.GetTimestamp() - tw0);
            if (VulkanPhaseProfile.Enabled)
                VulkanPhaseProfile.RecordOp();

        }

        /// <summary>True when no op is preparing and no explicit scope is open.</summary>
        private bool ShouldCloseLocked() =>
            _explicitDepth == 0 && _pendingPrep == 0 && _openRecords > 0;

        /// <summary>
        /// Ensures a command buffer is open for recording, rotating to the other
        /// (command buffer, fence) slot when the current one is still executing -
        /// that rotation is what lets a batch be built while the previous one runs.
        /// </summary>
        private void EnsureOpenLocked()
        {
            if (_cmdOpen)
                return;

            int slot = _activeSlot;
            if (_slotPending[slot])
            {
                int other = 1 - slot;
                if (!_slotPending[other])
                {
                    slot = other;
                }
                else
                {
                    // Both slots still executing: wait for this one (short - its
                    // batch was submitted already) rather than block every
                    // recording thread on a third wait.
                    WaitSlotFenceLocked(_fences[slot]);
                    _slotPending[slot] = false;
                }
            }

            var vk = _ctx.Vk;
            EnsureFenceLocked(slot);
            vk.ResetFences(_ctx.Device, 1, in _fences[slot]);
            EnsureCommandBufferLocked(slot);

            var begin = new CommandBufferBeginInfo
            {
                SType = StructureType.CommandBufferBeginInfo,
                Flags = CommandBufferUsageFlags.OneTimeSubmitBit
            };
            Result rb = vk.BeginCommandBuffer(_cmds[slot], in begin);
            if (rb != Result.Success)
                throw new InvalidOperationException($"begin cmd: {rb}");

            _activeSlot = slot;
            _cmdOpen = true;
            _openSeq = _nextSeq++;
            _openRecords = 0;
            _lastRecord = RecordKind.None;
        }

        /// <summary>
        /// Ends the open command buffer and submits it. Returns the batch
        /// sequence and its fence: the calling thread owns completion of that
        /// batch and publishes it to everyone waiting.
        /// </summary>
        private (long seq, Fence fence) CloseAndSubmitLocked()
        {
            var vk = _ctx.Vk;
            int slot = _activeSlot;
            var cmd = _cmds[slot];
            var fence = _fences[slot];

            Result er = vk.EndCommandBuffer(cmd);
            if (er != Result.Success)
                throw new InvalidOperationException($"end cmd: {er}");

            var localCmd = cmd;
            var submit = new SubmitInfo
            {
                SType = StructureType.SubmitInfo,
                CommandBufferCount = 1,
                PCommandBuffers = &localCmd
            };
            Result sr = vk.QueueSubmit(_ctx.Queue, 1, in submit, fence);
            if (sr != Result.Success)
                throw new InvalidOperationException($"queue submit: {sr}");

            long seq = _openSeq;
            int records = _openRecords;

            _slotPending[slot] = true;
            _inflightBatches++;
            SubmitCount++;
            if (records > 1)
                BatchedSubmits++;
            if (records > MaxBatchSize)
                MaxBatchSize = records;

            _cmdOpen = false;
            _openSeq = -1;
            _openRecords = 0;
            _lastRecord = RecordKind.None;
            return (seq, fence);
        }

        /// <summary>
        /// Waits for a submitted batch and publishes its completion. Runs
        /// deliberately WITHOUT the dispatch gate held, so other threads can
        /// keep recording while the GPU works.
        /// </summary>
        private void Complete(Fence? fence, long seq)
        {
            if (fence == null)
                return;

            Fence f = fence.Value;
            Result r = _ctx.Vk.WaitForFences(_ctx.Device, 1, in f, true, FenceTimeoutNanos);
            if (r != Result.Success)
                throw new InvalidOperationException($"fence wait: {r}");

            lock (_dispatchGate)
            {
                _inflightBatches--;
                if (seq > _completedSeq)
                    _completedSeq = seq;
                Monitor.PulseAll(_dispatchGate);
            }
        }

        /// <summary>
        /// Blocks until the batch this thread recorded into has completed - it is
        /// submitted and completed by whichever thread closes it.
        /// </summary>
        private void WaitForCompletion(long seq)
        {
            if (seq < 0 || seq <= _completedSeq)
                return;

            long deadline = Environment.TickCount64 + DispatchTimeoutMillis;
            lock (_dispatchGate)
            {
                while (_completedSeq < seq)
                {
                    long remaining = deadline - Environment.TickCount64;
                    if (remaining <= 0)
                        throw new TimeoutException(
                            $"Vulkan batch {seq} did not complete within {DispatchTimeoutMillis} ms.");
                    Monitor.Wait(_dispatchGate, (int)Math.Min(remaining, 50));
                }
            }
        }

        private void WaitSlotFenceLocked(Fence fence)
        {
            Result r = _ctx.Vk.WaitForFences(_ctx.Device, 1, in fence, true, FenceTimeoutNanos);
            if (r != Result.Success)
                throw new InvalidOperationException($"fence wait: {r}");
        }

        private void WaitForInflightDrainLocked()
        {
            long deadline = Environment.TickCount64 + DispatchTimeoutMillis;
            while (_inflightBatches > 0)
            {
                long remaining = deadline - Environment.TickCount64;
                if (remaining <= 0)
                    throw new TimeoutException("In-flight Vulkan batches did not drain in time.");
                Monitor.Wait(_dispatchGate, (int)Math.Min(remaining, 50));
            }
        }

        private void WaitForIdle()
        {
            long deadline = Environment.TickCount64 + DispatchTimeoutMillis;
            lock (_dispatchGate)
            {
                while (_inflightBatches > 0 || _openRecords > 0)
                {
                    long remaining = deadline - Environment.TickCount64;
                    if (remaining <= 0)
                        throw new TimeoutException("Vulkan did not go idle in time.");
                    Monitor.Wait(_dispatchGate, (int)Math.Min(remaining, 50));
                }
            }
        }

        private void FlushCore()
        {
            Fence? owned = null;
            long seq = -1;
            bool wait = false;

            lock (_dispatchGate)
            {
                if (_explicitDepth > 0 && _explicitOwner == Environment.CurrentManagedThreadId)
                    return; // inside our own scope: EndBatch is the completer

                if (_openRecords > 0)
                {
                    seq = _openSeq;
                    if (_pendingPrep == 0)
                        (seq, owned) = CloseAndSubmitLocked();
                    else
                        wait = true;
                }
                else if (_inflightBatches > 0)
                {
                    wait = true;
                }
            }

            if (owned != null)
                Complete(owned, seq);
            else if (wait)
                WaitForIdle();
        }

        private void EnsureFenceLocked(int slot)
        {
            if (_fences[slot].Handle != 0)
                return;

            var fenceInfo = new FenceCreateInfo { SType = StructureType.FenceCreateInfo };
            _ctx.Vk.CreateFence(_ctx.Device, in fenceInfo, null, out _fences[slot]);
        }

        private void EnsureCommandBufferLocked(int slot)
        {
            if (_cmds[slot].Handle != 0)
            {
                _ctx.Vk.ResetCommandBuffer(_cmds[slot], CommandBufferResetFlags.None);
                return;
            }

            var cmdAlloc = new CommandBufferAllocateInfo
            {
                SType = StructureType.CommandBufferAllocateInfo,
                CommandPool = _ctx.CommandPool,
                Level = CommandBufferLevel.Primary,
                CommandBufferCount = 1
            };
            Result r = _ctx.Vk.AllocateCommandBuffers(_ctx.Device, in cmdAlloc, out _cmds[slot]);
            if (r != Result.Success)
                throw new InvalidOperationException($"alloc cmd: {r}");
        }

        // ---- descriptor sets -------------------------------------------------

        private unsafe DescriptorSet AllocateAndBind(VulkanBuffer[] buffers, DescriptorSetLayout setLayout)
        {
            // Fast path: reuse a cached descriptor set when the exact same
            // buffer handles were bound last time for this layout. Handle
            // identity implies identical sizes (pool buckets are power-of-two
            // and buffers are never destroyed), so no re-write is needed -
            // this is what removes vkUpdateDescriptorSets from the hot path.
            DescriptorSetCacheKey key = new(buffers);
            if (_setCache.TryGetValue(setLayout.Handle, out var perLayout) &&
                perLayout.TryGetValue(key, out CachedDescriptorSet cached))
            {
                return cached.Set;
            }

            if (!TryAllocateSet(setLayout, out DescriptorSet set, out bool fromSpool))
                throw new InvalidOperationException("alloc set: descriptor pool exhausted.");

            UpdateBindings(set, buffers);

            if (!_setCache.TryGetValue(setLayout.Handle, out perLayout))
            {
                perLayout = new Dictionary<DescriptorSetCacheKey, CachedDescriptorSet>();
                _setCache[setLayout.Handle] = perLayout;
            }
            perLayout[key] = new CachedDescriptorSet(set, fromSpool);
            return set;
        }

        private unsafe bool TryAllocateSet(
            DescriptorSetLayout setLayout,
            out DescriptorSet set,
            out bool fromSpool)
        {
            var vk = _ctx.Vk;
            var allocInfo = new DescriptorSetAllocateInfo
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = _pool,
                DescriptorSetCount = 1,
                PSetLayouts = &setLayout
            };

            fromSpool = false;
            if (vk.AllocateDescriptorSets(_ctx.Device, in allocInfo, out set) == Result.Success)
                return true;

            // Recycling frees sets that in-flight command buffers still
            // reference, so it is only legal with nothing submitted. While a
            // batch is in flight the overflow pool absorbs the extra tuples.
            if (_inflightBatches == 0)
            {
                vk.ResetDescriptorPool(_ctx.Device, _pool, 0);
                PurgeCacheLocked(keepSpool: _spool != null);
                DescriptorPoolRecycles++;
                if (vk.AllocateDescriptorSets(_ctx.Device, in allocInfo, out set) == Result.Success)
                    return true;
            }
            else
            {
                EnsureSpoolLocked();
                allocInfo.DescriptorPool = _spool.Value;
                fromSpool = true;
                if (vk.AllocateDescriptorSets(_ctx.Device, in allocInfo, out set) == Result.Success)
                    return true;
            }

            // Pathological: let the in-flight batches finish, then recycle both.
            WaitForInflightDrainLocked();
            vk.ResetDescriptorPool(_ctx.Device, _pool, 0);
            if (_spool != null && _spool.Value.Handle != 0)
                vk.ResetDescriptorPool(_ctx.Device, _spool.Value, 0);
            PurgeCacheLocked(keepSpool: false);
            DescriptorPoolRecycles++;

            fromSpool = false;
            allocInfo.DescriptorPool = _pool;
            if (vk.AllocateDescriptorSets(_ctx.Device, in allocInfo, out set) == Result.Success)
                return true;

            set = default;
            return false;
        }

        private unsafe void EnsureSpoolLocked()
        {
            if (_spool != null)
                return;

            const uint spoolSets = 4096;
            var size = new DescriptorPoolSize
            {
                Type = DescriptorType.StorageBuffer,
                DescriptorCount = 4 * spoolSets
            };
            var info = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = spoolSets,
                PoolSizeCount = 1,
                PPoolSizes = &size
            };
            if (_ctx.Vk.CreateDescriptorPool(_ctx.Device, in info, null, out DescriptorPool spool)
                != Result.Success)
            {
                throw new InvalidOperationException("overflow descriptor pool failed.");
            }
            _spool = spool;
        }

        /// <summary>
        /// Drops the descriptor-set cache when it is safe: nothing recorded and
        /// nothing in flight, so no live command buffer can reference a set.
        /// Called between training steps via TrimIdleMemory so variable shapes
        /// cannot pin an ever-growing set of binding tuples.
        /// </summary>
        public void TrimDescriptorCache()
        {
            if (_disposed)
                return;

            lock (_dispatchGate)
            {
                if (_openRecords > 0 || _inflightBatches > 0 || _explicitDepth > 0)
                    return;
                if (_setCache.Count == 0)
                    return;

                _setCache.Clear();
                if (_spool != null && _spool.Value.Handle != 0)
                {
                    _ctx.Vk.DestroyDescriptorPool(_ctx.Device, _spool.Value, null);
                    _spool = null;
                }
            }
        }

        /// <summary>
        /// Drops cached sets that a pool reset invalidated. When
        /// <paramref name="keepSpool"/> is set only the primary pool's sets go;
        /// otherwise every cached set is stale.
        /// </summary>
        private void PurgeCacheLocked(bool keepSpool)
        {
            if (_setCache.Count == 0)
                return;
            if (!keepSpool)
            {
                _setCache.Clear();
                return;
            }

            var emptyLayouts = new List<ulong>();
            foreach (var entry in _setCache)
            {
                var perLayout = entry.Value;
                var stale = new List<DescriptorSetCacheKey>();
                foreach (var cached in perLayout)
                {
                    if (!cached.Value.FromSpool)
                        stale.Add(cached.Key);
                }
                foreach (var key in stale)
                    perLayout.Remove(key);
                if (perLayout.Count == 0)
                    emptyLayouts.Add(entry.Key);
            }
            foreach (var layout in emptyLayouts)
                _setCache.Remove(layout);
        }

        private unsafe void UpdateBindings(DescriptorSet set, VulkanBuffer[] buffers)
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

            // Pin both fixed arrays in a single scope before populating internal pointers
            fixed (DescriptorBufferInfo* ip = infos)
            fixed (WriteDescriptorSet* w = writes)
            {
                for (int i = 0; i < writes.Length; i++)
                {
                    writes[i].PBufferInfo = &ip[i];
                }

                vk.UpdateDescriptorSets(_ctx.Device, (uint)writes.Length, w, 0, null);
            }
        }

        // ---- teardown --------------------------------------------------------

        public void Dispose()
        {
            if (_disposed)
                return;

            // Whatever was recorded but not sent has to go out before the
            // command buffers and descriptor sets it references are destroyed.
            try
            {
                FlushCore();
            }
            catch (TimeoutException)
            {
                // The device never reported back; teardown must not throw.
            }

            _disposed = true;

            for (int slot = 0; slot < SlotCount; slot++)
            {
                if (_cmds[slot].Handle != 0)
                {
                    _ctx.Vk.FreeCommandBuffers(_ctx.Device, _ctx.CommandPool, 1, in _cmds[slot]);
                    _cmds[slot] = default;
                }
                if (_fences[slot].Handle != 0)
                {
                    _ctx.Vk.DestroyFence(_ctx.Device, _fences[slot], null);
                    _fences[slot] = default;
                }
            }

            foreach (var p in _pipelines.Values)
                _ctx.Vk.DestroyPipeline(_ctx.Device, p, null);
            foreach (var l in _layouts.Values)
                _ctx.Vk.DestroyPipelineLayout(_ctx.Device, l, null);
            foreach (var s in _setLayouts.Values)
                _ctx.Vk.DestroyDescriptorSetLayout(_ctx.Device, s, null);
            if (_spool != null && _spool.Value.Handle != 0)
                _ctx.Vk.DestroyDescriptorPool(_ctx.Device, _spool.Value, null);
            if (_pool.Handle != 0)
                _ctx.Vk.DestroyDescriptorPool(_ctx.Device, _pool, null);
        }
    }
}
