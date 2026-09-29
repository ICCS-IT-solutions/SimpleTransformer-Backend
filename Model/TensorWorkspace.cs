using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    public class TensorWorkspace : IDisposable
    {
        // Footprint of a pooled tensor in bytes. Buffer rather than Data: a
        // TensorView shares its parent's buffer, so charging the whole backing
        // array over-counts a view rather than under-counting it. For a cap
        // that is the safe direction to err in.
        private static long FootprintOf(TensorBase tensor)
        {
            float[] buffer = tensor.Buffer;
            return buffer == null ? 0L : (long)buffer.Length * sizeof(float);
        }

        // Reference-identity wrapper so the active set tracks tensor identity,
        // not value equality: two distinct tensors with equal shapes must not
        // compare equal, or releasing one would unregister the other.
        private readonly struct TensorEntry : IEquatable<TensorEntry>
        {
            public readonly TensorBase Tensor;

            public TensorEntry(TensorBase tensor) => Tensor = tensor;

            public bool Equals(TensorEntry other) => ReferenceEquals(Tensor, other.Tensor);
            public override bool Equals(object? obj) => obj is TensorEntry other && Equals(other);
            public override int GetHashCode() => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(Tensor);
        }

        private readonly ConcurrentDictionary<TensorShapeKey, ConcurrentBag<TensorBase>> _pool = new();
        // Tracks live borrows so Release can reject foreign or double-released
        // tensors (which would otherwise alias one buffer to two borrowers).
        // A concurrent dictionary doubles as the set: the value is unused.
        private readonly ConcurrentDictionary<TensorEntry, byte> _activeTensors = new();
        // Incremental accounting of pooled bytes and tensors. Maintained at
        // every pool insert/take rather than recomputed by walking the bags, so
        // the cap check below stays O(1) on the per-step hot path.
        private long _retainedBytes;
        private int _pooledTensorCount;
        // Serialises TrimRetainedTo against itself. The ConcurrentBag drains
        // are individually safe, but the evict-largest-first decision is a
        // read-modify-write over the whole pool and must not interleave.
        private readonly object _trimGate = new();
        // Cached pool cap and its refresh counter. Static because the cap
        // derives from process-wide config, not per-workspace state.
        private static long _cachedPoolCapBytes;
        private static long _borrowCount;
        private bool _isDisposed;

        /// <summary>
        /// Bytes currently retained by idle pooled tensors. Incremented when a
        /// tensor joins a bag and decremented when one leaves, so it tracks
        /// what the pool is pinning on the managed heap.
        /// </summary>
        public long RetainedBytes => Interlocked.Read(ref _retainedBytes);

        /// <summary>Number of idle tensors currently pooled for reuse.</summary>
        public int PooledTensorCount => Volatile.Read(ref _pooledTensorCount);

        /// <summary>Distinct activation shapes the pool is holding buckets for.</summary>
        public int PooledShapeCount => _pool.Count;

        /// <summary>
        /// Byte cap for the idle pool, derived from the configured quota. 0 when
        /// uncapped. This is the unconditional half of the memory fix: the pool
        /// previously had no bound at all, so a long run accumulated one pooled
        /// tensor per activation shape it ever saw and the valve above it only
        /// ever had to clean up after the fact.
        /// <para>
        /// Cached rather than recomputed per borrow: Borrow is called many times
        /// per training step across every layer, and
        /// <see cref="GC.GetGCMemoryInfo"/> is not free enough to sit on that
        /// path. The cache is refreshed on the same cadence the valve samples
        /// (<see cref="MemoryPressureSettings.CheckEveryNSteps"/>) so an edited
        /// config still takes effect within a few steps, without paying for the
        /// lookup on every single borrow.
        /// </para>
        /// </summary>
        private static long ResolvePoolCapBytes()
        {
            if (!MemoryPressureSettings.Enabled)
                return 0;

            double fraction = MemoryPressureSettings.WorkspaceCapFractionOfQuota;
            if (fraction <= 0.0)
                return 0;

            // Throttle: borrowCount is a plain counter, so concurrent borrows
            // may both recompute on the same boundary. That is harmless - they
            // compute the same value and the assignment is idempotent.
            if (Interlocked.Increment(ref _borrowCount) %
                Math.Max(1, MemoryPressureSettings.CheckEveryNSteps) != 0)
            {
                return Volatile.Read(ref _cachedPoolCapBytes);
            }

            long quota = MemoryPressureValve.ResolveQuotaBytes(
                MemoryPressureSettings.MaxQuotaBytes,
                MemoryPressureSettings.MinQuotaBytes,
                GC.GetGCMemoryInfo().TotalAvailableMemoryBytes);

            long cap = quota <= 0
                ? 0
                : (long)(quota * Math.Clamp(fraction, 0.0, 1.0));

            Volatile.Write(ref _cachedPoolCapBytes, cap);
            return cap;
        }

        /// <summary>
        /// The acceleration backend used for tensor math performed against tensors
        /// borrowed from this workspace. Defaults to the process-wide auto-selected
        /// backend (see <see cref="BackendSelector.SelectDefault"/>) when not provided.
        /// </summary>
        public IAccelerationBackend Backend { get; }

        public TensorWorkspace()
            : this(BackendSelector.SelectDefault())
        {
        }

        /// <summary>
        /// Creates a workspace that executes tensor math through the given backend.
        /// </summary>
        public TensorWorkspace(IAccelerationBackend? backend)
        {
            Backend = backend ?? BackendSelector.SelectDefault();
        }

        public TensorWorkspace(int capacityHint)
            : this(BackendSelector.SelectDefault())
        {
            _pool = new ConcurrentDictionary<TensorShapeKey, ConcurrentBag<TensorBase>>(
                Environment.ProcessorCount, capacityHint);
        }

        /// <summary>
        /// Borrows a tensor matching the shape, or creates a new one using <paramref name="factory"/> if unavailable.
        /// </summary>
        public TensorBase Borrow(ReadOnlySpan<int> shape, Func<int[], TensorBase> factory)
        {
            ThrowIfDisposed();
            var key = new TensorShapeKey(shape);

            TensorBase tensor;
            if (_pool.TryGetValue(key, out var bag) && bag.TryTake(out tensor))
            {
                // Leaving the pool: the bytes are now owned by the caller again
                // and must stop counting against the cap.
                Interlocked.Add(ref _retainedBytes, -FootprintOf(tensor));
                Interlocked.Decrement(ref _pooledTensorCount);
                tensor.Clear();
            }
            else
            {
                tensor = factory(shape.ToArray());
            }

            // Track active allocation for automatic sweep on Reset()
            _activeTensors.TryAdd(new TensorEntry(tensor), 0);

            // Enforce the pool cap on the way out rather than only under
            // pressure. Recomputed per borrow because the quota is read from
            // config, so a changed config takes effect without a restart.
            // Safe here: the tensor just handed to the caller is in
            // _activeTensors, which TrimRetainedTo never evicts from.
            long cap = ResolvePoolCapBytes();
            if (cap > 0 && Interlocked.Read(ref _retainedBytes) > cap)
                TrimRetainedTo(cap);

            return tensor;
        }

        /// <summary>
        /// Helper overload for 2D Tensors [rows, cols]
        /// </summary>
        public TensorBase Borrow(int rows, int cols, Func<int[], TensorBase> factory)
            => Borrow(stackalloc int[] { rows, cols }, factory);

        /// <summary>
        /// Helper overload for 3D Tensors [layers, rows, cols]
        /// </summary>
        public TensorBase Borrow(int layers, int rows, int cols, Func<int[], TensorBase> factory)
            => Borrow(stackalloc int[] { layers, rows, cols }, factory);

        /// <summary>
        /// Borrows or allocates a 1D Vector [length].
        /// </summary>
        public TensorBase Borrow1D(int length, Func<int[], TensorBase>? factory = null)
        {
            factory ??= shape => new Tensor(shape[0]);
            return Borrow(stackalloc int[] { length }, factory);
        }

        /// <summary>
        /// Borrows or allocates a 2D Matrix [rows, cols].
        /// </summary>
        public TensorBase Borrow2D(int rows, int cols, Func<int[], TensorBase>? factory = null)
        {
            factory ??= shape => new Tensor(shape[0], shape[1]);
            return Borrow(stackalloc int[] { rows, cols }, factory);
        }

        /// <summary>
        /// Borrows or allocates a 3D Tensor [layers/batch, rows, cols].
        /// </summary>
        public TensorBase Borrow3D(int layers, int rows, int cols, Func<int[], TensorBase>? factory = null)
        {
            factory ??= shape => new Tensor(shape[0], shape[1], shape[2]);
            return Borrow(stackalloc int[] { layers, rows, cols }, factory);
        }

        /// <summary>
        /// Borrows or allocates a 4D Tensor [batch, heads, sequence, dim] (useful for Multi-Head Attention).
        /// </summary>
        public TensorBase Borrow4D(int batch, int heads, int sequence, int dim, Func<int[], TensorBase>? factory = null)
        {
            factory ??= shape => new Tensor(shape[0], shape[1], shape[2], shape[3]);
            return Borrow(stackalloc int[] { batch, heads, sequence, dim }, factory);
        }

        /// <summary>
        /// Borrows a tensor matching the shape and layout of a reference tensor.
        /// </summary>
        public TensorBase BorrowLike(TensorBase reference, Func<int[], TensorBase>? factory = null)
        {
            factory ??= shape => new Tensor(shape);
            return Borrow(reference.Shape, factory);
        }
        /// <summary>
        /// Releases a tensor back to the pool for reuse. Tensors never borrowed
        /// from this workspace (layer outputs already released, foreign views)
        /// are ignored so a double-release cannot stock the pool with the same
        /// instance twice - that aliasing would let two borrowers silently share
        /// one buffer and corrupt each other's activations.
        /// </summary>
        public void Release(TensorBase? tensor)
        {
            if (tensor == null || _isDisposed) return;

            if (!_activeTensors.TryRemove(new TensorEntry(tensor), out _))
                return;

            var key = new TensorShapeKey(tensor.Shape);
            var bag = _pool.GetOrAdd(key, _ => new ConcurrentBag<TensorBase>());
            bag.Add(tensor);
            Interlocked.Add(ref _retainedBytes, FootprintOf(tensor));
            Interlocked.Increment(ref _pooledTensorCount);
        }
        
        /// <summary>
        /// Reclaims all borrowed tensors from the current pass, clears their memory,
        /// and returns them to the pool for reuse in the next step.
        /// <para>
        /// This is the bulk path where a whole step's activations land back in the
        /// pool at once, so the cap is enforced here as well as in
        /// <see cref="Borrow"/>. Enforcing it only on the borrow side would let
        /// the pool sit above its cap for the whole of a step.
        /// </para>
        /// </summary>
        public void Reset()
        {
            ThrowIfDisposed();

            foreach (var pair in _activeTensors)
            {
                if (!_activeTensors.TryRemove(pair.Key, out _))
                    continue;

                var tensor = pair.Key.Tensor;

                // Reset data state so previous intermediate results don't bleed over
                tensor.Clear();

                // Recycle into the pooled bags by shape key
                var key = new TensorShapeKey(tensor.Shape);
                var bag = _pool.GetOrAdd(key, _ => new ConcurrentBag<TensorBase>());
                bag.Add(tensor);
                Interlocked.Add(ref _retainedBytes, FootprintOf(tensor));
                Interlocked.Increment(ref _pooledTensorCount);
            }

            // A step's worth of activations has just landed in the pool at
            // once, so this is the cheapest moment to hold it to the cap.
            long cap = ResolvePoolCapBytes();
            if (cap > 0 && Interlocked.Read(ref _retainedBytes) > cap)
                TrimRetainedTo(cap);
        }        

        /// <summary>
        /// Evicts idle pooled tensors, largest first, until the pool retains at
        /// most <paramref name="targetBytes"/>. A target of 0 drains the pool
        /// completely (see <see cref="ReleasePooledMemory"/>). Returns how many
        /// were dropped.
        /// <para>
        /// Only tensors sitting in the bags are candidates - everything in
        /// <c>_activeTensors</c> is currently borrowed and is never touched, so
        /// this is safe to call from inside the training hot path. Eviction
        /// deliberately does not call <see cref="TensorBase.Dispose"/>:
        /// for a pooled <see cref="Tensor"/> that only zeroes the buffer, which
        /// costs a full memory pass to achieve nothing. Dropping the reference
        /// is what returns the array to the GC.
        /// </para>
        /// <para>
        /// Largest-first rather than arbitrary so the fewest tensors are dropped
        /// to free a given number of bytes - evicting one 64 MiB activation beats
        /// evicting two hundred 256 KiB ones, which would cost far more
        /// re-allocation on the steps that follow.
        /// </para>
        /// </summary>
        public int TrimRetainedTo(long targetBytes)
        {
            // targetBytes == 0 is a legitimate request to drain the pool, so it
            // is NOT an early-out here. The "nothing to do" case is a target at
            // or above what is already retained, checked under the gate below.
            if (_isDisposed)
                return 0;

            lock (_trimGate)
            {
                long retained = Interlocked.Read(ref _retainedBytes);
                if (retained <= targetBytes)
                    return 0;

                // Snapshot every bag by draining it, so the eviction set can be
                // decided before anything is handed back.
                var keepers = new Dictionary<TensorShapeKey, List<TensorBase>>();
                var candidates = new List<(TensorBase Tensor, long Bytes)>();

                foreach (var pair in _pool)
                {
                    var bag = pair.Value;
                    var drained = new List<TensorBase>();

                    while (bag.TryTake(out var pooled))
                        drained.Add(pooled);

                    if (drained.Count == 0)
                        continue;

                    keepers[pair.Key] = drained;

                    foreach (var pooled in drained)
                        candidates.Add((pooled, FootprintOf(pooled)));
                }

                candidates.Sort((a, b) => b.Bytes.CompareTo(a.Bytes));

                var evicted = new HashSet<TensorBase>(ReferenceEqualityComparer.Instance);
                long freed = 0;

                foreach (var candidate in candidates)
                {
                    if (retained - freed <= targetBytes)
                        break;

                    freed += candidate.Bytes;
                    evicted.Add(candidate.Tensor);
                }

                if (evicted.Count > 0)
                {
                    Interlocked.Add(ref _retainedBytes, -freed);
                    Interlocked.Add(ref _pooledTensorCount, -evicted.Count);
                }

                // Hand the survivors back, preserving their per-shape grouping.
                foreach (var pair in keepers)
                {
                    var bag = _pool.GetOrAdd(pair.Key, _ => new ConcurrentBag<TensorBase>());

                    foreach (var tensor in pair.Value)
                    {
                        if (!evicted.Contains(tensor))
                            bag.Add(tensor);
                    }

                    // Drop buckets that ended up empty so the shape key index
                    // does not accumulate one dead entry per shape ever seen.
                    if (bag.IsEmpty)
                        _pool.TryRemove(pair.Key, out _);
                }

                return evicted.Count;
            }
        }

        /// <summary>
        /// Drops every idle pooled tensor, returning its memory to the GC. Used
        /// on model unload; distinct from <see cref="Reset"/>, which keeps the
        /// pool warm for the next step. Returns the number of tensors dropped.
        /// </summary>
        public int ReleasePooledMemory()
        {
            if (_isDisposed)
                return 0;

            return TrimRetainedTo(0);
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            _activeTensors.Clear();

            foreach (var key in _pool.Keys)
            {
                if (_pool.TryRemove(key, out var bag))
                {
                    while (bag.TryTake(out var tensor))
                    {
                        tensor.Dispose();
                    }
                }
            }

            // Everything the pool held is gone; zero the accounting so a stale
            // reading cannot make a disposed workspace look like it is still
            // pinning memory.
            Interlocked.Exchange(ref _retainedBytes, 0);
            Interlocked.Exchange(ref _pooledTensorCount, 0);
        }

        private void ThrowIfDisposed()
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(TensorWorkspace));
        }

        /// <summary>
        /// Allocation-free shape key using Structural Equality
        /// </summary>
        internal readonly struct TensorShapeKey : IEquatable<TensorShapeKey>
        {
            private readonly int[] _dimensions;
            private readonly int _hashCode;

            public TensorShapeKey(ReadOnlySpan<int> shape)
            {
                _dimensions = shape.ToArray();

                // Compute hash code across arbitrary dimension lengths
                var hash = new HashCode();
                for (int i = 0; i < shape.Length; i++)
                {
                    hash.Add(shape[i]);
                }
                _hashCode = hash.ToHashCode();
            }

            public bool Equals(TensorShapeKey other)
            {
                if (_hashCode != other._hashCode) return false;
                return _dimensions.AsSpan().SequenceEqual(other._dimensions);
            }         

            public override bool Equals(object? obj) => obj is TensorShapeKey other && Equals(other);
            public override int GetHashCode() => _hashCode;

        }
    }
}