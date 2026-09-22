using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using SimpleTransformer.AccelerationEngine;

namespace SimpleTransformer.Model
{
    public class TensorWorkspace : IDisposable
    {
        private readonly ConcurrentDictionary<TensorShapeKey, ConcurrentBag<TensorBase>> _pool = new();
        private readonly ConcurrentBag<TensorBase> _activeTensors = new();
        private bool _isDisposed;

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
                tensor.Clear();
            }
            else
            {
                tensor = factory(shape.ToArray());
            }

            // Track active allocation for automatic sweep on Reset()
            _activeTensors.Add(tensor);
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
        /// Releases a tensor back to the pool for reuse.
        /// </summary>
        public void Release(TensorBase? tensor)
        {
            if (tensor == null || _isDisposed) return;

            var key = new TensorShapeKey(tensor.Shape);
            var bag = _pool.GetOrAdd(key, _ => new ConcurrentBag<TensorBase>());
            bag.Add(tensor);
        }
        
        /// <summary>
        /// Reclaims all borrowed tensors from the current pass, clears their memory,
        /// and returns them to the pool for reuse in the next step.
        /// </summary>
        public void Reset()
        {
            ThrowIfDisposed();

            while (_activeTensors.TryTake(out var tensor))
            {
                // Reset data state so previous intermediate results don't bleed over
                tensor.Clear();

                // Recycle into the pooled bags by shape key
                var key = new TensorShapeKey(tensor.Shape);
                var bag = _pool.GetOrAdd(key, _ => new ConcurrentBag<TensorBase>());
                bag.Add(tensor);
            }
        }        

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

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