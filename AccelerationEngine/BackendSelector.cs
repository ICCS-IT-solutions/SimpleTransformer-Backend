using System;
using System.Numerics;

namespace SimpleTransformer.AccelerationEngine
{
    public static class BackendSelector
    {
        public enum BackendType
        {
            CpuReference,
            CpuSimd,
            CpuAvx2,
            CpuAvx512,
            GpuVulkan,
            Auto // Automatically select the best available backend based on the current hardware and environment
        }

        public static IAccelerationBackend SelectBackend(BackendType type)
        {
            return type switch
            {
                BackendType.CpuReference => new CpuReference.CpuReferenceBackend(),
                BackendType.CpuSimd => new CpuSimd.CpuSimdBackend(),
                BackendType.CpuAvx2 => new CpuAvx2.CpuAvx2Backend(),
                BackendType.CpuAvx512 => new CpuAvx512.CpuAvx512Backend(),
                BackendType.GpuVulkan => new GpuVulkan.GpuVulkanBackend(),
                BackendType.Auto => SelectDefault(),
                _ => throw new ArgumentException($"Unsupported backend type: {type}"),
            };
        }

        private static readonly object _gate = new();
        private static IAccelerationBackend? _default;

        /// <summary>
        /// Returns (creating and caching on first use) the default backend chosen by the
        /// <see cref="BackendType.Auto"/> strategy. The default is process-wide and thread-safe.
        /// </summary>
        public static IAccelerationBackend SelectDefault()
        {
            // Fast path: the default backend is stable once created.
            var current = _default;
            if (current != null)
                return current;

            lock (_gate)
            {
                if (_default == null)
                {
                    // Auto strategy:
                    //  - Prefer the GPU (Vulkan) backend when a device is available.
                    //  - Otherwise prefer the SIMD backend when the JIT reports hardware
                    //    vector support, falling back to the pure managed reference backend.
                    // Note: CpuAvx2/CpuAvx512 are stubs and are never auto-selected.
                    _default = TryCreate(BackendType.GpuVulkan)
                        ?? (Vector.IsHardwareAccelerated
                            ? (IAccelerationBackend)new CpuSimd.CpuSimdBackend()
                            : new CpuReference.CpuReferenceBackend());
                }

                return _default;
            }
        }

        /// <summary>
        /// Probes a backend type by constructing it and checking
        /// <see cref="IAccelerationBackend.IsAvailable"/>. Returns null when
        /// construction throws or the backend reports itself unavailable.
        /// </summary>
        private static IAccelerationBackend? TryCreate(BackendType type)
        {
            try
            {
                var backend = SelectBackend(type);
                if (backend.IsAvailable)
                    return backend;

                backend.Dispose();
            }
            catch
            {
                // Probe failure — fall through to the next Auto candidate.
            }

            return null;
        }

        /// <summary>
        /// Overrides the process-wide default backend (used by the Auto strategy and by
        /// <see cref="Model.TensorWorkspace"/> when no explicit backend is provided).
        /// </summary>
        public static void SetDefault(IAccelerationBackend backend)
        {
            if (backend == null) throw new ArgumentNullException(nameof(backend));

            lock (_gate)
            {
                _default = backend;
            }
        }

        /// <summary>
        /// Clears the cached default so the next <see cref="SelectDefault"/> call re-runs the Auto strategy.
        /// </summary>
        public static void ResetDefault()
        {
            lock (_gate)
            {
                _default = null;
            }
        }
    }
}