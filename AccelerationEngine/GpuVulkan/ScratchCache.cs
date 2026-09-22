using System;
using System.Collections.Concurrent;

namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    /// <summary>
    /// Power-of-two pooled float[] scratch cache shared by all backend
    /// instances. Eliminates per-op managed allocations for pack/unpack
    /// staging arrays. Thread-safe.
    /// </summary>
    internal static class ScratchCache
    {
        private static readonly ConcurrentDictionary<int, ConcurrentBag<float[]>> _bags = new();

        public static float[] Rent(int count)
        {
            int bucket = BucketFor(count);
            if (_bags.TryGetValue(bucket, out var bag) && bag.TryTake(out var array))
                return array;
            return new float[bucket];
        }

        public static void Return(float[] array)
        {
            int bucket = BucketFor(array.Length);
            if (array.Length != bucket)
                return; // foreign array, let GC handle it
            var bag = _bags.GetOrAdd(bucket, _ => new ConcurrentBag<float[]>());
            if (bag.Count < 16)
                bag.Add(array);
        }

        private static int BucketFor(int count)
        {
            int bucket = 256;
            while (bucket < count)
                bucket <<= 1;
            return bucket;
        }
    }
}
