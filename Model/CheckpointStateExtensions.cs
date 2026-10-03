using System;
using System.Collections.Generic;
using System.IO;

namespace SimpleTransformer.Model
{
    /// <summary>
    /// Byte formats for the v5 checkpoint training-state trailer: the optimizer
    /// snapshot and the durable dropout-site state. Kept out of the optimizer
    /// and <see cref="DropoutSite"/> classes so they stay format-agnostic (see
    /// <see cref="OptimizerState"/> docs), and out of the header/weight logic in
    /// <see cref="TransformerModel"/> so schema versions stay readable in one
    /// place.
    /// </summary>
    public static class CheckpointStateExtensions
    {
        /// <summary>
        /// One dropout site's durable state as stored in the trailer: the
        /// architecture-derived key (used for matching on load), the durable
        /// salt, and the optimizer step the site was last keyed for.
        /// </summary>
        public readonly struct DropoutSiteState
        {
            public DropoutSiteState(string key, ulong salt, long step)
            {
                Key = key;
                Salt = salt;
                Step = step;
            }

            public string Key { get; }
            public ulong Salt { get; }
            public long Step { get; }
        }

        /// <summary>
        /// Writes an optimizer snapshot: kind, bias-correction step, then one
        /// entry per parameter (name + optional moment tensors). The layouts of
        /// AdamW and SGD are not interchangeable; the kind field is what lets
        /// <see cref="IOptimizer.ImportState"/> reject a cross-load.
        /// </summary>
        public static void WriteOptimizerState(BinaryWriter writer, OptimizerState state)
        {
            ArgumentNullException.ThrowIfNull(writer);
            ArgumentNullException.ThrowIfNull(state);

            writer.Write(state.Kind);
            writer.Write(state.StepCount);
            writer.Write(state.Parameters.Count);

            foreach (OptimizerParamState entry in state.Parameters)
            {
                writer.Write(entry.Name);
                WriteOptionalTensor(writer, entry.FirstMoment);
                WriteOptionalTensor(writer, entry.SecondMoment);
                WriteOptionalTensor(writer, entry.Velocity);
            }
        }

        /// <summary>
        /// Reads an optimizer snapshot written by
        /// <see cref="WriteOptimizerState"/>. Corrupt counts, unknown kinds and
        /// negative steps are rejected before any allocation happens.
        /// </summary>
        public static OptimizerState ReadOptimizerState(BinaryReader reader)
        {
            ArgumentNullException.ThrowIfNull(reader);

            int kind = reader.ReadInt32();
            if (kind != OptimizerStateKinds.AdamW && kind != OptimizerStateKinds.Sgd)
            {
                throw new InvalidDataException(
                    $"Corrupt checkpoint: unknown optimizer kind ({kind}).");
            }

            int stepCount = reader.ReadInt32();
            if (stepCount < 0)
            {
                throw new InvalidDataException(
                    $"Corrupt checkpoint: negative optimizer step ({stepCount}).");
            }

            int count = reader.ReadInt32();
            if (count < 0 || count > 1_000_000)
            {
                throw new InvalidDataException(
                    $"Corrupt checkpoint: invalid optimizer state entry count ({count}).");
            }

            var entries = new List<OptimizerParamState>(count);
            for (int i = 0; i < count; i++)
            {
                string name = reader.ReadString();
                TensorData? firstMoment = ReadOptionalTensor(reader);
                TensorData? secondMoment = ReadOptionalTensor(reader);
                TensorData? velocity = ReadOptionalTensor(reader);

                entries.Add(new OptimizerParamState
                {
                    Name = name,
                    FirstMoment = firstMoment,
                    SecondMoment = secondMoment,
                    Velocity = velocity
                });
            }

            return new OptimizerState
            {
                Kind = kind,
                StepCount = stepCount,
                Parameters = entries
            };
        }

        /// <summary>
        /// Writes the durable state of every dropout site: count, then
        /// (key, salt, keyed step) per site. Per-item mask layers are NOT
        /// written - they are derived from (salt, step, batch item) on use, so
        /// the trailer stays architecture-only regardless of batch size.
        /// </summary>
        public static void WriteDropoutSiteStates(BinaryWriter writer, IReadOnlyList<DropoutSite> sites)
        {
            ArgumentNullException.ThrowIfNull(writer);
            ArgumentNullException.ThrowIfNull(sites);

            writer.Write(sites.Count);
            for (int i = 0; i < sites.Count; i++)
            {
                writer.Write(sites[i].Key);
                writer.Write(sites[i].Salt);
                writer.Write(sites[i].Step);
            }
        }

        /// <summary>
        /// Reads the dropout-site states written by
        /// <see cref="WriteDropoutSiteStates"/>. Matching against the live
        /// model happens in <c>TransformerModel</c>, which owns the site list.
        /// </summary>
        public static List<DropoutSiteState> ReadDropoutSiteStates(BinaryReader reader)
        {
            ArgumentNullException.ThrowIfNull(reader);

            int count = reader.ReadInt32();
            if (count < 0 || count > 100_000)
            {
                throw new InvalidDataException(
                    $"Corrupt checkpoint: invalid dropout site count ({count}).");
            }

            var states = new List<DropoutSiteState>(count);
            for (int i = 0; i < count; i++)
            {
                string key = reader.ReadString();
                ulong salt = reader.ReadUInt64();
                long step = reader.ReadInt64();
                states.Add(new DropoutSiteState(key, salt, step));
            }

            return states;
        }

        private static void WriteOptionalTensor(BinaryWriter writer, TensorData? tensor)
        {
            bool hasValue = tensor != null;
            writer.Write(hasValue);
            if (hasValue)
            {
                TransformerModel.WriteTensor(writer, tensor!);
            }
        }

        private static TensorData? ReadOptionalTensor(BinaryReader reader)
            => reader.ReadBoolean() ? TransformerModel.ReadTensorOptimized(reader) : null;
    }
}
