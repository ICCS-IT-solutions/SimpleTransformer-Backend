using System;
using System.Threading.Tasks;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class GeluLayer : ILayer
    {
        public string Name => "gelu";
        private TensorBase? _lastInput;

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Rank != 2 && input.Rank != 3)
                throw new ArgumentException($"Input must be Rank 2 or Rank 3. Got Rank {input.Rank}.");

            _lastInput = input;

            // Borrow buffer matching exact input shape (handles Rank 2 or Rank 3 seamlessly)
            TensorBase output = workspace.BorrowLike(input);

            if (input.Rank == 2)
            {
                workspace.Backend.GeluInto(input, output);
            }
            else
            {
                int layers = input.Layers;
                Parallel.For(0, layers, layer =>
                {
                    TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, layer);
                    TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, layer);
                    workspace.Backend.GeluInto(inputSlice, outputSlice);
                });
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            if (_lastInput == null)
                throw new InvalidOperationException("Forward must be called before Backward.");

            if (gradient.Rank != _lastInput.Rank)
                throw new ArgumentException($"Gradient rank ({gradient.Rank}) must match input rank ({_lastInput.Rank}).");

            // Borrow input gradient buffer from workspace
            TensorBase inputGradient = workspace.BorrowLike(_lastInput);

            if (gradient.Rank == 2)
            {
                workspace.Backend.GeluBackwardInto(_lastInput, gradient, inputGradient);
            }
            else
            {
                int layers = gradient.Layers;
                Parallel.For(0, layers, layer =>
                {
                    TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(_lastInput, layer);
                    TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, layer);
                    TensorBase outGradSlice = TensorUtilitiesSimd.GetLayer(inputGradient, layer);

                    workspace.Backend.GeluBackwardInto(inputSlice, gradSlice, outGradSlice);
                });
            }

            return inputGradient;
        }
    }
}