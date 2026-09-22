using SimpleTransformer.AccelerationEngine.Common;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.CpuReference.Math;

public sealed class ElementWiseOperations
{
    public void AddInPlace(TensorBase target, TensorBase source)
    {
        TensorValidation.ValidateSameShape(target, source);

        ApplyInPlace(
            target,
            source,
            static (a, b) => a + b);
    }

    public void SubtractInPlace(TensorBase target, TensorBase source)
    {
        TensorValidation.ValidateSameShape(target, source);

        ApplyInPlace(
            target,
            source,
            static (a, b) => a - b);
    }

    public void MultiplyInPlace(TensorBase target, TensorBase source)
    {
        TensorValidation.ValidateSameShape(target, source);

        ApplyInPlace(
            target,
            source,
            static (a, b) => a * b);
    }

    public void DivideInPlace(TensorBase target, TensorBase source)
    {
        TensorValidation.ValidateSameShape(target, source);

        ApplyInPlace(
            target,
            source,
            static (a, b) => a / b);
    }


    public Tensor Add(TensorBase a, TensorBase b)
    {
        TensorValidation.ValidateSameShape(a, b);

        var result = new Tensor(a.Shape);

        AddInto(a, b, result);

        return result;
    }

    public Tensor Subtract(TensorBase a, TensorBase b)
    {
        TensorValidation.ValidateSameShape(a, b);

        var result = new Tensor(a.Shape);

        SubtractInto(a, b, result);

        return result;
    }

    public Tensor Multiply(TensorBase a, TensorBase b)
    {
        TensorValidation.ValidateSameShape(a, b);

        var result = new Tensor(a.Shape);

        MultiplyInto(a, b, result);

        return result;
    }

    public Tensor Divide(TensorBase a, TensorBase b)
    {
        TensorValidation.ValidateSameShape(a, b);

        var result = new Tensor(a.Shape);

        DivideInto(a, b, result);

        return result;
    }


    public void AddInto(
        TensorBase a,
        TensorBase b,
        TensorBase result)
    {
        TensorValidation.ValidateSameShape(a, b);
        TensorValidation.ValidateSameShape(a, result);

        Apply(
            a,
            b,
            result,
            static (x, y) => x + y);
    }

    public void SubtractInto(
        TensorBase a,
        TensorBase b,
        TensorBase result)
    {
        TensorValidation.ValidateSameShape(a, b);
        TensorValidation.ValidateSameShape(a, result);

        Apply(
            a,
            b,
            result,
            static (x, y) => x - y);
    }

    public void MultiplyInto(
        TensorBase a,
        TensorBase b,
        TensorBase result)
    {
        TensorValidation.ValidateSameShape(a, b);
        TensorValidation.ValidateSameShape(a, result);

        Apply(
            a,
            b,
            result,
            static (x, y) => x * y);
    }

    public void DivideInto(
        TensorBase a,
        TensorBase b,
        TensorBase result)
    {
        TensorValidation.ValidateSameShape(a, b);
        TensorValidation.ValidateSameShape(a, result);

        Apply(
            a,
            b,
            result,
            static (x, y) => x / y);
    }


    private static void ApplyInPlace(
        TensorBase target,
        TensorBase source,
        Func<float, float, float> operation)
    {
        Apply(
            target,
            source,
            target,
            operation);
    }


    private static void Apply(
        TensorBase a,
        TensorBase b,
        TensorBase result,
        Func<float, float, float> operation)
    {
        switch (a.Rank)
        {
            case 2:
                Apply2D(a, b, result, operation);
                break;

            case 3:
                Apply3D(a, b, result, operation);
                break;

            default:
                throw new ArgumentException(
                    $"Unsupported tensor rank: {a.Rank}.");
        }
    }


    private static void Apply2D(
        TensorBase a,
        TensorBase b,
        TensorBase result,
        Func<float, float, float> operation)
    {
        for (int row = 0; row < a.Rows; row++)
        {
            int aRowOffset =
                a.Offset + (row * a.Stride);

            int bRowOffset =
                b.Offset + (row * b.Stride);

            int resultRowOffset =
                result.Offset + (row * result.Stride);

            for (int col = 0; col < a.Cols; col++)
            {
                result.Data[resultRowOffset + col] =
                    operation(
                        a.Data[aRowOffset + col],
                        b.Data[bRowOffset + col]);
            }
        }
    }


    private static void Apply3D(
        TensorBase a,
        TensorBase b,
        TensorBase result,
        Func<float, float, float> operation)
    {
        for (int layer = 0; layer < a.Layers; layer++)
        {
            int aLayerOffset =
                a.Offset + (layer * a.LayerStride);

            int bLayerOffset =
                b.Offset + (layer * b.LayerStride);

            int resultLayerOffset =
                result.Offset + (layer * result.LayerStride);

            for (int row = 0; row < a.Rows; row++)
            {
                int aRowOffset =
                    aLayerOffset + (row * a.Stride);

                int bRowOffset =
                    bLayerOffset + (row * b.Stride);

                int resultRowOffset =
                    resultLayerOffset + (row * result.Stride);

                for (int col = 0; col < a.Cols; col++)
                {
                    result.Data[resultRowOffset + col] =
                        operation(
                            a.Data[aRowOffset + col],
                            b.Data[bRowOffset + col]);
                }
            }
        }
    }
}