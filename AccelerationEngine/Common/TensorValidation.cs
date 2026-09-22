using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.Common
{
    public static class TensorValidation
    {
        public static void ValidateSameShape(
            TensorBase a,
            TensorBase b)
        {
            if (a.Rank != b.Rank ||
                a.Layers != b.Layers ||
                a.Rows != b.Rows ||
                a.Cols != b.Cols)
            {
                throw new ArgumentException(
                    $"Tensor dimensions do not match: " +
                    $"{DescribeShape(a)} vs {DescribeShape(b)}.");
            }
        }


        public static string DescribeShape(TensorBase tensor)
        {
            return tensor.Rank switch
            {
                2 => $"({tensor.Rows}x{tensor.Cols})",

                3 => $"({tensor.Layers}x{tensor.Rows}x{tensor.Cols})",

                _ => $"Rank {tensor.Rank}"
            };
        }
    }
}