using SimpleTransformer.AccelerationEngine;
using SimpleTransformer.Model;

namespace SimpleTransformer.AccelerationEngine.CpuAvx2
{
    public class CpuAvx2Backend : IAccelerationBackend
    {
        public string Name => throw new NotImplementedException();

        public bool IsGpuAccelerated => throw new NotImplementedException();

        public void ApplyMaskInPlace(TensorBase scores, TensorBase mask)
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
        {
            throw new NotImplementedException();
        }

        public void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
        {
            throw new NotImplementedException();
        }

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
        {
            throw new NotImplementedException();
        }

        public void ElementWiseMultiplyInto(TensorBase a, TensorBase b, TensorBase result)
        {
            throw new NotImplementedException();
        }

        public void GeluBackwardInto(TensorBase input, TensorBase outputGradient, TensorBase inputGradient)
        {
            throw new NotImplementedException();
        }

        public void GeluInPlace(TensorBase tensor)
        {
            throw new NotImplementedException();
        }

        public void GeluInto(TensorBase input, TensorBase result)
        {
            throw new NotImplementedException();
        }

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1E-05F)
        {
            throw new NotImplementedException();
        }

        public void LayerNormInto(TensorBase input, TensorBase gamma, TensorBase beta, TensorBase result, float epsilon = 1E-05F)
        {
            throw new NotImplementedException();
        }

        public void MatMul(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
        {
            throw new NotImplementedException();
        }

        public void MatMulAccumulate(TensorBase a, TensorBase b, TensorBase result, bool transposeA = false, bool transposeB = false)
        {
            throw new NotImplementedException();
        }

        public void ScaleInPlace(TensorBase tensor, float scalar)
        {
            throw new NotImplementedException();
        }

        public void SoftmaxBackwardInto(TensorBase softmaxOutput, TensorBase outputGradient, TensorBase inputGradient)
        {
            throw new NotImplementedException();
        }

        public void SoftmaxInPlace(TensorBase tensor)
        {
            throw new NotImplementedException();
        }

        public void TransposeInto(TensorBase source, TensorBase destination)
        {
            throw new NotImplementedException();
        }

        public void CopyInto(TensorBase source, TensorBase destination)
        {
            throw new NotImplementedException();
        }

        public void Fill(TensorBase tensor, float value)
        {
            throw new NotImplementedException();
        }

        public void Synchronize()
        {
            throw new NotImplementedException();
        }
    }
}