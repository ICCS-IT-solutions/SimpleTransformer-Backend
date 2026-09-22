namespace SimpleTransformer.Model
{
    public class Tensor : TensorBase
    {
        private readonly int _rank;
        private readonly int _rows;
        private readonly int _cols;
        private readonly int _stride;
        private readonly int _layers;
        private readonly int[] _shape;
        public override float[] Buffer => Data;
        public override int Length =>Data.Length;
        public override int Offset => 0;
        public override float[] Data { get; }
        public override int[] Shape => _shape;
        public override int Rank => _rank;
        public override int Layers => _layers;
        public override int Rows => _rows;
        public override int Cols => _cols;
        public override int Stride => _stride;
        public override int Size
        {
            get => Rank switch
            {
                1 => Cols,
                2 => Rows * Cols,
                3 => Layers * Rows * Cols,
                _ => 0
            };
        }

        public Tensor(params int[] shape)
        {
            _shape = shape;
            _rank = shape.Length;
            switch (Rank)
            {
                case 1:
                    // Rank-1/2 tensors are a single "layer": report Layers = 1 so
                    // plain Tensors match TensorView (which always sets _layers = 1
                    // for rank < 3). Leaving this 0 made Layers-based shape checks
                    // reject valid Tensor + TensorView pairs (e.g. MHA backward).
                    _layers = 1;
                    _cols = shape[0];
                    _stride = _cols;
                    break;

                case 2:
                    _layers = 1;
                    _rows = shape[0];
                    _cols = shape[1];
                    _stride = _cols;
                    break;

                case 3:
                    _layers = shape[0];
                    _rows = shape[1];
                    _cols = shape[2];
                    _stride = _cols;
                    break;
            }            
            int size = 1;
            foreach (var dim in shape)
            {
                if(dim <= 0) throw new ArgumentException("All dimensions must be positive.");
                size *= dim;
            }
            Data = new float[size];
        }
        // Stacked matrix indexer (3D)
        public override float this[int layer, int row, int col]
        {
            get
            {
                if (Rank != 3) 
                    throw new ArgumentException($"Tensor must be Rank 3 (current rank: {Rank}).");
                if (layer < 0 || layer >= _layers || row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException($"Index [{layer}, {row}, {col}] out of bounds for shape [{_layers}, {_rows}, {_cols}].");
                
                return Buffer[Offset + layer * (_rows * _stride) + row * _stride + col];
            }
            set
            {
                if (Rank != 3) 
                    throw new ArgumentException($"Tensor must be Rank 3 (current rank: {Rank}).");
                if (layer < 0 || layer >= _layers || row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException($"Index [{layer}, {row}, {col}] out of bounds for shape [{_layers}, {_rows}, {_cols}].");

                Buffer[Offset + layer * (_rows * _stride) + row * _stride + col] = value;
            }
        }

        // Matrix indexer (2D)
        public override float this[int row, int col]
        {
            get
            {
                if (Rank != 2) 
                    throw new ArgumentException($"Tensor must be Rank 2 (current rank: {Rank}).");
                if (row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException($"Index [{row}, {col}] out of bounds for shape [{_rows}, {_cols}].");
                
                return Buffer[Offset + row * _stride + col];
            }
            set
            {
                if (Rank != 2) 
                    throw new ArgumentException($"Tensor must be Rank 2 (current rank: {Rank}).");
                if (row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException($"Index [{row}, {col}] out of bounds for shape [{_rows}, {_cols}].");
                
                Buffer[Offset + row * _stride + col] = value;
            }
        }

        // Vector indexer (1D)
        public override float this[int index]
        {
            get
            {
                if (Rank != 1) 
                    throw new ArgumentException($"Tensor must be Rank 1 (current rank: {Rank}).");
                if (index < 0 || index >= _cols) 
                    throw new IndexOutOfRangeException($"Index [{index}] out of bounds for vector of length {_cols}.");
                
                return Buffer[Offset + index];
            }
            set
            {
                if (Rank != 1) 
                    throw new ArgumentException($"Tensor must be Rank 1 (current rank: {Rank}).");
                if (index < 0 || index >= _cols) 
                    throw new IndexOutOfRangeException($"Index [{index}] out of bounds for vector of length {_cols}.");
                
                Buffer[Offset + index] = value;
            }
        }


        public override Tensor Clone()
        {
            // Allocate an identical concrete shape container
            var clone = new Tensor((int[])Shape.Clone());

            // Execute a fast, direct hardware-level block memory copy
            this.ReadOnlySpan.CopyTo(clone.Span);

            return clone;
        }
        public override void Dispose()
        {
            //Clear any sensitive data in the buffer if needed
            if (Data != null)
            {
                Array.Clear(Data, 0, Data.Length);
            } 
            base.Dispose();
        }
    }
}