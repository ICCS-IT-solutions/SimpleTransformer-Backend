namespace SimpleTransformer.AccelerationEngine.GpuVulkan
{
    public static class VulkanShaders
    {
        public const string Scale = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InBuf { float x[]; };
layout(push_constant) uniform Push { uint n; float alpha; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    x[i] = x[i] * p.alpha;
}";

        public const string Fill = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InBuf { float x[]; };
layout(push_constant) uniform Push { uint n; float value; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    x[i] = p.value;
}";

        public const string AddInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    a[i] = a[i] + b[i];
}";

        public const string AddInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(set = 0, binding = 2) buffer RBuf { float r[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    r[i] = a[i] + b[i];
}";

        public const string MulInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    a[i] = a[i] * b[i];
}";

        public const string MulInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(set = 0, binding = 2) buffer RBuf { float r[]; };
layout(push_constant) uniform Push { uint n; float _pad; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    r[i] = a[i] * b[i];
}";

        // Tiled GEMM: 16x16 workgroup, 32-byte push constants.
        // Inputs must be packed dense row-major per batch layer.
        // A: batch * aRows x aCols, B: batch * bRows x bCols, R: batch * m x n.
        public const string MatMul = @"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) buffer ABuf { float a[]; };
layout(set = 0, binding = 1) buffer BBuf { float b[]; };
layout(set = 0, binding = 2) buffer RBuf { float r[]; };
layout(push_constant) uniform Push {
    uint m; uint n; uint k; uint batch;
    uint transposeA; uint transposeB; uint accumulate; uint _pad;
} p;
shared float tileA[16][16];
shared float tileB[16][16];
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint lidY = gl_LocalInvocationID.y;
    uint lidX = gl_LocalInvocationID.x;
    uint batch = gl_WorkGroupID.z;
    if (batch >= p.batch) return;

    uint aRows = p.transposeA != 0 ? p.k : p.m;
    uint aCols = p.transposeA != 0 ? p.m : p.k;
    uint bRows = p.transposeB != 0 ? p.n : p.k;
    uint bCols = p.transposeB != 0 ? p.k : p.n;

    uint aBase = batch * aRows * aCols;
    uint bBase = batch * bRows * bCols;
    uint rBase = batch * p.m * p.n;

    float sum = 0.0;
    uint tiles = (p.k + 15u) / 16u;
    for (uint t = 0u; t < tiles; t++) {
        uint aRow = row;
        uint aCol = t * 16u + lidX;
        float av = 0.0;
        if (aRow < p.m && aCol < p.k) {
            uint ai = p.transposeA != 0 ? aCol * aCols + aRow : aRow * aCols + aCol;
            av = a[aBase + ai];
        }
        tileA[lidY][lidX] = av;

        uint bRow = t * 16u + lidY;
        uint bCol = col;
        float bv = 0.0;
        if (bRow < p.k && bCol < p.n) {
            uint bi = p.transposeB != 0 ? bCol * bCols + bRow : bRow * bCols + bCol;
            bv = b[bBase + bi];
        }
        tileB[lidY][lidX] = bv;

        barrier();

        for (uint e = 0u; e < 16u; e++) {
            sum += tileA[lidY][e] * tileB[e][lidX];
        }

        barrier();
    }

    if (row < p.m && col < p.n) {
        uint ri = rBase + row * p.n + col;
        r[ri] = p.accumulate != 0 ? r[ri] + sum : sum;
    }
}";

        public const string GeluInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer T { float x[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    for (uint c = 0u; c < p.cols; c++) {
        uint i = r * p.cols + c;
        float v = x[i];
        // Saturate extreme inputs (host SIMD does the same).
        if (isnan(v)) { continue; }
        if (v >= 12.0) { x[i] = v; continue; }
        if (v <= -12.0) { x[i] = 0.0; continue; }
        float t = tanh(0.7978845608 * (v + 0.044715 * v * v * v));
        x[i] = 0.5 * v * (1.0 + t);
    }
}";

        public const string GeluInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer R { float r[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= p.rows) return;
    for (uint c = 0u; c < p.cols; c++) {
        uint i = row * p.cols + c;
        float v = a[i];
        // Saturate extreme inputs (host SIMD does the same): avoids v^3
        // overflow (Inf) which then yields NaN via 0.5*v*(1+tanh(Inf)).
        if (isnan(v)) { r[i] = v; continue; }
        if (v >= 12.0) { r[i] = v; continue; }
        if (v <= -12.0) { r[i] = 0.0; continue; }
        float t = tanh(0.7978845608 * (v + 0.044715 * v * v * v));
        r[i] = 0.5 * v * (1.0 + t);
    }
}";

        public const string GeluBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer X { float x[]; };
layout(set = 0, binding = 1) buffer DY { float dy[]; };
layout(set = 0, binding = 2) buffer DX { float dx[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= p.rows) return;
    for (uint c = 0u; c < p.cols; c++) {
        uint i = row * p.cols + c;
        float v = x[i];
        float t = tanh(0.7978845608 * (v + 0.044715 * v * v * v));
        float dt = 0.7978845608 * (1.0 + 3.0 * 0.044715 * v * v);
        dx[i] = dy[i] * (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * dt);
    }
}";

        public const string LayerNormInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer T { float x[]; };
layout(set = 0, binding = 1) buffer G { float g[]; };
layout(set = 0, binding = 2) buffer B { float b[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float eps; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    uint base_ = r * p.cols;
    float sum = 0.0;
    for (uint c = 0u; c < p.cols; c++) sum += x[base_ + c];
    float mean = sum / float(p.cols);
    float var_ = 0.0;
    for (uint c = 0u; c < p.cols; c++) {
        float d = x[base_ + c] - mean;
        var_ += d * d;
    }
    var_ /= float(p.cols);
    float inv = inversesqrt(var_ + p.eps);
    for (uint c = 0u; c < p.cols; c++) {
        uint i = base_ + c;
        x[i] = (x[i] - mean) * inv * g[c] + b[c];
    }
}";

        public const string LayerNormInto = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer G { float g[]; };
layout(set = 0, binding = 2) buffer B { float b[]; };
layout(set = 0, binding = 3) buffer R { float r[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float eps; uint _f; } p;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= p.rows) return;
    uint base_ = row * p.cols;
    float sum = 0.0;
    for (uint c = 0u; c < p.cols; c++) sum += a[base_ + c];
    float mean = sum / float(p.cols);
    float var_ = 0.0;
    for (uint c = 0u; c < p.cols; c++) {
        float d = a[base_ + c] - mean;
        var_ += d * d;
    }
    var_ /= float(p.cols);
    float inv = inversesqrt(var_ + p.eps);
    for (uint c = 0u; c < p.cols; c++) {
        uint i = base_ + c;
        r[i] = (a[i] - mean) * inv * g[c] + b[c];
    }
}";

        public const string SoftmaxInPlace = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer T { float x[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    uint base_ = r * p.cols;
    float m = x[base_];
    for (uint c = 1u; c < p.cols; c++) m = max(m, x[base_ + c]);
    float s = 0.0;
    for (uint c = 0u; c < p.cols; c++) {
        float e = exp(x[base_ + c] - m);
        x[base_ + c] = e;
        s += e;
    }
    for (uint c = 0u; c < p.cols; c++) x[base_ + c] /= s;
}";

        public const string SoftmaxBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer DY { float dy[]; };
layout(set = 0, binding = 2) buffer DX { float dx[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= p.rows) return;
    uint base_ = r * p.cols;
    float dot = 0.0;
    for (uint c = 0u; c < p.cols; c++) dot += dy[base_ + c] * s[base_ + c];
    for (uint c = 0u; c < p.cols; c++) {
        uint i = base_ + c;
        dx[i] = s[i] * (dy[i] - dot);
    }
}";

        public const string ApplyMask = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer M { float m[]; };
layout(push_constant) uniform Push { uint n; float _v; uint _a; uint _b; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    if (m[i] == 0.0) s[i] = -1e9;
}";

        public const string Transpose = @"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer D { float d[]; };
layout(push_constant) uniform Push { uint rows; uint cols; float _e; uint _f; } p;
void main() {
    uint r = gl_GlobalInvocationID.y;
    uint c = gl_GlobalInvocationID.x;
    if (r >= p.rows || c >= p.cols) return;
    d[c * p.rows + r] = s[r * p.cols + c];
}";

        public const string Copy = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer S { float s[]; };
layout(set = 0, binding = 1) buffer D { float d[]; };
layout(push_constant) uniform Push { uint n; float _v; uint _a; uint _b; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    d[i] = s[i];
}";
    }
}
