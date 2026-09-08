// EXP-001: FP32, batch-one CPU executors. No allocation or packing in kernels.
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

extern "C" int configure_threads() {
    return BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
}

static float dot(const float* a, const float* b, int n) {
    float sum = 0;
    #pragma clang loop vectorize(enable)
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// All gate/up weights: [intermediate, hidden]. Dense down: [hidden, intermediate].
// Irregular down: [intermediate, hidden]. Block down: [blocks, hidden, B].
// mode 0: Accelerate dense; 1: native neuron; 2: native block; 3: Accelerate block.
extern "C" double ffn(int mode, int h, int m, int b, const float* x,
    const float* gate, const float* up, const float* down,
    const uint8_t* mask, float* scratch, float* output) {
    const auto start = std::chrono::steady_clock::now();
    float* g = scratch;
    float* u = scratch + m;
    std::fill(output, output + h, 0.f);
    if (mode == 0) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, h, 1, gate, h, x, 1, 0, g, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, h, 1, up, h, x, 1, 0, u, 1);
        for (int i = 0; i < m; ++i) g[i] = (g[i] / (1.f + std::exp(-g[i]))) * u[i];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, h, m, 1, down, m, g, 1, 0, output, 1);
    } else if (mode == 1) {
        for (int i = 0; i < m; ++i) {
            if (!mask[i]) continue;
            const float gate_value = dot(gate + size_t(i)*h, x, h);
            const float activation = (gate_value / (1.f + std::exp(-gate_value))) * dot(up + size_t(i)*h, x, h);
            const float* column = down + size_t(i)*h;
            for (int j = 0; j < h; ++j) output[j] += column[j] * activation;
        }
    } else {
        for (int block = 0; block < m/b; ++block) {
            if (!mask[block]) continue;
            const int first = block*b;
            if (mode == 3) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, b, h, 1, gate + size_t(first)*h, h, x, 1, 0, g, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, b, h, 1, up + size_t(first)*h, h, x, 1, 0, u, 1);
                for (int k = 0; k < b; ++k) g[k] = (g[k] / (1.f + std::exp(-g[k]))) * u[k];
                cblas_sgemv(CblasRowMajor, CblasNoTrans, h, b, 1, down + size_t(block)*h*b, b, g, 1, 1, output, 1);
            } else {
                for (int k = 0; k < b; ++k) {
                    const float v = dot(gate + size_t(first+k)*h, x, h);
                    g[k] = (v / (1.f + std::exp(-v))) * dot(up + size_t(first+k)*h, x, h);
                }
                const float* tile = down + size_t(block)*h*b;
                for (int j = 0; j < h; ++j) output[j] += dot(tile + size_t(j)*b, g, b);
            }
        }
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// Diagnostic only: scan/compact a byte mask into a caller-owned index buffer.
extern "C" double scan_mask(const uint8_t* mask, int n, int* indices, int* count) {
    const auto start = std::chrono::steady_clock::now();
    int used = 0;
    for (int i = 0; i < n; ++i) if (mask[i]) indices[used++] = i;
    *count = used;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
