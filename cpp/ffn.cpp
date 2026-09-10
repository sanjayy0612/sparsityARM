// EXP-001: FP32, batch-one CPU executors. No allocation or packing in kernels.
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <arm_neon.h>

extern "C" int configure_threads() {
    return BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
}

static float dot(const float* a, const float* b, int n) {
    float sum = 0;
    #pragma clang loop vectorize(enable)
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static float dot_neon(const float* a, const float* b, int n) {
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a+i), vld1q_f32(b+i));
        s1 = vfmaq_f32(s1, vld1q_f32(a+i+4), vld1q_f32(b+i+4));
    }
    float sum = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// All gate/up weights: [intermediate, hidden]. Dense down: [hidden, intermediate].
// Irregular down: [intermediate, hidden]. Block down: [blocks, hidden, B].
// mode 0: Accelerate dense; 1: native neuron; 2: native block;
// 3: Accelerate block; 4: explicit NEON B8 block.
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
            } else if (mode == 4) {
                for (int k = 0; k < 8; ++k) {
                    const float v = dot_neon(gate + size_t(first+k)*h, x, h);
                    g[k] = (v / (1.f + std::exp(-v)))
                           * dot_neon(up + size_t(first+k)*h, x, h);
                }
                const float* tile = down + size_t(block)*h*8;
                const float32x4_t values0 = vld1q_f32(g);
                const float32x4_t values1 = vld1q_f32(g+4);
                for (int j = 0; j < h; ++j) {
                    const float* row = tile + size_t(j)*8;
                    output[j] += vaddvq_f32(vaddq_f32(
                        vmulq_f32(vld1q_f32(row), values0),
                        vmulq_f32(vld1q_f32(row+4), values1)));
                }
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

// Diagnostic staged B-block executor. Five timings are output initialization,
// mask compaction, gate/up/activation, down projection, and total milliseconds.
extern "C" void profile_block_staged(int h, int m, int b, const float* x,
    const float* gate, const float* up, const float* down, const uint8_t* mask,
    float* scratch, float* output, int* indices, int* active, int* runs,
    double* timings) {
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    std::fill(output, output + h, 0.f);
    const auto init_end = clock::now();

    int used = 0;
    int run_count = 0;
    bool previous = false;
    for (int block = 0; block < m/b; ++block) {
        if (mask[block]) {
            indices[used++] = block;
            if (!previous) ++run_count;
            previous = true;
        } else {
            previous = false;
        }
    }
    const auto scan_end = clock::now();

    float* activation = scratch;
    for (int index = 0; index < used; ++index) {
        const int first = indices[index] * b;
        for (int k = 0; k < b; ++k) {
            const float v = dot(gate + size_t(first+k)*h, x, h);
            activation[first+k] = (v / (1.f + std::exp(-v)))
                                  * dot(up + size_t(first+k)*h, x, h);
        }
    }
    const auto projection_end = clock::now();

    for (int index = 0; index < used; ++index) {
        const int block = indices[index];
        const float* tile = down + size_t(block)*h*b;
        const float* values = activation + block*b;
        for (int j = 0; j < h; ++j) output[j] += dot(tile + size_t(j)*b, values, b);
    }
    const auto down_end = clock::now();
    *active = used;
    *runs = run_count;
    timings[0] = std::chrono::duration<double, std::milli>(init_end-total_start).count();
    timings[1] = std::chrono::duration<double, std::milli>(scan_end-init_end).count();
    timings[2] = std::chrono::duration<double, std::milli>(projection_end-scan_end).count();
    timings[3] = std::chrono::duration<double, std::milli>(down_end-projection_end).count();
    timings[4] = std::chrono::duration<double, std::milli>(down_end-total_start).count();
}
