#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int q8_block = 32;

const ggml_type_traits_cpu * q8_traits() {
    return ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
}

size_t row_bytes(int columns) {
    return ggml_row_size(GGML_TYPE_Q8_0, columns);
}

float dot_q8(const uint8_t * lhs, const uint8_t * rhs, int length) {
    float result = 0.0f;
    q8_traits()->vec_dot(length, &result, 0, lhs, 0, rhs, 0, 1);
    return result;
}

}

extern "C" size_t armsparse_q8_0_row_bytes(int columns) {
    return columns > 0 && columns % q8_block == 0 ? row_bytes(columns) : 0;
}

extern "C" int armsparse_q8_0_quantize_rows(
        const float * source, int rows, int columns, uint8_t * destination) {
    if (source == nullptr || destination == nullptr || rows <= 0 || columns <= 0 || columns % q8_block != 0) {
        return -1;
    }
    const size_t bytes = row_bytes(columns);
    for (int row = 0; row < rows; ++row) {
        q8_traits()->from_float(source + size_t(row) * columns, destination + size_t(row) * bytes, columns);
    }
    return 0;
}

// mode 0: dense; mode 1: B32 sparse; mode 2: masked-dense correctness control.
extern "C" double armsparse_q8_0_ffn(
        int mode, int hidden, int intermediate, const float * input,
        const uint8_t * gate, const uint8_t * up, const uint8_t * down,
        const uint8_t * mask, float * activation, uint8_t * quantized_input,
        uint8_t * quantized_activation, float * output) {
    if (mode < 0 || mode > 2 || hidden <= 0 || intermediate <= 0 || hidden % q8_block != 0 ||
            intermediate % q8_block != 0) {
        return -1.0;
    }

    const auto start = std::chrono::steady_clock::now();
    const size_t hidden_row_bytes = row_bytes(hidden);
    const size_t intermediate_row_bytes = row_bytes(intermediate);
    q8_traits()->from_float(input, quantized_input, hidden);
    std::fill(activation, activation + intermediate, 0.0f);
    std::fill(output, output + hidden, 0.0f);

    for (int neuron = 0; neuron < intermediate; ++neuron) {
        if (mode == 1 && mask[neuron / q8_block] == 0) {
            continue;
        }
        const float gate_value = dot_q8(gate + size_t(neuron) * hidden_row_bytes, quantized_input, hidden);
        const float up_value = dot_q8(up + size_t(neuron) * hidden_row_bytes, quantized_input, hidden);
        activation[neuron] = gate_value / (1.0f + std::exp(-gate_value)) * up_value;
    }

    if (mode == 2) {
        for (int block = 0; block < intermediate / q8_block; ++block) {
            if (mask[block] == 0) {
                std::fill(activation + block * q8_block, activation + (block + 1) * q8_block, 0.0f);
            }
        }
    }

    if (mode != 1) {
        q8_traits()->from_float(activation, quantized_activation, intermediate);
        for (int row = 0; row < hidden; ++row) {
            output[row] = dot_q8(down + size_t(row) * intermediate_row_bytes, quantized_activation, intermediate);
        }
    } else {
        const size_t block_bytes = row_bytes(q8_block);
        for (int block = 0; block < intermediate / q8_block; ++block) {
            if (mask[block] == 0) {
                continue;
            }
            q8_traits()->from_float(activation + block * q8_block,
                                    quantized_activation + size_t(block) * block_bytes, q8_block);
            for (int row = 0; row < hidden; ++row) {
                output[row] += dot_q8(down + size_t(row) * intermediate_row_bytes + size_t(block) * block_bytes,
                                      quantized_activation + size_t(block) * block_bytes, q8_block);
            }
        }
    }

    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
