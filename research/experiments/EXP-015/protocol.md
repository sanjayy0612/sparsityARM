# EXP-015 - Quantization-aligned B32 replay

Status: approved for execution.

## Question

Can fixed B32 sparsity beat matched dense execution when both paths use the
pinned llama.cpp Q8_0 representation and Apple ARM dot-product kernel?

## Design

- Synthetic one-token 2048 -> 8192 -> 2048 SwiGLU, matching Llama 3.2 1B FFN geometry.
- Q8_0 weights and dynamic Q8_0 activation quantization through the pinned
  llama.cpp CPU implementation.
- B32 aligns one neuron block with one Q8_0 group in each down-projection row.
- Fixed clustered masks from 10% through 80% sparsity in 10-point increments.
- Matched dense and sparse kernels use identical quantized weights and dot-product primitive.
- Three independent seeds, eight masks per point, 20 warmups, and 200 retained
  paired measurements per implementation and point.
- Alternate dense-first and sparse-first execution order within paired measurements.
- Validate sparse output against a masked-dense Q8_0 control before timing.

## Interpretation boundary

This is an executor microbenchmark. Masks are synthetic, selector time is absent,
and the measurement is not end-to-end token generation. A runtime crossover only
shows mechanical B32 feasibility. Existing Llama 3.2 1B quality evidence already
shows substantial degradation for direct B32 masks at moderate sparsity, so speed
alone cannot establish a useful inference configuration.

The benchmark calls llama.cpp's Q8_0 ARM dot-product primitive directly. It does
not use the full GGML graph scheduler or its multi-row matrix multiplication path,
so EXP-014 remains the production dense end-to-end baseline.
