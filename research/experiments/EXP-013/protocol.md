# EXP-013 — Weight-only INT8 executor feasibility

Status: implementation complete; benchmark not started.

## Question

Does matched weight-only INT8 execution move the B8 runtime break-even below
the 30–40% FP32 boundary on Apple M2?

## Design

- Synthetic 2048 -> 8192 -> 2048 SwiGLU, one token and one CPU thread.
- Symmetric per-tensor INT8 weights; FP32 input, activations and accumulation.
- Matched custom INT8 dense and B8 sparse kernels using the same quantization.
- B8 clustered equal-work masks from 10% through 80% sparsity.
- Record error against the corresponding dequantized-weight FP32 computation.
- Three independent weight/input seeds, eight masks per sparsity, 20 warmups
  and 200 retained timings per dense/sparse kernel and case.

This is a narrow kernel feasibility experiment. Per-tensor weight-only INT8 is
not a production Llama quantization scheme, and the custom dense kernel is not
a llama.cpp-class optimized quantized baseline. A positive result would require
later validation against a strong runtime; a negative result can reject this
specific implementation only.
