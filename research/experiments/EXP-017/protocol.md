# EXP-017 - TinyLlama-geometry B8 runtime check

Status: implementation complete; benchmark not started.

## Question

At TinyLlama's quality-compatible B8/20% point, does block execution beat its
matched dense control on Apple M2?

## Locked design

- Synthetic one-token 2048 -> 5632 -> 2048 SwiGLU matching TinyLlama 1.1B geometry.
- B8 fixed clustered equal-work masks at 20% target sparsity.
- FP32: Accelerate dense versus the existing native packed-B8 executor.
- INT8: matched custom symmetric per-tensor weight-only dense and packed-B8
  executors with FP32 activations and accumulation.
- Three independent weight/input seeds and eight masks per seed.
- Twenty warmups and 200 retained paired measurements per executor.
- Alternate dense-first and sparse-first order within paired measurements.
- Validate FP32 and INT8 sparse outputs against independent NumPy masked-dense
  references before timing.

## Interpretation boundary

This tests executor behavior at the active-neuron budget supported by EXP-016;
it does not replay the exact TinyLlama masks. Clustered masks are a favorable
locality control and selector cost is absent.

The FP32 dense path is an optimized Accelerate baseline. The custom INT8 dense
path is only a matched quantization control and is not a production llama.cpp
baseline. A custom-INT8 speedup cannot establish deployable token-generation
speed. No model weights or language-quality metric are used in this experiment.
