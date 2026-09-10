# EXP-014 — Strong quantized runtime baseline

Status: integration scaffold implemented; model conversion and benchmark not run.

## Objective

Establish a reproducible CPU-only Q8_0 Llama 3.2 1B baseline using official
`llama.cpp` before modifying any production quantized kernel.

## Pinned runtime

- Upstream: `ggml-org/llama.cpp` Git submodule.
- Commit: `fa6769818708afd9807b22183ccda112fd563427`.
- Release build with Metal disabled and native Apple ARM CPU features enabled.
- Required tools: `llama-bench` and `llama-quantize`; conversion uses the pinned
  `convert_hf_to_gguf.py`.

## Baseline

- Pinned local `meta-llama/Llama-3.2-1B` snapshot.
- Direct Q8_0 GGUF conversion, with model and converter hashes retained.
- `llama-bench`: prompt length 128, generation 32, one CPU thread, zero GPU
  layers, five repetitions and one-second inter-test delay.

This phase establishes dense prompt-processing and token-generation behavior.
It makes no sparsity claim. Sparse modification is forbidden until the dense
artifact is reproducible and validated.
