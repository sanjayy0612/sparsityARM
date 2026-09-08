# EXP-002 preflight

Status: ready for user-authorized execution. EXP-002 has not been run.

Checked on 2026-09-08 on the target Apple M2, macOS 26.5.1, 8 GiB RAM.

## Checkpoint

The pinned official `meta-llama/Llama-3.2-1B` snapshot is complete at revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`.

- `model.safetensors`: 2,471,645,608 bytes; SHA-256
  `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.
- `config.json`: SHA-256
  `bfb5d39c327ee5c3a65e3574596ff30b052a445cfc59a25bdf3b4c0c5a44f4d2`.
- `tokenizer.json`: SHA-256
  `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`.
- Safetensors opened successfully with 146 tensors and PyTorch-format metadata.
- Layers 0 and 15 were sampled directly: gate/up `[8192, 2048]`, down
  `[2048, 8192]`, all BF16. The embedding tensor is `[128256, 2048]`, BF16.
- A meta-device `AutoModel` construction verified all 16 layers and every FFN
  gate/up/down shape without loading the real weights or running inference.

## Capture path

- The official tokenizer loaded entirely offline from the pinned snapshot.
- All eight prompt IDs are unique. Each tokenizes to exactly 64 tokens including
  BOS, leaving 56 analyzed positions after excluding positions 0–7.
- A two-layer local BF16 Llama exercised the exact MLP and down-projection hooks,
  CPU/no-cache forward path, finite-output checks and bitwise SwiGLU
  recomputation without changing module inputs.
- BF16 CPU execution was also smoke-tested independently.
- The capture refuses the wrong snapshot revision, unexpected geometry,
  non-CPU/non-BF16 parameters, missing hooks, nonfinite values, recomputation
  differences and process RSS above 5.5 GiB.
- `psutil.swap_memory()` raises `OSError` on this macOS build. Preflight found
  and fixed this before execution; capture now records swap as null when the OS
  API is unavailable instead of aborting.

All prompts reaching the 64-token cap means this pilot studies equal-length
prefixes. It does not study context-length dependence; this is recorded as a
limitation rather than silently changing the approved prompt protocol.

## Analysis and validation path

- Unit tests cover clustered/scattered coverage, block sizes 8/16/32/64,
  sparsities 0/20/30/40/100%, exact expansion preservation, fixed block budgets,
  tie permutations, zero-mass activations and nonfinite rejection.
- A temporary synthetic integration fixture with eight prompts × 16 layers
  passed through the exact analyzer and independent validator: 128 capture
  artifacts, 128 layer/token instances and 1,536 metric rows.
- The validator independently checked source/capture/analysis hashes, artifact
  counts, token positions, mask shapes, neuron budgets, block budgets, exact
  block expansion and every reported aggregate mean.
- The full repository suite passes: 61 tests. `pip check`, Python compilation
  and `git diff --check` pass.
- At least 27 GiB disk was free during preflight; protocol requires 2 GiB.

## Execution boundary

No real checkpoint forward pass, activation capture, mask analysis or EXP-002
result directory was created during preflight. When authorized, execute capture,
analysis and validation sequentially using the commands in `protocol.md`.
Operational capture time is not a performance measurement.
