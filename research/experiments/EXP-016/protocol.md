# EXP-016 - TinyLlama quality validation

Status: implementation and model preflight complete; benchmark not started.

## Question

Does TinyLlama 1.1B exhibit the same mismatch seen on Llama 3.2 1B, where
contiguous block sparsity loses acceptable language quality before executor
speedups become plausible?

## Model and hardware

- Official `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` pretrained checkpoint.
- Pinned revision: `59f6f375b26bde864a6ca194a9a3044570490064`.
- Verified geometry: hidden 2048, intermediate 5632, 22 layers, SwiGLU.
- Apple M2 CPU only, BF16 weights, one Torch intra-op and inter-op thread.
- Resident-memory ceiling: 5.5 GiB on the 8 GiB development machine.
- Record and enforce resident memory after every condition, after lazy model
  pages have been exercised; a pre-forward RSS reading is not accepted.

The pretrained 3T checkpoint is used instead of the chat-tuned derivative so
the WikiText perplexity screen is not confounded by instruction tuning.

## Fixed screen

- Same frozen 16-record WikiText-2 corpus and 128-token cap used by EXP-004.
- Dense baseline.
- Per-token post-SwiGLU neuron top-k at 30% sparsity as a noncontiguous oracle control.
- Isolated output-contribution ranking at B8 and B32, each at 10%, 20%, and 30% sparsity.
- Same pre-registered acceptance gate: at most 5% perplexity increase relative to dense.

## Boundaries

All masks inspect the current post-SwiGLU activation and are oracle/reference
masks. Selector cost is excluded. Evaluation duration is operational metadata,
not an inference-latency measurement. This experiment tests model quality only;
it cannot establish a TinyLlama speedup.

The screen intentionally does not repeat every previous selector, layout, or
kernel experiment. If B8 or B32 passes at a useful sparsity, a later matched
TinyLlama-geometry executor benchmark will test its runtime. If neither does,
the cross-model negative result is sufficient to stop further kernel work.

Model configuration source: https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/59f6f375b26bde864a6ca194a9a3044570490064/config.json
