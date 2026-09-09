# EXP-003 — Quality under activation-aware block masks

Status: implementation complete; real run awaits protocol approval.

## Question

At the same realized sparsity, how much language-model quality is lost by
selecting whole contiguous FFN blocks instead of individual neurons?

This is a **quality experiment**, not a latency experiment. The reference masks
inspect the dense post-SwiGLU activation, so their selector cost and the dense
gate/up projections have already been paid. No timing emitted by this experiment
may be presented as deployable sparse-inference performance.

## Comparisons

- Dense baseline.
- Activation-magnitude neuron top-k reference.
- Direct block top-k, scored by the sum of absolute activation within each block.
- Uniform random blocks as a negative control.

Planned sparsities are 20%, 30%, and 40%; planned block sizes are 8, 16, 32,
and 64. Selection is independently recomputed for every token and layer. The
fixed model is official Llama 3.2 1B revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`, BF16 on one Apple M2 CPU thread.

## Primary metric

Token-weighted causal cross-entropy and perplexity relative to the dense model.
Per-record negative log-likelihood and token counts are retained so aggregation
can be independently checked. Random controls use seed 17.

## Required approval before the real run

The following are intentionally not chosen by implementation code:

1. corpus name, exact revision, and locally materialized JSONL file;
2. evaluation record count and maximum tokens per record;
3. maximum acceptable relative perplexity increase, fixed before observing runs.

The runner requires this threshold explicitly and hashes the exact corpus. A
small synthetic/toy preflight may verify mechanics but cannot support H5.

## Interpretation gates

- A block configuration is quality-viable only if it meets the pre-registered
  threshold and materially outperforms its random-block control.
- This experiment can support or reject H5 only in its recorded corpus scope.
- It cannot establish end-to-end speed, predictor quality, or general benchmark
  accuracy.
