# EXP-003 — Quality under activation-aware block masks

Status: screening protocol approved 2026-09-09; run in progress.

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

## Approved screening run

Before observing EXP-003 results, the human approved:

- official `Salesforce/wikitext` WikiText-2 raw test split, repository revision
  `f776294184f13b8ff2337b3841cf9269a6216d1e`;
- the first 16 non-heading test rows containing at least 128 Llama tokens;
- truncation to at most 128 tokens per record;
- 30% target sparsity;
- dense, neuron top-k, B=8/16/32/64 direct-block masks, and corresponding
  random-block controls;
- at most 5% relative perplexity increase as the screening quality threshold.

The downloaded Parquet SHA-256 is
`3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925`.
The runner requires the threshold explicitly and hashes the materialized corpus.
This small screening sample is directional and cannot alone establish H5 broadly.

## Interpretation gates

- A block configuration is quality-viable only if it meets the pre-registered
  threshold and materially outperforms its random-block control.
- This experiment can support or reject H5 only in its recorded corpus scope.
- It cannot establish end-to-end speed, predictor quality, or general benchmark
  accuracy.
