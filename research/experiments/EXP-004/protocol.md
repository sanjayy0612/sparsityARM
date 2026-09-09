# EXP-004 — Contribution-aware block selection

Status: implementation complete; real-run grid awaits human approval.

## Question

Can a score that includes `down_proj` weights identify contiguous FFN blocks
whose removal preserves substantially more quality than activation magnitude
alone?

## Scores

1. **Activation sum:** sum of absolute post-SwiGLU activations in each block.
2. **Weight proxy:** activation L2 norm multiplied by the Frobenius norm of the
   corresponding `down_proj` columns. This is cheap but ignores alignment.
3. **Output norm:** squared L2 norm of `W_down,b @ h_b`. A per-block Gram matrix
   computes this exactly for each isolated block without materializing all full
   output vectors.

Output norm is an oracle contribution score, not the exact optimal subset:
different block output vectors may reinforce or cancel each other, and finding
the globally best subset is combinatorial.

## Proposed screening grid

- Same pinned Llama 3.2 1B model, WikiText-2 corpus, 16 records × 128 tokens,
  CPU/BF16/one thread, and ≤5% relative-perplexity gate as EXP-003.
- Block sizes: 8 and 16.
- Sparsities: 10%, 20%, and 30%.
- Dense baseline plus all three scores: 19 total conditions.

All scoring observes dense post-SwiGLU activations. Consequently EXP-004 is a
quality/attainability study—not a deployable selector or latency benchmark.

## Decision rule

A configuration is promising only if it meets the fixed 5% perplexity gate.
Output norm should outperform activation sum before predictor or executor work
is justified. The small corpus remains screening evidence, not broad validation.
