# EXP-008 — Hot/cold reordered block quality screen

Status: implementation complete; held-out run not started.

## Question

Does the fixed hot/cold frequency layout from EXP-007 preserve Llama 3.2 1B
language quality under direct block-aware masking better than the original and
LSH layouts tested previously?

## Fixed inputs

- Model: pinned `meta-llama/Llama-3.2-1B` revision
  `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Layout: `hot_cold` permutation from the validated EXP-007 artifact.
- Layout calibration: EXP-002 prompts P01–P04 only.
- Held-out corpus: the same 16×128-token WikiText-2 screen used by EXP-003,
  EXP-004 and EXP-006, totaling 2,032 predicted tokens.
- CPU-only BF16 evaluation with one Torch thread.

## Conditions

- Dense baseline.
- Output-contribution-ranked blocks at B8 and B16.
- Target block sparsity 20% and 30%.
- Existing acceptance gate: no more than 5% relative perplexity increase.

B32 is excluded because EXP-007 still found at least 98.21% mixed held-out B32
blocks and previous quality deteriorated strongly as block size increased.

## Boundaries

The mask uses the post-SwiGLU activation and exact down-projection block norm,
so it is an oracle/reference quality test—not a deployable selector. Evaluation
duration is operational metadata and must not be reported as sparse inference
latency. A passing condition may advance to replay-executor measurement; a
failing screen must not.
