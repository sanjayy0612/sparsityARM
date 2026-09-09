# EXP-006 — Coactivation-aware neuron reordering

Status: held-out B8/B16/B32 at 20/30% approved 2026-09-09; run in progress.

## Question

Can a fixed, function-preserving neuron permutation place similarly activating
neurons next to each other, allowing 20–30% contiguous block sparsity to retain
quality better than the original Llama neuron order?

## Calibration and leakage control

Learn one fixed permutation per layer from the retained EXP-002 prompt
activations only. WikiText-2 remains held out for quality evaluation. The layout
must not be refit after seeing held-out results.

The baseline layout uses centered absolute activation patterns, 32 deterministic
random-hyperplane projections, and lexicographic signature ordering. This is a
cheap locality-sensitive method, not an optimal clustering claim. Its ordering
is hierarchical so B8 groups remain nested within B16 and B32 groups.

## Correctness invariant

Reorder `gate_proj` rows, `up_proj` rows, and `down_proj` columns by the same
permutation. Dense FFN output must remain numerically identical before any
sparsity is introduced.

## Approved held-out screen

- Pinned Llama 3.2 1B, CPU BF16, the same 16×128-token WikiText-2 screen.
- Output-contribution block scoring on the reordered layout.
- B8/B16/B32 at 20% and 30% sparsity.
- Pre-existing ≤5% relative perplexity gate.

This is oracle/reference quality research. Layout learning occurs offline;
post-SwiGLU scoring remains non-deployable and no evaluation duration is a
sparse-inference speed result.
