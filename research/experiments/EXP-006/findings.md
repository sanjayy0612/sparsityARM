# EXP-006 reordered-layout findings

Run: `m2-reordered-screening-20260909`

## Observation

The fixed calibration-only layout passed dense-equivalence checks in all 16
layers; maximum relative L2 error was `3.04e-7`. Dense perplexity reproduced at
22.774209 over 2,032 predicted held-out tokens.

| block | sparsity | perplexity | relative to dense | passes ≤5% gate |
|---:|---:|---:|---:|:---:|
| 8 | 20% | 24.7242 | +8.562% | no |
| 8 | 30% | 27.0788 | +18.901% | no |
| 16 | 20% | 26.0054 | +14.188% | no |
| 16 | 30% | 30.1641 | +32.448% | no |
| 32 | 20% | 28.3416 | +24.446% | no |
| 32 | 30% | 34.6912 | +52.327% | no |

All model, corpus, layout and committed-source hashes and every token-weighted
loss aggregate were independently validated.

## Interpretation

The LSH layout improved every matched B8/B16 output-norm result versus the
original layout, but not enough to pass:

- B8/20%: 24.7721 → 24.7242 perplexity.
- B8/30%: 27.3673 → 27.0788.
- B16/20%: 26.6550 → 26.0054.
- B16/30%: 31.7040 → 30.1641.

The relative benefit grew for coarse/high-sparsity cases, yet the absolute
quality-speed intersection remains empty. B32 lacks a matched original-layout
output-norm baseline, so no reordering improvement is claimed for B32.

## Conclusion and limitation

This cheap LSH baseline does not make 20–30% contiguous sparsity quality-safe.
Do not proceed to executor timing for these masks. Before trying more layout
heuristics, the next step should be a literature-grounded review of balanced
coactivation clustering/reordering methods and an explicit clustering-quality
diagnostic on calibration versus held-out activations.

This remains a small held-out screen with an oracle post-SwiGLU score. It does
not establish deployable prediction, broad quality, or runtime performance.
