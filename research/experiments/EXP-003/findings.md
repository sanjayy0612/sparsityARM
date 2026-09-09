# EXP-003 screening findings

Run: `m2-screening-20260909`

## Observation

All ten preregistered conditions completed over 16 WikiText-2 test excerpts and
2,032 predicted tokens per condition. Dense perplexity was 22.7742.

| condition | perplexity | relative to dense | passes ≤5% gate |
|---|---:|---:|:---:|
| neuron top-k, 30% | 22.8210 | +0.21% | yes |
| block B8, 30% | 27.6284 | +21.31% | no |
| block B16, 30% | 31.3421 | +37.62% | no |
| block B32, 30% | 36.9118 | +62.08% | no |
| block B64, 30% | 40.6239 | +78.38% | no |
| random B8, 30% | 372.6979 | +1536.49% | no |
| random B16, 30% | 328.2134 | +1341.16% | no |
| random B32, 30% | 1764.1003 | +7646.04% | no |
| random B64, 30% | 336.8844 | +1379.24% | no |

Every aggregate was independently recomputed from saved per-record negative
log-likelihoods. The corpus, source, and model hashes were also checked.

## Interpretation

Activation-magnitude neuron top-k preserved screening perplexity at 30%
sparsity. Direct contiguous blocks scored by summed absolute activation did not:
even B8 missed the preregistered threshold substantially, and degradation grew
monotonically through B64.

The ranked block masks were far better than their one-seed random controls, so
the activation score carries useful information. That is insufficient evidence
that the current masks are usable: none passed the absolute quality gate.

## Conclusion and limitation

Do not proceed directly to optimizing a 30%-sparse executor for these masks.
The next smallest experiment should test whether lower block sparsity (starting
at 10% and 20%) or a contribution-aware block score can meet the quality gate.

This is a 16-excerpt screening experiment, not a full WikiText-2 evaluation.
The masks are post-SwiGLU oracle/reference masks, random controls use only seed
17, and no timing here is deployable sparse-inference performance.
