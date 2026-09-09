# EXP-004 staged screening findings

Run: `m2-oracle-screening-20260909`

## Observation

The seven approved conditions completed over the same 16 WikiText-2 excerpts
and 2,032 predicted tokens per condition as EXP-003. Dense perplexity reproduced
exactly at 22.774209.

| block | sparsity | perplexity | relative to dense | passes ≤5% gate |
|---:|---:|---:|---:|:---:|
| 8 | 10% | 23.3481 | +2.520% | yes |
| 8 | 20% | 24.7721 | +8.773% | no |
| 8 | 30% | 27.3673 | +20.168% | no |
| 16 | 10% | 23.9486 | +5.157% | no |
| 16 | 20% | 26.6550 | +17.040% | no |
| 16 | 30% | 31.7040 | +39.210% | no |

All block masks were ranked by the squared L2 norm of each isolated block's
`down_proj` output contribution. Every aggregate and all corpus, source, and
model hashes were independently validated.

## Interpretation

B8 at 10% sparsity is the first contiguous-block configuration to satisfy the
pre-registered screening quality gate. B16 at 10% narrowly missed by 0.157
percentage points; every 20% and 30% configuration failed clearly.

At 30%, output-norm scoring only slightly improved B8 over EXP-003 activation
sum (27.3673 versus 27.6284 perplexity) and made B16 slightly worse (31.7040
versus 31.3421). It therefore has not demonstrated a consistent advantage over
activation sum. A direct same-run comparison at the viable 10% region is still
required.

## Conclusion and limitation

The next smallest test is activation-sum and weight-proxy scoring at B8/10% and
B16/10%, reusing the same protocol. Executor optimization should target B8/10%
only after that comparison identifies whether the expensive oracle score is
actually necessary.

This is a small screening corpus and post-SwiGLU oracle selection. It does not
establish deployable selection, broad quality retention, or inference speed.
