# EXP-008 hot/cold reordered quality findings

Run: `m2-hot-cold-screening-20260910`

The run completed from clean commit
`25c30b725ce98f9abf7973f9b734d64ee398053f`. The independent validator checked
the model, corpus, layout and committed-source hashes; all per-example loss
aggregates; all 16 dense-equivalence checks; and the fixed 5% acceptance gate.
Maximum dense-equivalence relative L2 error was `3.04e-7`.

| block | sparsity | perplexity | relative to dense | passes <=5% gate |
|---:|---:|---:|---:|:---:|
| dense | 0% | 22.7742 | — | — |
| 8 | 20% | 24.3227 | +6.799% | no |
| 8 | 30% | 26.3231 | +15.583% | no |
| 16 | 20% | 25.6049 | +12.429% | no |
| 16 | 30% | 28.7863 | +26.399% | no |

## Comparison with earlier layouts

Hot/cold ordering produced lower perplexity than both original order and the
EXP-006 LSH order in all four matched configurations. For example, B8/20%
improved from 24.7721 (original) and 24.7242 (LSH) to 24.3227. B16/30%
improved from 31.7040 and 30.1641 to 28.7863.

## Interpretation and decision boundary

The EXP-007 activation-concentration improvement translated into consistently
better held-out perplexity, so activation-frequency ordering is a useful fixed
layout. Nevertheless, no tested 20–30% condition met the preregistered quality
gate. It therefore does not provide a mask configuration eligible for executor
timing at those sparsities.

Do not continue accumulating simple layout heuristics. The remaining plausible
region is approximately B8 at 10–20% sparsity, but EXP-005 already found the
current B8/10% native block executor 37–39% slower than dense. The next research
decision should prioritize profiling and improving that executor before paying
for another narrow quality sweep. This experiment remains an oracle/reference
quality screen and provides no deployable selector or latency evidence.
