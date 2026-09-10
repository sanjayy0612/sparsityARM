# EXP-017 TinyLlama-geometry B8 runtime findings

Run: `m2-tinyllama-b8-s20-20260911-r1`

## Observation

All three independent runs completed at 20.028% realized B8 sparsity. Numerical
errors against the corresponding masked-dense references were below 6.48e-7.

| run | FP32 dense (ms) | FP32 B8 (ms) | B8 vs dense | INT8 dense (ms) | INT8 B8 (ms) | B8 vs dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6360 | 3.3031 | 25.31% slower | 4.6405 | 3.8928 | 16.11% faster |
| 2 | 2.6486 | 3.5093 | 32.49% slower | 4.6421 | 3.8892 | 16.22% faster |
| 3 | 2.6908 | 3.4979 | 30.00% slower | 4.6279 | 3.8793 | 16.18% faster |

## Interpretation

TinyLlama's B8/20% quality-compatible point does not beat the optimized FP32
dense executor. The custom INT8 block path consistently beats its matched custom
INT8 dense control, showing that skipped work can matter inside that kernel
family. However, the custom INT8 B8 path still takes approximately 3.88-3.89 ms,
well above the 2.64-2.69 ms Accelerate FP32 dense baseline. The matched INT8 win
therefore does not establish a useful system speedup.

This reproduces the central quality-speed mismatch on TinyLlama geometry: the
quality-compatible block budget is insufficient for the current practical B8
executor to outperform a strong dense baseline.

## Boundaries

The masks are synthetic clustered equal-work controls, not captured TinyLlama
masks. The custom per-tensor weight-only INT8 implementation is not a production
quantized runtime. This experiment measures a one-token FFN microkernel and does
not support an end-to-end token-generation claim.

## Decision

Do not invest further in this custom INT8 implementation or a bespoke B8 Q8_0
format. Together, EXP-015 through EXP-017 show no verified quality-compatible
speedup against a strong practical dense baseline. The next work should synthesize
the negative systems result and decide whether a learned block-aware selector or
weight reordering is justified as a separate research phase.
