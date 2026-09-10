# EXP-015 Q8_0 B32 replay findings

## Result

The fixed-mask B32 executor did not produce a reliable speedup over its matched
dense Q8_0 control at any measured sparsity from 10% through 80%.

| Target sparsity | Dense median range (ms) | B32 median range (ms) | Dense / B32 speedup range |
| ---: | ---: | ---: | ---: |
| 10% | 1.391-1.435 | 3.247-3.386 | 0.419-0.429x |
| 20% | 1.312-1.455 | 2.581-3.266 | 0.445-0.508x |
| 30% | 1.349-1.394 | 2.567-2.729 | 0.511-0.525x |
| 40% | 1.348-1.398 | 2.285-2.469 | 0.566-0.590x |
| 50% | 1.324-1.386 | 1.941-2.158 | 0.642-0.682x |
| 60% | 1.319-1.447 | 1.690-2.057 | 0.704-0.791x |
| 70% | 1.309-1.417 | 1.384-1.613 | 0.879-0.946x |
| 80% | 1.363-1.529 | 1.146-1.584 | 0.965-1.189x |

At 80%, two runs favored B32 by 8.4% and 15.9%, while the third found B32
3.6% slower. This is not a reproducible crossover. Sparse outputs agreed with
the masked-dense Q8_0 control to relative L2 error at most 3.43e-7.

## Interpretation

Matching the neuron block to Q8_0's 32-value storage group is insufficient by
itself. The dense primitive amortizes one call over an entire row, whereas this
minimal sparse path invokes the same primitive for scattered active groups and
accumulates partial results. That dispatch and accumulation cost dominates until
extreme sparsity.

The result does not prove that an optimized fused B32 kernel cannot cross over.
It rejects the minimal group-skipping implementation as a useful path. Building
a new fused quantized kernel is outside the agreed scope because existing Llama
3.2 1B quality measurements already show severe B32 degradation at moderate
sparsity: EXP-003 measured +62.08% screening perplexity at 30% sparsity.

## Boundary

This is a synthetic FFN executor measurement using llama.cpp Q8_0 primitives,
not a full GGML graph or end-to-end token-generation benchmark. Masks are fixed
and synthetic, and selector overhead is absent. EXP-014 remains the production
dense llama.cpp baseline.

## Decision

Stop the custom quantized-kernel branch here. The next useful experiment is a
bounded TinyLlama 1.1B validation of the quality-compatible block-sparsity ceiling
and its relationship to the already measured executor break-even behavior.
