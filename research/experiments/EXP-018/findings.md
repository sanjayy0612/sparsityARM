# EXP-018 TinyLlama-geometry B8 break-even findings

Run: `m2-tinyllama-b8-break-even-20260911-r1`

## Observation

All 15 cases completed from clean commit `15bfa68`. The validator checked the
committed sources, case hashes, equal-work masks, numerical correctness, and
all retained medians.

| sparsity | native B8 speedup range versus Accelerate dense |
| ---: | ---: |
| 20% | -25.0% to -21.7% |
| 30% | -18.7% to -13.3% |
| 40% | -5.2% to -1.1% |
| 50% | +10.7% to +17.1% |
| 60% | +38.8% to +42.1% |

Negative values mean B8 was slower. The first measured crossover bound is 50%
sparsity; the exact crossover lies between the 40% and 50% grid points and was
not interpolated.

## Quality-speed comparison

EXP-016 found that TinyLlama B8 passed the 5% screening perplexity gate at 20%
sparsity and failed at 30%. EXP-018 finds that the executor remains slower at
20%, 30%, and 40%, and first wins at 50%. Therefore the current TinyLlama B8
configuration has no verified quality-speed intersection.

## Interpretation

Changing from Llama 3.2 1B to TinyLlama improves the quality boundary but does
not close the systems gap. In fact, the smaller 5632-neuron FFN requires more
structured sparsity on this grid than the earlier 8192-neuron synthetic shape,
whose first measured FP32 crossover was 40% in EXP-012. Fixed block overhead is
harder to amortize on the smaller matrix.

## Boundaries and decision

These are favorable synthetic clustered masks with no selector cost, not exact
TinyLlama activation masks. This is a one-token FP32 FFN microbenchmark, not
end-to-end inference.

The present executor-plus-oracle direction has now answered its immediate
question negatively on both primary and secondary models. Further kernel tuning
without a credible path to roughly 50% quality-compatible B8 sparsity would be
drift. Continue only through a separately approved hypothesis that can move the
quality boundary substantially, such as trained block-aware selection or a
reordering method with a preregistered stop condition.
