# EXP-012 B8 break-even findings

Run: `m2-b8-break-even-20260910`

The sweep completed from clean commit
`103617d26e69556e0e38086787bd60feb9469621`. The validator checked all 24 case
hashes, committed sources, equal-work masks and retained medians.

| sparsity | native B8 speedup range versus dense |
|---:|---:|
| 10% | -28.7% to -26.8% |
| 20% | -20.5% to -15.8% |
| 30% | -11.7% to -8.9% |
| 40% | +1.6% to +3.1% |
| 50% | +19.1% to +22.0% |
| 60% | +44.6% to +48.0% |
| 70% | +89.3% to +93.6% |
| 80% | +178.6% to +193.3% |

Negative values mean the block executor was slower. The first measured
crossover is 40% sparsity; the exact break-even lies somewhere between the
30% and 40% grid points and was not interpolated or claimed more precisely.

## Research conclusion

The current FP32 native B8 executor requires roughly 40% structured sparsity
before it provides even a small reproducible advantage over Accelerate dense.
The real-model quality experiments found large degradation by 30% block
sparsity, including +15.583% perplexity for the strongest hot/cold B8 layout.
Consequently, the project has no demonstrated quality–speed intersection for
this executor and mask family.

This does not reject ARM block sparsity generally. It rejects the present
combination of FP32 kernels, Llama 3.2 1B oracle block masks and tested layouts.
The next justified direction must materially move either boundary: a matched
quantized dense/sparse kernel study to lower runtime break-even, or a learned
block selector/training method to raise quality-compatible sparsity. Given the
project's executor focus and M2 constraint, quantized kernels are the more
aligned next direction.
