# EXP-001 findings

The primary sweep completed on the local Apple M2 with 8 GiB RAM. All 72
cases passed numerical validation, and all source/data hashes and raw sample
counts passed the ASTRA validator. The expanded unit suite passed 36 tests.

See the [complete measured table](../../results/EXP-001/m2-primary-20260906/summary.md),
[per-executor CSV](../../results/EXP-001/m2-primary-20260906/summary.csv), and
[manifest](../../results/EXP-001/m2-primary-20260906/manifest.json).

## Observations

- Scattered neuron-mask expansion erased nearly all sparsity. B=16/32/64
  activated every block for all sampled inputs at every target sparsity.
- Several equal-work clustered configurations had lower native-block median
  latency in every run: B=64 at target 20% and 30%, and B=32 at target 40%.
  The full table retains every losing and winning configuration.
- No configuration beat base irregular execution in every run while computing
  additional neurons. The primary H2 tradeoff is not established.
- Dense-relative speedups varied substantially across independent runs. The
  repeatability screen is descriptive and does not supply confidence intervals
  or prove a cache/SIMD mechanism.

## Hypothesis assessment

H1 remains a hypothesis with configuration-dependent observations; operation
reduction alone did not reliably predict latency. H2 is not supported by the
extra-work controls in this sweep. H3 has granularity-dependent measurements,
but no established general optimum. H4 and H5 remain untested.

## Interpretation and next proposed measurement

There is a measurable equal-work execution benefit in some cases, but it is
not enough to justify assuming that expanding real contextual masks will help.
The next useful uncertainty to resolve is whether real Llama 3.2 1B FFN masks
have enough contiguous coverage to preserve meaningful block sparsity. Record
oracle activation masks and measure their coverage before building a predictor
or claiming an end-to-end speedup. This is a proposed next research step; no
model download, model integration or quality evaluation occurred in EXP-001.

Keep thermal/core-scheduling variability and the backend/layout differences in
scope when interpreting these results. No cache counters or SIMD utilization
were measured. This experiment does not establish novelty or publication readiness.
