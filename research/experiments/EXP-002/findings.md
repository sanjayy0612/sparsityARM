# EXP-002 findings

EXP-002 completed on the Apple M2 using the pinned official Llama 3.2 1B
revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. The capture used CPU BF16
teacher-forced prefill for eight fixed 64-token prompts. Positions 8–63 from all
16 layers produced 7,168 layer/token activations.

All 128 activation files were bitwise-equal to independently recomputed SwiGLU
activations. The independent validator checked the capture and analysis hashes,
mask budgets, expansion, token positions and every aggregate across 86,016
metric rows. Peak capture RSS was 1.342 GiB. The 129.7-second operational
capture duration is not an inference benchmark.

See the [measured table](../../results/EXP-002/m2-primary-20260908/analysis/summary.md),
[aggregate metrics](../../results/EXP-002/m2-primary-20260908/analysis/aggregate.json),
and [capture manifest](../../results/EXP-002/m2-primary-20260908/manifest.json).

## Observations

- Expanding the magnitude-selected neuron set to every touched contiguous block
  removed nearly all intended sparsity. At B=8, mean surviving block sparsity
  was 0.000218%, 0.006907% and 0.065817% for target neuron sparsities of 20%,
  30% and 40%. The median was zero in all three cases.
- At B=32 and B=64, expansion yielded exactly zero block sparsity for every
  layer/token at every tested target. B=16 also yielded zero everywhere at 20%
  and 30%; at 40% its mean was 0.000027%.
- The same masks under random neuron permutations produced nearly identical
  B=8 means: 0.000304%, 0.006639% and 0.064573%. The current model order showed
  no practically useful contiguous clustering under this expansion metric.
- Directly selecting blocks at a fixed budget preserved approximately the
  requested sparsity, but changed neuron membership. At B=8 it retained mean
  activation L1 mass of 90.0508%, 83.2581% and 75.4741% at 20%, 30% and 40%
  sparsity. Larger blocks retained less under this score and budget.
- Neuron top-k retained 98.5884%, 96.4000% and 92.8896% activation L1 mass at
  those targets. Activation L1 mass is a diagnostic surrogate, not model
  quality or down-projection output error.

## Interpretation

The naive strategy of selecting individual neurons and expanding their mask to
fixed contiguous blocks is not viable for these masks: almost every block
contains at least one selected neuron. This agrees with the failure mode seen
using scattered synthetic masks in EXP-001 and now demonstrates it on the
pilot Llama 3.2 1B activation corpus.

This does not reject block execution or H2 generally. It rejects this mask
conversion strategy at 20–40% neuron sparsity. A block-aware selector can retain
the execution structure, but it selects a different neuron set and loses more
activation mass. Its actual model-quality effect must be measured before replay
performance is meaningful.

Natural neuron order did not provide a useful spatial-locality advantage over
the permutation control. Weight/neuron reordering based on co-activation is a
possible later research direction, but it is not supported as a solution until
measured against an unchanged-order baseline and its packing/runtime costs.

## Hypothesis assessment and decision gate

- H2 remains unverified. EXP-002 removes naive neuron-to-block expansion from
  the next executor experiment.
- H3 remains unverified. An execution optimum cannot be selected before a
  quality-constrained block-aware mask exists.
- H5 remains untested. Retained activation mass cannot substitute for
  perplexity or task evaluation.
- H1 and H4 were not tested by this experiment.

The next smallest experiment should evaluate block-aware masking on Llama 3.2
1B using dense replay: measure per-layer FFN output error and end-to-end
perplexity for B=8/16/32/64 at 20/30/40% block sparsity. Only configurations
that satisfy a predefined quality threshold should be replayed through native
execution kernels. That experiment requires a human-approved quality metric,
dataset and acceptance threshold before execution.

## Limits

The prompts are a small, locally authored convenience sample, all truncated to
the same 64-token length. Captures are BF16 and teacher-forced. Tokens and
layers are correlated, and this is one checkpoint on one machine. No selector
latency, kernel latency, generation, perplexity, task accuracy, cache counters,
energy, or cross-hardware behavior was measured.
