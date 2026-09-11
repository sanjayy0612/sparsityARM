# ARM-Sparse

ARM-Sparse is a reproducible investigation of one systems question:

> Can dynamic FFN block sparsity preserve useful LLM quality **and** beat an
> optimized dense CPU baseline on an Apple M2?

## Answer from the completed investigation

**Not in the tested configuration.** Packed B8 execution became faster only at
sparsity levels that failed the pre-registered 5% relative-perplexity gate.
The project found a real kernel break-even, but no verified quality–speed
intersection.

![Quality-speed gap](paper/figures/quality-speed-frontier.svg)

| Model | Highest measured B8 sparsity passing quality gate | First measured B8 speedup point | Verified overlap |
|---|---:|---:|:---:|
| Llama 3.2 1B | 10% | 40% | No |
| TinyLlama 1.1B | 20% | 50% | No |

This is a bounded negative result, not proof that block sparsity can never
work. It applies to single-token, one-thread Apple M2 execution; oracle
output-contribution masks; packed FP32 B8 kernels; and the recorded evaluation
corpus. Selector cost was excluded.

## Five decisive experiments

- **EXP-004 — Llama quality frontier:** B8 passed at 10% sparsity
  (+2.520% perplexity) and failed at 20% (+8.773%).
- **EXP-005 — Real-mask replay:** the quality-compatible Llama B8/10% masks
  were 37–39% slower than optimized dense execution.
- **EXP-012 — Llama runtime frontier:** B8 first beat Accelerate dense at the
  40% measured grid point (+1.6% to +3.1% across three runs).
- **EXP-016 — TinyLlama validation:** B8 passed at 10% and 20%, then failed at
  30%; B32 was less quality-efficient.
- **EXP-018 — TinyLlama runtime frontier:** B8 first beat dense at the 50%
  measured grid point (+10.7% to +17.1% across three runs).

The other experiments are retained as diagnostics, controls, implementation
iterations, and quantized-runtime probes. They are not hidden; they simply do
not carry the main conclusion.

## Reproduce and audit

On the recorded macOS environment, the complete verification command is:

```bash
make verify-package
```

It validates the five core evidence bundles plus supporting production-baseline
artifacts, regenerates the summary/figures deterministically, and runs the test
suite. See [REPRODUCING.md](REPRODUCING.md) for setup and boundaries.

Generated research outputs:

- [Technical report source](paper/main.tex)
- [Technical report PDF](paper/arm-sparse-report.pdf)
- [Public technical article](docs/article.md)
- [Machine-readable summary](research/package/summary.json)
- [Claim audit](research/package/claim-audit.md)

## Repository map

- `armsparse/` — tested Python reference implementations.
- `cpp/` — native Apple M2 FFN executors.
- `benchmarks/` — experiment runners.
- `research/` — protocols, decisions, literature, claims, and immutable results.
- `research_tools/` — validators and deterministic packaging tools.
- `paper/` — canonical LaTeX report, figures, tables, and PDF preview.
- `tests/` — correctness and regression checks.

## Research interpretation

The useful result is the measured mismatch: blockification can reduce kernel
latency, but the tested models lost acceptable quality before enough work was
removed. Future work should target the quality frontier—through training-aware
block structure or a selector/layout co-design—not more micro-optimization of
the current B8 oracle-mask executor.

The project is informed by contextual-sparsity and memory-layout work including
DejaVu, LLM in a Flash, PowerInfer, PowerInfer-2, and Dynamic Input Pruning. It
does not claim to invent contextual sparsity or neuron grouping.

## AI assistance

SOLARIS/Codex assisted with implementation, experiment execution, validation,
analysis, and writing. All quantitative statements in the package are derived
from committed artifacts; AI-generated prose is not treated as evidence.
