# Reproducing the ARM-Sparse investigation

## Supported environment

The recorded results were produced on an Apple M2, ARM64 macOS, with CPU-only
execution. Native kernels require Xcode command-line tools and Accelerate.
Model downloads are not required to validate the committed result package.

## One-command verification

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,torch]'
make verify-package
```

`make verify-package` performs three checks:

1. validates the committed manifests and measurements for EXP-004, EXP-005,
   EXP-012, EXP-014, EXP-016, and EXP-018;
2. verifies that the generated CSV, JSON, SVG figures, and LaTeX table exactly
   match their source artifacts;
3. runs the full automated test suite.

## Regenerate publication artifacts

```bash
make package
```

This rewrites only deterministic derived artifacts. Raw experiment bundles are
never modified by packaging.

## Measurement boundary

- Hardware: Apple M2 CPU.
- Primary runtime comparison: packed native FP32 B8 FFN versus optimized
  Accelerate dense FFN.
- Workload: single-token FFN calls, one thread, recorded warmups/repetitions.
- Masks: replayed/oracle masks; selector time is excluded.
- Quality: relative perplexity on 16 fixed excerpts (2,032 predicted tokens),
  with a 5% acceptance gate.
- Statistics: per-run medians and three-run ranges. They are not confidence
  intervals.

## What cannot be inferred

The artifacts do not establish an end-to-end token-generation speedup, a
deployable selector, generality beyond Apple M2, or impossibility of block
sparsity under trained block-aware models. The result is a controlled
quality-versus-executor break-even investigation.
