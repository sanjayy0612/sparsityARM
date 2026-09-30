# Reproducing the ARM-Sparse investigation

## Supported environment

The recorded results were produced on an Apple M2, ARM64 macOS, with CPU-only
execution. Native kernels require Xcode command-line tools and Accelerate.
Model downloads are not required to validate the committed result package
(see "What a fresh clone verifies" below).

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

## What a fresh clone verifies

Raw inputs are git-ignored (`research/data/**/*.jsonl|npz|gguf|parquet`,
`research/results/**/sources/`, `*.npz`, `*.log`) and the Hugging Face
snapshots live in `~/.cache`. Manifests also record absolute paths from the
original machine; the validators rebase these onto the current checkout.

Always verified, strictly, from committed files: manifest structure and
status, case-artifact hashes, source-file hashes (via `git show` of the
recorded commit, including EXP-005 when `sources/` is absent), recomputed
perplexities, NLL, relative increases and thresholds (EXP-004/016), recomputed
medians and p95 (EXP-005/012/018), correctness tolerances, EXP-014 rows,
EXP-016 model metadata, and the generated package artifacts.

Skipped with a `SKIPPED (not present in this checkout)` line when the file is
absent (hash recorded in the manifest is printed): the WikiText corpus
(EXP-004/016), the Hugging Face model files (EXP-004), the Q8_0 GGUF (EXP-014),
and the EXP-005 mask bank plus raw `.npz` run and weight arrays (including the
mask shape / 922-active-block check). If such a file is present it is always
checked, and a mismatch fails. `verify_core.py` ends with the number of
skipped checks.

To require every input (for example on the machine that produced the data):

```bash
ARMSPARSE_STRICT_INPUTS=1 make verify-package
# or: PYTHONPATH=. .venv/bin/python research_tools/verify_core.py --strict
```

In strict mode any missing input fails the run.

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
