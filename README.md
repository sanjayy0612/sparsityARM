# ARM-Sparse

ARM-Sparse studies whether input-dependent FFN sparsity can produce actual
wall-clock inference gains on Apple Silicon CPUs. ASTRA maintains research
state and evidence; LUNA will manage scientific writing when verified evidence
is ready. These are parts of one project.

## Current scope

Development targets the available Apple M2 with 8 GiB RAM. The model ladder is
synthetic SwiGLU → Llama 3.2 1B → TinyLlama 1.1B → optional Llama 3.2 3B if
memory and runtime permit. CPU measurements are primary; no GPU is required.

EXP-001 is complete: native CPU execution for a synthetic 2048 → 8192 → 2048
FFN, comparing Accelerate dense, irregular skipping, native packed blocks and
Accelerate packed blocks. Python kernels remain tested correctness references.
Real-model integration, deployable selection and hardware-counter profiling
have not been implemented. LUNA tooling and the LaTeX manuscript are deferred.

## Install and test

Native benchmarks require macOS 15+ and Xcode command-line tools. Dependencies
are declared in `pyproject.toml`; the torch extra supports the Python reference
tests, not full-model loading.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,torch]'
python -m pytest -q
```

## Run the current experiment

```bash
.venv/bin/python benchmarks/run_exp001.py
.venv/bin/python astra/validate_exp001.py research/results/EXP-001/<run-directory>
.venv/bin/python astra/summarize_exp001.py research/results/EXP-001/<run-directory>
```

The runner builds the native library and saves raw timings, weights, masks,
source snapshots, build settings and environment metadata. See the
[protocol](research/experiments/EXP-001/protocol.md) for measurement boundaries.
Large NPZ artifacts stay local and are excluded from Git; preserve them with
their manifests when archiving results.

## Evidence and next step

Read the [full measured results](research/results/EXP-001/m2-primary-20260906/summary.md)
and [research assessment](research/experiments/EXP-001/findings.md).
Some equal-work clustered configurations favored blocks. A repeatable benefit
while computing additional neurons was not established. These are synthetic
executor observations, not end-to-end LLM speedups.

The next proposed measurement is real Llama 3.2 1B mask coverage and locality,
before predictor work. Hypotheses and evidence-backed observations are tracked
in [research state](research/project.yaml), [hypotheses](research/hypotheses.yaml)
and [claims](research/claims.yaml).

## Repository

- `armsparse/`: Python reference kernels, masks and tested reference runtime.
- `cpp/`: native CPU executors.
- `benchmarks/`: EXP-001 runner and native bridge; `results/` preserves legacy measurements.
- `astra/`: experiment integrity checks and result aggregation.
- `research/`: protocols, decisions, hypotheses, claims and measured artifacts.
- `tests/`: reference and native correctness checks.

Historical experiment source snapshots remain immutable inside result bundles.
They document what actually ran and are not active implementation entry points.

## Attribution

The project is informed by DejaVu, LLM in a Flash and NimbleEdge
`sparse_transformers`. It does not claim to invent contextual sparsity.
Novelty and publication claims require further verified literature and evidence.
