# ARM-Sparse

ARM-Sparse is a reproducible research prototype for comparing dense, contextual sparse, and contiguous block-sparse Llama FFN execution on CPU—particularly Arm CPUs. It implements the reference algorithms first; benchmarks record real measurements only.

## Status

The repository contains the Phase 0–7 Python reference path: deterministic selection/masking, structured blockification, FFN reference kernels, one-time weight packing, CPU model loading, and JSON benchmark infrastructure. The C++ hot kernel and hardware-specific tuning remain the next implementation phase.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,torch]'
python -m pytest
python scripts/collect_system_info.py
```

## Benchmark

Supply a local Hugging Face-compatible Llama model path or model ID (access may be required for Meta Llama models):

```bash
python benchmarks/benchmark_dense.py --model /path/to/Llama-3.2-3B-Instruct
python benchmarks/benchmark_sparse.py --model /path/to/Llama-3.2-3B-Instruct --sparsity 0.30
python benchmarks/benchmark_armsparse.py --model /path/to/Llama-3.2-3B-Instruct --sparsity 0.30 --block-size 32
```

`sparse` and `armsparse` are reference FFN kernels today, intentionally suitable for correctness and profiling rather than a speed claim. Results are written to `benchmarks/results/`.

## Attribution

This project is inspired by DejaVu contextual sparsity, LLM in a Flash, and NimbleEdge `sparse_transformers`. ARM-Sparse's proposed contribution is the measured conversion of dynamic selections into Arm-friendly contiguous blocks; it does not claim contextual sparsity as a new idea.

See [PLAN.md](PLAN.md), [docs/architecture.md](docs/architecture.md), and [RESULTS.md](RESULTS.md).
