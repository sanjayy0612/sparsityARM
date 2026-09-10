# EXP-001: synthetic CPU SwiGLU execution

Status: initial protocol, authorized by the user's instruction to implement and collect results on 2026-09-06. This supersedes the older 3B-first roadmap for the initial experiment.

## Question and scope

Measure dense, irregular neuron and contiguous block execution on the local M2 CPU. Primary shape is batch=1, tokens=1, hidden=2048, intermediate=8192, FP32. No model is downloaded. Masks are synthetic controls, not contextual predictions or oracle importance estimates. No language-quality or end-to-end inference claim is possible.

## Cases

Block sizes 8/16/32/64 and requested sparsities 20/30/40%, with two mask families:

- Scattered: uniformly sample a fixed number of active neurons without replacement. Expand to every block touching an active neuron, without dropping any selected neuron. This directly measures extra-work inflation.
- Clustered: uniformly sample a fixed number of active full blocks. Expand these into individual neuron indices for the irregular executor. Both executors compute exactly the same neuron set. This is a favorable locality control, not evidence that real masks have such clustering.

Each case includes Accelerate dense, native irregular on the base mask, native irregular on the expanded mask, native block, and Accelerate block. The expanded irregular control isolates changed neuron count from the change of executor. In clustered cases the two irregular variants are identical controls measured independently.

Gate/up are row-major in every variant. Irregular execution uses a once-transposed neuron-major down matrix. Block execution uses once-packed [blocks, hidden, B] down tiles. Thus the irregular baseline already has contiguous down-column access. The study compares execution layouts and algorithms together; it cannot attribute differences solely to cache behavior or SIMD. Native gate/up dot products share code. Accelerate block is a backend control using GEMV for each block.

## Measurement protocol

- Three independent runs, seeds 1701/1702/1703. Each run uses the same weights across all 24 cases.
- Eight saved input/mask pairs per case, cycled through warmups and samples.
- Twenty warmups per variant, then 200 measurements per variant.
- Randomize case order within each run and variant order within each sample round. Retain the complete schedule.
- Time inside C++ with steady_clock; include mask scanning, branches, output initialization and arithmetic. Exclude Python/ctypes dispatch, packing, synthetic mask construction and correctness validation.
- Every native executor is serial. Accelerate uses checked BLASSetThreading(SINGLE_THREADED). No MPS. No core pinning; heterogeneous CPU scheduling and thermal/background load remain limitations.
- Report median and p95 (NumPy linear interpolation) for each case and run. Do not pool independent runs into a single p95 or discard slow samples.
- Retain every timing. Record exact active counts and projection MACs=3*hidden*active_neurons; FLOPs=2*MACs. This excludes nonlinearities and overhead. Idealized projection-only speedup is 1/(1-realized_sparsity).
- Report down-weight packing separately, and mask scan/compaction as a standalone diagnostic. Scan timings are not subtracted from kernel times and are not claimed to be an additive overhead decomposition. No deployable selector exists; selector cost is null, never zero.
- Record canonical and additional packed weight bytes. Peak process RSS includes validation temporaries and all variants, not just one deployment layout. macOS ru_maxrss is in bytes.
- Save full FP32 weights once per run, plus inputs/masks per case. Save source snapshots, build command, binary hash, git commit, working-tree diff, environment metadata and artifact hashes.

## Correctness and acceptance

Before timing, validate every input/mask/variant against an independent FP64 NumPy masked-dense result with rtol=2e-4, atol=2e-5. Fail the case immediately on disagreement. Unit tests also cover empty/full/partial masks and all block sizes. Reference computes all projections and masks the resulting activation, whereas native sparse paths actually skip work.

Completion means: native correctness passes; all planned cases are retained; provenance validation passes; results and limitations are reported. A speedup is not an acceptance requirement. Synthetic results do not establish H4/H5, deployable performance, novelty, or general ARM hardware behavior.

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/run_exp001.py
.venv/bin/python research_tools/validate_exp001.py research/results/EXP-001/<run-directory>
```

Smoke tests may override shape, sample count and run count. They must remain labeled by their manifest configuration and cannot replace the primary sweep.
