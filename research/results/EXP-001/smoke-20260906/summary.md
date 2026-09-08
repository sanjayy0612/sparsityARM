# EXP-001 measured results

Shape: 32 → 128 → 32; CPU FP32, one thread.

24 cases; 1 independent seeds; 2 warmups and 5 samples per executor/case. All raw samples retained.

Synthetic executor measurements only. No real model, selector, language-quality evaluation, cache counters or end-to-end generation was measured.

## Latency and equal-work controls

Cells are the median of independent-run medians (ms). Per-run p95 values are in summary.csv and case JSON; no pooled p95 is reported. I = irregular base mask; IE = irregular expanded mask; BN = native block; BA = Accelerate block.

| Mask family | Target skip | B | Realized block skip | Dense | I | IE | BN | BA | BN speedup vs dense range | BN speedup vs I range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scattered | 20% | 8 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.002 | 0.003 | 0.585–0.585× | 0.756–0.756× |
| scattered | 20% | 16 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.722–0.722× | 0.945–0.945× |
| scattered | 20% | 32 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 | 0.823–0.823× | 0.971–0.971× |
| scattered | 20% | 64 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.863–0.863× | 1.035–1.035× |
| scattered | 30% | 8 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.002 | 0.003 | 0.463–0.463× | 0.574–0.574× |
| scattered | 30% | 16 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.595–0.595× | 0.667–0.667× |
| scattered | 30% | 32 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.788–0.788× | 1.000–1.000× |
| scattered | 30% | 64 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.002 | 0.001 | 0.684–0.684× | 0.790–0.790× |
| scattered | 40% | 8 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.002 | 0.003 | 0.565–0.565× | 0.652–0.652× |
| scattered | 40% | 16 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 | 0.743–0.743× | 0.772–0.772× |
| scattered | 40% | 32 | 0.0000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.834–0.834× | 0.866–0.866× |
| scattered | 40% | 64 | 0.0000% | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.774–0.774× | 0.839–0.839× |
| clustered | 20% | 8 | 18.7500% | 0.001 | 0.002 | 0.002 | 0.002 | 0.003 | 0.805–0.805× | 1.139–1.139× |
| clustered | 20% | 16 | 25.0000% | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 | 0.930–0.930× | 1.036–1.036× |
| clustered | 20% | 32 | 25.0000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 1.000–1.000× | 1.291–1.291× |
| clustered | 20% | 64 | 0.0000% | 0.001 | 0.002 | 0.002 | 0.002 | 0.001 | 0.667–0.667× | 1.083–1.083× |
| clustered | 30% | 8 | 31.2500% | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.866–0.866× | 0.933–0.933× |
| clustered | 30% | 16 | 25.0000% | 0.001 | 0.002 | 0.002 | 0.001 | 0.001 | 0.999–0.999× | 1.692–1.692× |
| clustered | 30% | 32 | 25.0000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 1.039–1.039× | 1.120–1.120× |
| clustered | 30% | 64 | 50.0000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 1.286–1.286× | 1.000–1.000× |
| clustered | 40% | 8 | 37.5000% | 0.001 | 0.002 | 0.002 | 0.001 | 0.002 | 0.824–0.824× | 1.089–1.089× |
| clustered | 40% | 16 | 37.5000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 1.273–1.273× | 1.227–1.227× |
| clustered | 40% | 32 | 50.0000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 1.319–1.319× | 0.955–0.955× |
| clustered | 40% | 64 | 50.0000% | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 1.561–1.561× | 1.624–1.624× |

## Memory, correctness and overhead

Peak process RSS including all layouts and FP64 validation: 37.5–37.8 MiB. This is not per-executor deployment memory.

Maximum recorded relative L2 error against FP64 reference: 2.42e-07. Every saved input/mask passed rtol=2e-4, atol=2e-5 before timing.

Packing, synthetic-mask construction and standalone scan/compaction timings are retained per case. The scan diagnostic is not subtracted from executor time. Deployable selector cost is unknown, represented as null.

## Interpretation boundaries

- Scattered-to-block expansion may erase almost all sparsity; compare realized active counts before interpreting latency.
- Clustered masks deliberately favor block execution and use equal neuron sets across executors. They are not representative model masks until validated against real activations.
- IE versus BN compares the same expanded neurons. BN versus BA also changes the arithmetic backend. No isolated cache/SIMD causal claim follows from these timings.
- A configuration faster than dense here is not a deployable LLM speedup; selector, attention, normalization, runtime integration and quality remain unmeasured.
- Single-thread macOS scheduling and thermal/background activity are uncontrolled; results do not establish optimality on other ARM CPUs.

Sources: manifest.json, its hashed case_artifacts, raw case JSON, and saved weights/input/mask NPZ files. summary.csv contains one row per executor/case.
