# EXP-001 measured results

Shape: 2048 → 8192 → 2048; CPU FP32, one thread.

72 cases; 3 independent seeds; 20 warmups and 200 samples per executor/case. All raw samples retained.

Synthetic executor measurements only. No real model, selector, language-quality evaluation, cache counters or end-to-end generation was measured.

## Latency and equal-work controls

Cells are the median of independent-run medians (ms). Per-run p95 values are in summary.csv and case JSON; no pooled p95 is reported. I = irregular base mask; IE = irregular expanded mask; BN = native block; BA = Accelerate block.

| Mask family | Target skip | B | Realized block skip | Dense | I | IE | BN | BA | BN speedup vs dense range | BN speedup vs I range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scattered | 20% | 8 | 0.0000% | 4.483 | 5.063 | 5.804 | 8.401 | 9.777 | 0.496–0.735× | 0.580–0.693× |
| scattered | 20% | 16 | 0.0000% | 4.188 | 4.071 | 4.213 | 4.792 | 6.760 | 0.665–0.948× | 0.763–0.874× |
| scattered | 20% | 32 | 0.0000% | 4.381 | 4.962 | 5.683 | 5.631 | 5.140 | 0.757–0.969× | 0.861–0.906× |
| scattered | 20% | 64 | 0.0000% | 4.409 | 4.737 | 5.355 | 5.406 | 5.242 | 0.611–1.066× | 0.875–1.023× |
| scattered | 30% | 8 | 0.0163% | 4.360 | 5.275 | 6.545 | 9.415 | 11.401 | 0.419–0.728× | 0.554–0.669× |
| scattered | 30% | 16 | 0.0000% | 4.269 | 4.771 | 5.798 | 6.723 | 7.849 | 0.631–0.841× | 0.710–0.837× |
| scattered | 30% | 32 | 0.0000% | 4.347 | 5.239 | 6.305 | 6.367 | 5.608 | 0.638–0.993× | 0.782–0.909× |
| scattered | 30% | 64 | 0.0000% | 4.379 | 4.531 | 5.280 | 5.170 | 5.061 | 0.618–1.011× | 0.810–0.933× |
| scattered | 40% | 8 | 0.0651% | 4.220 | 4.053 | 4.491 | 6.047 | 8.278 | 0.391–0.698× | 0.487–0.670× |
| scattered | 40% | 16 | 0.0000% | 4.357 | 5.438 | 7.566 | 8.561 | 9.251 | 0.501–0.960× | 0.630–0.864× |
| scattered | 40% | 32 | 0.0000% | 4.595 | 4.757 | 6.031 | 5.913 | 5.556 | 0.573–1.053× | 0.710–0.954× |
| scattered | 40% | 64 | 0.0000% | 4.418 | 4.697 | 5.783 | 5.699 | 5.181 | 0.740–1.066× | 0.794–0.995× |
| clustered | 20% | 8 | 20.0195% | 4.490 | 4.316 | 4.479 | 5.968 | 7.594 | 0.482–0.953× | 0.668–0.763× |
| clustered | 20% | 16 | 19.9219% | 4.419 | 5.046 | 5.075 | 5.737 | 6.680 | 0.624–1.108× | 0.871–0.918× |
| clustered | 20% | 32 | 19.9219% | 4.277 | 4.417 | 4.419 | 4.580 | 4.074 | 0.731–1.218× | 0.954–0.992× |
| clustered | 20% | 64 | 20.3125% | 4.277 | 4.618 | 4.617 | 4.459 | 4.114 | 0.952–1.326× | 1.027–1.081× |
| clustered | 30% | 8 | 29.9805% | 4.527 | 4.436 | 4.423 | 6.200 | 7.278 | 0.583–0.967× | 0.671–0.741× |
| clustered | 30% | 16 | 30.0781% | 4.222 | 3.111 | 3.087 | 3.424 | 4.790 | 0.737–1.233× | 0.873–0.912× |
| clustered | 30% | 32 | 30.0781% | 4.383 | 4.062 | 4.034 | 3.996 | 3.748 | 0.899–1.376× | 0.998–1.028× |
| clustered | 30% | 64 | 29.6875% | 4.268 | 4.212 | 4.189 | 4.035 | 3.673 | 1.007–1.471× | 1.012–1.067× |
| clustered | 40% | 8 | 40.0391% | 4.842 | 3.531 | 3.543 | 5.114 | 5.847 | 0.833–1.147× | 0.691–0.749× |
| clustered | 40% | 16 | 40.0391% | 4.307 | 3.185 | 3.152 | 3.238 | 4.411 | 0.937–1.391× | 0.863–0.984× |
| clustered | 40% | 32 | 39.8438% | 4.671 | 3.097 | 3.066 | 2.972 | 3.503 | 0.997–1.674× | 1.031–1.042× |
| clustered | 40% | 64 | 39.8438% | 4.412 | 3.207 | 3.188 | 3.223 | 3.113 | 1.291–1.773× | 0.995–1.092× |

## Repeatability screen

The following configurations have a lower native-block median than base-irregular median in every independent run. This is a descriptive screen of the complete grid, not a significance test or proof of an optimum.

- clustered, target 20%, B=64: 1.027–1.081× versus base irregular
- clustered, target 30%, B=64: 1.012–1.067× versus base irregular
- clustered, target 40%, B=32: 1.031–1.042× versus base irregular

Configurations passing the same screen while computing additional neurons:

None. H2's extra-computation benefit is not established by this sweep.

## Memory, correctness and overhead

Peak process RSS including all layouts and FP64 validation: 469.7–500.9 MiB. This is not per-executor deployment memory.

Maximum recorded relative L2 error against FP64 reference: 1.72e-06. Every saved input/mask passed rtol=2e-4, atol=2e-5 before timing.

Packing, synthetic-mask construction and standalone scan/compaction timings are retained per case. The scan diagnostic is not subtracted from executor time. Deployable selector cost is unknown, represented as null.

## Interpretation boundaries

- Scattered-to-block expansion may erase almost all sparsity; compare realized active counts before interpreting latency.
- Clustered masks deliberately favor block execution and use equal neuron sets across executors. They are not representative model masks until validated against real activations.
- IE versus BN compares the same expanded neurons. BN versus BA also changes the arithmetic backend. No isolated cache/SIMD causal claim follows from these timings.
- A configuration faster than dense here is not a deployable LLM speedup; selector, attention, normalization, runtime integration and quality remain unmeasured.
- Single-thread macOS scheduling and thermal/background activity are uncontrolled; results do not establish optimality on other ARM CPUs.

Sources: manifest.json, its hashed case_artifacts, raw case JSON, and saved weights/input/mask NPZ files. summary.csv contains one row per executor/case.
