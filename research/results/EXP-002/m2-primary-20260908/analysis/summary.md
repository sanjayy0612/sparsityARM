# EXP-002: Llama 3.2 1B magnitude-mask coverage

Model revision: `4e20de362430cd3b72f300e6b0f18e50e7166e08`. Eight fixed prompts, all 16 layers, CPU BF16 teacher-forced prefill.

Analyzed 7168 layer/token activations. Token positions 8 onward, capped at 64 input tokens per prompt. Every captured activation was bitwise equal to independently recomputed SwiGLU activation.

Means below weight every analyzed layer/token equally. Layer min/max are ranges of layer means, not independent-run uncertainty. The corpus is a pilot convenience sample.

| Target skip | B | Expanded skip mean | Expanded skip layer range | Permuted skip mean | Budget-block actual skip | Neuron retained L1 | Budget-block retained L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | 8 | 0.000218% | 0.000000%–0.000654% | 0.000304% | 20.0195% | 98.5884% | 90.0508% |
| 20% | 16 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 19.9219% | 98.5884% | 87.7206% |
| 20% | 32 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 19.9219% | 98.5884% | 85.8100% |
| 20% | 64 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 20.3125% | 98.5884% | 83.9961% |
| 30% | 8 | 0.006907% | 0.005014%–0.008937% | 0.006639% | 29.9805% | 96.4000% | 83.2581% |
| 30% | 16 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 30.0781% | 96.4000% | 79.9211% |
| 30% | 32 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 30.0781% | 96.4000% | 77.3646% |
| 30% | 64 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 29.6875% | 96.4000% | 75.7676% |
| 40% | 8 | 0.065817% | 0.061471%–0.071498% | 0.064573% | 40.0391% | 92.8896% | 75.4741% |
| 40% | 16 | 0.000027% | 0.000000%–0.000436% | 0.000045% | 40.0391% | 92.8896% | 71.5558% |
| 40% | 32 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 39.8438% | 92.8896% | 68.7140% |
| 40% | 64 | 0.000000% | 0.000000%–0.000000% | 0.000000% | 39.8438% | 92.8896% | 66.4439% |

## Limits

These are magnitude-reference masks, not optimal importance estimates. L1 retention is not language quality. Fixed-budget block selection changes which neurons survive; expansion preserves them but may erase sparsity.

This experiment does not measure sparse execution speed, selector cost, generation quality or end-to-end inference. BF16 captures and teacher-forced prefill must not be conflated with EXP-001 FP32 timings. No claim across ARM processors follows.

Recorded peak capture RSS: 1.342 GiB. Operational capture time: 129.7 seconds (not an inference benchmark).
