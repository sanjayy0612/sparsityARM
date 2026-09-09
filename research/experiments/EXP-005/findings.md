# EXP-005 replay executor findings

Run: `m2-replay-20260909`

## Observation

All three runs completed with 200 retained timings per executor and passed the
independent FP64 correctness check. Each replay mask activated 7,376 of 8,192
neurons (9.961% realized sparsity).

| run | dense median (ms) | irregular median (ms) | B8 block median (ms) | block vs dense |
|---:|---:|---:|---:|---:|
| 1 | 4.0058 | 3.9311 | 5.5442 | 38.41% slower |
| 2 | 4.1293 | 3.9892 | 5.6679 | 37.26% slower |
| 3 | 4.1339 | 4.0960 | 5.7364 | 38.77% slower |

The native irregular executor was only 1.90%, 3.51%, and 0.93% faster than
dense across the three runs, far below the idealized 11.06% projection speedup.
The native B8 executor was also 40.05–42.08% slower than equal-work irregular.

## Interpretation

The only EXP-004 quality-compatible configuration does not pass the standalone
executor gate in the current portable FP32 kernel. At merely 10% skipped work,
block-loop and accumulation costs exceed the saved arithmetic. The result also
supports H1 in this narrow scope: irregular sparsity underdelivered substantially
relative to theoretical operation reduction.

## Conclusion and limitation

Do not claim a B8/10% M2 speedup and do not integrate this kernel into full
Llama inference. The current quality/speed intersection is empty: B8/10% keeps
screening quality but loses latency; 20–30% could save more work but failed the
quality gate.

This result is for one portable FP32 implementation and one Apple M2. It does
not reject better packed microkernels, quantization, neuron reordering, or masks
that preserve quality at higher sparsity. It is standalone FFN evidence, not
end-to-end token latency.
