# EXP-011 contiguous-run coalescing findings

Run: `m2-coalesced-b8-20260910`

The run completed from clean commit
`7b676e727694b6471ae442dcb82745dabde1301c`. The validator checked source and
mask hashes and recomputed every median and p95 from 3,200 retained timings per
implementation and run.

| run | Accelerate dense | native B8 | coalesced B8 | vs native | vs dense |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.0619 ms | 5.7492 ms | 8.1881 ms | 29.79% slower | 50.39% slower |
| 2 | 4.0471 ms | 5.8229 ms | 8.1430 ms | 28.49% slower | 50.30% slower |
| 3 | 4.0427 ms | 5.7233 ms | 8.1101 ms | 29.43% slower | 50.15% slower |

## Conclusion

Contiguous-run coalescing fails decisively for these masks. Their median 93
active runs require roughly 279 small BLAS calls per FFN, and the larger
per-run operations do not recover that dispatch cost. This mechanism is an
interpretation supported by the run count and timing result, not a direct
per-call hardware-counter measurement.

Do not optimize this path further at approximately 10% sparsity. Together with
EXP-009 and EXP-010, the evidence now rejects mask scanning, simple explicit
SIMD and active-run BLAS coalescing as solutions to the current dense gap. A
meaningful next step requires changing a larger premise—such as quantized
packed kernels, substantially higher quality-compatible sparsity, or a stronger
existing sparse runtime baseline—rather than another small loop variation.

This remains standalone FP32 executor evidence, not end-to-end Llama inference.
