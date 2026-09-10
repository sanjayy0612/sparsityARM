# EXP-010 explicit NEON B8 findings

Run: `m2-neon-b8-20260910`

The run completed from clean commit
`4122347db94afa3c8419a573b9eafbfbef3d376a`. The independent validator checked
committed-source and mask hashes and recomputed every median and p95 from 3,200
retained timings per implementation in each of three runs.

| run | Accelerate dense | native B8 | explicit NEON B8 | NEON vs native | NEON vs dense |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.7872 ms | 5.7207 ms | 5.9344 ms | 3.60% slower | 36.18% slower |
| 2 | 4.1503 ms | 5.9593 ms | 6.2296 ms | 4.34% slower | 33.38% slower |
| 3 | 4.1729 ms | 5.8261 ms | 6.2444 ms | 6.70% slower | 33.17% slower |

## Interpretation

This direct intrinsic implementation does not improve the executor. The
existing optimized build likely already vectorizes these simple dot products
and schedules them more effectively; assembly inspection would be required to
confirm that mechanism. Explicit SIMD alone is therefore not the missing
ingredient.

Do not make mode 4 the production path. Preserve it as a measured negative
control. The next plausible executor experiment is not more hand-written dot
products: it should reduce kernel granularity and call overhead by coalescing
contiguous active B8 runs into larger matrix operations while retaining the
interleaved locality lesson from EXP-009. With a median 93 runs per mask, this
still has meaningful dispatch risk and must be benchmarked rather than assumed.

This remains a standalone FP32 executor result at approximately 9.96% sparsity,
not end-to-end Llama inference or evidence of a deployable selector.
