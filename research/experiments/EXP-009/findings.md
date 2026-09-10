# EXP-009 B8 executor profiling findings

Run: `m2-b8-profile-20260910`

The run completed from clean commit
`83e82ea1f44d1342fa26c85f8b4731d3fa9cb9fe`. The validator checked committed
source hashes, the replay-mask hash, all retained raw timings and every
recomputed median and p95 aggregate.

| run | interleaved | staged | gate/up + SwiGLU | down projection | mask scan |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.6395 ms | 6.0886 ms | 2.4999 ms | 3.5713 ms | 0.00125 ms |
| 2 | 5.3017 ms | 5.8559 ms | 2.3857 ms | 3.4274 ms | 0.00113 ms |
| 3 | 5.4688 ms | 5.9024 ms | 2.4164 ms | 3.4604 ms | 0.00117 ms |

Across runs, the down projection consumed 58.53–58.66% of staged median
latency and gate/up plus SwiGLU consumed 40.74–41.06%. Mask compaction consumed
about 0.02%; output initialization was negligible. Each mask contained 922
active blocks arranged into a median 93 contiguous runs.

The staged diagnostic was 7.93–10.45% slower than the original interleaved
executor. Separating projection from down projection therefore harms this
implementation, plausibly through lost cache locality; this is an
interpretation rather than a hardware-counter observation.

## Decision

Do not optimize mask indexing or split the phases. The measured target is the
arithmetic path, led by the B8 down projection. The next implementation should
retain interleaving and test a focused vectorized/fused B8 arithmetic kernel
against both the existing native block executor and Accelerate dense. Because
only about 10% of blocks are skipped, the kernel needs a large constant-factor
improvement merely to close the existing dense gap.

This is a standalone FP32 diagnostic with one replay bank. It is not an
end-to-end Llama result, hardware-counter profile, quality result or deployable
selector measurement.
