# EXP-010 — Explicit NEON B8 executor

Status: implementation complete; benchmark not started.

## Hypothesis

Explicit ARM NEON vectorization of the measured gate/up and B8 down-projection
arithmetic can reduce interleaved B8 latency relative to the compiler-vectorized
native kernel. The primary success criterion remains beating Accelerate dense,
not merely beating the previous sparse implementation.

## Locked comparison

- Apple M2 CPU, FP32, one thread, 2048 -> 8192 -> 2048 SwiGLU, one token.
- The same 16 real B8/10% replay masks as EXP-005 and EXP-009.
- Accelerate dense versus existing native B8 versus explicit NEON B8.
- Three independent weight/input seeds; 20 warmups and 200 retained timings per
  mask and implementation per run.
- Packing, mask creation, Python dispatch and correctness checks remain outside
  the native timed region.

Every implementation must match the independent FP64 NumPy reference before
timing. Raw measurements, medians, p95 values, sources, compiler command and
binary hash must be retained. This is standalone executor evidence, not
end-to-end Llama inference.
