# EXP-009 — B8 executor phase profiling

Status: instrumentation implemented; measurement not started.

## Question

Which measured component prevents the quality-relevant B8 executor from
beating optimized dense execution on Apple M2?

## Scope

- Synthetic Llama 3.2 1B FFN geometry, FP32, batch/token one, one CPU thread.
- Replay the retained real B8 masks used by EXP-005.
- Profile output initialization, mask compaction, gate/up plus SwiGLU, and down
  projection separately.
- Record active-block count and contiguous active-run count.

The diagnostic executor stages projection and down-projection phases so only
five clock reads occur per invocation. Its total latency must be compared with
the original interleaved B8 executor; its component timings describe this
staged diagnostic, not the original kernel exactly.

## Boundary

EXP-009 identifies an optimization target. It does not demonstrate an
optimization, end-to-end inference speedup, deployable selector, or model
quality. No NEON or other low-level kernel should be introduced until this
measurement identifies the dominant cost.
