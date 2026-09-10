# EXP-011 — Contiguous active-run coalescing

Status: implementation complete; benchmark not started.

## Hypothesis

Combining adjacent active B8 blocks into larger Accelerate matrix-vector calls
can amortize per-block dispatch and improve arithmetic efficiency relative to
the native B8 loop while skipping exactly the same neurons.

## Locked comparison

- Apple M2, FP32, one thread, 2048 -> 8192 -> 2048 SwiGLU, one token.
- Same 16 B8/10% real replay masks as EXP-005, EXP-009 and EXP-010.
- Accelerate dense, existing native packed B8 and coalesced-run B8.
- Three weight/input seeds, 20 warmups, 200 retained timings per mask and mode.
- Exact mask equality and independent FP64 numerical validation before timing.

The coalesced kernel uses original dense down-projection storage because a
contiguous neuron run is a strided submatrix of that layout. Its gate/up and
down calls are interleaved per run. With a median 93 runs per mask, dispatch
overhead remains a known risk. This is standalone executor evidence only.
