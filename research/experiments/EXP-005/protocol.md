# EXP-005 — B8/10% replay executor gate

Status: implementation and 3-run protocol approved 2026-09-09; execution pending.

## Question

Does the only EXP-004 quality-compatible configuration—B8 at approximately
10% sparsity—reduce standalone FFN latency on the Apple M2?

## Scope

- Synthetic FP32 Llama 3.2 1B geometry: 2048 → 8192 → 2048, batch/token=1.
- Replay 16 real output-norm masks reconstructed from one retained EXP-002 token
  across all 16 model layers. Each mask keeps 922 of 1024 B8 blocks (9.961%
  realized block sparsity).
- Three independent weight/input seeds, 20 warmups and 200 retained timings per
  executor per run, one CPU thread.
- Compare Accelerate dense, native irregular, native packed-block, and
  Accelerate-per-block implementations. Native irregular and block executors
  compute exactly the same neurons.

Mask selection, weight packing, Python dispatch, and correctness validation are
outside the timed region. The mask scan/branches and output initialization are
inside. Every variant must first match an independent FP64 reference.

## Decision rule

The configuration advances only if native packed-block latency beats optimized
dense latency reproducibly across all three runs. This is a standalone executor
gate, not end-to-end Llama speed or a deployable-selector result.
