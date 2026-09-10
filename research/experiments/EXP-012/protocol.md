# EXP-012 — B8 performance break-even frontier

Status: implementation complete; sweep not started.

## Question

At what equal-work B8 sparsity does the existing native block executor become
faster than Accelerate dense on Apple M2, and does that point overlap the
quality-compatible sparsity region observed in real-model experiments?

## Locked sweep

- Synthetic 2048 -> 8192 -> 2048 SwiGLU, FP32, one token and one CPU thread.
- B8 clustered masks at 10%, 20%, 30%, 40%, 50%, 60%, 70% and 80% sparsity.
- Accelerate dense and native B8 are the primary comparison; other EXP-001
  reference modes remain recorded but are secondary.
- Three weight/input seeds, eight masks per case, 20 warmups and 200 retained
  timings per executor and case.

Clustered masks guarantee identical neuron work between native irregular and
block execution. They are synthetic controls, not realistic selector outputs.
The break-even point is descriptive for this FP32 kernel and hardware only.
