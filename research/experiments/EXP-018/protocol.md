# EXP-018 - TinyLlama-geometry B8 break-even frontier

Status: implementation complete; sweep not started.

## Question

At what B8 sparsity does the native FP32 block executor become faster than
Accelerate dense for TinyLlama's 2048 -> 5632 -> 2048 FFN geometry?

## Locked sweep

- Synthetic one-token 2048 -> 5632 -> 2048 SwiGLU, FP32, one CPU thread.
- B8 clustered equal-work masks at 20%, 30%, 40%, 50%, and 60% sparsity.
- Accelerate dense versus native packed-B8 is the primary comparison.
- Three independent weight/input seeds and eight masks per case.
- Twenty warmups and 200 retained timings per executor and case.
- Randomized executor order within every retained iteration.
- Independent FP64 NumPy masked-dense correctness validation before timing.

## Decision rule

The first grid point where native B8 beats Accelerate dense in all three runs
is the measured crossover bound. No interpolation between grid points will be
reported as an observation.

Compare that bound with EXP-016's TinyLlama quality screen: B8 passed the 5%
perplexity gate at 20% sparsity and failed at 30%. If runtime crosses only above
30%, the current TinyLlama configuration has no verified quality-speed overlap.

## Boundaries

Clustered masks are favorable synthetic equal-work controls, not captured model
masks. Selector cost is absent. This is a one-token FFN microbenchmark and does
not establish end-to-end token-generation speed.
