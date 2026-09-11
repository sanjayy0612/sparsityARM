# I Tried to Accelerate Sparse LLM FFNs on an Apple M2. Here Is Where It Failed.

The tempting argument for LLM sparsity is simple: if 30% of feed-forward
neurons are unnecessary for a token, skip them and expect roughly 30% less
work. CPUs make that arithmetic misleading. Sparse indices, irregular memory
accesses, weak vector utilization, and dispatch overhead can consume the saved
time.

ARM-Sparse tested a hardware-aware compromise: execute contiguous blocks of
neurons. Blocks deliberately compute some extra neurons, but permit packed
weights, sequential reads, simpler indexing, and SIMD-friendly inner loops.
The central question was not whether blocks reduce FLOPs. It was whether one
block configuration could both retain model quality and beat optimized dense
execution on the only guaranteed machine: an Apple M2 MacBook.

## The experiment that mattered

The final comparison used B8 blocks on two small Llama-family models. Quality
and runtime were deliberately separated:

- Oracle output-contribution masks estimated the attainable quality frontier.
- Replay masks isolated executor cost from selector cost.
- A packed native block kernel was compared with Apple's optimized Accelerate
  dense path.
- A configuration passed quality only if relative perplexity increased by no
  more than 5% on the fixed evaluation corpus.

This separation prevents an oracle mask from being mislabeled as deployable
end-to-end inference.

## What worked mechanically

Block sparsity eventually accelerated the FFN kernel. On Llama 3.2 1B geometry,
the first positive measured grid point was 40% sparsity, with a +1.6% to +3.1%
speedup range across three runs. On TinyLlama 1.1B geometry, the first positive
point was 50%, with +10.7% to +17.1%.

So the executor hypothesis was not nonsense. Remove enough blocks and packed
contiguous execution can overcome its overhead.

## Where it failed

Quality failed earlier. Llama passed the 5% gate at 10% B8 sparsity but failed
at 20%. TinyLlama passed at 20% but failed at 30%. The quality-compatible Llama
B8/10% masks were also 37–39% slower than optimized dense execution when
replayed through the native kernel.

The two frontiers never met:

| Model | Quality ceiling | First speedup point |
|---|---:|---:|
| Llama 3.2 1B | 10% | 40% |
| TinyLlama 1.1B | 20% | 50% |

This is the main result. ARM-Sparse found a speed region and a quality region,
but no verified overlap.

## Why the negative result matters

Optimizing the current kernel further is unlikely to close a 30-percentage-
point sparsity gap by itself. The next bottleneck is structural: useful FFN
contributions are not sufficiently aligned with fixed contiguous B8 groups in
these unmodified models.

A stronger continuation would therefore change the model/selection interface:
train for block-aligned activation, co-design neuron layout and selection, or
learn a cheap selector that explicitly values hardware-efficient groups. It
should not simply add another hand-tuned loop and assume quality will follow.

## Scope and honesty

This investigation used one Apple M2, one-thread single-token FFN benchmarks,
two approximately 1B-parameter models, a small fixed quality corpus, and oracle
masks whose selection cost was excluded. It is not an impossibility proof and
not an end-to-end inference claim.

What it does provide is a reproducible answer to a bounded engineering
question: for the tested untrained block layouts, dynamic FFN block sparsity did
not deliver quality-preserving wall-clock acceleration.
