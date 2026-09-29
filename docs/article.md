# I Tried to Accelerate Sparse LLM FFNs on an Apple M2. Here Is What Actually Worked, and Why It Was Small.

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

For the packed block kernel, ARM-Sparse found a speed region and a quality
region, but no verified overlap.

## The result hiding in the data

The benchmarks also timed a simple per-neuron executor on the *same* masks.
It stores down-projection weights neuron-major, so each active neuron is one
long contiguous row, while the packed kernel turns every block into 2,048
tiny length-8 dot products. Doing identical arithmetic, the per-neuron
executor was faster than dense at every measured sparsity, including the
quality ceilings:

| Model | Quality ceiling | Per-neuron speedup at ceiling |
|---|---:|---:|
| Llama 3.2 1B | 10% | +0.9% to +3.5% (real masks), +2.6% to +7.3% (synthetic) |
| TinyLlama 1.1B | 20% | +5.1% to +7.3% (synthetic) |

So a quality-compatible speedup exists, but it is small. On scattered masks
that were not block-aligned, the same per-neuron kernel was usually slower
than dense. Block alignment helps; the packed tile layout was the wrong way to
exploit it.

## Why the gain is small

At 10–20% sparsity, even a perfect kernel can skip only 10–20% of the weight
bytes, which is at most an 11–25% FFN speedup. The per-neuron kernel already
captures a meaningful share of that. The next bottleneck is structural:
useful FFN contributions are not sufficiently aligned with fixed contiguous B8
groups in these unmodified models, so quality fails long before larger
savings become possible.

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
question: for the tested untrained models, packed block kernels did not deliver
quality-preserving acceleration, while block-aligned masks with neuron-major
weights delivered a small FFN-only gain that the quality ceiling caps.
