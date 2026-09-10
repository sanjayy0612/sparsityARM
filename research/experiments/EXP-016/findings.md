# EXP-016 TinyLlama quality findings

Run: `m2-tinyllama-screening-20260910-r1`

## Observation

All eight approved conditions completed over the same 16 WikiText-2 excerpts
and 2,032 predicted tokens per condition. Dense TinyLlama perplexity was
18.017000.

| condition | perplexity | relative to dense | passes <=5% gate |
| --- | ---: | ---: | :---: |
| neuron top-k, 30% | 18.0689 | +0.288% | yes |
| output norm B8, 10% | 18.2576 | +1.336% | yes |
| output norm B8, 20% | 18.5857 | +3.157% | yes |
| output norm B8, 30% | 19.2554 | +6.874% | no |
| output norm B32, 10% | 18.6692 | +3.620% | yes |
| output norm B32, 20% | 19.8739 | +10.306% | no |
| output norm B32, 30% | 21.9412 | +21.781% | no |

The maximum recorded resident memory after a condition was 2.034 GiB, below
the approved 5.5 GiB ceiling. All corpus, model, metadata, source, metric, and
acceptance checks passed the independent validator.

## Interpretation

TinyLlama has substantial useful unstructured activation sparsity: its 30%
neuron oracle changed screening perplexity by only 0.288%. Contiguous grouping
introduces a clear quality cost, but TinyLlama is more tolerant of B8 than the
previous Llama 3.2 1B screen. Its measured B8 quality-compatible region reaches
20% but not 30% on this grid. B32 reaches only 10%.

This does not establish a speedup. EXP-015 found the minimal Q8_0 B32 executor
slower at 10% through 70% sparsity, so TinyLlama's B32 quality-compatible point
does not intersect that measured mechanical speed region. The B8/20% result is
the only new candidate worth a matched TinyLlama-geometry executor check.

## Conclusion and limitation

The Llama 3.2 1B quality ceiling is not universal: TinyLlama preserves screening
quality under more B8 sparsity. However, the result does not rescue B32 or prove
the ARM-Sparse hypothesis. The next smallest experiment is a matched 2048 ->
5632 -> 2048 B8 executor measurement at 20% sparsity, with dense and sparse
quantization held constant.

These are post-SwiGLU oracle masks on a small screening corpus. Selector cost is
absent, evaluation time is not inference latency, and no downstream task quality
or end-to-end token-generation claim is supported.
