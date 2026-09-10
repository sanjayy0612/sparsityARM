# EXP-014 llama.cpp Q8_0 dense baseline findings

Accepted run: `m2-q8-dense-20260910-r1.json`

The pinned Llama 3.2 1B snapshot converted successfully to a 1,321,078,720-byte
Q8_0 GGUF. The CPU-only `llama.cpp` build identified Apple M2, Accelerate,
ARM dot-product and i8mm support. Metal was disabled and no GPU layers ran.

| workload | average | standard deviation | repetitions |
|---|---:|---:|---:|
| 128-token prompt processing | 125.113 tokens/s | 5.447 | 5 |
| 32-token generation | 40.416 tokens/s | 1.760 | 5 |

The accepted artifact ran from clean ARM-Sparse commit
`13c08ed39d17fe65884b37b514a5011980ea7fc9` and pinned upstream `llama.cpp`
commit `fa6769818708afd9807b22183ccda112fd563427`. Model, converter, runner and
binary hashes are retained and validated.

An earlier preliminary benchmark lacked ARM-Sparse source provenance and was
excluded; it remains outside the repository and is not evidence.

## Interpretation

This is the strong dense quantized baseline that EXP-013 lacked. It proves the
model and runtime operate within the available M2 environment. It provides no
sparsity speedup or quality comparison. The next implementation must instrument
or modify the pinned GGML Q8_0 FFN path so dense and sparse variants share this
runtime, then replay fixed masks before attempting selector integration.
