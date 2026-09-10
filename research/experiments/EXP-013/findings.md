# EXP-013 matched INT8 feasibility findings

Run: `m2-int8-break-even-20260910`

The sweep completed from clean commit
`0142b2e4c1335452fdad2430fcf5945c3e4ea532`. The validator checked committed
sources, numerical errors and all medians and p95 values from retained timings.
Relative error against the corresponding dequantized-weight reference was
below `7.3e-7` in every case.

| sparsity | INT8 B8 speedup over matched INT8 dense |
|---:|---:|
| 10% | 5.9–6.3% |
| 20% | 18.3–19.2% |
| 30% | 34.7–35.8% |
| 40% | 55.9–58.2% |
| 50% | 86.9–89.2% |

The matched custom INT8 crossover is already below 10% sparsity. This shows
that reducing weight traffic can make skipped B8 work visible within the same
simple quantized implementation.

## Critical limitation

The custom INT8 dense kernel took roughly 6.8–7.3 ms and B8/10% took roughly
6.4 ms. The established Accelerate FP32 dense baseline is roughly 4.0 ms.
Therefore EXP-013 does **not** demonstrate a practical speedup over the best
current dense baseline. The apparent low break-even depends on a weak custom
INT8 dense comparator that performs INT8-to-FP32 conversion during dot products
and is not representative of llama.cpp-class quantized kernels.

The next valid step is integration or comparison with a strong existing
quantized ARM runtime. It would be scientifically invalid to claim that
quantization solved ARM-Sparse from this matched microbenchmark alone.
