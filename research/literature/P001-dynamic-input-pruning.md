# P001 — Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking

Authors: Marco Federici, Davide Belli, Mart van Baalen, Amir Jalalirad, Andrii Skliar, Bence Major, Markus Nagel, Paul Whatmough. MLSys 2025. Checked: 2026-09-08.

Primary sources: [official proceedings entry](https://proceedings.mlsys.org/paper_files/paper/2025/hash/afd6374c7f2839cba22f537f15f4f760-Abstract-Conference.html), [paper](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Paper-Conference.pdf), [supplement](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Supplemental-Conference.pdf), [official code](https://github.com/Qualcomm-AI-research/dynamic-sparsity). Page pointers below are one-based PDF pages.

## Verified mechanism

- GLU pruning selects per-token top-k absolute post-GLU activations, masking corresponding down-projection columns; up/gate remain dense. DIP additionally selects absolute MLP inputs for up/gate columns, then selects down columns from the resulting approximate GLU activations. These are coordinate/column masks, not contiguous B-neuron block masks. [Paper §3.1–3.2, p.4, Eq.4; §4, p.5, Eq.7–8](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Paper-Conference.pdf#page=4).
- DIP-CA reweights activation scores using previous DRAM residency before top-k. It deliberately permits reranking to favor cached neurons. Consequently, original magnitude top-k membership is not preserved; retaining strongest activations is a tuning motivation, not a superset guarantee. This is software-managed DRAM caching of Flash-resident weights. [Paper §5.2, pp.6–7, Eq.10, Algorithm 1](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Paper-Conference.pdf#page=6).
- Throughput uses simulated memory transfers and omits NPU compute. Models are Phi-3 Mini/Medium, Llama 3 8B, and Mistral 7B; evaluation includes WikiText-2 perplexity and MMLU. [Paper §6.1, pp.7–8](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Paper-Conference.pdf#page=7).

## Setup and evidence limits

The simulator defaults to Apple A18, 60 GB/s DRAM and 1 GB/s Flash. Unpruned layers and KV cache occupy DRAM first; remaining capacity is distributed uniformly among MLP caches. These assumptions do not establish M2 CPU cache-line behavior or actual sparse-kernel speed. [Supplement Appendix A, p.1](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Supplemental-Conference.pdf#page=1).

The official repository README identifies `glu_pruning`, `dip`, and `weighting_current_cache`, and distinguishes model evaluation from optional hardware simulation. Its `precision` setting permits simulated quantized throughput; this is not evidence of measured BF16 CPU throughput. [Code README, Usage](https://github.com/Qualcomm-AI-research/dynamic-sparsity#usage). Implementation-file retrieval failed through the browser; this record verifies the algorithm from the paper and the exposed workflow from README, not executable code internals. No experiments were run for this review.

## Relation to EXP-002 — interpretation, not novelty assessment

Our specified experiment captures CPU M2 BF16 Llama 3.2 1B post-SwiGLU activations. Its magnitude selector therefore overlaps most directly with the paper's GLU-pruning baseline, rather than reproducing full DIP's additional input pruning.

Expanding selected neurons to containing B8/16/32/64 blocks preserves the selected set while increasing active density. Fixed-budget block selection instead changes membership. These answer different questions and require reporting both retained magnitude coverage and actual density. A permutation control probes whether original neuron order contributes spatial locality. DIP-CA instead trades score ranking against temporal DRAM residency. The reviewed sources do not establish the outcome of our block-expansion or permutation tests. This bounded review cannot establish novelty, model-quality preservation, or an M2 inference speedup.
