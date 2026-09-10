# P003 — VLM in a Flash / Neuron Chunking

- Title: *VLM in a flash: I/O-Efficient Sparsification of Vision-Language Model via Neuron Chunking*
- Authors: Kichang Yang et al.
- Status: 2025 preprint / OpenReview record
- Stable source: https://arxiv.org/abs/2511.18692
- Relevance: high mechanism overlap; contiguous neuron chunks and offline ordering.

Verified points used by ARM-Sparse:

- The method chooses contiguous chunks using importance divided by profiled I/O
  latency and greedily excludes overlaps.
- Its offline hot/cold reordering marks each sample's top 50% importance neurons
  active, counts per-neuron activation frequency, and sorts by that frequency.
- The authors report this simple frequency order as comparable to their tested
  coactivation-based alternatives for I/O efficiency.

Limitations: VLMs, Jetson devices and flash-I/O latency differ from Llama 3.2 1B
SwiGLU arithmetic on Apple M2. The source is recent and not treated as replicated
evidence. ARM-Sparse uses hot/cold sorting only as a baseline to test locally.
