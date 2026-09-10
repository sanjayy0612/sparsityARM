# P002 — LLM in a Flash

- Title: *LLM in a flash: Efficient Large Language Model Inference with Limited Memory*
- Authors: Keivan Alizadeh et al.
- Venue: ACL 2024
- Stable source: https://aclanthology.org/2024.acl-long.678/
- Relevance: high; Apple hardware and contiguous sparse weight access.

Verified points used by ARM-Sparse:

- Row-column bundling stores the weight pieces needed by one neuron together to
  increase sequential read size; it is primarily a flash-I/O technique.
- Appendix coactivation bundling paired neurons with their closest coactive
  neighbor. The paper reports a negative result: highly active neurons became
  the closest partner of many neurons, causing repeated loading.
- This is direct counterevidence against treating nearest-neighbor coactivation
  pairing as an obviously effective contiguous layout.

Limitation for ARM-Sparse: the paper studies flash-to-DRAM movement, mostly
ReLU-family activation sparsity, rather than SwiGLU CPU arithmetic on M2.
