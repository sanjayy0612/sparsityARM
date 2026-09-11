# P004 — Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

- **Authors:** Zichang Liu et al.
- **Venue/year:** ICML 2023, Proceedings of Machine Learning Research 202
- **Primary source:** https://proceedings.mlr.press/v202/liu23am.html
- **Verification status:** verified from publisher proceedings

## Relevance

DejaVu establishes contextual sparsity: lightweight predictors select
input-dependent subsets of MLP parameters and attention heads. ARM-Sparse does
not claim this idea as novel. Its narrower focus is whether blockifying dynamic
FFN work improves CPU execution enough to compensate for additional compute.
