# P005 — PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU

- **Authors:** Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen
- **Venue/year:** SOSP 2024
- **Primary source:** https://doi.org/10.1145/3694715.3695964
- **Verification status:** verified from ACM DOI record

## Relevance

PowerInfer exploits hot/cold neuron activation locality and heterogeneous
CPU/GPU execution. Its hardware, scheduling problem, and end-to-end design
differ from ARM-Sparse's CPU-only block-granularity experiment, but it is a
necessary baseline for claims about activation frequency and neuron locality.
