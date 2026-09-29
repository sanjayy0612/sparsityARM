# ARM-Sparse package claim audit

| Package claim | Registry claim | Direct evidence | Status |
|---|---|---|---|
| Llama B8/10% passed the quality gate | C008 | EXP-004 manifest | Verified |
| Llama B8/20% and 30% failed the quality gate | C009 | EXP-004 manifest | Verified |
| Quality-compatible Llama B8/10% replay: packed B8 37–39% slower | C010 | EXP-005 manifest and three cases | Verified |
| Same real B8/10% masks: per-neuron executor 0.9–3.5% faster than dense in every run | — (paper) | EXP-005 findings and manifest | Verified |
| Llama B8 first beat dense at the 40% grid point | C018 | EXP-012 manifest and cases | Verified |
| TinyLlama B8 passed through 20% and failed at 30% | C022 | EXP-016 manifest | Verified |
| TinyLlama B8 first beat dense at the 50% grid point | C024 | EXP-018 manifest and cases | Verified |
| Per-neuron executor faster than dense at every measured block-aligned sparsity (EXP-012, EXP-018) | — (paper) | EXP-012/018 case summaries (`irregular_native`), `summary.json` | Verified |
| Packed B8 has no verified quality-speed intersection for either model | Derived from C008–C010, C018, C022, C024 | `summary.json` | Verified bounded conclusion |
| Per-neuron executor on B8 masks has a small verified intersection (Llama B8/10%, TinyLlama B8/20%) | Derived from C008, C022 and the rows above | `summary.json` | Verified bounded conclusion |

## Citation audit

| Literature statement | Record | Primary source verified |
|---|---|:---:|
| Contextual sparsity and lightweight prediction predate ARM-Sparse | P004 | Yes |
| Flash-oriented row/column bundling motivates contiguous access | P002 | Yes |
| Hot/cold activation locality and heterogeneous sparse execution predate ARM-Sparse | P005 | Yes |
| Neuron clusters as mobile scheduling units predate ARM-Sparse | P006 | Yes |
| Cache-aware dynamic input pruning targets modern mobile inference | P001 | Yes |

## Unsupported claims deliberately excluded

- No end-to-end LLM generation speedup is claimed.
- No deployable selector performance is claimed.
- No general result beyond Apple M2 is claimed.
- No impossibility result for all block-sparse methods is claimed.
- No novelty claim is made for contextual sparsity, neuron grouping, or packed
  contiguous weight access by themselves.
