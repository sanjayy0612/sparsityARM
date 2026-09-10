# EXP-007 hot/cold layout findings

Run: `m2-layout-diagnostic-20260910-r1`

The first attempt stopped before producing an analysis because CPU-name
collection through `sysctl` was unavailable in the execution sandbox. Its
orphaned local layout file is not evidence. The corrected run completed from
clean commit `bd05bb7f2e90129057992d0f203a19a35fe6c8a7` and its 576 rows and three
16-by-8192 permutations passed independent structural and hash checks.

## Held-out observations

Hot/cold ordering learned from P01–P04 improved direct block-selection L1 mass
on P05–P08 at every tested configuration relative to original model order.

| block | sparsity | original retained L1 | hot/cold retained L1 | change |
|---:|---:|---:|---:|---:|
| 8 | 20% | 90.061% | 91.485% | +1.424 pp |
| 8 | 30% | 83.274% | 85.448% | +2.174 pp |
| 16 | 20% | 87.732% | 89.904% | +2.172 pp |
| 16 | 30% | 79.935% | 83.185% | +3.250 pp |
| 32 | 20% | 85.820% | 88.739% | +2.918 pp |
| 32 | 30% | 77.378% | 81.673% | +4.294 pp |

The fraction of mixed selected/unselected boundary blocks also fell relative
to original order by 0.686–3.890 percentage points, depending on configuration.
However, boundary fractions remained very high: 79.343–99.313%.

Exact expansion remained unusable. Mean surviving held-out sparsity under the
hot/cold layout was at most 0.02921% (B8 at 30% target) and was exactly zero in
four of six aggregate configurations.

## Interpretation

Activation frequency generalizes enough across this small prompt split to make
directly selected blocks contain more activation magnitude. It does **not**
make per-token neuron masks naturally block sparse. Therefore hot/cold ordering
is a candidate for a direct block-aware quality screen, not a solution for
losslessly expanding irregular masks.

These metrics are activation proxies. They are not perplexity, task quality,
runtime, selector accuracy or deployable inference evidence. Advancing the
hot/cold layout to a full model quality screen requires the planned human
decision.
