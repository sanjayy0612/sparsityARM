# EXP-007 — Held-out layout concentration diagnostic

Status: implementation complete; diagnostic not yet run.

## Question

Before another expensive perplexity sweep, do fixed layouts learned on P01–P04
actually make important neurons more contiguous on held-out P05–P08 activations?

## Layouts

- Original model order.
- EXP-006 random-hyperplane LSH, refit only on P01–P04.
- Hot/cold ordering: mark each calibration sample's top 50% magnitude neurons
  active, count activation frequency per neuron, then sort by decreasing count.

The hot/cold baseline follows the method described by *VLM in a Flash*.
Apple's *LLM in a Flash* reported that naïve closest-friend coactivation
bundling caused highly active neurons to be loaded repeatedly; therefore that
method is intentionally excluded.

## Metrics

For B8/B16/B32 and 20/30% target sparsity on both calibration and held-out
activations:

- sparsity surviving exact expansion of neuron top-k masks;
- fraction of boundary blocks containing mixed selected/unselected neurons;
- activation L1 mass retained by direct block selection.

These are cheap layout diagnostics, not language quality or runtime results.
Only a layout that clearly improves held-out concentration should advance to a
full WikiText perplexity run, and that advancement remains a human-approved
research decision rather than an automatic pass/fail result.

## Reproducibility

The runner verifies every EXP-002 activation artifact against its recorded
SHA-256, refuses to overwrite outputs, uses stable index-based tie breaking,
and records source hashes, Git state, hardware, software and layout hashes.
