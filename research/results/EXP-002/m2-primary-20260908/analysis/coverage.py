"""Offline magnitude-mask coverage; these scores are not a quality oracle."""
import numpy as np


def top_mask(scores, keep, tie_order):
    """Largest scores with a fixed randomized tie order, independent of index."""
    if scores.ndim != 2 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scores must be finite nonnegative [tokens, neurons]")
    width = scores.shape[1]
    if not 0 <= keep <= width or sorted(tie_order.tolist()) != list(range(width)):
        raise ValueError("invalid keep count or tie permutation")
    ranked = np.argsort(-scores[:, tie_order], axis=1, kind="stable")[:, :keep]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, tie_order[ranked], True, axis=1)
    return mask


def coverage(activations, block_size, sparsity, seed=17):
    scores = np.abs(activations.astype(np.float64))
    n, width = scores.shape
    if not n or width % block_size or not 0 <= sparsity <= 1:
        raise ValueError("invalid shape or sparsity")
    rng = np.random.default_rng(seed)
    base = top_mask(scores, round(width*(1-sparsity)), rng.permutation(width))
    blocks = base.reshape(n, -1, block_size).any(axis=2)
    expanded = np.repeat(blocks, block_size, axis=1)
    grouped_scores = scores.reshape(n, -1, block_size).sum(axis=2)
    selected_blocks = top_mask(grouped_scores, round(grouped_scores.shape[1]*(1-sparsity)),
                               rng.permutation(grouped_scores.shape[1]))
    budget_mask = np.repeat(selected_blocks, block_size, axis=1)
    mass = scores.sum(axis=1)
    # Zero-mass rows retain all of their zero signal; flag separately for analysis.
    denom = np.where(mass > 0, mass, 1)
    retained = lambda mask: np.where(mass > 0, (scores*mask).sum(axis=1)/denom, 1.)
    permuted = []
    # Identical base mask, permuted neuron order: locality control, not another model.
    for _ in range(3):
        shuffled = base[:, rng.permutation(width)]
        permuted.append(1-shuffled.reshape(n, -1, block_size).any(axis=2).mean(axis=1))
    return {
        "base_mask": base, "expanded_block_mask": blocks, "budget_block_mask": selected_blocks,
        "base_sparsity": 1-base.mean(axis=1),
        "expanded_sparsity": 1-expanded.mean(axis=1),
        "budget_sparsity": 1-budget_mask.mean(axis=1),
        "permuted_expanded_sparsity": np.array(permuted),
        "neuron_retained_l1": retained(base), "budget_retained_l1": retained(budget_mask),
        "zero_mass": mass == 0,
    }
