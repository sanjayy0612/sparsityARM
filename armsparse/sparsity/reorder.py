"""Function-preserving FFN neuron layouts learned from calibration activations."""
from __future__ import annotations

import numpy as np
import torch


def lsh_permutation(activations: np.ndarray, projection_dim: int = 32,
                    seed: int = 17) -> np.ndarray:
    """Order neurons by random-hyperplane signatures of activation variation.

    Lexicographic signature order is hierarchical: nearby B8 groups remain
    adjacent when interpreted as B16/B32 groups. This is a deterministic
    locality-sensitive baseline, not an optimal balanced clustering algorithm.
    """
    if activations.ndim != 2 or activations.shape[0] < 2 or projection_dim < 1:
        raise ValueError("activations must be [samples, neurons] with at least two samples")
    if not np.isfinite(activations).all():
        raise ValueError("activations must be finite")
    patterns = np.abs(activations.astype(np.float32, copy=False))
    patterns = patterns - patterns.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(patterns, axis=0, keepdims=True)
    patterns = patterns / np.maximum(norms, np.float32(1e-12))
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((patterns.shape[0], projection_dim), dtype=np.float32)
    embedding = patterns.T @ projection
    signatures = embedding >= 0
    # np.lexsort uses the last key as primary, so bit zero is deliberately last.
    keys = tuple(signatures[:, bit] for bit in reversed(range(projection_dim)))
    permutation = np.lexsort(keys).astype(np.int32)
    if not np.array_equal(np.sort(permutation), np.arange(activations.shape[1])):
        raise RuntimeError("learned layout is not a permutation")
    return permutation


def hot_cold_permutation(activations: np.ndarray, active_fraction: float = 0.5) -> np.ndarray:
    """Sort neurons by how often they enter each sample's top activation set."""
    if activations.ndim != 2 or activations.shape[0] < 1 or not 0 < active_fraction < 1:
        raise ValueError("invalid activations or active fraction")
    if not np.isfinite(activations).all():
        raise ValueError("activations must be finite")
    scores = np.abs(activations.astype(np.float32, copy=False))
    keep = round(scores.shape[1] * active_fraction)
    if keep < 1 or keep >= scores.shape[1]:
        raise ValueError("active fraction selects an invalid number of neurons")
    # Stable ordering makes equal-score selection deterministic by neuron index.
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :keep]
    frequency = np.zeros(scores.shape[1], dtype=np.int64)
    np.add.at(frequency, indices.reshape(-1), 1)
    # Stable sorting provides a deterministic original-index tie break.
    return np.argsort(-frequency, kind="stable").astype(np.int32)


def permute_ffn_weights(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                        permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same hidden-neuron permutation to all three SwiGLU projections."""
    width = gate.shape[0]
    if gate.shape != up.shape or down.shape[1] != width or permutation.shape != (width,):
        raise ValueError("incompatible FFN weights or permutation")
    if not torch.equal(torch.sort(permutation).values,
                       torch.arange(width, device=permutation.device)):
        raise ValueError("permutation must contain every neuron exactly once")
    return gate[permutation], up[permutation], down[:, permutation]
