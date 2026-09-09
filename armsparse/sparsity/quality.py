"""Reference masks for quality experiments, not deployable selectors."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskCondition:
    method: str
    sparsity: float
    block_size: int | None = None
    seed: int = 17

    def __post_init__(self):
        if self.method not in {"dense", "neuron", "block", "block_weight_proxy",
                               "block_output_norm", "random_block"}:
            raise ValueError(f"unknown mask method: {self.method}")
        if not 0 <= self.sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")
        if self.method.startswith("block") or self.method == "random_block":
            if not self.block_size:
                raise ValueError("block methods require a positive block size")

    @property
    def name(self) -> str:
        if self.method == "dense":
            return "dense"
        suffix = f"-B{self.block_size}" if self.block_size else ""
        return f"{self.method}{suffix}-s{round(self.sparsity * 100)}"


def _top_mask(scores: torch.Tensor, keep: int) -> torch.Tensor:
    if scores.ndim < 1 or keep < 1 or keep > scores.shape[-1]:
        raise ValueError("invalid scores or keep count")
    # Stable sorting makes ties reproducible by selecting lower indices first.
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :keep]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(-1, indices, True)


def block_statistics(down_weight: torch.Tensor, block_size: int, method: str) -> torch.Tensor:
    """Precompute weight terms for contribution-aware block scores.

    ``block_weight_proxy`` returns each block's Frobenius norm.
    ``block_output_norm`` returns W_b^T W_b, allowing ||W_b h_b||^2 to be
    evaluated without materializing every full output contribution.
    """
    if down_weight.ndim != 2 or down_weight.shape[1] % block_size:
        raise ValueError("down weight must be [output, intermediate] with divisible blocks")
    weight = down_weight.detach().float().reshape(down_weight.shape[0], -1, block_size)
    if method == "block_weight_proxy":
        return weight.square().sum(dim=(0, 2)).sqrt()
    if method == "block_output_norm":
        return torch.einsum("obk,obl->bkl", weight, weight)
    raise ValueError(f"statistics are not defined for {method}")


def block_scores(activation: torch.Tensor, condition: MaskCondition,
                 statistics: torch.Tensor | None = None) -> torch.Tensor:
    block_size = condition.block_size
    if block_size is None or activation.shape[-1] % block_size:
        raise ValueError("activation width must be divisible by block size")
    values = activation.float().reshape(*activation.shape[:-1], -1, block_size)
    if condition.method == "block":
        return values.abs().sum(dim=-1)
    if statistics is None:
        raise ValueError(f"{condition.method} requires precomputed block statistics")
    if statistics.device != activation.device:
        raise ValueError("block statistics and activation must use the same device")
    if condition.method == "block_weight_proxy":
        return values.square().sum(dim=-1).sqrt() * statistics
    if condition.method == "block_output_norm":
        # Squared norm is sufficient for ranking and avoids a square root.
        return torch.einsum("...bk,bkl,...bl->...b", values, statistics, values).clamp_min_(0)
    raise ValueError(f"scores are not defined for {condition.method}")


def activation_mask(activation: torch.Tensor, condition: MaskCondition, layer: int = 0,
                    statistics: torch.Tensor | None = None) -> torch.Tensor:
    """Return a token-dependent mask with the same shape as ``activation``.

    ``neuron`` and ``block`` are oracle/reference masks because they inspect the
    post-SwiGLU activation. ``random_block`` is a deterministic negative control.
    """
    if condition.method == "dense":
        return torch.ones_like(activation, dtype=torch.bool)
    width = activation.shape[-1]
    if condition.method == "neuron":
        return _top_mask(activation.abs(), round(width * (1 - condition.sparsity)))
    block_size = condition.block_size
    if block_size is None or width % block_size:
        raise ValueError("activation width must be divisible by block size")
    blocks = width // block_size
    keep = round(blocks * (1 - condition.sparsity))
    shape = (*activation.shape[:-1], blocks, block_size)
    if condition.method == "random_block":
        generator = torch.Generator(device=activation.device)
        generator.manual_seed(condition.seed + layer)
        scores = torch.rand((*activation.shape[:-1], blocks), generator=generator,
                            device=activation.device)
        block_mask = _top_mask(scores, keep)
    else:
        block_mask = _top_mask(block_scores(activation, condition, statistics), keep)
    return block_mask.unsqueeze(-1).expand(shape).reshape_as(activation)


def apply_activation_mask(activation: torch.Tensor, condition: MaskCondition, layer: int = 0,
                          statistics: torch.Tensor | None = None) -> torch.Tensor:
    return activation if condition.method == "dense" else activation * activation_mask(
        activation, condition, layer, statistics
    )
