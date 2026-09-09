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
        if self.method not in {"dense", "neuron", "block", "random_block"}:
            raise ValueError(f"unknown mask method: {self.method}")
        if not 0 <= self.sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")
        if self.method in {"block", "random_block"} and not self.block_size:
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


def activation_mask(activation: torch.Tensor, condition: MaskCondition, layer: int = 0) -> torch.Tensor:
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
    if condition.method == "block":
        block_mask = _top_mask(activation.abs().reshape(shape).sum(dim=-1), keep)
    else:
        generator = torch.Generator(device=activation.device)
        generator.manual_seed(condition.seed + layer)
        scores = torch.rand((*activation.shape[:-1], blocks), generator=generator,
                            device=activation.device)
        block_mask = _top_mask(scores, keep)
    return block_mask.unsqueeze(-1).expand(shape).reshape_as(activation)


def apply_activation_mask(activation: torch.Tensor, condition: MaskCondition, layer: int = 0) -> torch.Tensor:
    return activation if condition.method == "dense" else activation * activation_mask(
        activation, condition, layer
    )
