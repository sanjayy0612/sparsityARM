from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Mask:
    values: torch.Tensor
    target_sparsity: float

    @property
    def active_count(self) -> int:
        return int(self.values.sum().item())

    @property
    def realized_sparsity(self) -> float:
        return 1.0 - self.active_count / self.values.numel()


def make_mask(scores: torch.Tensor, sparsity: float) -> Mask:
    """Keep top-k entries over the final dimension, deterministically."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    width = scores.shape[-1]
    keep = round(width * (1.0 - sparsity))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if keep:
        indices = torch.topk(scores, keep, dim=-1, largest=True, sorted=False).indices
        mask.scatter_(-1, indices, True)
    return Mask(mask, sparsity)
