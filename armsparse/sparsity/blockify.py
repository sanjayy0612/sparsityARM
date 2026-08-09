from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class BlockMask:
    values: torch.Tensor  # [..., num_blocks]
    block_size: int
    original_width: int
    target_sparsity: float

    @property
    def active_blocks(self) -> int:
        return int(self.values.sum().item())

    @property
    def realized_sparsity(self) -> float:
        return 1.0 - self.active_blocks / self.values.numel()

    def expand(self) -> torch.Tensor:
        return self.values.repeat_interleave(self.block_size, dim=-1)[..., : self.original_width]


def blockify(neuron_scores: torch.Tensor, block_size: int, target_sparsity: float,
             strategy: str = "top_blocks") -> BlockMask:
    if block_size <= 0 or not 0 <= target_sparsity <= 1:
        raise ValueError("invalid block size or sparsity")
    width = neuron_scores.shape[-1]
    blocks = math.ceil(width / block_size)
    padded = torch.nn.functional.pad(neuron_scores.abs(), (0, blocks * block_size - width))
    grouped = padded.reshape(*padded.shape[:-1], blocks, block_size)
    if strategy == "top_blocks":
        scores = grouped.sum(dim=-1)
    elif strategy == "expand_neurons":
        keep_neurons = round(width * (1 - target_sparsity))
        selected = torch.zeros_like(neuron_scores, dtype=torch.bool)
        if keep_neurons:
            selected.scatter_(-1, torch.topk(neuron_scores.abs(), keep_neurons, dim=-1).indices, True)
        scores = torch.nn.functional.pad(selected, (0, blocks * block_size - width)).reshape(*selected.shape[:-1], blocks, block_size).any(-1).float()
    else:
        raise ValueError("strategy must be 'top_blocks' or 'expand_neurons'")
    keep = round(blocks * (1 - target_sparsity))
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    if keep:
        chosen.scatter_(-1, torch.topk(scores, keep, dim=-1).indices, True)
    return BlockMask(chosen, block_size, width, target_sparsity)
