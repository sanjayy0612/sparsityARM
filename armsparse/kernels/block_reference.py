from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PackedFFNWeights:
    """Block-major view of Llama-style FFN weights, packed once per layer."""
    gate_blocks: torch.Tensor
    up_blocks: torch.Tensor
    down_blocks: torch.Tensor
    block_size: int
    intermediate_size: int

    @classmethod
    def from_weights(cls, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                     block_size: int) -> "PackedFFNWeights":
        if gate.shape != up.shape or down.shape[1] != gate.shape[0]:
            raise ValueError("incompatible FFN dimensions")
        intermediate = gate.shape[0]
        pad = (-intermediate) % block_size
        if pad:
            gate, up = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (gate, up))
            down = torch.nn.functional.pad(down, (0, pad))
        blocks = gate.shape[0] // block_size
        return cls(gate.reshape(blocks, block_size, -1).contiguous(),
                   up.reshape(blocks, block_size, -1).contiguous(),
                   down.reshape(down.shape[0], blocks, block_size).permute(1, 0, 2).contiguous(),
                   block_size, intermediate)


def block_sparse_ffn(hidden: torch.Tensor, packed: PackedFFNWeights,
                     block_mask: torch.Tensor) -> torch.Tensor:
    """Vectorized reference block executor; selected blocks use contiguous packed rows."""
    if hidden.ndim != 2 or block_mask.ndim != 1:
        raise ValueError("reference executor expects [tokens, hidden] and [blocks]")
    if block_mask.numel() != packed.gate_blocks.shape[0]:
        raise ValueError("block mask has wrong number of blocks")
    output = hidden.new_zeros((hidden.shape[0], packed.down_blocks.shape[1]))
    for block_id in torch.nonzero(block_mask, as_tuple=False).flatten().tolist():
        gate = torch.nn.functional.linear(hidden, packed.gate_blocks[block_id])
        up = torch.nn.functional.linear(hidden, packed.up_blocks[block_id])
        activation = torch.nn.functional.silu(gate) * up
        output += torch.nn.functional.linear(activation, packed.down_blocks[block_id])
    return output
