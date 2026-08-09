import torch
from .dense import dense_ffn


def sparse_ffn(hidden: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor,
               down_weight: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Correctness reference. A supplied mask selects intermediate entries."""
    gate = torch.nn.functional.linear(hidden, gate_weight)
    up = torch.nn.functional.linear(hidden, up_weight)
    activation = torch.nn.functional.silu(gate) * up
    if mask is None:
        return torch.nn.functional.linear(activation, down_weight)
    if mask.shape != activation.shape:
        raise ValueError(f"mask shape {mask.shape} must equal activation shape {activation.shape}")
    return torch.nn.functional.linear(activation * mask.to(activation.dtype), down_weight)
