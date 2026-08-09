import torch


def dense_ffn(hidden: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor,
              down_weight: torch.Tensor) -> torch.Tensor:
    gate = torch.nn.functional.linear(hidden, gate_weight)
    up = torch.nn.functional.linear(hidden, up_weight)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_weight)
