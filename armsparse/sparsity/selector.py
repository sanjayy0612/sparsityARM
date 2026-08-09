import torch


def activation_scores(gate_values: torch.Tensor) -> torch.Tensor:
    """Oracle/reference importance score: absolute gated-MLP preactivation."""
    if gate_values.ndim < 1:
        raise ValueError("gate_values must have an intermediate dimension")
    return gate_values.detach().abs()
