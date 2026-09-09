import numpy as np
import pytest
import torch

from armsparse.sparsity.reorder import lsh_permutation, permute_ffn_weights


def test_lsh_layout_is_deterministic_permutation():
    activations = np.random.default_rng(17).standard_normal((40, 64), dtype=np.float32)
    first = lsh_permutation(activations, projection_dim=8, seed=9)
    second = lsh_permutation(activations, projection_dim=8, seed=9)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(64))


def test_consistent_weight_permutation_preserves_swiglu_output():
    torch.manual_seed(17)
    hidden, intermediate = 5, 16
    x = torch.randn(hidden)
    gate, up = torch.randn(intermediate, hidden), torch.randn(intermediate, hidden)
    down = torch.randn(hidden, intermediate)
    permutation = torch.randperm(intermediate)
    expected = down @ (torch.nn.functional.silu(gate @ x) * (up @ x))
    pg, pu, pd = permute_ffn_weights(gate, up, down, permutation)
    actual = pd @ (torch.nn.functional.silu(pg @ x) * (pu @ x))
    torch.testing.assert_close(actual, expected)


def test_invalid_permutation_is_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        permute_ffn_weights(torch.ones(4, 2), torch.ones(4, 2), torch.ones(2, 4),
                            torch.tensor([0, 1, 1, 3]))
