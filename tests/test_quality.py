import pytest
import torch

from armsparse.sparsity.quality import MaskCondition, activation_mask, apply_activation_mask


def test_neuron_mask_keeps_largest_magnitudes_exactly():
    values = torch.tensor([[1.0, -8.0, 3.0, -2.0]])
    condition = MaskCondition("neuron", 0.5)
    assert activation_mask(values, condition).tolist() == [[False, True, True, False]]
    assert apply_activation_mask(values, condition).tolist() == [[0.0, -8.0, 3.0, 0.0]]


def test_block_mask_uses_total_activation_mass_not_largest_single_neuron():
    values = torch.tensor([[5.0, 0.0, 3.0, 3.0]])
    mask = activation_mask(values, MaskCondition("block", 0.5, 2))
    assert mask.tolist() == [[False, False, True, True]]


def test_block_mask_is_token_dependent_and_exact_budget():
    values = torch.tensor([[9.0, 8.0, 1.0, 1.0], [1.0, 1.0, 9.0, 8.0]])
    mask = activation_mask(values, MaskCondition("block", 0.5, 2))
    assert mask.tolist() == [[True, True, False, False], [False, False, True, True]]
    assert torch.all(mask.sum(dim=-1) == 2)


def test_random_control_is_reproducible_and_layer_specific():
    values = torch.ones((3, 16))
    condition = MaskCondition("random_block", 0.5, 2, seed=17)
    first = activation_mask(values, condition, layer=0)
    assert torch.equal(first, activation_mask(values, condition, layer=0))
    assert not torch.equal(first, activation_mask(values, condition, layer=1))
    assert torch.all(first.sum(dim=-1) == 8)


def test_invalid_block_geometry_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        activation_mask(torch.ones(1, 10), MaskCondition("block", 0.2, 4))
