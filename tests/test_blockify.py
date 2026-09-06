import torch
from armsparse.sparsity import blockify

def test_blocks_expand_at_boundaries():
    block=blockify(torch.tensor([1.,2.,9.,8.,0.]),2,.5)
    assert block.values.shape == (3,)
    assert block.expand().shape == (5,)
    assert block.active_blocks == 2

def test_zero_sparsity_keeps_all_blocks():
    assert blockify(torch.ones(10),8,0).values.all()

def test_expansion_preserves_selected_neurons_even_when_all_blocks_needed():
    scores = torch.tensor([9., 0., 8., 0., 7., 0., 6., 0.])
    block = blockify(scores, 2, .5, strategy="expand_neurons")
    assert block.values.all()
    assert block.realized_sparsity == 0

def test_realized_sparsity_counts_unpadded_neurons():
    block = blockify(torch.tensor([0., 0., 0., 0., 9.]), 2, 2/3)
    assert block.expand().tolist() == [False, False, False, False, True]
    assert block.realized_sparsity == .8
