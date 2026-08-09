import torch
from armsparse.sparsity import blockify

def test_blocks_expand_at_boundaries():
    block=blockify(torch.tensor([1.,2.,9.,8.,0.]),2,.5)
    assert block.values.shape == (3,)
    assert block.expand().shape == (5,)
    assert block.active_blocks == 2

def test_zero_sparsity_keeps_all_blocks():
    assert blockify(torch.ones(10),8,0).values.all()
