import torch
from armsparse.sparsity import make_mask

def test_topk_count_and_zero_sparsity():
    mask=make_mask(torch.tensor([1.,2.,3.,4.]),0.0)
    assert mask.active_count == 4 and mask.realized_sparsity == 0

def test_half_sparsity_is_deterministic():
    scores=torch.tensor([1.,4.,3.,2.])
    assert torch.equal(make_mask(scores,.5).values,torch.tensor([False,True,True,False]))
