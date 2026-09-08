import numpy as np
import pytest
from armsparse.sparsity.coverage import coverage, top_mask


def test_known_clustered_and_scattered_coverage():
    clustered = np.array([[10, 9, 0, 0, 8, 7, 0, 0]], dtype=np.float32)
    scattered = np.array([[10, 0, 9, 0, 8, 0, 7, 0]], dtype=np.float32)
    c = coverage(clustered, 2, .5)
    s = coverage(scattered, 2, .5)
    assert c["expanded_sparsity"][0] == .5
    assert s["expanded_sparsity"][0] == 0
    assert s["budget_sparsity"][0] == .5
    assert s["budget_retained_l1"][0] == 19/34
    assert c["neuron_retained_l1"][0] == 1


@pytest.mark.parametrize("sparsity", [0, .2, .3, .4, 1])
@pytest.mark.parametrize("b", [8,16,32,64])
def test_expansion_preserves_base_and_budget(sparsity, b):
    a = np.random.default_rng(2).normal(size=(3, 128)).astype(np.float32)
    result = coverage(a,b,sparsity)
    assert np.all(np.repeat(result["expanded_block_mask"],b,axis=1) >= result["base_mask"])
    assert np.all(result["base_mask"].sum(axis=1) == round(128*(1-sparsity)))
    assert np.all(result["budget_block_mask"].sum(axis=1) == round((128//b)*(1-sparsity)))


def test_ties_use_explicit_permutation_and_zero_mass_is_finite():
    tie = np.array([2,0,3,1])
    assert top_mask(np.ones((1,4)),2,tie).tolist() == [[True,False,True,False]]
    result = coverage(np.zeros((1,8)),2,.5)
    assert result["zero_mass"][0]
    assert result["budget_retained_l1"][0] == 1


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        coverage(np.full((1,8), np.nan),2,.5)
