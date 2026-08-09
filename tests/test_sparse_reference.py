import torch
from armsparse.kernels import dense_ffn, sparse_ffn, PackedFFNWeights, block_sparse_ffn

def weights():
    torch.manual_seed(2); return torch.randn(3,4),torch.randn(8,4),torch.randn(8,4),torch.randn(4,8)

def test_zero_sparse_matches_dense():
    x,g,u,d=weights(); assert torch.allclose(dense_ffn(x,g,u,d),sparse_ffn(x,g,u,d,torch.ones(3,8,dtype=torch.bool)))

def test_packed_all_blocks_matches_dense():
    x,g,u,d=weights(); packed=PackedFFNWeights.from_weights(g,u,d,4)
    assert torch.allclose(dense_ffn(x,g,u,d),block_sparse_ffn(x,packed,torch.ones(2,dtype=torch.bool)),atol=1e-6)
