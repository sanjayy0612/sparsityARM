import importlib.util
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from armsparse.sparsity.quality import MaskCondition


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_exp006.py"
    spec = importlib.util.spec_from_file_location("run_exp006", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reordered_mask_hook_changes_output_and_restores_dense():
    module = _module()
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attention_dropout=0)).eval()
    permutations = torch.stack([torch.randperm(64), torch.randperm(64)])
    ids = torch.tensor([[1, 2, 3, 4]])
    dense = model(ids, use_cache=False).logits
    condition = MaskCondition("block_output_norm", 0.5, 8)
    with module.reordered_masked_mlp(model, condition, permutations):
        sparse = model(ids, use_cache=False).logits
    assert not torch.equal(dense, sparse)
    assert torch.equal(dense, model(ids, use_cache=False).logits)
    assert max(item["relative_l2_error"] for item in module.dense_equivalence(
        model, permutations)) < 2e-5
