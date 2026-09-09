import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from armsparse.sparsity.quality import MaskCondition


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_exp003.py"
    spec = importlib.util.spec_from_file_location("run_exp003", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_corpus_validates_records(tmp_path):
    module = _module()
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"id": "a", "text": "useful text"}) + "\n")
    assert module.load_corpus(path) == [{"id": "a", "text": "useful text"}]
    path.write_text('{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n')
    with pytest.raises(ValueError, match="unique"):
        module.load_corpus(path)


def test_mask_hook_changes_output_and_is_removed():
    module = _module()
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attention_dropout=0)).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    dense = model(ids, use_cache=False).logits
    with module.masked_mlp(model, MaskCondition("block", 0.5, 8)):
        sparse = model(ids, use_cache=False).logits
    restored = model(ids, use_cache=False).logits
    assert not torch.equal(dense, sparse)
    assert torch.equal(dense, restored)


def test_condition_parser():
    module = _module()
    assert module.parse_condition("dense") == MaskCondition("dense", 0)
    assert module.parse_condition("block:32:0.4") == MaskCondition("block", 0.4, 32)
    with pytest.raises(Exception):
        module.parse_condition("bad")
