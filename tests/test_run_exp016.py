import importlib.util
from pathlib import Path

import pytest
from transformers import LlamaConfig


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_exp016.py"
    spec = importlib.util.spec_from_file_location("run_exp016", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_tinyllama_geometry():
    module = _module()
    valid = LlamaConfig(hidden_size=2048, intermediate_size=5632, num_hidden_layers=22)
    module.validate_geometry(valid)
    invalid = LlamaConfig(hidden_size=2048, intermediate_size=5632, num_hidden_layers=21)
    with pytest.raises(ValueError, match="unexpected TinyLlama geometry"):
        module.validate_geometry(invalid)
