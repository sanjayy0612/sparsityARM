import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/analyze_exp007_layouts.py"
    spec = importlib.util.spec_from_file_location("analyze_exp007_layouts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layout_metrics_reward_contiguous_neuron_masks():
    module = _module()
    activation = np.array([[9, 8, 7, 6, 1, 2, 3, 4]], dtype=np.float32)
    original = module.metrics(activation, np.arange(8), block_size=2, sparsity=0.5)
    scattered = module.metrics(activation, np.array([0, 4, 1, 5, 2, 6, 3, 7]), 2, 0.5)
    assert original["mean_expanded_sparsity"] > scattered["mean_expanded_sparsity"]
    assert original["mean_boundary_block_fraction"] < scattered["mean_boundary_block_fraction"]


def test_layout_metrics_reject_invalid_permutation():
    module = _module()
    with pytest.raises(ValueError, match="exactly once"):
        module.metrics(np.ones((2, 8), dtype=np.float32), np.zeros(8, dtype=int), 2, 0.5)
