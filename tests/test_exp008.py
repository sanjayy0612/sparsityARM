import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_exp008.py"
    spec = importlib.util.spec_from_file_location("run_exp008", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_hot_cold_layout_checks_hash_and_permutations(tmp_path):
    module = _module()
    layout = tmp_path / "layout.npz"
    values = np.tile(np.arange(8192, dtype=np.int32), (16, 1))
    np.savez(layout, hot_cold=values)
    metadata = tmp_path / "analysis.json"
    metadata.write_text(json.dumps({"experiment_id": "EXP-007", "status": "completed",
                                    "layouts_sha256": module.sha(layout)}))
    actual, _ = module.load_hot_cold_layout(layout, metadata)
    np.testing.assert_array_equal(actual.numpy(), values)


def test_load_hot_cold_layout_rejects_hash_mismatch(tmp_path):
    module = _module()
    layout = tmp_path / "layout.npz"
    np.savez(layout, hot_cold=np.tile(np.arange(8192), (16, 1)))
    metadata = tmp_path / "analysis.json"
    metadata.write_text(json.dumps({"experiment_id": "EXP-007", "status": "completed",
                                    "layouts_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash"):
        module.load_hot_cold_layout(layout, metadata)
