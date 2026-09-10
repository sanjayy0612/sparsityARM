import importlib.util
import ctypes
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_ffn", ROOT / "benchmarks/native_ffn.py")
native_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_module)


@pytest.fixture(scope="module")
def native():
    if sys.platform != "darwin":
        pytest.skip("Accelerate backend requires macOS")
    return native_module.NativeFFN(native_module.build())


@pytest.mark.parametrize("b", [8, 16, 32, 64])
@pytest.mark.parametrize("sparsity", [0., .3, 1.])
@pytest.mark.parametrize("family", ["clustered", "scattered"])
def test_native_matches_independent_masked_reference(native, b, sparsity, family):
    rng = np.random.default_rng(41)
    h, m = 17, 128  # odd hidden dimension exercises vector remainder handling
    x = rng.standard_normal(h).astype(np.float32)
    gate = (rng.standard_normal((m, h))*.1).astype(np.float32)
    up = (rng.standard_normal((m, h))*.1).astype(np.float32)
    down = (rng.standard_normal((h, m))*.1).astype(np.float32)
    base, blocks = native_module.mask_pair(rng, m, b, sparsity, family)
    packed = down.reshape(h, m//b, b).transpose(1, 0, 2).copy()
    scratch, out = np.empty(2*m, np.float32), np.empty(h, np.float32)
    for mode, weights, mask in [(0, down, np.ones(m, np.uint8)),
                                (1, down.T.copy(), base), (2, packed, blocks), (3, packed, blocks)]:
        elapsed = native.execute(mode, x, gate, up, weights, mask, b, scratch, out)
        expected_mask = np.ones(m) if mode == 0 else (base if mode == 1 else np.repeat(blocks, b))
        expected = native_module.reference(x, gate, up, down, expected_mask)
        np.testing.assert_allclose(out, expected, rtol=2e-4, atol=2e-5)
        assert elapsed >= 0


def test_scattered_expansion_preserves_base_and_clustered_is_equal():
    for family in ("scattered", "clustered"):
        base, blocks = native_module.mask_pair(np.random.default_rng(17), 8192, 32, .3, family)
        expanded = np.repeat(blocks, 32)
        assert np.all(expanded >= base)
        if family == "clustered":
            np.testing.assert_array_equal(expanded, base)
        else:
            assert base.sum() == round(8192*.7)
            assert expanded.sum() > base.sum()


def test_scan_diagnostic_returns_selected_indices(native):
    mask = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    indices = np.empty(len(mask), np.int32)
    count = ctypes.c_int()
    native.lib.scan_mask(mask, len(mask), indices, ctypes.byref(count))
    assert count.value == 3
    np.testing.assert_array_equal(indices[:count.value], [1, 3, 4])


def test_staged_block_profile_matches_reference_and_reports_structure(native):
    rng = np.random.default_rng(909)
    h, m, b = 17, 64, 8
    x = rng.standard_normal(h).astype(np.float32)
    gate = (rng.standard_normal((m, h)) * .1).astype(np.float32)
    up = (rng.standard_normal((m, h)) * .1).astype(np.float32)
    dense_down = (rng.standard_normal((h, m)) * .1).astype(np.float32)
    packed = dense_down.reshape(h, m//b, b).transpose(1, 0, 2).copy()
    blocks = np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
    scratch, output = np.empty(2*m, np.float32), np.empty(h, np.float32)
    profile = native.profile_block_staged(x, gate, up, packed, blocks, b, scratch, output,
                                          np.empty(m//b, np.int32))
    expected = native_module.reference(x, gate, up, dense_down, np.repeat(blocks, b))
    np.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-5)
    assert profile["active_blocks"] == 5
    assert profile["active_runs"] == 3
    assert profile["total_ms"] >= sum(profile[key] for key in (
        "output_init_ms", "mask_scan_ms", "gate_up_activation_ms", "down_projection_ms")) * .999
