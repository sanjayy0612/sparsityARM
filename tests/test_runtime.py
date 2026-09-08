import pytest
import torch

from armsparse.config import RuntimeConfig
from armsparse.runtime import ArmSparseRuntime


def _inputs():
    torch.manual_seed(4)
    return torch.randn(1, 4), torch.randn(8, 4), torch.randn(8, 4), torch.randn(4, 8)


def test_prepared_weights_match_fallback_and_skip_repacking():
    hidden, gate, up, down = _inputs()
    runtime = ArmSparseRuntime(RuntimeConfig(mode="armsparse", sparsity=0.5, block_size=8))
    prepared = runtime.prepare_ffn(gate, up, down)

    expected, fallback_metadata = runtime.execute(hidden, gate, up, down)
    actual, prepared_metadata = runtime.execute(hidden, gate, up, down, packed=prepared)

    assert torch.allclose(actual, expected)
    assert fallback_metadata["packing_ms"] > 0
    assert prepared_metadata["packing_ms"] == 0


def test_prepare_rejects_modes_that_do_not_use_packing():
    _, gate, up, down = _inputs()
    runtime = ArmSparseRuntime(RuntimeConfig(mode="dense"))
    with pytest.raises(ValueError, match="only used in armsparse mode"):
        runtime.prepare_ffn(gate, up, down)
