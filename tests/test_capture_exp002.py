import importlib.util
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaModel


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/capture_exp002.py"
    spec = importlib.util.spec_from_file_location("capture_exp002", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_swap_failure_is_recorded_as_unavailable(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.psutil, "swap_memory", lambda: (_ for _ in ()).throw(OSError()))
    assert module.swap_used() is None


def test_capture_hooks_observe_swiglu_without_changing_inputs():
    torch.manual_seed(17)
    model = LlamaModel(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attention_dropout=0)).to(dtype=torch.bfloat16).eval()
    captured = {}
    handles = []

    def input_hook(layer):
        def hook(module, inputs):
            captured.setdefault(layer, {})["x"] = inputs[0].detach().clone()
        return hook

    def activation_hook(layer):
        def hook(module, inputs):
            captured.setdefault(layer, {})["activation"] = inputs[0].detach().clone()
        return hook

    for layer, block in enumerate(model.layers):
        handles.append(block.mlp.register_forward_pre_hook(input_hook(layer)))
        handles.append(block.mlp.down_proj.register_forward_pre_hook(activation_hook(layer)))
    try:
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([[1,2,3,4,5,6,7,8,9]]),
                           use_cache=False).last_hidden_state
            for layer, values in captured.items():
                mlp = model.layers[layer].mlp
                recomputed = torch.nn.functional.silu(mlp.gate_proj(values["x"])) * mlp.up_proj(values["x"])
                assert torch.equal(recomputed, values["activation"])
        assert set(captured) == {0, 1}
        assert torch.isfinite(output).all()
    finally:
        for handle in handles:
            handle.remove()
