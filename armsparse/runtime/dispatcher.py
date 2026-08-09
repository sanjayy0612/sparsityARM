from time import perf_counter
import torch
from ..config import RuntimeConfig
from ..kernels import PackedFFNWeights, dense_ffn, sparse_ffn, block_sparse_ffn
from ..sparsity import activation_scores, make_mask, blockify


class ArmSparseRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def execute(self, hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
                down: torch.Tensor) -> tuple[torch.Tensor, dict]:
        start = perf_counter()
        if self.config.mode == "dense":
            output = dense_ffn(hidden, gate, up, down)
            return output, {"mode": "dense", "selector_ms": 0.0, "ffn_ms": (perf_counter()-start)*1000}
        score_start = perf_counter()
        scores = activation_scores(torch.nn.functional.linear(hidden, gate))
        if self.config.mode == "sparse":
            mask = make_mask(scores, self.config.sparsity).values
            selected, realized, blocks = mask, 1 - mask.float().mean().item(), None
            packed = None
        else:
            block = blockify(scores.mean(dim=0), self.config.block_size, self.config.sparsity, self.config.block_strategy)
            selected, realized, blocks = block.values, block.realized_sparsity, block.active_blocks
            packed = PackedFFNWeights.from_weights(gate, up, down, self.config.block_size)
        selector_ms = (perf_counter() - score_start) * 1000
        ffn_start = perf_counter()
        output = sparse_ffn(hidden, gate, up, down, selected) if packed is None else block_sparse_ffn(hidden, packed, selected)
        return output, {"mode": self.config.mode, "target_sparsity": self.config.sparsity,
                        "realized_sparsity": realized, "block_size": self.config.block_size if blocks is not None else None,
                        "active_blocks": blocks, "selector_ms": selector_ms, "ffn_ms": (perf_counter()-ffn_start)*1000}
