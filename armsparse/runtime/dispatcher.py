from time import perf_counter
import torch
from ..config import RuntimeConfig
from ..kernels import PackedFFNWeights, dense_ffn, sparse_ffn, block_sparse_ffn
from ..sparsity import activation_scores, make_mask, blockify


class ArmSparseRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def prepare_ffn(self, gate: torch.Tensor, up: torch.Tensor,
                    down: torch.Tensor) -> PackedFFNWeights:
        """Pack one FFN layer for reuse by repeated ``execute`` calls."""
        if self.config.mode != "armsparse":
            raise ValueError("FFN packing is only used in armsparse mode")
        return PackedFFNWeights.from_weights(gate, up, down, self.config.block_size)

    def execute(self, hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
                down: torch.Tensor, *, packed: PackedFFNWeights | None = None
                ) -> tuple[torch.Tensor, dict]:
        start = perf_counter()
        if self.config.mode == "dense":
            output = dense_ffn(hidden, gate, up, down)
            return output, {"mode": "dense", "selector_ms": 0.0, "packing_ms": 0.0,
                            "ffn_ms": (perf_counter()-start)*1000}
        score_start = perf_counter()
        scores = activation_scores(torch.nn.functional.linear(hidden, gate))
        if self.config.mode == "sparse":
            mask = make_mask(scores, self.config.sparsity).values
            selected, realized, blocks = mask, 1 - mask.float().mean().item(), None
            executor_weights = None
            packing_ms = 0.0
        else:
            block = blockify(scores.mean(dim=0), self.config.block_size, self.config.sparsity, self.config.block_strategy)
            selected, realized, blocks = block.values, block.realized_sparsity, block.active_blocks
        selector_ms = (perf_counter() - score_start) * 1000
        if self.config.mode == "armsparse":
            packing_start = perf_counter()
            executor_weights = packed or self.prepare_ffn(gate, up, down)
            packing_ms = (perf_counter() - packing_start) * 1000 if packed is None else 0.0
        ffn_start = perf_counter()
        output = (sparse_ffn(hidden, gate, up, down, selected) if executor_weights is None
                  else block_sparse_ffn(hidden, executor_weights, selected))
        return output, {"mode": self.config.mode, "target_sparsity": self.config.sparsity,
                        "realized_sparsity": realized, "block_size": self.config.block_size if blocks is not None else None,
                        "active_blocks": blocks, "selector_ms": selector_ms,
                        "packing_ms": packing_ms, "ffn_ms": (perf_counter()-ffn_start)*1000}
