from dataclasses import dataclass
from typing import Literal


Mode = Literal["dense", "sparse", "armsparse"]


@dataclass(frozen=True)
class RuntimeConfig:
    mode: Mode = "dense"
    sparsity: float = 0.0
    block_size: int = 32
    block_strategy: str = "top_blocks"

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("sparsity must be in [0, 1]")
        if self.block_size not in (8, 16, 32, 64, 128):
            raise ValueError("block_size must be one of 8, 16, 32, 64, 128")
