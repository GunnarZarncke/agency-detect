from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Kind:
    """An agent 'kind' for pooled training / held-out evaluation."""

    name: str
    num_agents: int
    variant_mode: str
    copies_per_role: int = 3
    interaction_strength: float = 0.05
    decoy_vars: int = 8


ALL_KINDS: List[Kind] = [
    Kind("easy3_redundant", num_agents=3, variant_mode="redundant", decoy_vars=6),
    Kind("med5_rich", num_agents=5, variant_mode="rich", decoy_vars=8),
    Kind("hard8_complex", num_agents=8, variant_mode="complex", decoy_vars=8),
]

TRAIN_KINDS: List[Kind] = [k for k in ALL_KINDS if k.name != "hard8_complex"]
HELDOUT_KINDS: List[Kind] = [k for k in ALL_KINDS if k.name == "hard8_complex"]

EVAL_WINDOWS: List[int] = [250, 125, 60]
