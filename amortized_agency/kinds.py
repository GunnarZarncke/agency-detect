from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

Backend = Literal["sim", "external"]


@dataclass(frozen=True)
class Kind:
    """An agent 'kind' for pooled training / held-out evaluation."""

    name: str
    num_agents: int
    variant_mode: str
    copies_per_role: int = 3
    interaction_strength: float = 0.05
    decoy_vars: int = 8
    backend: Backend = "sim"
    external_key: str | None = None

    def __post_init__(self) -> None:
        if self.backend == "external" and not self.external_key:
            raise ValueError(f"Kind {self.name!r}: external backend requires external_key")
        if self.backend == "sim" and self.external_key:
            raise ValueError(f"Kind {self.name!r}: sim backend must not set external_key")


# E13 sim spectrum (unchanged default benchmarks).
ALL_KINDS: List[Kind] = [
    Kind("easy3_redundant", num_agents=3, variant_mode="redundant", decoy_vars=6),
    Kind("med5_rich", num_agents=5, variant_mode="rich", decoy_vars=8),
    Kind("hard8_complex", num_agents=8, variant_mode="complex", decoy_vars=8),
]

TRAIN_KINDS: List[Kind] = [k for k in ALL_KINDS if k.name != "hard8_complex"]
HELDOUT_KINDS: List[Kind] = [k for k in ALL_KINDS if k.name == "hard8_complex"]

# E16 extended pool: E15 externals + Melting Pot (opt-in via scripts; not in ALL_KINDS).
EXTERNAL_KINDS: List[Kind] = [
    Kind(
        "physics_cartpole",
        num_agents=1,
        variant_mode="external",
        backend="external",
        external_key="physics_cartpole",
        decoy_vars=4,
    ),
    Kind(
        "physics_cartpole_track",
        num_agents=1,
        variant_mode="external",
        backend="external",
        external_key="physics_cartpole_track",
        decoy_vars=4,
    ),
    Kind(
        "physics_cartpole_x3",
        num_agents=3,
        variant_mode="external",
        backend="external",
        external_key="physics_cartpole_x3",
        decoy_vars=8,
    ),
    Kind(
        "rock_sample_5x5",
        num_agents=1,
        variant_mode="external",
        backend="external",
        external_key="rock_sample_5x5",
        decoy_vars=4,
    ),
    Kind(
        "grid_pomdp_3x3",
        num_agents=2,
        variant_mode="external",
        backend="external",
        external_key="grid_pomdp_3x3",
        decoy_vars=4,
    ),
    Kind(
        "grid_pomdp_5x5",
        num_agents=2,
        variant_mode="external",
        backend="external",
        external_key="grid_pomdp_5x5",
        decoy_vars=4,
    ),
    Kind(
        "melting_pot_cooking_ring",
        num_agents=2,  # ring substrates are 2-player; updated on first successful build
        variant_mode="external",
        backend="external",
        external_key="melting_pot_cooking_ring",
        decoy_vars=4,
    ),
]

EXTENDED_TRAIN_KINDS: List[Kind] = TRAIN_KINDS + [
    k
    for k in EXTERNAL_KINDS
    if k.name in ("physics_cartpole", "physics_cartpole_track", "rock_sample_5x5", "grid_pomdp_3x3")
]

EXTENDED_HELDOUT_KINDS: List[Kind] = HELDOUT_KINDS + [
    k
    for k in EXTERNAL_KINDS
    if k.name in ("grid_pomdp_5x5", "melting_pot_cooking_ring")
]

EXTENDED_ALL_KINDS: List[Kind] = EXTENDED_TRAIN_KINDS + [
    k for k in EXTENDED_HELDOUT_KINDS if k not in EXTENDED_TRAIN_KINDS
]

# E13 MI ceiling: use REFERENCE_WINDOWS for primary benchmarks; EVAL_WINDOWS for
# the short-window band where MI collapses (see amortized_agency/benchmark.py).
EVAL_WINDOWS: List[int] = [250, 125, 60]
