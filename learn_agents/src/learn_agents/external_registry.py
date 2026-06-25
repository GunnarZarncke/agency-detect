"""Registry of external trace builders for amortized pool / benchmarks.

Keys match ``Kind.external_key``. Builders return ``SimulationResult`` with S/A/I roles.
"""

from __future__ import annotations

from typing import Callable, Dict

from learn_agents.grid_pomdp import GridPomdpConfig, roll_grid_pomdp
from learn_agents.learn_agents import SimulationResult
from learn_agents.physics_pomdp import roll_cartpole_multi, roll_cartpole_partial_obs
from learn_agents.rock_sample import RockSampleConfig, roll_rock_sample

ExternalBuilder = Callable[[int, int], SimulationResult]


def _physics(seed: int, t_steps: int) -> SimulationResult:
    del t_steps  # one continuous rollout; horizon is env-defined
    return roll_cartpole_partial_obs(seed=seed)


def _physics_track(seed: int, t_steps: int) -> SimulationResult:
    del t_steps
    return roll_cartpole_partial_obs(seed=seed, policy="track", theta_ref=0.12, normalize=False)


def _physics_x3(seed: int, t_steps: int) -> SimulationResult:
    del t_steps
    return roll_cartpole_multi(seed=seed, num_agents=3, n_decoy_env=8)


def _rock(seed: int, t_steps: int) -> SimulationResult:
    return roll_rock_sample(RockSampleConfig(seed=seed, max_steps=min(t_steps, 100)))


def _grid3(seed: int, t_steps: int) -> SimulationResult:
    return roll_grid_pomdp(
        GridPomdpConfig(grid=3, view=3, num_agents=2, max_steps=min(t_steps, 250), seed=seed)
    )


def _grid5(seed: int, t_steps: int) -> SimulationResult:
    return roll_grid_pomdp(
        GridPomdpConfig(grid=5, view=3, num_agents=2, max_steps=min(t_steps, 250), seed=seed)
    )


def _melting_pot_ring(seed: int, t_steps: int) -> SimulationResult:
    from learn_agents.melting_pot import MeltingPotConfig, roll_melting_pot

    return roll_melting_pot(
        MeltingPotConfig(substrate_name="collaborative_cooking__ring", max_steps=t_steps, seed=seed)
    )


EXTERNAL_BUILDERS: Dict[str, ExternalBuilder] = {
    "physics_cartpole": _physics,
    "physics_cartpole_track": _physics_track,
    "physics_cartpole_x3": _physics_x3,
    "rock_sample_5x5": _rock,
    "grid_pomdp_3x3": _grid3,
    "grid_pomdp_5x5": _grid5,
    "melting_pot_cooking_ring": _melting_pot_ring,
}


def build_external_trace(key: str, *, seed: int, t_steps: int) -> SimulationResult:
    if key not in EXTERNAL_BUILDERS:
        raise KeyError(f"unknown external_key {key!r}; known: {sorted(EXTERNAL_BUILDERS)}")
    return EXTERNAL_BUILDERS[key](seed, t_steps)
