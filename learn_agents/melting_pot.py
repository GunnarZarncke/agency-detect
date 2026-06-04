"""Melting Pot substrate logger (structured obs, not RGB).

Maps per-player vector observations to S/A/I columns compatible with ``external_traces.pack_trace``.
Requires optional ``meltingpot`` (see requirements-dev.txt). Not invoked in default CI.

Design:
  S — egocentric / public vector fields present in the player's observation dict
  I — simulator state fields omitted from S (e.g. true position when obs is partial)
  A — discrete action id per step

Do not log RGB tensors as agent variables (high-N, no blanket structure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from learn_agents.external_traces import TraceColumn, pack_trace
from learn_agents.learn_agents import SimulationResult

# Substrates vetted for multi-agent structured logging (expand after smoke tests).
SUPPORTED_SUBSTRATES: Tuple[str, ...] = (
    "collaborative_cooking__ring",
    "collaborative_cooking__circuit",
    "clean_up_2x2",
)

_SKIP_OBS_KEYS = frozenset({"RGB", "rgb", "RGB2", "WORLD.RGB", "WORLD.RGB2"})


@dataclass
class MeltingPotConfig:
    substrate_name: str = "collaborative_cooking__ring"
    max_steps: int = 500
    seed: int = 0
    roles: Tuple[str, ...] | None = None  # default: one default role per player
    n_decoy_env: int = 4


def _require_meltingpot():
    try:
        from meltingpot import substrate  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "meltingpot is not installed. Install optional deps, e.g.\n"
            "  pip install 'dm-env' meltingpot\n"
            "See learn_agents/EXPERIMENTS.md (E16) and requirements-dev.txt."
        ) from e
    return substrate


def _numeric_leaves(prefix: str, obj: Any, out: List[Tuple[str, float]]) -> None:
    """Flatten numeric observation leaves; skip image-like keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _SKIP_OBS_KEYS or "RGB" in str(k).upper():
                continue
            _numeric_leaves(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _numeric_leaves(f"{prefix}[{i}]", v, out)
    else:
        try:
            out.append((prefix, float(obj)))
        except (TypeError, ValueError):
            pass


def _split_sensor_internal(
    leaves: Sequence[Tuple[str, float]],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Heuristic: WORLD.* and egocentric public fields → S; latent-style → I."""
    sensors: List[Tuple[str, float]] = []
    internals: List[Tuple[str, float]] = []
    for name, val in leaves:
        low = name.lower()
        if any(x in low for x in ("position", "pos", "inventory", "held", "latent", "state")):
            if "ego" in low or "obs" in low:
                sensors.append((name, val))
            else:
                internals.append((name, val))
        else:
            sensors.append((name, val))
    if not internals and len(sensors) > 4:
        internals = sensors[len(sensors) // 2 :]
        sensors = sensors[: len(sensors) // 2]
    return sensors, internals


def roll_melting_pot(cfg: MeltingPotConfig | None = None) -> SimulationResult:
    """Roll out a Melting Pot substrate and pack a multi-agent trace."""
    cfg = cfg or MeltingPotConfig()
    if cfg.substrate_name not in SUPPORTED_SUBSTRATES:
        raise ValueError(
            f"substrate {cfg.substrate_name!r} not in SUPPORTED_SUBSTRATES {SUPPORTED_SUBSTRATES}"
        )

    substrate = _require_meltingpot()
    # Probe player count without a long rollout.
    probe = substrate.build(cfg.substrate_name, roles=["default"] * 2)
    try:
        n_players = len(probe.observation_spec())
    finally:
        probe.close()

    roles = list(cfg.roles) if cfg.roles else ["default"] * n_players
    if len(roles) != n_players:
        raise ValueError(f"roles length {len(roles)} != num_players {n_players}")

    env = substrate.build(cfg.substrate_name, roles=roles)
    rng = np.random.default_rng(cfg.seed)
    try:
        timestep = env.reset()
        T = cfg.max_steps
        per_player_s: List[List[List[float]]] = [[] for _ in range(n_players)]
        per_player_i: List[List[List[float]]] = [[] for _ in range(n_players)]
        per_player_a: List[List[float]] = [[] for _ in range(n_players)]
        sensor_names: List[List[str]] = [[] for _ in range(n_players)]
        internal_names: List[List[str]] = [[] for _ in range(n_players)]

        for _ in range(T):
            if timestep.last():
                break
            obs = timestep.observation
            for p in range(n_players):
                leaves: List[Tuple[str, float]] = []
                _numeric_leaves("", obs[p], leaves)
                sens, internal = _split_sensor_internal(leaves)
                if not sensor_names[p] and sens:
                    sensor_names[p] = [n for n, _ in sens]
                if not internal_names[p] and internal:
                    internal_names[p] = [n for n, _ in internal]
                per_player_s[p].append([v for _, v in sens] or [0.0])
                per_player_i[p].append([v for _, v in internal] or [0.0])
                act = int(rng.integers(0, 8))
            actions = [
                int(rng.integers(0, env.action_spec()[p].num_values)) for p in range(n_players)
            ]
            for p in range(n_players):
                per_player_a[p].append(float(actions[p]))
            timestep = env.step(actions)

        actual_T = len(per_player_a[0])
        if actual_T == 0:
            raise RuntimeError(f"melting pot substrate {cfg.substrate_name} produced zero steps")

        cols: List[TraceColumn] = []
        for p in range(n_players):
            s_arr = np.asarray(per_player_s[p], dtype=np.float32)
            i_arr = np.asarray(per_player_i[p], dtype=np.float32)
            a_arr = np.asarray(per_player_a[p], dtype=np.float32)
            for j in range(s_arr.shape[1]):
                name = sensor_names[p][j] if j < len(sensor_names[p]) else f"s{j}"
                cols.append(
                    TraceColumn(f"agent{p}.sensor.{name}", p, "sensor", s_arr[:, j])
                )
            for j in range(i_arr.shape[1]):
                name = internal_names[p][j] if j < len(internal_names[p]) else f"i{j}"
                cols.append(
                    TraceColumn(f"agent{p}.internal.{name}", p, "internal", i_arr[:, j])
                )
            cols.append(TraceColumn(f"agent{p}.action.id", p, "action", a_arr))

        return pack_trace(
            cols,
            num_agents=n_players,
            seed=cfg.seed,
            source=f"melting_pot_{cfg.substrate_name}",
            n_decoy_env=cfg.n_decoy_env,
            rng=rng,
        )
    finally:
        env.close()
