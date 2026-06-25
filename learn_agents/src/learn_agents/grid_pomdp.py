"""Small multi-agent grid POMDP (Melting-Pot-style stepping stone).

5×5 world, 2 agents, 3×3 egocentric observation (no global x,y in S).
I = true (x, y) per agent. Longer episodes than CartPole; tractable N for MI/UAD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from learn_agents.external_traces import TraceColumn, pack_trace
from learn_agents.learn_agents import SimulationResult


@dataclass
class GridPomdpConfig:
    grid: int = 5
    view: int = 3
    num_agents: int = 2
    max_steps: int = 250
    seed: int = 0


def _local_patch(world: np.ndarray, pos: Tuple[int, int], view: int) -> np.ndarray:
    """Egocentric view: channels = cell type (empty, wall, resource, agent id norm)."""
    g = world.shape[0]
    half = view // 2
    patch = np.zeros((view, view, 4), dtype=np.float32)
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            pi, pj = pos[0] + di, pos[1] + dj
            ii, jj = di + half, dj + half
            if pi < 0 or pj < 0 or pi >= g or pj >= g:
                patch[ii, jj, 1] = 1.0
            else:
                patch[ii, jj, 0] = 1.0 - world[pi, pj]
                if world[pi, pj] > 0.5:
                    patch[ii, jj, 2] = world[pi, pj]
                if world[pi, pj] >= 2.0:
                    patch[ii, jj, 3] = (world[pi, pj] - 2.0) / max(1, 10)
    return patch.reshape(-1)


class GridPomdpEnv:
    def __init__(self, cfg: GridPomdpConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.world = np.zeros((cfg.grid, cfg.grid), dtype=np.float32)
        self.positions: List[Tuple[int, int]] = []

    def reset(self) -> None:
        g = self.cfg.grid
        self.world = np.zeros((g, g), dtype=np.float32)
        for _ in range(6):
            self.world[self.rng.integers(0, g), self.rng.integers(0, g)] = 1.0
        self.positions = []
        for a in range(self.cfg.num_agents):
            while True:
                p = (int(self.rng.integers(0, g)), int(self.rng.integers(0, g)))
                if self.world[p] < 2.0:
                    self.world[p] = 2.0 + a
                    self.positions.append(p)
                    break

    def step(self, actions: List[int]) -> None:
        g = self.cfg.grid
        for a, act in enumerate(actions):
            x, y = self.positions[a]
            if act == 0:
                x = max(0, x - 1)
            elif act == 1:
                x = min(g - 1, x + 1)
            elif act == 2:
                y = max(0, y - 1)
            elif act == 3:
                y = min(g - 1, y + 1)
            elif act == 4 and self.world[x, y] > 0.5:
                self.world[x, y] = 0.0
            self.positions[a] = (x, y)
            self.world[x, y] = 2.0 + a
        if self.rng.random() < 0.05:
            self.world[self.rng.integers(0, g), self.rng.integers(0, g)] = max(
                self.world[self.rng.integers(0, g), self.rng.integers(0, g)], 1.0
            )


def roll_grid_pomdp(cfg: GridPomdpConfig | None = None) -> SimulationResult:
    cfg = cfg or GridPomdpConfig()
    env = GridPomdpEnv(cfg)
    env.reset()
    T = cfg.max_steps
    n_ch = 4
    sens = [np.zeros((T, n_ch), dtype=np.float32) for _ in range(cfg.num_agents)]
    ix = [np.zeros(T, dtype=np.float32) for _ in range(cfg.num_agents)]
    iy = [np.zeros(T, dtype=np.float32) for _ in range(cfg.num_agents)]
    act = [np.zeros(T, dtype=np.float32) for _ in range(cfg.num_agents)]
    for t in range(T):
        actions = [int(env.rng.integers(0, 5)) for _ in range(cfg.num_agents)]
        for a in range(cfg.num_agents):
            patch = _local_patch(env.world, env.positions[a], cfg.view).reshape(
                cfg.view, cfg.view, n_ch
            )
            sens[a][t] = patch.mean(axis=(0, 1))
            ix[a][t] = float(env.positions[a][0])
            iy[a][t] = float(env.positions[a][1])
            act[a][t] = float(actions[a])
        env.step(actions)
    trace_cols: List[TraceColumn] = []
    for a in range(cfg.num_agents):
        for c in range(n_ch):
            trace_cols.append(
                TraceColumn(f"agent{a}.sensor.ego_ch{c}", a, "sensor", sens[a][:, c])
            )
        trace_cols.extend(
            [
                TraceColumn(f"agent{a}.internal.x", a, "internal", ix[a]),
                TraceColumn(f"agent{a}.internal.y", a, "internal", iy[a]),
                TraceColumn(f"agent{a}.action.move", a, "action", act[a]),
            ]
        )
    return pack_trace(
        trace_cols,
        num_agents=cfg.num_agents,
        seed=cfg.seed,
        source=f"grid_pomdp_{cfg.grid}x{cfg.grid}_a{cfg.num_agents}",
    )
