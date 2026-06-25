"""Small RockSample POMDP (5×5 grid, K rocks) for external trace benchmarks.

True state I = (robot x, y, rock goodness×K). Observation S = noisy rock reading when sensing,
else a sentinel. Actions A = move / sample / sense_i. Longer horizon than card games (~100 steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from learn_agents.external_traces import TraceColumn, pack_trace
from learn_agents.learn_agents import SimulationResult

SENSE_NONE = 0.0
SENSE_GOOD = 1.0
SENSE_BAD = -1.0


@dataclass
class RockSampleConfig:
    map_size: Tuple[int, int] = (5, 5)
    n_rocks: int = 3
    rock_positions: Tuple[Tuple[int, int], ...] = ((1, 1), (3, 3), (4, 4))
    max_steps: int = 100
    sensor_efficiency: float = 20.0
    seed: int = 0


def _dist(pos: Tuple[int, int], rock: Tuple[int, int]) -> float:
    return float(abs(pos[0] - rock[0]) + abs(pos[1] - rock[1]))


class RockSampleEnv:
    """Minimal RockSample for trace logging (not a Gym env)."""

    def __init__(self, cfg: RockSampleConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.pos = (0, 0)
        self.rocks: List[bool] = []
        self.t = 0

    def reset(self) -> None:
        self.pos = (0, 0)
        self.rocks = [self.rng.random() > 0.5 for _ in range(self.cfg.n_rocks)]
        self.t = 0

    def step(self, action: int) -> Tuple[float, bool]:
        """action: 0..3 move N/E/S/W, 4 sample, 5..5+K-1 sense rock."""
        cfg = self.cfg
        w, h = cfg.map_size
        obs_val = SENSE_NONE
        done = False
        if action < 4:
            dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
            self.pos = (min(w - 1, max(0, self.pos[0] + dx)), min(h - 1, max(0, self.pos[1] + dy)))
            if self.pos[0] == w - 1:
                done = True
        elif action == 4:
            for i, rp in enumerate(cfg.rock_positions):
                if self.pos == rp:
                    if not self.rocks[i]:
                        done = True  # bad sample ends episode
                    break
        else:
            rock_i = action - 5
            if 0 <= rock_i < cfg.n_rocks:
                d = _dist(self.pos, cfg.rock_positions[rock_i])
                p_correct = 1.0 - np.exp(-d / cfg.sensor_efficiency)
                correct = self.rng.random() < p_correct
                reading = self.rocks[rock_i] if correct else (not self.rocks[rock_i])
                obs_val = SENSE_GOOD if reading else SENSE_BAD
        self.t += 1
        if self.t >= cfg.max_steps:
            done = True
        return obs_val, done


def roll_rock_sample(cfg: RockSampleConfig | None = None) -> SimulationResult:
    cfg = cfg or RockSampleConfig()
    env = RockSampleEnv(cfg)
    env.reset()
    K = cfg.n_rocks
    n_actions = 5 + K
    T = cfg.max_steps
    s_obs = np.zeros(T, dtype=np.float32)
    a = np.zeros(T, dtype=np.float32)
    ix = np.zeros(T, dtype=np.float32)
    iy = np.zeros(T, dtype=np.float32)
    rocks = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        ix[t], iy[t] = env.pos
        rocks[t] = np.asarray(env.rocks, dtype=np.float32)
        action = env.rng.integers(0, n_actions)
        s_obs[t], done = env.step(int(action))
        a[t] = float(action)
        if done:
            T = t + 1
            s_obs, a, ix, iy, rocks = s_obs[:T], a[:T], ix[:T], iy[:T], rocks[:T]
            break
    cols: List[TraceColumn] = [
        TraceColumn("agent0.sensor.rock_reading", 0, "sensor", s_obs),
        TraceColumn("agent0.internal.pos_x", 0, "internal", ix),
        TraceColumn("agent0.internal.pos_y", 0, "internal", iy),
        TraceColumn("agent0.action.id", 0, "action", a),
    ]
    for k in range(K):
        cols.append(TraceColumn(f"agent0.internal.rock{k}", 0, "internal", rocks[:, k]))
    return pack_trace(cols, num_agents=1, seed=cfg.seed, source=f"rock_sample_{cfg.map_size[0]}x{cfg.map_size[1]}_k{K}")
