from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MachineRunConfig:
    duration_s: float = 1800.0  # 30 minutes
    dt: float = 1.0
    max_cores: int = 4

    # Background stressor (exogenous world): 2 cores, burst spans multiple ticks.
    stressor_cores: int = 2
    stressor_duty: float = 0.45
    stressor_min_ticks: int = 4
    stressor_max_ticks: int = 12

    # cpu_regulator (A0): drive system CPU toward a *reachable* setpoint.
    cpu_regulator_target: float = 55.0  # %
    cpu_regulator_cores: int = 2

    # deadline_burster (A1): periodic multi-core bursts spanning several seconds.
    burster_cores: int = 2
    burster_min_burst_ticks: int = 4
    burster_max_burst_ticks: int = 8
    burster_gap_min_ticks: int = 30
    burster_gap_max_ticks: int = 70

    # mem_grabber (A2): large, random-filled (incompressible) allocations held for
    # several ticks so global RAM actually moves beyond the process's idle footprint.
    mem_target_gb: float = 2.5
    mem_chunk_mb: int = 256
    mem_hold_ticks: int = 10
    mem_gap_min_ticks: int = 20
    mem_gap_max_ticks: int = 50
    mem_min_free_gb: float = 2.0  # safety: stop allocating below this free RAM

    output_dir: Path = Path("results/intention/machine_runs")

    @property
    def n_ticks(self) -> int:
        return int(self.duration_s / self.dt)
