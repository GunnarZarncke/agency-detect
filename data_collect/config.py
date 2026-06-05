from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MachineRunConfig:
    duration_s: float = 1800.0  # 30 minutes
    dt: float = 1.0
    max_cores: int = 4
    stressor_cores: int = 2
    stressor_duty: float = 0.45  # fraction of time stressor is active
    cpu_regulator_target: float = 42.0  # A1 setpoint (%)
    output_dir: Path = Path("results/intention/machine_runs")

    @property
    def n_ticks(self) -> int:
        return int(self.duration_s / self.dt)
