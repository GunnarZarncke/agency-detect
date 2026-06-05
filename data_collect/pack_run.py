"""Load machine JSONL logs into a learn_agents SimulationResult."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from learn_agents.external_traces import TraceColumn, pack_trace
from learn_agents.learn_agents import SimulationResult

AGENT_SPECS: Sequence[Tuple[str, bool]] = (
    ("cpu_regulator", True),
    ("deadline_burster", True),
    ("mem_grabber", True),
    ("fixed_worker", False),
    ("bystander", False),
)


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _series(rows: List[Dict[str, object]], key: str, *, default: float = 0.0) -> np.ndarray:
    return np.array([float(r.get(key, default)) for r in rows], dtype=np.float64)


def pack_machine_run(
    run_dir: Path,
    *,
    seed: int = 0,
    source: str = "machine_dataset",
) -> SimulationResult:
    system_rows = _load_jsonl(run_dir / "system.jsonl")
    T = len(system_rows)
    if T == 0:
        raise ValueError(f"empty system trace in {run_dir}")

    columns: List[TraceColumn] = []
    columns.append(
        TraceColumn("world.stressor_active", -1, "env", _series(system_rows, "stressor_active"))
    )
    time_phase = np.array(
        [math.sin(2.0 * math.pi * t / max(T, 1)) for t in range(T)], dtype=np.float64
    )
    columns.append(TraceColumn("world.time_phase", -1, "env", time_phase))
    columns.append(
        TraceColumn("resource.cpu_percent", -1, "env", _series(system_rows, "cpu_percent"))
    )
    columns.append(
        TraceColumn("resource.ram_used_frac", -1, "env", _series(system_rows, "ram_used_frac"))
    )

    for agent_id, (name, _gt) in enumerate(AGENT_SPECS):
        rows = _load_jsonl(run_dir / f"agent_{name}.jsonl")
        if len(rows) != T:
            raise ValueError(f"agent {name}: {len(rows)} ticks != system {T}")
        columns.append(TraceColumn(f"agent{agent_id}.sensor", agent_id, "sensor", _series(rows, "sensor")))
        columns.append(
            TraceColumn(f"agent{agent_id}.internal", agent_id, "internal", _series(rows, "internal"))
        )
        columns.append(TraceColumn(f"agent{agent_id}.action", agent_id, "action", _series(rows, "action")))

    result = pack_trace(
        columns,
        num_agents=len(AGENT_SPECS),
        seed=seed,
        source=source,
        n_decoy_env=0,
        normalize=False,
    )
    meta = result.metadata
    var_names = list(meta["var_names"])
    cpu_idx = var_names.index("resource.cpu_percent")
    ram_idx = var_names.index("resource.ram_used_frac")
    stressor_idx = var_names.index("world.stressor_active")
    phase_idx = var_names.index("world.time_phase")

    meta["critical_outcomes"] = [
        {"name": "resource.cpu_percent", "index": cpu_idx, "direction": "lower_is_better"},
        {"name": "resource.ram_used_frac", "index": ram_idx, "direction": "lower_is_better"},
    ]
    meta["world_var_indices"] = [stressor_idx, phase_idx]
    meta["resource_var_indices"] = [cpu_idx, ram_idx]
    meta["outcome_influence_ground_truth"] = {
        str(i): gt for i, (_name, gt) in enumerate(AGENT_SPECS)
    }
    meta["agent_labels"] = {str(i): name for i, (name, _gt) in enumerate(AGENT_SPECS)}
    meta["run_dir"] = str(run_dir)
    return result
