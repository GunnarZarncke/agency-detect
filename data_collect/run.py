"""Orchestrate a real-machine collection run."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import List

from data_collect.config import MachineRunConfig
from data_collect.workers import (
    run_bystander,
    run_cpu_regulator,
    run_deadline_burster,
    run_fixed_worker,
    run_mem_grabber,
    run_stressor,
)


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def record_system(
    *,
    log_path: Path,
    schedule_path: Path,
    system_live_path: Path,
    dt: float,
    n_ticks: int,
) -> None:
    import psutil

    psutil.cpu_percent(interval=0.2)
    for t in range(n_ticks):
        t0 = time.perf_counter()
        active = 0
        if schedule_path.exists():
            try:
                active = int(json.loads(schedule_path.read_text()).get("active", 0))
            except (json.JSONDecodeError, OSError):
                active = 0
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        ram_frac = vm.percent / 100.0
        live = {"t": t, "cpu_percent": cpu, "ram_used_frac": ram_frac, "stressor_active": active}
        system_live_path.write_text(json.dumps(live))
        _append_jsonl(
            log_path,
            {
                "t": t,
                "stressor_active": active,
                "cpu_percent": cpu,
                "ram_used_frac": ram_frac,
            },
        )
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))


def collect_machine_run(cfg: MachineRunConfig, *, seed: int = 0) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = run_dir / "schedule.json"
    system_live_path = run_dir / "system_live.json"
    schedule_path.write_text(json.dumps({"t": 0, "active": 0}))
    system_live_path.write_text(json.dumps({"cpu_percent": 0.0, "ram_used_frac": 0.0}))

    n_ticks = cfg.n_ticks
    dt = cfg.dt
    procs: List[Process] = []

    procs.append(
        Process(
            target=run_stressor,
            kwargs={
                "log_path": run_dir / "stressor.jsonl",
                "schedule_path": schedule_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "n_cores": cfg.stressor_cores,
                "duty": cfg.stressor_duty,
                "seed": seed,
            },
        )
    )
    procs.append(
        Process(
            target=run_cpu_regulator,
            kwargs={
                "log_path": run_dir / "agent_cpu_regulator.jsonl",
                "system_live_path": system_live_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "target": cfg.cpu_regulator_target,
                "seed": seed + 1,
            },
        )
    )
    procs.append(
        Process(
            target=run_deadline_burster,
            kwargs={
                "log_path": run_dir / "agent_deadline_burster.jsonl",
                "dt": dt,
                "n_ticks": n_ticks,
                "seed": seed + 2,
            },
        )
    )
    procs.append(
        Process(
            target=run_mem_grabber,
            kwargs={
                "log_path": run_dir / "agent_mem_grabber.jsonl",
                "dt": dt,
                "n_ticks": n_ticks,
                "seed": seed + 3,
            },
        )
    )
    procs.append(
        Process(
            target=run_fixed_worker,
            kwargs={
                "log_path": run_dir / "agent_fixed_worker.jsonl",
                "dt": dt,
                "n_ticks": n_ticks,
                "seed": seed + 4,
            },
        )
    )
    procs.append(
        Process(
            target=run_bystander,
            kwargs={
                "log_path": run_dir / "agent_bystander.jsonl",
                "schedule_path": schedule_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "seed": seed + 5,
            },
        )
    )

    for p in procs:
        p.start()

    record_system(
        log_path=run_dir / "system.jsonl",
        schedule_path=schedule_path,
        system_live_path=system_live_path,
        dt=dt,
        n_ticks=n_ticks,
    )

    for p in procs:
        p.join()

    meta = {
        "run_id": run_id,
        "duration_s": cfg.duration_s,
        "dt": cfg.dt,
        "n_ticks": n_ticks,
        "max_cores": cfg.max_cores,
        "stressor_cores": cfg.stressor_cores,
        "seed": seed,
        "started": run_id,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
