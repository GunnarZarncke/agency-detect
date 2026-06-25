"""Orchestrate a real-machine collection run."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List

from data_collect.config import MachineRunConfig
from data_collect.workers import (
    run_bystander,
    run_cpu_regulator,
    run_deadline_burster,
    run_fixed_worker,
    run_mem_grabber,
    run_stressor,
)

GB = 1024**3


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def record_system(
    *,
    log_path: Path,
    proc_log_path: Path,
    schedule_path: Path,
    system_live_path: Path,
    pid_map: Dict[str, int],
    dt: float,
    n_ticks: int,
) -> None:
    """Sample global CPU/RAM and per-process owned CPU/RSS each tick.

    Owned CPU is the agent process's own cpu_times delta (its in-process core);
    helper burn subprocesses are short-lived and surface only in the global meter
    — that gap is the "effect beyond own footprint" we want to detect.
    """
    import psutil

    psutil.cpu_percent(interval=0.2)
    handles: Dict[str, psutil.Process] = {}
    prev_cpu_s: Dict[str, float] = {}
    for name, pid in pid_map.items():
        try:
            p = psutil.Process(pid)
            handles[name] = p
            ct = p.cpu_times()
            prev_cpu_s[name] = float(ct.user + ct.system)
        except psutil.Error:
            pass

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
        available_gb = vm.available / GB
        system_live_path.write_text(
            json.dumps(
                {
                    "t": t,
                    "cpu_percent": cpu,
                    "ram_used_frac": ram_frac,
                    "available_gb": available_gb,
                    "stressor_active": active,
                }
            )
        )
        _append_jsonl(
            log_path,
            {
                "t": t,
                "stressor_active": active,
                "cpu_percent": cpu,
                "ram_used_frac": ram_frac,
                "available_gb": available_gb,
            },
        )

        proc_row: Dict[str, object] = {"t": t}
        for name, p in handles.items():
            try:
                ct = p.cpu_times()
                cur = float(ct.user + ct.system)
                owned_cpu = max(0.0, (cur - prev_cpu_s.get(name, cur)) / dt * 100.0)
                prev_cpu_s[name] = cur
                rss_mb = p.memory_info().rss / (1024**2)
            except psutil.Error:
                owned_cpu, rss_mb = 0.0, 0.0
            proc_row[f"{name}.cpu"] = owned_cpu
            proc_row[f"{name}.rss_mb"] = rss_mb
        _append_jsonl(proc_log_path, proc_row)

        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


def collect_machine_run(cfg: MachineRunConfig, *, seed: int = 0) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = run_dir / "schedule.json"
    system_live_path = run_dir / "system_live.json"
    schedule_path.write_text(json.dumps({"t": 0, "active": 0}))
    system_live_path.write_text(
        json.dumps({"cpu_percent": 0.0, "ram_used_frac": 0.0, "available_gb": 8.0})
    )

    n_ticks = cfg.n_ticks
    dt = cfg.dt

    specs = [
        (
            "stressor",
            run_stressor,
            {
                "log_path": run_dir / "stressor.jsonl",
                "schedule_path": schedule_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "n_cores": cfg.stressor_cores,
                "duty": cfg.stressor_duty,
                "min_ticks": cfg.stressor_min_ticks,
                "max_ticks": cfg.stressor_max_ticks,
                "seed": seed,
            },
        ),
        (
            "cpu_regulator",
            run_cpu_regulator,
            {
                "log_path": run_dir / "agent_cpu_regulator.jsonl",
                "system_live_path": system_live_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "target": cfg.cpu_regulator_target,
                "cores": cfg.cpu_regulator_cores,
                "seed": seed + 1,
            },
        ),
        (
            "deadline_burster",
            run_deadline_burster,
            {
                "log_path": run_dir / "agent_deadline_burster.jsonl",
                "dt": dt,
                "n_ticks": n_ticks,
                "cores": cfg.burster_cores,
                "min_burst_ticks": cfg.burster_min_burst_ticks,
                "max_burst_ticks": cfg.burster_max_burst_ticks,
                "gap_min_ticks": cfg.burster_gap_min_ticks,
                "gap_max_ticks": cfg.burster_gap_max_ticks,
                "seed": seed + 2,
            },
        ),
        (
            "mem_grabber",
            run_mem_grabber,
            {
                "log_path": run_dir / "agent_mem_grabber.jsonl",
                "system_live_path": system_live_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "target_gb": cfg.mem_target_gb,
                "chunk_mb": cfg.mem_chunk_mb,
                "hold_ticks": cfg.mem_hold_ticks,
                "gap_min_ticks": cfg.mem_gap_min_ticks,
                "gap_max_ticks": cfg.mem_gap_max_ticks,
                "min_free_gb": cfg.mem_min_free_gb,
                "seed": seed + 3,
            },
        ),
        (
            "fixed_worker",
            run_fixed_worker,
            {
                "log_path": run_dir / "agent_fixed_worker.jsonl",
                "dt": dt,
                "n_ticks": n_ticks,
                "seed": seed + 4,
            },
        ),
        (
            "bystander",
            run_bystander,
            {
                "log_path": run_dir / "agent_bystander.jsonl",
                "schedule_path": schedule_path,
                "dt": dt,
                "n_ticks": n_ticks,
                "seed": seed + 5,
            },
        ),
    ]

    procs: List[Process] = []
    pid_map: Dict[str, int] = {}
    for name, target, kwargs in specs:
        p = Process(target=target, kwargs=kwargs)
        p.start()
        procs.append(p)
        if name != "stressor":
            pid_map[name] = p.pid

    record_system(
        log_path=run_dir / "system.jsonl",
        proc_log_path=run_dir / "proc_usage.jsonl",
        schedule_path=schedule_path,
        system_live_path=system_live_path,
        pid_map=pid_map,
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
        "mem_target_gb": cfg.mem_target_gb,
        "seed": seed,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
