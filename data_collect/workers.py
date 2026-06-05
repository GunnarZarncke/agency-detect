"""Per-role worker loops (each runs in its own process)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

from data_collect.burn import burn_cores


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def run_stressor(
    *,
    log_path: Path,
    schedule_path: Path,
    dt: float,
    n_ticks: int,
    n_cores: int,
    duty: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    on = False
    next_flip = 0
    for t in range(n_ticks):
        if t >= next_flip:
            on = rng.random() < duty
            next_flip = t + int(rng.uniform(20, 60) / dt)
        schedule_path.write_text(json.dumps({"t": t, "active": int(on)}))
        cpu_self = 0.0
        if on:
            burn_cores(n_cores, dt * 0.85)
            cpu_self = float(n_cores)
        else:
            time.sleep(dt)
        _append_jsonl(
            log_path,
            {"t": t, "sensor": float(on), "internal": float(t), "action": float(on), "cpu_self": cpu_self},
        )


def run_cpu_regulator(
    *,
    log_path: Path,
    system_live_path: Path,
    dt: float,
    n_ticks: int,
    target: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    internal = 0.0
    for t in range(n_ticks):
        cpu = target
        if system_live_path.exists():
            try:
                cpu = float(json.loads(system_live_path.read_text()).get("cpu_percent", target))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                cpu = target
        err = target - cpu
        internal = 0.9 * internal + 0.1 * err
        if err > 8.0:
            action = 1.0
            burn_cores(1, dt * 0.7)
        elif err < -8.0:
            action = -1.0
            time.sleep(dt * 0.9)
        else:
            action = 0.0
            time.sleep(dt * 0.85)
        _append_jsonl(
            log_path,
            {
                "t": t,
                "sensor": cpu,
                "internal": internal,
                "action": action + 0.05 * rng.random(),
            },
        )


def run_deadline_burster(*, log_path: Path, dt: float, n_ticks: int, seed: int) -> None:
    rng = random.Random(seed)
    internal = 0.0
    next_burst = int(rng.uniform(40, 90))
    bursting = False
    burst_left = 0
    for t in range(n_ticks):
        if not bursting and t >= next_burst:
            bursting = True
            burst_left = int(rng.uniform(3, 8))
            next_burst = t + int(rng.uniform(50, 100))
        action = 0.0
        if bursting:
            action = 1.0
            burn_cores(1, min(dt, 0.9))
            burst_left -= 1
            if burst_left <= 0:
                bursting = False
        else:
            time.sleep(dt * 0.9)
        internal = 0.85 * internal + 0.15 * action
        _append_jsonl(
            log_path,
            {
                "t": t,
                "sensor": float(internal),
                "internal": internal,
                "action": action,
            },
        )


def run_mem_grabber(*, log_path: Path, dt: float, n_ticks: int, seed: int) -> None:
    import psutil

    rng = random.Random(seed)
    chunks: list[bytearray] = []
    target_chunks = 40  # ~400 MB
    internal = 0.0
    for t in range(n_ticks):
        vm = psutil.virtual_memory()
        free_gb = vm.available / (1024**3)
        action = 0.0
        if len(chunks) < target_chunks and free_gb > 1.5:
            chunks.append(bytearray(10_000_000))
            action = 1.0
        elif len(chunks) > 0 and free_gb < 0.8:
            chunks.pop()
            action = -1.0
        internal = float(len(chunks))
        time.sleep(dt * 0.95)
        rss_mb = len(chunks) * 10.0
        _append_jsonl(
            log_path,
            {
                "t": t,
                "sensor": free_gb,
                "internal": internal,
                "action": action + 0.02 * rng.random(),
                "rss_mb": rss_mb,
            },
        )


def run_fixed_worker(*, log_path: Path, dt: float, n_ticks: int, seed: int) -> None:
    rng = random.Random(seed)
    internal = 0.0
    for t in range(n_ticks):
        internal = 0.95 * internal + 0.05 * rng.random()
        x = sum(i * i for i in range(200))
        _ = x
        time.sleep(dt)
        _append_jsonl(
            log_path,
            {"t": t, "sensor": internal, "internal": internal, "action": 0.2},
        )


def run_bystander(*, log_path: Path, schedule_path: Path, dt: float, n_ticks: int, seed: int) -> None:
    """Reads shared stressor schedule; logs heavily when W active but does not burn CPU."""
    rng = random.Random(seed)
    scratch = Path("/tmp/agency_detect_bystander_scratch.bin")
    internal = 0.0
    for t in range(n_ticks):
        active = 0
        if schedule_path.exists():
            try:
                active = int(json.loads(schedule_path.read_text()).get("active", 0))
            except (json.JSONDecodeError, OSError):
                active = 0
        action = float(active)
        if active:
            # disk/log activity, not CPU burn
            scratch.write_bytes(bytes(int(rng.randint(5000, 20000))))
        else:
            time.sleep(dt * 0.9)
        internal = 0.8 * internal + 0.2 * action
        _append_jsonl(
            log_path,
            {
                "t": t,
                "sensor": float(active),
                "internal": internal,
                "action": action,
            },
        )
