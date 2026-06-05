"""Per-role worker loops (each runs in its own process).

System metrics are read from the parent's ``system_live.json`` (psutil's
cpu_count is blocked in sandboxed children); the parent recorder samples global
and per-process usage.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from data_collect.burn import burn_cores, burn_inproc


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _read_live(path: Path, key: str, default: float) -> float:
    if path.exists():
        try:
            return float(json.loads(path.read_text()).get(key, default))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return default
    return default


def _sleep_remaining(t0: float, dt: float) -> None:
    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


def run_stressor(
    *,
    log_path: Path,
    schedule_path: Path,
    dt: float,
    n_ticks: int,
    n_cores: int,
    duty: float,
    min_ticks: int,
    max_ticks: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    on = False
    next_flip = 0
    for t in range(n_ticks):
        t0 = time.perf_counter()
        if t >= next_flip:
            on = rng.random() < duty
            next_flip = t + rng.randint(min_ticks, max_ticks)
        schedule_path.write_text(json.dumps({"t": t, "active": int(on)}))
        if on:
            burn_cores(n_cores, dt * 0.9)
        else:
            _sleep_remaining(t0, dt)
        _append_jsonl(
            log_path,
            {"t": t, "sensor": float(on), "internal": float(t), "action": float(on)},
        )
        _sleep_remaining(t0, dt)


def run_cpu_regulator(
    *,
    log_path: Path,
    system_live_path: Path,
    dt: float,
    n_ticks: int,
    target: float,
    cores: int,
    seed: int,
) -> None:
    """Homeostatic: add load when CPU below setpoint, idle when above.

    Anti-correlates with the exogenous stressor (disturbance rejection toward setpoint).
    """
    rng = random.Random(seed)
    internal = 0.0
    for t in range(n_ticks):
        t0 = time.perf_counter()
        cpu = _read_live(system_live_path, "cpu_percent", target)
        err = target - cpu
        internal = 0.9 * internal + 0.1 * err
        if err > 5.0:
            action = 1.0
            burn_cores(cores, dt * 0.9)
        elif err < -5.0:
            action = -1.0
            _sleep_remaining(t0, dt)
        else:
            action = 0.0
            burn_inproc(dt * 0.45)
        _append_jsonl(
            log_path,
            {"t": t, "sensor": cpu, "internal": internal, "action": action + 0.03 * rng.random()},
        )
        _sleep_remaining(t0, dt)


def run_deadline_burster(
    *,
    log_path: Path,
    dt: float,
    n_ticks: int,
    cores: int,
    min_burst_ticks: int,
    max_burst_ticks: int,
    gap_min_ticks: int,
    gap_max_ticks: int,
    seed: int,
) -> None:
    """Goal-driven multi-core bursts spanning several seconds; ignores system load."""
    rng = random.Random(seed)
    internal = 0.0
    next_burst = rng.randint(gap_min_ticks // 2, gap_max_ticks // 2)
    burst_left = 0
    for t in range(n_ticks):
        t0 = time.perf_counter()
        if burst_left <= 0 and t >= next_burst:
            burst_left = rng.randint(min_burst_ticks, max_burst_ticks)
            next_burst = t + burst_left + rng.randint(gap_min_ticks, gap_max_ticks)
        if burst_left > 0:
            action = 1.0
            burn_cores(cores, dt * 0.9)
            burst_left -= 1
        else:
            action = 0.0
            _sleep_remaining(t0, dt)
        internal = 0.85 * internal + 0.15 * action
        _append_jsonl(
            log_path,
            {"t": t, "sensor": float(internal), "internal": internal, "action": action},
        )
        _sleep_remaining(t0, dt)


def run_mem_grabber(
    *,
    log_path: Path,
    system_live_path: Path,
    dt: float,
    n_ticks: int,
    target_gb: float,
    chunk_mb: int,
    hold_ticks: int,
    gap_min_ticks: int,
    gap_max_ticks: int,
    min_free_gb: float,
    seed: int,
) -> None:
    """Allocate large, random-filled (incompressible) RAM in chunks held for several
    ticks so global memory pressure moves beyond the process's idle footprint."""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    chunk_bytes = int(chunk_mb * 1024 * 1024)
    target_chunks = max(1, int(target_gb * 1024 / chunk_mb))
    chunks: list[np.ndarray] = []
    phase = "alloc"  # alloc -> hold -> release -> gap
    hold_left = 0
    gap_left = 0
    for t in range(n_ticks):
        t0 = time.perf_counter()
        free_gb = _read_live(system_live_path, "available_gb", 8.0)
        action = 0.0
        if phase == "alloc":
            if len(chunks) < target_chunks and free_gb > min_free_gb:
                chunks.append(rng.integers(0, 256, size=chunk_bytes, dtype=np.uint8))
                action = 1.0
            else:
                phase = "hold"
                hold_left = hold_ticks
        elif phase == "hold":
            hold_left -= 1
            if hold_left <= 0:
                phase = "release"
        elif phase == "release":
            if chunks:
                chunks.pop()
                action = -1.0
            else:
                phase = "gap"
                gap_left = prng.randint(gap_min_ticks, gap_max_ticks)
        elif phase == "gap":
            gap_left -= 1
            if gap_left <= 0:
                phase = "alloc"
        held_mb = len(chunks) * chunk_mb
        _sleep_remaining(t0, dt)
        _append_jsonl(
            log_path,
            {"t": t, "sensor": free_gb, "internal": float(held_mb), "action": action, "phase": phase},
        )
        _sleep_remaining(t0, dt)


def run_fixed_worker(*, log_path: Path, dt: float, n_ticks: int, seed: int) -> None:
    rng = random.Random(seed)
    internal = 0.0
    for t in range(n_ticks):
        t0 = time.perf_counter()
        internal = 0.95 * internal + 0.05 * rng.random()
        x = sum(i * i for i in range(200))
        _ = x
        _sleep_remaining(t0, dt)
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
        t0 = time.perf_counter()
        active = 0
        if schedule_path.exists():
            try:
                active = int(json.loads(schedule_path.read_text()).get("active", 0))
            except (json.JSONDecodeError, OSError):
                active = 0
        action = float(active)
        if active:
            scratch.write_bytes(bytes(int(rng.randint(5000, 20000))))
        internal = 0.8 * internal + 0.2 * action
        _sleep_remaining(t0, dt)
        _append_jsonl(
            log_path,
            {"t": t, "sensor": float(active), "internal": internal, "action": action},
        )
