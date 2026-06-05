"""Multi-core CPU burn helpers (separate processes to bypass GIL).

The current process always burns one core (so it is attributable to the agent's
own ``cpu_times``); extra cores run in helper subprocesses that show up only in
the system-wide meter — the "effect beyond own footprint" we want to detect.
"""

from __future__ import annotations

import time
from multiprocessing import Process


def _burn_until(stop_time: float) -> None:
    x = 0.0
    while time.perf_counter() < stop_time:
        x += 1.0
        x *= 1.0000001


def burn_inproc(duration_s: float) -> None:
    """Busy-loop in the calling process for ~duration_s (one core, attributable)."""
    if duration_s <= 0:
        return
    _burn_until(time.perf_counter() + duration_s)


def burn_cores(n_cores: int, duration_s: float) -> None:
    """Burn n_cores for duration_s: 1 in-process + (n_cores-1) helper subprocesses."""
    if n_cores <= 0 or duration_s <= 0:
        return
    stop = time.perf_counter() + duration_s
    helpers = [Process(target=_burn_until, args=(stop,)) for _ in range(max(0, n_cores - 1))]
    for p in helpers:
        p.start()
    burn_inproc(duration_s)
    for p in helpers:
        p.join()
