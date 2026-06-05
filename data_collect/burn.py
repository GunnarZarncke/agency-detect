"""Multi-core CPU burn helpers (separate processes to bypass GIL)."""

from __future__ import annotations

import time
from multiprocessing import Process


def _burn_until(stop_time: float) -> None:
    x = 0
    while time.perf_counter() < stop_time:
        x += 1


def burn_cores(n_cores: int, duration_s: float) -> None:
    if n_cores <= 0 or duration_s <= 0:
        return
    stop = time.perf_counter() + duration_s
    procs = [Process(target=_burn_until, args=(stop,)) for _ in range(n_cores)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
