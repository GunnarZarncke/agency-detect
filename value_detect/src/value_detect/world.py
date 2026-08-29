"""Wrap Gunnar's read-only handle-world for passive value-discovery runs.

We import ``SyntheticHandleWorld`` from ``uad_handles`` and never copy or modify it.
The only trace we analyse is the ``x`` array of observations (columns are ``VAR_NAMES``);
the world's second ``x_next`` array is a same-step preview readout and is deliberately ignored.

The noise defaults below were read from the world's source on 2026-08-07 and are
re-checked against the live class by :func:`verify_world_defaults` before any run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from uad_handles.minimal import (
    IDX,
    NO_OP,
    TRUE_LOOP,
    VAR_NAMES,
    SyntheticHandleWorld,
    generate_passive,
)

# generate_passive() burns in this many steps before recording (read from source).
BURN_IN = 100

# Default noise parameters of SyntheticHandleWorld, read from source (verified 2026-08-07).
# alias_noise = 0.0 means the alias columns are exact copies of the causally-effective
# "line" values, which the world's dynamics reuse as the true drivers.
DEFAULT_NOISE: Dict[str, float] = {
    "sensor_noise": 0.05,
    "belief_noise": 0.03,
    "action_noise": 0.04,
    "env_noise": 0.03,
    "alias_noise": 0.0,
    "handle_obs_noise": 0.06,
    "distractor_noise": 0.06,
    "goal_flip_rate": 0.015,
}


@dataclass
class Trace:
    """A recorded passive run: one row per timestep, columns named as ``VAR_NAMES``."""

    frame: pd.DataFrame
    seed: int
    n_steps: int
    burn_in: int
    noise: Dict[str, float]


def verify_world_defaults() -> Dict[str, object]:
    """Check the live world class still matches our recorded defaults.

    Returns a report with any mismatches, so a drift between the design notes and
    Gunnar's code is surfaced rather than silently trusted.
    """
    world = SyntheticHandleWorld(seed=0)
    mismatches = {}
    for name, expected in DEFAULT_NOISE.items():
        actual = getattr(world, name, None)
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "var_names": list(VAR_NAMES),
        "true_loop": [VAR_NAMES[i] for i in TRUE_LOOP],
    }


def passive_trace(seed: int = 0, n_steps: int = 2000) -> Trace:
    """Generate one passive rollout and return its observation table.

    Also asserts the passive invariants the design relies on: no handle operations
    fire, so the operation columns stay idle throughout.
    """
    data = generate_passive(seed, n_steps)

    # Passive invariants: every step is a no-op, so ops never touch the agent.
    noop_code = data.op_names.index(NO_OP)
    if not np.all(data.op_type == noop_code):
        raise AssertionError("Passive trace contains a non-no-op operation.")
    if not np.all(data.op_handle == -1):
        raise AssertionError("Passive trace contains an active operation handle.")

    frame = pd.DataFrame(np.asarray(data.x), columns=list(VAR_NAMES)).astype(int)
    return Trace(
        frame=frame,
        seed=seed,
        n_steps=n_steps,
        burn_in=BURN_IN,
        noise=dict(DEFAULT_NOISE),
    )


def record_trace(trace: Trace, outdir: Union[str, Path]) -> Dict[str, str]:
    """Save the trace as CSV plus a JSON metadata sidecar; return the written paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"trace_seed{trace.seed}.csv"
    trace.frame.to_csv(csv_path, index_label="t")

    meta: Dict[str, object] = {
        "seed": trace.seed,
        "n_steps": trace.n_steps,
        "burn_in": trace.burn_in,
        "variables": list(VAR_NAMES),
        "true_loop": [VAR_NAMES[i] for i in TRUE_LOOP],
        "noise": trace.noise,
        "source_world": "uad_handles.minimal.SyntheticHandleWorld (read-only, imported)",
        "analysed_array": "x (observations); x_next preview deliberately unused",
    }
    meta_path = outdir / f"trace_seed{trace.seed}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {"csv": str(csv_path), "meta": str(meta_path)}


def variable_names() -> List[str]:
    return list(VAR_NAMES)
