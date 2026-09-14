#!/usr/bin/env python3
"""V2-5 — the V2 benchmark sweep (docs/V2_5_PREREGISTRATION.md; run only after SJ locks).

Worlds × tests × {main, nocore, scramble} × seeds, per the registered feasibility map
and shift budgets. Stores scores + floors (+ partitions and diagnostics) per unit; all
criteria are evaluated by the aggregator against the locked registration.
"""

from __future__ import annotations

import os

os.environ["PYTHONHASHSEED"] = "0"

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.agentblocks import BlockScorer, detect_blocks_swept
from value_detect.floors import ConventionScorer, null_floors, shift_null_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v2_5"

# world -> (env_var, tests); shift budgets per (world, test).
WORLDS = {
    "anchor": ("E", ["fused", "fused_bestkey", "grown_keys", "keyring", "menu"]),
    "colony": (None, ["fused_bestkey", "grown_keys", "keyring", "menu"]),
    "deep_synergy": (None, ["fused_bestkey", "grown_keys", "keyring", "menu"]),
    "slow_meter": ("E", ["fused_bestkey", "grown_keys", "keyring", "menu"]),
    # Registered contingency: partitioned colony, block tests only, for attribution.
    "colony0": (None, ["keyring", "menu"]),
}
SHIFTS_DEFAULT = 200
SHIFTS_COLONY_KEYED = 25  # pooled across seeds at aggregation (registered)
NFUSED = 2_000_000


def _frame(world: str, condition: str, seed: int, n: int) -> pd.DataFrame:
    if world == "anchor":
        if condition == "nocore":
            return vd.passive_trace_nocore(seed=seed, n_steps=n).frame
        base = vd.passive_trace(seed=seed, n_steps=n).frame
    elif world == "colony":
        base = vd.colony_frame(seed=seed, n_steps=n, disconnect_goals=(condition == "nocore"))
    elif world == "colony0":
        base = vd.colony_frame(seed=seed, n_steps=n, coupling=0.0,
                               disconnect_goals=(condition == "nocore"))
    elif world == "deep_synergy":
        base = vd.deep_synergy_frame(seed=seed, n_steps=n, disconnect_goals=(condition == "nocore"))
    else:
        base = vd.slow_meter_frame(seed=seed, n_steps=n, disconnect_goal=(condition == "nocore"))
    if condition == "scramble":
        return vd.scramble_frame(base, seed=seed + 9000)
    return base


def run_unit(task) -> str:
    world, condition, seed, n20, jobsafe = task
    outpath = OUTDIR / f"{world}_{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {world} {condition} s{seed}"
    t0 = time.time()
    env_var, tests = WORLDS[world]
    frame = _frame(world, condition, seed, n20)
    rec = {"world": world, "condition": condition, "seed": seed, "tests": {}}

    part = None
    if "keyring" in tests:
        part = detect_blocks_swept(frame)
        rec["partition"] = {k: part[k] for k in ("agents", "env", "orphans", "dial")}

    for test in tests:
        shifts = SHIFTS_COLONY_KEYED if (world == "colony" and test in ("fused_bestkey", "grown_keys")) else SHIFTS_DEFAULT
        if test == "fused":
            ffu = _frame(world, condition, seed, NFUSED)
            scorer = ConventionScorer(ffu, "fused", env_var=env_var)
            shifts = 50  # V1 pooled-fused spec
        elif test in ("keyring", "menu"):
            scorer = BlockScorer(frame, test, env_var=env_var, partition=part)
        else:
            scorer = ConventionScorer(frame, test, env_var=env_var)
        scores = scorer.score_all()
        pooled_combo = world == "colony" and test in ("fused_bestkey", "grown_keys")
        if pooled_combo:
            # Registration pools these floors across seeds: store RAW samples (same RNG
            # stream null_floors would use) and derive this seed's stats from them.
            samples = shift_null_samples(scorer, n_shifts=shifts, seed=seed * 97 + 13)
            floors = _stats_from_samples(samples)
        else:
            floors = null_floors(scorer, n_shifts=shifts, seed=seed * 97 + 13)
        entry = {"scores": scores.reset_index().to_dict("records"),
                 "floors": floors.reset_index().to_dict("records"),
                 "n_shifts": shifts}
        if pooled_combo:
            entry["null_samples"] = samples.to_dict("records")
        if hasattr(scorer, "lost_mass"):
            entry["lost_mass"] = scorer.lost_mass
        rec["tests"][test] = entry
        del scorer

    outpath.write_text(json.dumps(_jsonable(rec)))
    return f"done {world} {condition} s{seed} in {(time.time()-t0)/60:.1f}m"


def _stats_from_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-seed floor stats derived from raw samples (columns match null_floors)."""
    rows = []
    for v, g in samples.groupby("variable", sort=False):
        row = {"variable": v, "n_shifts": len(g)}
        for name in ("push_in", "out_sys", "out_env", "total_flow"):
            arr = g[name].to_numpy(dtype=float)
            finite = np.isfinite(arr).any()
            row[f"{name}_p95"] = float(np.nanpercentile(arr, 95)) if finite else float("nan")
            row[f"{name}_p995"] = float(np.nanpercentile(arr, 99.5)) if finite else float("nan")
            row[f"{name}_mean"] = float(np.nanmean(arr)) if finite else float("nan")
            row[f"{name}_sd"] = float(np.nanstd(arr)) if finite else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--n20", type=int, default=20_000)
    parser.add_argument("--worlds", type=str, default="anchor,colony,deep_synergy,slow_meter")
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    worlds = args.worlds.split(",")
    tasks = [(w, cond, s, args.n20, True)
             for w in worlds for s in range(args.seeds)
             for cond in (("main", "nocore") if w == "colony0" else ("main", "nocore", "scramble"))]
    # Cheap worlds first so early results are inspectable while the colony grinds.
    order = {"anchor": 0, "slow_meter": 1, "colony0": 2, "deep_synergy": 3, "colony": 4}
    tasks.sort(key=lambda t: order.get(t[0], 9))
    print(f"[v2_5] {len(tasks)} units, jobs={args.jobs}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap(run_unit, tasks), 1):
            el = time.time() - t0
            print(f"[v2_5 {i}/{len(tasks)}] {msg} | {el/60:.0f}m elapsed", flush=True)
    print(f"V2-5 SWEEP COMPLETE in {(time.time()-t0)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
