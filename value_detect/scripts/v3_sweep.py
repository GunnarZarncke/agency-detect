#!/usr/bin/env python3
"""V3 — the curve-ball sweep (docs/V3_REGISTRATION.md, LOCKED).

Worlds: puppetfast, puppetslow, alias{000,005,010,020,050}. Tests: any-block (`menu`),
own-block (`keyring`). Conditions per registration. Plus yardstick units (both probes,
all variables) on puppetfast, puppetslow, alias000, alias050.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.agentblocks import BlockScorer, detect_blocks_swept
from value_detect.floors import null_floors
from value_detect.yardstick import yardstick_scores, yardstick_verdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "v3_5"

ALIAS = {"alias000": 0.0, "alias005": 0.005, "alias010": 0.01,
         "alias020": 0.02, "alias050": 0.05}
TESTS = ("keyring", "menu")


def _frame(world: str, condition: str, seed: int, n: int) -> pd.DataFrame:
    if world == "puppetfast":
        base = vd.colony_frame(seed=seed, n_steps=n, puppet=("fast", 3),
                               disconnect_goals=(condition == "nocore"))
    elif world == "puppetslow":
        base = vd.colony_frame(seed=seed, n_steps=n, puppet=("slow", 3),
                               disconnect_goals=(condition == "nocore"))
    else:
        base = vd.alias_colony_frame(seed=seed, n_steps=n, alias_noise=ALIAS[world],
                                     disconnect_goals=(condition == "nocore"))
    if condition == "scramble":
        return vd.scramble_frame(base, seed=seed + 9000)
    return base


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


def run_world(task) -> str:
    world, condition, seed, n20, shifts = task
    outpath = OUTDIR / f"{world}_{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {world} {condition} s{seed}"
    t0 = time.time()
    frame = _frame(world, condition, seed, n20)
    part = detect_blocks_swept(frame)
    rec = {"world": world, "condition": condition, "seed": seed,
           "partition": {k: part[k] for k in ("agents", "env", "orphans", "dial")}, "tests": {}}
    for test in TESTS:
        scorer = BlockScorer(frame, test, env_var=None, partition=part)
        scores = scorer.score_all()
        floors = null_floors(scorer, n_shifts=shifts, seed=seed * 97 + 13)
        rec["tests"][test] = {"scores": scores.reset_index().to_dict("records"),
                              "floors": floors.reset_index().to_dict("records"),
                              "lost_mass": scorer.lost_mass}
        del scorer
    if world == "puppetslow" and condition == "main":
        for lag in (2, 3):
            sc = BlockScorer(frame, "menu", env_var=None, partition=part, lag=lag)
            rec[f"menu_lag{lag}"] = sc.score_all().reset_index().to_dict("records")
            del sc
    outpath.write_text(json.dumps(_jsonable(rec)))
    return f"done {world} {condition} s{seed} (dial {part['dial']}) in {(time.time()-t0)/60:.1f}m"


def run_yardstick(task) -> str:
    world, seed = task
    outpath = OUTDIR / f"yardstick_{world}_seed{seed}.json"
    if outpath.exists():
        return f"skip yardstick {world} s{seed}"
    t0 = time.time()
    kw, n = {}, 2000
    if world == "puppetfast":
        kw = {"puppet": ("fast", 3)}
    elif world == "puppetslow":
        kw = {"puppet": ("slow", 3)}
        n = 6000
    else:
        kw = {"alias_noise": ALIAS[world]}
    cols = [f"{c}{i}" for i in range(8) for c in "BSAEGD"] + ["W"]
    if world.startswith("alias"):
        cols += ["G2_alias", "G5_alias"]
    rec = {"world": world, "seed": seed, "targets": {}}
    for target in cols:
        s = yardstick_scores(seed=seed, target=target, n_steps=n, n_perms=300, **kw)
        rec["targets"][target] = {"scores": s, "verdict": yardstick_verdict(s)}
    outpath.write_text(json.dumps(_jsonable(rec)))
    return f"done yardstick {world} s{seed} in {(time.time()-t0)/60:.1f}m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--n20", type=int, default=20_000)
    parser.add_argument("--shifts", type=int, default=200)
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in range(args.seeds):
        for world in ("puppetfast", "puppetslow"):
            for cond in ("main", "nocore", "scramble"):
                tasks.append(("w", (world, cond, seed, args.n20, args.shifts)))
        for world in ALIAS:
            tasks.append(("w", (world, "main", seed, args.n20, args.shifts)))
        for cond in ("nocore", "scramble"):
            tasks.append(("w", ("alias010", cond, seed, args.n20, args.shifts)))
        for world in ("puppetfast", "puppetslow", "alias000", "alias050"):
            tasks.append(("y", (world, seed)))
    fn = {"w": run_world, "y": run_yardstick}
    print(f"[v3] {len(tasks)} units, jobs={args.jobs}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        results = [pool.apply_async(fn[k], (p,)) for k, p in tasks]
        for i, r in enumerate(results, 1):
            print(f"[v3 {i}/{len(tasks)}] {r.get()} | {(time.time()-t0)/60:.0f}m", flush=True)
    print(f"V3 SWEEP COMPLETE in {(time.time()-t0)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
