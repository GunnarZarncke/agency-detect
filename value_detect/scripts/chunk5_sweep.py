#!/usr/bin/env python3
"""Chunk 5 — the pre-registered sweep (docs/SUCCESS_CRITERIA.md, LOCKED 2026-08-10).

Per seed (default 20):
  * MAIN world: six tests ({as-is, lookalike-free} × {pairwise, fused, fused+best-key}),
    scores + procedure-mirroring shift floors + V/U/T criteria evaluation;
  * NO-CORE control (C1): same six tests, signature must vanish;
  * SCRAMBLE control (C2): same six tests, floor-calibration + no signature at 99.5th;
  * `goal_progress` CONTRAST (C3): Gunnar's older simulator (1 solar + 1 steel),
    pairwise + fused+best-key, floors for the *_goal columns;
  * stability extras (descriptive): run lengths 2k/5k (+200k fused on the lookalike-free
    config), lags 2 and 3.

Run lengths: 20k (pairwise, fused+best-key); 2M (fused). Shifts: 200/200/50 (per
convention, as locked). Resumable: existing per-seed JSONs are skipped. Parallel via
--jobs. All artifacts under results/chunk5/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.criteria import (
    evaluate_goalprogress_test,
    evaluate_main_test,
    evaluate_nocore_test,
    evaluate_scramble_test,
)
from value_detect.floors import ConventionScorer, shift_null_floors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "chunk5"

CONVENTIONS = ["pairwise", "fused", "fused_bestkey"]
CONFIGS = ["asis", "noalias"]


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
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict("records")
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def _frames_for(condition: str, seed: int, n20: int, nfused: int):
    if condition == "main":
        f20 = vd.passive_trace(seed=seed, n_steps=n20).frame
        ffu = vd.passive_trace(seed=seed, n_steps=nfused).frame
    elif condition == "nocore":
        f20 = vd.passive_trace_nocore(seed=seed, n_steps=n20).frame
        ffu = vd.passive_trace_nocore(seed=seed, n_steps=nfused).frame
    elif condition == "scramble":
        f20 = vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=n20).frame, seed=seed + 9000)
        ffu = vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=nfused).frame, seed=seed + 9500)
    else:
        raise ValueError(condition)
    return f20, ffu


def run_seed_condition(task) -> str:
    (seed, condition, outdir, n20, nfused, shifts, do_stability) = task
    outpath = Path(outdir) / f"{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {condition} seed {seed} (exists)"
    t0 = time.time()
    f20_a, ffu_a = _frames_for(condition, seed, n20, nfused)

    record = {"seed": seed, "condition": condition, "n20": n20, "nfused": nfused, "tests": {}}
    for config in CONFIGS:
        f20 = f20_a if config == "asis" else vd.drop_aliases(f20_a)
        ffu = ffu_a if config == "asis" else vd.drop_aliases(ffu_a)
        for conv in CONVENTIONS:
            frame = ffu if conv == "fused" else f20
            scorer = ConventionScorer(frame, conv, env_var="E")
            scores = scorer.score_all()
            floors = shift_null_floors(scorer, n_shifts=shifts[conv], seed=seed * 97 + 13)
            if condition == "main":
                ev = evaluate_main_test(scores, floors, is_bestkey=(conv == "fused_bestkey"))
            elif condition == "nocore":
                ev = evaluate_nocore_test(scores, floors)
            else:
                ev = evaluate_scramble_test(scores, floors)
            record["tests"][f"{config}_{conv}"] = {
                "scores": scores, "floors": floors, "criteria": ev,
            }
            del scorer

    if do_stability and condition == "main":
        stab = {}
        for config in CONFIGS:
            base20 = f20_a if config == "asis" else vd.drop_aliases(f20_a)
            basefu = ffu_a if config == "asis" else vd.drop_aliases(ffu_a)
            for n_short in (2000, 5000):
                sub = base20.iloc[:n_short]
                for conv in ("pairwise", "fused_bestkey"):
                    stab[f"{config}_{conv}_n{n_short}"] = ConventionScorer(sub, conv, env_var="E").score_all()
            for lag in (2, 3):
                for conv in ("pairwise", "fused_bestkey"):
                    stab[f"{config}_{conv}_lag{lag}"] = ConventionScorer(base20, conv, env_var="E", lag=lag).score_all()
                stab[f"{config}_fused_lag{lag}"] = ConventionScorer(basefu, conv2 := "fused", env_var="E", lag=lag).score_all()
            if config == "noalias":
                stab["noalias_fused_n200k"] = ConventionScorer(basefu.iloc[:200_000], "fused", env_var="E").score_all()
        record["stability"] = stab

    outpath.write_text(json.dumps(_jsonable(record)))
    return f"done {condition} seed {seed} in {time.time()-t0:.0f}s"


def run_c3(task) -> str:
    (seed, outdir, steps, shifts_bestkey) = task
    outpath = Path(outdir) / f"c3_seed{seed}.json"
    if outpath.exists():
        return f"skip c3 seed {seed} (exists)"
    t0 = time.time()
    np.random.seed(seed)  # Gunnar's simulator uses global numpy randomness.
    from agency_detect.agents import generate_decoupled_trace

    trace = generate_decoupled_trace(steps=steps, n_solar_panels=1, factory_materials=["steel"])
    frame = pd.DataFrame(trace).astype(int)
    goal_cols = [c for c in frame.columns if c.endswith("_goal")]
    record = {"seed": seed, "steps": steps, "variables": list(frame.columns), "goal_cols": goal_cols, "tests": {}}
    for conv in ("pairwise", "fused_bestkey"):
        scorer = ConventionScorer(frame, conv, env_var=None)
        scores = scorer.score_all()
        floors = shift_null_floors(scorer, n_shifts=shifts_bestkey, seed=seed * 89 + 7, variables=goal_cols)
        ev = evaluate_goalprogress_test(scores, floors, goal_cols)
        record["tests"][conv] = {"scores": scores, "floors": floors, "criteria": ev}
    outpath.write_text(json.dumps(_jsonable(record)))
    return f"done c3 seed {seed} in {time.time()-t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n20", type=int, default=20_000)
    parser.add_argument("--nfused", type=int, default=2_000_000)
    parser.add_argument("--c3-steps", type=int, default=20_000)
    parser.add_argument("--shifts-pairwise", type=int, default=200)
    parser.add_argument("--shifts-bestkey", type=int, default=200)
    parser.add_argument("--shifts-fused", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="1 seed, small lengths and shift counts")
    args = parser.parse_args()

    if args.smoke:
        args.seeds, args.n20, args.nfused, args.c3_steps = 1, 4000, 100_000, 4000
        args.shifts_pairwise = args.shifts_bestkey = 20
        args.shifts_fused = 10

    outdir = args.outdir if not args.smoke else args.outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    shifts = {"pairwise": args.shifts_pairwise, "fused_bestkey": args.shifts_bestkey, "fused": args.shifts_fused}

    tasks = []
    for seed in range(args.seeds):
        for condition in ("main", "nocore", "scramble"):
            tasks.append(("world", (seed, condition, str(outdir), args.n20, args.nfused, shifts, not args.smoke)))
        tasks.append(("c3", (seed, str(outdir), args.c3_steps, args.shifts_bestkey)))

    total = len(tasks)
    print(f"[chunk5] {total} work units, jobs={args.jobs}, outdir={outdir}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        results = [
            pool.apply_async(run_seed_condition if kind == "world" else run_c3, (payload,))
            for kind, payload in tasks
        ]
        for r in results:
            msg = r.get()
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[chunk5 {done}/{total}] {msg} | elapsed {elapsed/60:.1f}m, eta {eta/60:.0f}m", flush=True)

    print(f"SWEEP COMPLETE in {(time.time()-t0)/60:.1f} minutes -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
