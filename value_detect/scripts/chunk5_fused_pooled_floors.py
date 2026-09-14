#!/usr/bin/env python3
"""Chunk 5 — spec-compliance pass: fused-convention floors POOLED across seeds.

The locked criteria specify fused floors as ≥50 shifts per seed pooled across seeds
(≥1000 samples per score). The sweep stored only per-seed percentiles; this pass
regenerates the identical null samples (same RNG streams) and saves them raw, so the
aggregator can pool them as locked. Fused convention only; other conventions' per-seed
≥200-shift floors already comply.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import value_detect as vd
from value_detect.floors import ConventionScorer, shift_null_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "results" / "chunk5"


def run_unit(task) -> str:
    seed, condition, nfused, n_shifts = task
    outpath = OUTDIR / f"fusednulls_{condition}_seed{seed}.json"
    if outpath.exists():
        return f"skip {condition} seed {seed}"
    t0 = time.time()
    if condition == "main":
        ffu = vd.passive_trace(seed=seed, n_steps=nfused).frame
    elif condition == "nocore":
        ffu = vd.passive_trace_nocore(seed=seed, n_steps=nfused).frame
    else:
        ffu = vd.scramble_frame(vd.passive_trace(seed=seed, n_steps=nfused).frame, seed=seed + 9500)
    record = {"seed": seed, "condition": condition, "samples": {}}
    for config in ("asis", "noalias"):
        frame = ffu if config == "asis" else vd.drop_aliases(ffu)
        scorer = ConventionScorer(frame, "fused", env_var="E")
        samples = shift_null_samples(scorer, n_shifts=n_shifts, seed=seed * 97 + 13)
        record["samples"][config] = samples.to_dict("records")
        del scorer
    outpath.write_text(json.dumps(record))
    return f"done {condition} seed {seed} in {time.time()-t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--nfused", type=int, default=2_000_000)
    parser.add_argument("--shifts", type=int, default=50)
    args = parser.parse_args()

    tasks = [(seed, cond, args.nfused, args.shifts)
             for seed in range(args.seeds) for cond in ("main", "nocore", "scramble")]
    print(f"[pooled-floors] {len(tasks)} units, jobs={args.jobs}", flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_unit, tasks), 1):
            el = time.time() - t0
            print(f"[pooled-floors {i}/{len(tasks)}] {msg} | {el/60:.1f}m elapsed", flush=True)
    print(f"POOLED FLOOR SAMPLES COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
