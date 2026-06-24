#!/usr/bin/env python3
"""Fast first scaling benchmark for handle-aware UAD.

This module builds on `uad_handles.scaling` but avoids the expensive full
rescoring of every candidate after every intervention. It uses the same synthetic
world and passive UAD score, then treats each handle-operation as a targeted
piece of evidence that updates only candidates whose claimed role uses the tested
handle.

It is deliberately a first scaling scaffold, not a definitive benchmark.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from uad_handles import scaling as core


def score_plain(schema: core.VariableSchema, cands: List[core.Candidate], passive: core.DataSet) -> pd.DataFrame:
    return core.score_candidates(schema, cands, passive, use_interventions=False)


def operation_candidate_effect(c: core.Candidate, op_data: core.DataSet, op_name: str, handle: int) -> float:
    """Candidate-specific evidence from one tested handle operation."""
    b, s, a, g, e = c.as_tuple()
    got = core._combined_noop_and(op_data, op_name, handle)
    if got is None:
        return 0.0
    if op_name == "sensor_flip" and s == handle:
        X, Xn, active = got
        return 30.0 * (core.cmi(active, Xn[:, a], X[:, [e, g]]) - 0.020)
    if op_name == "action_block" and a == handle:
        X, Xn, active = got
        normal = active == 0
        blocked = active == 1
        if normal.sum() > 25 and blocked.sum() > 25:
            base_ctrl = core.cmi(Xn[normal, e], X[normal, a], X[normal, e])
            block_ctrl = core.cmi(Xn[blocked, e], X[blocked, a], X[blocked, e])
            return 4.0 * ((base_ctrl - block_ctrl) - 0.12)
    if op_name == "goal_flip" and g == handle:
        X, Xn, active = got
        return 4.0 * (core.cmi(active, X[:, a], X[:, [b, e]]) - 0.002)
    return 0.0


def choose_next(df: pd.DataFrame, tested: set[Tuple[str, int]]) -> Tuple[str, int]:
    # Attack the current highest-scoring explanation's claimed handles, action first.
    for _, row in df.head(30).iterrows():
        for role, op in [("a", "action_block"), ("s", "sensor_flip"), ("g", "goal_flip")]:
            h = int(row[role])
            if (op, h) not in tested:
                return op, h
    row = df.iloc[0]
    return "action_block", int(row["a"])


def apply_targeted_test(
    schema: core.VariableSchema,
    df: pd.DataFrame,
    cands: List[core.Candidate],
    op_data: core.DataSet,
    op_name: str,
    handle: int,
) -> pd.DataFrame:
    df = df.copy()
    if "active_bonus" not in df.columns:
        df["active_bonus"] = 0.0
    if op_name == "sensor_flip":
        relevant = df.index[df["s"] == handle].to_numpy()
    elif op_name == "action_block":
        relevant = df.index[df["a"] == handle].to_numpy()
    else:
        relevant = df.index[df["g"] == handle].to_numpy()
    # Need map from row to candidate. df preserves b/s/a/g/e, so reconstruct candidate.
    effects = []
    for ix in relevant:
        row = df.loc[ix]
        c = core.Candidate(int(row.b), int(row.s), int(row.a), int(row.g), int(row.e))
        effects.append(operation_candidate_effect(c, op_data, op_name, handle))
    if len(relevant):
        df.loc[relevant, "active_bonus"] += np.array(effects)
    df["score"] = df["passive"] + df["active_bonus"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def rank_of_true(df: pd.DataFrame) -> int:
    arr = np.where(df["is_true"].to_numpy())[0]
    return int(arr[0] + 1) if len(arr) else -1


def run_one(
    schema: core.VariableSchema,
    seed: int,
    passive_n: int,
    max_rounds: int,
    intervention_batch: int,
    candidate_cap: int,
    world_kwargs: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cands = core.enumerate_candidates(schema, cap=candidate_cap, seed=seed)
    passive = core.generate_passive(schema, seed=seed, n=passive_n, **world_kwargs)
    t0 = time.perf_counter()
    df = score_plain(schema, cands, passive)
    plain_time = time.perf_counter() - t0
    df["active_bonus"] = 0.0
    history = []
    tested: set[Tuple[str, int]] = set()
    for r in range(max_rounds + 1):
        top = df.iloc[0]
        history.append({
            "seed": seed,
            "n_alias": schema.n_alias,
            "n_distractor": schema.n_distractor,
            "n_random": schema.n_random,
            "n_variables": len(schema.names),
            "candidate_space_upper": core.total_candidate_space(schema),
            "candidate_sampled": len(df),
            "passive_n": passive_n,
            "round": r,
            "intervention_rows": r * intervention_batch,
            "true_rank": rank_of_true(df),
            "top_true": bool(top["is_true"]),
            "top_candidate": top["candidate"],
            "top_score": float(top["score"]),
            "true_score": float(df.loc[df["is_true"], "score"].iloc[0]),
            "plain_score_time_sec": plain_time if r == 0 else 0.0,
            "tested_handles": len(tested),
            "operation": "-",
            "handle": "-",
        })
        if r == max_rounds:
            break
        op, handle = choose_next(df, tested)
        tested.add((op, handle))
        op_data = core.generate_operation(schema, seed=seed + 10000 + 131 * r + 11 * handle, op_name=op, handle=handle, n=intervention_batch, **world_kwargs)
        df = apply_targeted_test(schema, df, cands, op_data, op, handle)
        history[-1]["operation"] = op
        history[-1]["handle"] = schema.names[handle]
    return pd.DataFrame(history), df.head(30).copy(), score_plain(schema, cands, passive).head(30).copy()


def run(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    alias_counts = [int(x) for x in args.alias_counts.split(",") if x]
    passive_ns = [int(x) for x in args.passive_ns.split(",") if x]
    world_kwargs = {
        "alias_noise": args.alias_noise,
        "handle_obs_noise": args.handle_obs_noise,
        "sensor_noise": args.sensor_noise,
        "action_noise": args.action_noise,
    }
    all_hist = []
    total_jobs = len(alias_counts) * len(passive_ns) * args.seeds
    job = 0
    for n_alias in alias_counts:
        schema = core.VariableSchema.make(n_alias=n_alias, n_distractor=max(args.min_distractors, n_alias), n_random=max(args.min_random, n_alias // 2))
        for passive_n in passive_ns:
            for seed in range(args.seeds):
                job += 1
                if args.verbose:
                    print(f"[{job}/{total_jobs}] aliases={n_alias} passive_n={passive_n} seed={seed}", flush=True)
                hist, active_top, passive_top = run_one(schema, seed, passive_n, args.max_rounds, args.intervention_batch, args.candidate_cap, world_kwargs)
                all_hist.append(hist)
                if seed == 0 and passive_n == passive_ns[0]:
                    passive_top.to_csv(outdir / f"top_passive_alias{n_alias}.csv", index=False)
                    active_top.to_csv(outdir / f"top_active_alias{n_alias}_round{args.max_rounds}.csv", index=False)
    history = pd.concat(all_hist, ignore_index=True)
    history.to_csv(outdir / "scaling_fast_history.csv", index=False)
    summary = history.groupby(["n_alias", "passive_n", "round"], as_index=False).agg(
        success=("top_true", "mean"),
        median_rank=("true_rank", "median"),
        mean_rank=("true_rank", "mean"),
        n_variables=("n_variables", "first"),
        candidate_space_upper=("candidate_space_upper", "first"),
        candidate_sampled=("candidate_sampled", "first"),
    )
    summary.to_csv(outdir / "scaling_fast_summary.csv", index=False)
    summary[summary["round"] == args.max_rounds].to_csv(outdir / "scaling_fast_final_round_summary.csv", index=False)

    if plt is not None and not args.no_plots:
        largest_passive = max(passive_ns)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for n_alias in alias_counts:
            sub = summary[(summary.n_alias == n_alias) & (summary.passive_n == largest_passive)]
            ax.plot(sub["round"], sub["success"], marker="o", label=f"aliases={n_alias}")
        ax.set_xlabel("Active handle tests")
        ax.set_ylabel("Exact true-loop recovery rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Recovery versus intervention budget (passive_n={largest_passive})")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(outdir / "fast_success_vs_rounds.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for passive_n in passive_ns:
            p0 = summary[(summary.passive_n == passive_n) & (summary.round == 0)]
            pf = summary[(summary.passive_n == passive_n) & (summary.round == args.max_rounds)]
            ax.plot(p0.n_alias, p0.success, linestyle="--", marker="o", label=f"plain T={passive_n}")
            ax.plot(pf.n_alias, pf.success, marker="o", label=f"active r={args.max_rounds}, T={passive_n}")
        ax.set_xlabel("Number of passive alias pairs")
        ax.set_ylabel("Exact recovery rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("False handles require handle tests, not just more passive samples")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(outdir / "fast_success_vs_aliases.png", dpi=180); plt.close(fig)

    readme = f"""# Fast handle-UAD scaling benchmark\n\nThis is the first scaling scaffold. It uses targeted handle evidence rather than\nfull posterior recomputation after every intervention.\n\nParameters:\n- alias counts: {alias_counts}\n- passive sample counts: {passive_ns}\n- seeds: {args.seeds}\n- max active tests: {args.max_rounds}\n- candidate cap: {args.candidate_cap}\n- intervention batch: {args.intervention_batch}\n\nFiles:\n- `scaling_fast_history.csv`: row per config/seed/round.\n- `scaling_fast_summary.csv`: aggregate success/rank by alias count, passive N, and round.\n- `scaling_fast_final_round_summary.csv`: final intervention budget snapshot.\n- `top_passive_alias*.csv`, `top_active_alias*.csv`: example top candidates.\n\nRound 0 is plain passive UAD. Rounds >0 add handle-operation tests.\n"""
    (outdir / "README.md").write_text(readme)
    zip_path = outdir.with_suffix(".zip")
    pkg = Path(__file__).resolve().parent
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pkg / "scaling_fast.py", arcname="uad_handles/scaling_fast.py")
        zf.write(pkg / "scaling.py", arcname="uad_handles/scaling.py")
        for p in outdir.iterdir():
            if p.is_file():
                zf.write(p, arcname=p.name)
    return history, summary


def main() -> None:
    p = argparse.ArgumentParser()
    from uad_handles import default_outdir

    p.add_argument("--outdir", type=Path, default=default_outdir("scaling_fast"))
    p.add_argument("--alias-counts", default="0,1,2,4,8")
    p.add_argument("--passive-ns", default="300,800")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--intervention-batch", type=int, default=80)
    p.add_argument("--candidate-cap", type=int, default=5000)
    p.add_argument("--min-distractors", type=int, default=2)
    p.add_argument("--min-random", type=int, default=2)
    p.add_argument("--alias-noise", type=float, default=0.0)
    p.add_argument("--handle-obs-noise", type=float, default=0.08)
    p.add_argument("--sensor-noise", type=float, default=0.05)
    p.add_argument("--action-noise", type=float, default=0.04)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    t = time.perf_counter()
    history, summary = run(args)
    print(f"Wrote outputs to {args.outdir} in {time.perf_counter()-t:.2f}s")
    print("\nRound 0 / passive:")
    print(summary[summary["round"] == 0].to_string(index=False))
    print(f"\nFinal round {args.max_rounds}:")
    print(summary[summary["round"] == args.max_rounds].to_string(index=False))


if __name__ == "__main__":
    main()
