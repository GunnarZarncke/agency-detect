#!/usr/bin/env python3
"""MI vs learned+UAD variable ARI for all agent families.

MI: measured live (simulate + mi_cluster_variable_labels), with per-trace timing.
Learned+UAD: cached spotlight JSON when available; ``--run-spotlight-missing`` runs live
peels for families without cache (saves under ``results/learn_agents/spotlight_runs/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

def _bootstrap_repo() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "repo_bootstrap.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError("agency-detect repo root not found")
    import repo_bootstrap

    return repo_bootstrap.install(here)

REPO_ROOT = _bootstrap_repo()

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score


from amortized_agency.benchmark import EVAL_T_STEPS, MI_REFERENCE_ARI  # noqa: E402
from learn_agents.grid_pomdp import GridPomdpConfig, roll_grid_pomdp  # noqa: E402
from learn_agents.learn_agents import (  # noqa: E402
    TraceSimulationConfig,
    mi_cluster_variable_labels,
    simulate_known_agent_trace,
)
from learn_agents.physics_pomdp import roll_cartpole_partial_obs  # noqa: E402
from learn_agents.rock_sample import RockSampleConfig, roll_rock_sample  # noqa: E402

WINDOW = 250
SEEDS_DEFAULT = [0, 1, 2]

E14_BASE = dict(
    num_agents=5,
    copies_per_role=3,
    decoy_vars=8,
    interaction_strength=0.05,
    agent_variant_mode="rich",
    episodic=False,
)
E14_EXTENSIONS: Dict[str, Dict] = {
    "telemetry_none": {},
    "telemetry_periodic": dict(shared_period=200, shared_periodic_strength=0.5),
    "telemetry_heavytail": dict(innovation_dist="t", innovation_df=3.0, innovation_strength=0.6),
    "telemetry_regime": dict(episodic=True, episode_len=120, episode_gap=60),
    "telemetry_saturate": dict(agent_variant_mode="complex"),
    "telemetry_all": dict(
        shared_period=200,
        shared_periodic_strength=0.5,
        innovation_dist="t",
        innovation_df=3.0,
        innovation_strength=0.6,
        episodic=True,
        episode_len=120,
        episode_gap=60,
        agent_variant_mode="complex",
    ),
}

# Cached spotlight runs (learned+UAD). Paths relative to REPO_ROOT.
SPOTLIGHT_CACHE: Dict[str, List[str]] = {
    "hard8_complex": sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "results/spotlight/e12b").glob("spotlight_complex_*.json")
    ),
    "med5_rich": [
        "results/spotlight/e11/spotlight_e11_rich_agents_cpr3_k24_p16.json",
    ],
}

SPOTLIGHT_RUNS_DIR = REPO_ROOT / "results/learn_agents/spotlight_runs"


def _mi_ari(trace: np.ndarray, var_agent: np.ndarray, num_agents: int, window: int) -> float:
    agent_cols = np.where(var_agent >= 0)[0]
    sub = trace[:window, agent_cols]
    true_ids = var_agent[agent_cols]
    labels = mi_cluster_variable_labels(sub, num_clusters=num_agents)
    active = labels >= 0
    if active.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(true_ids[active], labels[active]))


def _spotlight_ari(report: Dict[str, Any]) -> float:
    agent_clusters = {int(k): list(v) for k, v in report["sim_metadata"]["agent_clusters"].items()}
    n_vars = int(report["sim_metadata"]["num_vars"])
    true = np.full(n_vars, -1, dtype=np.int64)
    for aid, idxs in agent_clusters.items():
        for i in idxs:
            true[i] = aid
    pred = np.full(n_vars, -1, dtype=np.int64)
    for p in report["passes"]:
        if not p.get("admitted"):
            continue
        aid = int(p.get("best_agent_id", -1))
        if aid < 0:
            continue
        for vi in p.get("cluster_var_indices", []):
            pred[int(vi)] = aid
    mask = true >= 0
    active = pred[mask] >= 0
    if active.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(true[mask][active], pred[mask][active]))


def _spotlight_peel_seconds(report: Dict[str, Any]) -> Optional[float]:
    timing = report.get("timing") or {}
    if "peel_total_sec" in timing:
        return float(timing["peel_total_sec"])
    return None


def _load_spotlight_stats(paths: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not paths:
        return None
    aris: List[float] = []
    recalls: List[float] = []
    seconds: List[float] = []
    sources: List[str] = []
    for rel in paths:
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        report = json.loads(p.read_text())
        aris.append(_spotlight_ari(report))
        recalls.append(float(report["summary"]["cumulative_recall"]))
        t = _spotlight_peel_seconds(report)
        if t is not None:
            seconds.append(t)
        sources.append(rel)
    if not aris:
        return None
    out: Dict[str, Any] = {
        "learned_uad_ari_mean": float(np.mean(aris)),
        "learned_uad_ari_std": float(np.std(aris)),
        "learned_uad_recall_mean": float(np.mean(recalls)),
        "learned_uad_sources": sources,
        "learned_uad_note": f"cached spotlight ({len(aris)} run(s))",
    }
    if seconds:
        out["learned_uad_sec_mean"] = float(np.mean(seconds))
        out["learned_uad_sec_std"] = float(np.std(seconds))
    return out


def _telemetry_mi_from_cache(combo: str) -> Optional[float]:
    p = REPO_ROOT / "results/learn_agents/telemetry_extensions/telemetry_extension_detectability.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    key = "none" if combo == "telemetry_none" else combo.replace("telemetry_", "")
    for row in data.get("rows", []):
        if row.get("combo") == key and row.get("window") == WINDOW:
            return float(row["ari_mean"])
    return None


def _spotlight_cfg(overrides: Dict, seed: int, *, fast: bool) -> "SpotlightConfig":
    from dataclasses import replace

    from agent_spotlight.config import SpotlightConfig

    n = int(overrides["num_agents"])
    epoch_kw = dict(pretrain_epochs=25, refine_epochs=20) if fast else dict()
    sim_keys = {
        "num_agents",
        "copies_per_role",
        "decoy_vars",
        "agent_variant_mode",
        "interaction_strength",
        "episodic",
        "episode_len",
        "episode_gap",
        "shared_period",
        "shared_periodic_strength",
        "innovation_dist",
        "innovation_strength",
        "innovation_df",
    }
    spot_kw = {k: overrides[k] for k in sim_keys if k in overrides}
    return replace(
        SpotlightConfig(verbose=False, seed=seed, T=EVAL_T_STEPS, **epoch_kw),
        max_passes=n,
        proposal_mi_k=max(12, n * 2),
        **spot_kw,
    )


def _run_spotlight_live(
    name: str,
    overrides: Dict,
    seed: int,
    *,
    fast: bool,
) -> Dict[str, Any]:
    from agent_spotlight.peel import run_spotlight_peel

    t0 = time.perf_counter()
    cfg = _spotlight_cfg(overrides, seed, fast=fast)
    report = run_spotlight_peel(cfg)
    elapsed = time.perf_counter() - t0
    SPOTLIGHT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = SPOTLIGHT_RUNS_DIR / f"{name}_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "learned_uad_ari_mean": _spotlight_ari(report),
        "learned_uad_ari_std": 0.0,
        "learned_uad_recall_mean": float(report["summary"]["cumulative_recall"]),
        "learned_uad_sec_mean": float(report.get("timing", {}).get("peel_total_sec", elapsed)),
        "learned_uad_sources": [str(out.relative_to(REPO_ROOT))],
        "learned_uad_note": f"live spotlight seed={seed}",
    }


def _simulate_cfg(overrides: Dict, seed: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    t0 = time.perf_counter()
    cfg = TraceSimulationConfig(T=EVAL_T_STEPS, seed=seed, **overrides)
    result = simulate_known_agent_trace(cfg)
    sim_s = time.perf_counter() - t0
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    return result.trace, var_agent, cfg.num_agents, sim_s


def eval_sim_row(
    name: str,
    overrides: Dict,
    seeds: Sequence[int],
    *,
    spotlight_paths: Optional[List[str]] = None,
    mi_cached: Optional[float] = None,
    run_spotlight_missing: bool = False,
    spotlight_fast: bool = True,
) -> Dict[str, Any]:
    mi_aris: List[float] = []
    mi_seconds: List[float] = []
    for seed in seeds:
        trace, var_agent, num_agents, sim_s = _simulate_cfg(overrides, seed)
        w = min(WINDOW, trace.shape[0])
        t0 = time.perf_counter()
        mi_aris.append(_mi_ari(trace, var_agent, num_agents, w))
        mi_seconds.append(sim_s + (time.perf_counter() - t0))

    row: Dict[str, Any] = {
        "agent": name,
        "num_agents": int(overrides.get("num_agents", E14_BASE["num_agents"])),
        "window": WINDOW,
        "n_seeds": len(seeds),
        "mi_ari_mean": float(np.mean(mi_aris)),
        "mi_ari_std": float(np.std(mi_aris)),
        "mi_sec_mean": float(np.mean(mi_seconds)),
        "mi_sec_std": float(np.std(mi_seconds)),
    }
    if mi_cached is not None:
        row["mi_ari_cached"] = mi_cached
    frozen = MI_REFERENCE_ARI.get((name, WINDOW))
    if frozen is not None:
        row["mi_ari_frozen_e13"] = frozen

    cache = _load_spotlight_stats(spotlight_paths or [])
    if cache and not run_spotlight_missing:
        row.update(cache)
    elif run_spotlight_missing and not (spotlight_paths or []):
        lu_aris: List[float] = []
        lu_secs: List[float] = []
        recalls: List[float] = []
        sources: List[str] = []
        for seed in seeds:
            live = _run_spotlight_live(name, overrides, seed, fast=spotlight_fast)
            lu_aris.append(live["learned_uad_ari_mean"])
            lu_secs.append(live["learned_uad_sec_mean"])
            recalls.append(live["learned_uad_recall_mean"])
            sources.extend(live["learned_uad_sources"])
        row.update(
            {
                "learned_uad_ari_mean": float(np.mean(lu_aris)),
                "learned_uad_ari_std": float(np.std(lu_aris)),
                "learned_uad_recall_mean": float(np.mean(recalls)),
                "learned_uad_sec_mean": float(np.mean(lu_secs)),
                "learned_uad_sec_std": float(np.std(lu_secs)),
                "learned_uad_sources": sources,
                "learned_uad_note": f"live spotlight ({len(seeds)} seeds)",
            }
        )
    elif cache:
        row.update(cache)
    else:
        row["learned_uad_ari_mean"] = None
        row["learned_uad_sec_mean"] = None
        row["learned_uad_note"] = "no cached spotlight run"
    return row


def eval_external_row(name: str, builder: Callable[[int], Any], seeds: Sequence[int]) -> Dict[str, Any]:
    mi_aris: List[float] = []
    mi_seconds: List[float] = []
    num_agents = 0
    for seed in seeds:
        t0 = time.perf_counter()
        result = builder(seed)
        build_s = time.perf_counter() - t0
        trace = result.trace
        var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
        num_agents = int(result.metadata["config"].num_agents)
        w = min(WINDOW, trace.shape[0])
        t1 = time.perf_counter()
        mi_aris.append(_mi_ari(trace, var_agent, num_agents, w))
        mi_seconds.append(build_s + (time.perf_counter() - t1))
    return {
        "agent": name,
        "num_agents": num_agents,
        "window": WINDOW,
        "n_seeds": len(seeds),
        "mi_ari_mean": float(np.mean(mi_aris)),
        "mi_ari_std": float(np.std(mi_aris)),
        "mi_sec_mean": float(np.mean(mi_seconds)),
        "mi_sec_std": float(np.std(mi_seconds)),
        "learned_uad_ari_mean": None,
        "learned_uad_sec_mean": None,
        "learned_uad_note": "spotlight not wired for external traces",
    }


def run_all(
    seeds: Sequence[int],
    *,
    run_spotlight_missing: bool = False,
    spotlight_fast: bool = True,
) -> List[Dict]:
    rows: List[Dict] = []
    pool = {
        "easy3_redundant": dict(
            num_agents=3,
            agent_variant_mode="redundant",
            decoy_vars=6,
            copies_per_role=3,
            interaction_strength=0.05,
            episodic=False,
        ),
        "med5_rich": dict(
            num_agents=5,
            agent_variant_mode="rich",
            decoy_vars=8,
            copies_per_role=3,
            interaction_strength=0.05,
            episodic=False,
        ),
        "hard8_complex": dict(
            num_agents=8,
            agent_variant_mode="complex",
            decoy_vars=8,
            copies_per_role=3,
            interaction_strength=0.05,
            episodic=False,
        ),
    }
    spotlight_map = {
        "easy3_redundant": [],
        "med5_rich": SPOTLIGHT_CACHE.get("med5_rich", []),
        "hard8_complex": SPOTLIGHT_CACHE.get("hard8_complex", []),
    }
    for name, ov in pool.items():
        print(f"=== {name} ===", flush=True)
        paths = spotlight_map.get(name, [])
        if run_spotlight_missing and not paths:
            print("  (running live spotlight)", flush=True)
        rows.append(
            eval_sim_row(
                name,
                ov,
                seeds,
                spotlight_paths=paths,
                run_spotlight_missing=run_spotlight_missing,
                spotlight_fast=spotlight_fast,
            )
        )
        r = rows[-1]
        lu = r.get("learned_uad_ari_mean")
        lu_s = "—" if lu is None else f"{lu:.3f}"
        print(
            f"  MI ARI={r['mi_ari_mean']:.3f}  ({r['mi_sec_mean']:.2f}s/trace)  LU={lu_s}",
            flush=True,
        )

    for name in E14_EXTENSIONS:
        print(f"=== {name} ===", flush=True)
        ov = {**E14_BASE, **E14_EXTENSIONS[name]}
        cached_mi = _telemetry_mi_from_cache(name)
        rows.append(
            eval_sim_row(
                name,
                ov,
                seeds,
                mi_cached=cached_mi,
                run_spotlight_missing=run_spotlight_missing,
                spotlight_fast=spotlight_fast,
            )
        )
        r = rows[-1]
        print(f"  MI ARI={r['mi_ari_mean']:.3f}  ({r['mi_sec_mean']:.2f}s/trace)", flush=True)

    print("=== external POMDPs ===", flush=True)
    rows.append(eval_external_row("physics_cartpole", lambda s: roll_cartpole_partial_obs(seed=s), seeds))
    rows.append(eval_external_row("rock_sample_5x5", lambda s: roll_rock_sample(RockSampleConfig(seed=s)), seeds))
    rows.append(
        eval_external_row(
            "grid_pomdp_3x3",
            lambda s: roll_grid_pomdp(GridPomdpConfig(grid=3, view=3, num_agents=2, max_steps=250, seed=s)),
            seeds,
        )
    )
    rows.append(
        eval_external_row(
            "grid_pomdp_5x5",
            lambda s: roll_grid_pomdp(GridPomdpConfig(grid=5, view=3, num_agents=2, max_steps=250, seed=s)),
            seeds,
        )
    )
    return rows


def _fmt_ari(mean: Optional[float], std: Optional[float] = None) -> str:
    if mean is None:
        return "—"
    if std is not None and std > 0:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def _fmt_sec(mean: Optional[float]) -> str:
    if mean is None:
        return "—"
    if mean < 1.0:
        return f"{mean * 1000:.0f} ms"
    if mean < 120:
        return f"{mean:.1f} s"
    return f"{mean / 60:.1f} min"


def markdown_table(rows: List[Dict]) -> str:
    lines = [
        "| Agent family | n | MI ARI @W=250 | MI time / trace | Learned+UAD ARI | LU time | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lu_ari = _fmt_ari(r.get("learned_uad_ari_mean"), r.get("learned_uad_ari_std"))
        note = r.get("learned_uad_note", "")
        if r.get("learned_uad_recall_mean") is not None and r.get("learned_uad_ari_mean") is not None:
            note = f"recall={r['learned_uad_recall_mean']:.2f}; {note}"
        if r.get("mi_ari_cached") is not None:
            note = f"MI cache W=250={r['mi_ari_cached']:.3f}; {note}"
        if r["agent"] == "med5_rich" and r.get("learned_uad_ari_mean") is not None:
            note = "LU proxy: rich 8-agent E11 run; " + note
        lines.append(
            f"| {r['agent']} | {r['num_agents']} | "
            f"{_fmt_ari(r['mi_ari_mean'], r['mi_ari_std'])} | "
            f"{_fmt_sec(r.get('mi_sec_mean'))} | "
            f"{lu_ari} | "
            f"{_fmt_sec(r.get('learned_uad_sec_mean'))} | "
            f"{note} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/learn_agents/agent_ari_table.json",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out in place (default: archive existing, write timestamped file)",
    )
    p.add_argument(
        "--run-spotlight-missing",
        action="store_true",
        help="Run live spotlight for families without cached JSON (slow)",
    )
    p.add_argument(
        "--no-spotlight-fast",
        action="store_true",
        help="Use full spotlight train epochs (50/40) when running live",
    )
    args = p.parse_args()
    rows = run_all(
        args.seeds,
        run_spotlight_missing=args.run_spotlight_missing,
        spotlight_fast=not args.no_spotlight_fast,
    )
    md = markdown_table(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": WINDOW,
        "eval_t_steps": EVAL_T_STEPS,
        "seeds": list(args.seeds),
        "mi_protocol": "mi_cluster_variable_labels on agent cols, trace[:W]; timed per seed",
        "learned_uad_protocol": (
            "cached + live spotlight for missing families"
            if args.run_spotlight_missing
            else "cached agent_spotlight peels only"
        ),
        "markdown_table": md,
        "rows": rows,
    }
    from learn_agents.safe_results import write_json

    written = write_json(payload, args.out, force=args.force)
    print(f"\nWrote {written}\n")
    print(md)


if __name__ == "__main__":
    main()
