#!/usr/bin/env python3
"""
Sweep num_agents from 1..N to locate breaking points for:
  - raw MI partition recall (coarse initializer quality)
  - latent baseline candidate Recall@K
  - latent + MI-refine candidate Recall@K

Uses one fixed hyperparameter regime (clean, low-noise simulator).
Each agent count: train once, evaluate baseline slots, then MI-refine and re-evaluate.
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
import copy
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(REPO_ROOT / "learn_agents" / "scripts"))

from agency_detect.markov_blanket import MarkovBlanketValidator
from agency_detect.config import DetectionConfig
from learn_agents.learn_agents import (
    ModelConfig,
    RefineConfig,
    TraceSimulationConfig,
    TrainConfig,
    encode_trace,
    mi_cluster_variable_labels,
    mi_partition_search,
    refine_model_with_mi,
    simulate_known_agent_trace,
    train_model,
)
from evaluate_latent_candidates_with_uad import (
    build_trace_dicts,
    discretize_trace,
    evaluate_candidate_hits,
    generate_latent_candidates,
    jaccard,
    map_candidate_to_raw_vars,
    recall_at_k,
    strict_precision_recall_at_k,
    summarize_candidate_mapping,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep agent counts 1..N for latent/MI breaking points.")
    p.add_argument("--min-agents", type=int, default=1)
    p.add_argument("--max-agents", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--T", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--refine-epochs", type=int, default=25)
    p.add_argument("--copies-per-role", type=int, default=2)
    p.add_argument(
        "--decoy-vars",
        type=int,
        default=None,
        help="Fixed decoy count (ignored if --decoy-fraction is set).",
    )
    p.add_argument(
        "--decoy-fraction",
        type=float,
        default=None,
        help="Fraction of total variables that are decoys: decoy/(agent+decoy).",
    )
    p.add_argument("--lambda-align", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--slot-dim", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-json", type=str, default="results/learn_agents/agent_count/agent_count_sweep.json")
    return p.parse_args()


def slots_for_agents(num_agents: int) -> int:
    return max(4, num_agents * 3)


def agent_var_count(num_agents: int, copies_per_role: int) -> int:
    return num_agents * 3 * copies_per_role


def decoy_count_from_fraction(num_agents: int, copies_per_role: int, fraction: float) -> int:
    """decoy_vars such that decoy / (agent_vars + decoy) ≈ fraction."""
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        raise ValueError("decoy-fraction must be in [0, 1)")
    agent_vars = agent_var_count(num_agents, copies_per_role)
    if agent_vars == 0:
        return 0
    decoy = int(round(fraction * agent_vars / (1.0 - fraction)))
    if fraction > 0 and decoy < 1:
        decoy = 1
    return decoy


def resolve_decoy_vars(num_agents: int, args: argparse.Namespace) -> int:
    if args.decoy_fraction is not None:
        return decoy_count_from_fraction(num_agents, args.copies_per_role, args.decoy_fraction)
    return int(args.decoy_vars or 0)


def mi_partition_recall(
    labels: np.ndarray,
    agent_clusters: Dict[int, List[int]],
    hit_thresh: float = 0.30,
) -> Dict[str, float]:
    """How well MI variable labels cover true agent clusters (agent-level recall)."""
    agent_ids = sorted(agent_clusters.keys())
    if not agent_ids:
        return {"recall": 0.0, "mean_best_jaccard": 0.0, "n_mi_clusters": 0.0}

    valid_labels = sorted(set(int(x) for x in labels if x >= 0))
    if not valid_labels:
        return {"recall": 0.0, "mean_best_jaccard": 0.0, "n_mi_clusters": 0.0}

    best_jaccards: List[float] = []
    covered: set[int] = set()
    for lbl in valid_labels:
        idxs = np.where(labels == lbl)[0].tolist()
        best_agent, best_j = -1, 0.0
        for a, truth in agent_clusters.items():
            jj = jaccard(idxs, truth)
            if jj > best_j:
                best_j, best_agent = jj, int(a)
        best_jaccards.append(best_j)
        if best_j >= hit_thresh and best_agent >= 0:
            covered.add(best_agent)

    return {
        "recall": len(covered) / len(agent_ids),
        "mean_best_jaccard": float(np.mean(best_jaccards)),
        "n_mi_clusters": float(len(valid_labels)),
    }


def evaluate_candidates(
    model,
    trace: np.ndarray,
    metadata: Dict[str, Any],
    top_k: int,
    max_pairs: int = 40,
) -> Dict[str, Any]:
    var_names = list(metadata["var_names"])
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}
    agent_ids = sorted(agent_clusters.keys())

    latent = encode_trace(model, trace)
    candidates = generate_latent_candidates(latent, max_pairs=max_pairs)
    avg_assign = latent["assign"].mean(axis=0)
    for c in candidates:
        c["raw_vars"] = map_candidate_to_raw_vars(c["slots"], avg_assign, var_names, 0.20, 4)

    disc = discretize_trace(trace, bins=8)
    trace_dict = build_trace_dicts(disc, var_names)
    validator = MarkovBlanketValidator(DetectionConfig())

    for c in candidates:
        with redirect_stdout(io.StringIO()):
            result = validator.validate_cluster(c["raw_vars"], var_names, disc, trace_dict)
        blanket = result["blanket_validation"]
        c["strict_valid"] = blanket["valid"]
        c["strict_violation"] = float(blanket["violation"])

    candidates.sort(
        key=lambda x: (x["proposal_score"], 1.0 if x["strict_valid"] is True else 0.0, -x["strict_violation"]),
        reverse=True,
    )
    evaluate_candidate_hits(candidates, agent_clusters, var_to_idx, 0.30)
    mapping = summarize_candidate_mapping(
        candidates, var_to_idx, np.asarray(metadata["var_agent"], dtype=int), inspect_top_k=min(20, top_k)
    )
    post = strict_precision_recall_at_k(candidates, agent_ids, top_k)
    return {
        "pre_recall_at_k": recall_at_k(candidates, agent_ids, top_k),
        "post_recall_at_k": post["recall"],
        "post_precision_at_k": post["precision"],
        "strict_count_at_k": post["strict_count"],
        "mapping": mapping,
    }


def run_one(num_agents: int, args: argparse.Namespace) -> Dict[str, Any]:
    decoy_vars = resolve_decoy_vars(num_agents, args)
    agent_vars = agent_var_count(num_agents, args.copies_per_role)
    sim_cfg = TraceSimulationConfig(
        seed=args.seed,
        T=args.T,
        num_agents=num_agents,
        copies_per_role=args.copies_per_role,
        decoy_vars=decoy_vars,
        process_noise=0.02,
        observation_noise=0.01,
        redundancy_noise=0.0,
        interaction_strength=0.45,
        confound_strength=0.0,
        leakage_strength=0.0,
        mixing_strength=0.0,
        episodic=False,
    )
    sim = simulate_known_agent_trace(sim_cfg)
    trace = sim.trace
    metadata = sim.metadata
    n_slots = slots_for_agents(num_agents)
    n_vars = trace.shape[1]

    mi_part = mi_partition_search(trace, fixed_k=None)
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}
    mi_stats = mi_partition_recall(mi_part.labels, agent_clusters)
    mi_stats["best_k"] = mi_part.best_k

    model_cfg = ModelConfig(
        num_vars=n_vars,
        window=args.window,
        num_slots=n_slots,
        slot_dim=args.slot_dim,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_agency_regularizer=False,
        device=args.device,
    )

    print(
        f"\n--- num_agents={num_agents} | agent_vars={agent_vars} decoys={decoy_vars} "
        f"({decoy_vars / max(n_vars, 1):.0%} of {n_vars}) | slots={n_slots} | MI recall={mi_stats['recall']:.2f} ---"
    )
    model, train_hist = train_model(trace, model_cfg, train_cfg)
    baseline = evaluate_candidates(model, trace, metadata, args.top_k)

    model_ref = copy.deepcopy(model)
    refine_cfg = RefineConfig(
        epochs=args.refine_epochs,
        lambda_align=args.lambda_align,
        device=args.device,
        mi_fixed_k=None,
    )
    with redirect_stdout(io.StringIO()):
        model_ref, refine_meta = refine_model_with_mi(model_ref, trace, refine_cfg=refine_cfg)
    refined = evaluate_candidates(model_ref, trace, metadata, args.top_k)

    row = {
        "num_agents": num_agents,
        "num_vars": n_vars,
        "agent_vars": agent_vars,
        "decoy_vars": decoy_vars,
        "decoy_fraction_actual": float(decoy_vars / max(n_vars, 1)),
        "num_slots": n_slots,
        "vars_per_agent": int(3 * args.copies_per_role),
        "mi_partition": mi_stats,
        "train_final_loss": float(train_hist["loss"][-1]),
        "baseline": baseline,
        "mi_refine": refined,
        "refine_final_align": float(refine_meta["history"]["align"][-1]),
    }
    print(
        f"  baseline R@{args.top_k}={baseline['pre_recall_at_k']:.2f} "
        f"post={baseline['post_recall_at_k']:.2f}/{baseline['post_precision_at_k']:.2f} | "
        f"refine R@{args.top_k}={refined['pre_recall_at_k']:.2f} "
        f"post={refined['post_recall_at_k']:.2f}/{refined['post_precision_at_k']:.2f}"
    )
    return row


def summarize_breaking_points(rows: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    def first_drop(key_path: str, thresh: float) -> int | None:
        for r in rows:
            val = r
            for part in key_path.split("."):
                val = val[part]
            if val < thresh:
                return int(r["num_agents"])
        return None

    return {
        "mi_recall_below_1.0": first_drop("mi_partition.recall", 0.999),
        "baseline_pre_recall_below_0.5": first_drop("baseline.pre_recall_at_k", 0.5),
        "refine_pre_recall_below_0.5": first_drop("mi_refine.pre_recall_at_k", 0.5),
        "refine_beats_baseline_first": next(
            (int(r["num_agents"]) for r in rows if r["mi_refine"]["pre_recall_at_k"] > r["baseline"]["pre_recall_at_k"] + 0.05),
            None,
        ),
        "refine_worse_than_mi_first": next(
            (
                int(r["num_agents"])
                for r in rows
                if r["mi_partition"]["recall"] > 0.5 and r["mi_refine"]["pre_recall_at_k"] < r["mi_partition"]["recall"] - 0.1
            ),
            None,
        ),
        "top_k": top_k,
    }


def print_table(rows: List[Dict[str, Any]], top_k: int) -> None:
    hdr = (
        f"{'K':>3} {'vars':>4} {'dec':>3} {'d%':>4} {'slots':>5} | {'MI_R':>5} {'MI_J':>5} | "
        f"{'B_pre':>5} {'B_post':>6} | {'R_pre':>5} {'R_post':>6} {'R_prec':>6} | {'maj':>5}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        mi = r["mi_partition"]
        b = r["baseline"]
        rf = r["mi_refine"]
        print(
            f"{r['num_agents']:>3} {r['num_vars']:>4} {r['decoy_vars']:>3} "
            f"{100 * r['decoy_fraction_actual']:>3.0f}% {r['num_slots']:>5} | "
            f"{mi['recall']:>5.2f} {mi['mean_best_jaccard']:>5.2f} | "
            f"{b['pre_recall_at_k']:>5.2f} {b['post_recall_at_k']:>6.2f} | "
            f"{rf['pre_recall_at_k']:>5.2f} {rf['post_recall_at_k']:>6.2f} {rf['post_precision_at_k']:>6.2f} | "
            f"{rf['mapping'].get('majority_frac_mean', 0):>5.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.decoy_fraction is None and args.decoy_vars is None:
        args.decoy_vars = 0
    rows: List[Dict[str, Any]] = []
    for k in range(args.min_agents, args.max_agents + 1):
        rows.append(run_one(k, args))

    breaking = summarize_breaking_points(rows, args.top_k)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "rows": rows,
        "breaking_points": breaking,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_table(rows, args.top_k)
    print("\nBreaking points:", json.dumps(breaking, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
