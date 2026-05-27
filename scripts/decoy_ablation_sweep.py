#!/usr/bin/env python3
"""
Decoy type and intensity ablation with variable-K MI (no assumed agent count).

Reports MI partition recall (fixed-K vs MDL search), latent baseline, and MI-refine
candidate Recall@K for anchor agent counts (default 3 and 8).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from learn_agents.learn_agents import (
    ModelConfig,
    RefineConfig,
    TraceSimulationConfig,
    TrainConfig,
    factorize_background,
    mi_partition_search,
    refine_model_with_mi,
    simulate_known_agent_trace,
    train_model,
)
from learn_agents_agent_count_sweep import (
    agent_var_count,
    decoy_count_from_fraction,
    evaluate_candidates,
    mi_partition_recall,
    slots_for_agents,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decoy type/intensity ablation sweep.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--T", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--refine-epochs", type=int, default=15)
    p.add_argument("--agent-counts", nargs="+", type=int, default=[3, 8])
    p.add_argument("--decoy-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.5])
    p.add_argument("--decoy-modes", nargs="+", default=["noise", "confound", "ar1", "mixed"])
    p.add_argument("--confound-weights", nargs="+", type=float, default=[0.2, 0.5, 1.0, 2.0])
    p.add_argument("--ar1-rhos", nargs="+", type=float, default=[0.0, 0.7, 0.93, 0.99])
    p.add_argument("--intensity-fraction", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--skip-intensity", action="store_true")
    p.add_argument("--output-json", type=str, default="results/decoy_ablation_sweep.json")
    return p.parse_args()


def run_condition(
    num_agents: int,
    decoy_fraction: float,
    decoy_mode: str,
    decoy_ar1_rho: float,
    decoy_confound_weight: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    decoy_vars = decoy_count_from_fraction(num_agents, 2, decoy_fraction)
    sim_cfg = TraceSimulationConfig(
        seed=args.seed,
        T=args.T,
        num_agents=num_agents,
        copies_per_role=2,
        decoy_vars=decoy_vars,
        decoy_mode=decoy_mode,
        decoy_ar1_rho=decoy_ar1_rho,
        decoy_confound_weight=decoy_confound_weight,
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
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}

    mi_fixed = mi_partition_search(trace, fixed_k=num_agents)
    mi_var = mi_partition_search(trace, fixed_k=None, k_selection="mdl")
    mi_trace, _bg = factorize_background(trace, n_components=1)
    mi_down = mi_partition_search(
        mi_trace,
        fixed_k=None,
        k_selection="downstream",
        avg_assign=None,
    )
    mi_fixed_stats = mi_partition_recall(mi_fixed.labels, agent_clusters)
    mi_var_stats = mi_partition_recall(mi_var.labels, agent_clusters)
    mi_down_stats = mi_partition_recall(mi_down.labels, agent_clusters)

    n_slots = slots_for_agents(num_agents)
    model_cfg = ModelConfig(num_vars=trace.shape[1], window=16, num_slots=n_slots, slot_dim=16)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=128, use_agency_regularizer=False)

    print(
        f"  agents={num_agents} mode={decoy_mode} frac={decoy_fraction:.0%} "
        f"decoys={decoy_vars} MI_K={mi_var.best_k} Kds={mi_down.best_k} "
        f"fixed_R={mi_fixed_stats['recall']:.2f} var_R={mi_var_stats['recall']:.2f} "
        f"ds_R={mi_down_stats['recall']:.2f}"
    )

    model, _ = train_model(trace, model_cfg, train_cfg)
    baseline = evaluate_candidates(model, trace, metadata, args.top_k)

    def refine_with(
        fixed_k: Optional[int],
        *,
        k_selection: str = "downstream",
        background_factorize: bool = True,
    ) -> Dict[str, Any]:
        m = copy.deepcopy(model)
        rcfg = RefineConfig(
            epochs=args.refine_epochs,
            lambda_align=2.0,
            mi_fixed_k=fixed_k,
            mi_k_selection=k_selection,
            mi_background_factorize=background_factorize,
        )
        with redirect_stdout(io.StringIO()):
            m, meta = refine_model_with_mi(m, trace, refine_cfg=rcfg)
        ev = evaluate_candidates(m, trace, metadata, args.top_k)
        ev["mi_best_k"] = meta.get("mi_best_k")
        return ev

    refine_fixed = refine_with(num_agents, k_selection="mdl", background_factorize=False)
    refine_var = refine_with(None, k_selection="mdl", background_factorize=False)
    refine_down = refine_with(None, k_selection="downstream", background_factorize=True)

    return {
        "num_agents": num_agents,
        "decoy_fraction_target": decoy_fraction,
        "decoy_fraction_actual": decoy_vars / max(trace.shape[1], 1),
        "decoy_vars": decoy_vars,
        "decoy_mode": decoy_mode,
        "decoy_ar1_rho": decoy_ar1_rho,
        "decoy_confound_weight": decoy_confound_weight,
        "mi_fixed_k": num_agents,
        "mi_fixed": {**mi_fixed_stats, "best_k": mi_fixed.best_k},
        "mi_variable": {**mi_var_stats, "best_k": mi_var.best_k, "k_scores": mi_var.k_scores},
        "mi_downstream": {
            **mi_down_stats,
            "best_k": mi_down.best_k,
            "k_scores": mi_down.k_scores,
            "k_downstream_scores": mi_down.k_downstream_scores,
        },
        "baseline": baseline,
        "refine_fixed_k": refine_fixed,
        "refine_variable_k": refine_var,
        "refine_downstream_k": refine_down,
    }


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []

    print("=== Phase 1: decoy type ablation ===")
    for mode in args.decoy_modes:
        for num_agents in args.agent_counts:
            for frac in args.decoy_fractions:
                rows.append(
                    run_condition(
                        num_agents, frac, mode,
                        decoy_ar1_rho=0.93,
                        decoy_confound_weight=1.0,
                        args=args,
                    )
                )

    if not args.skip_intensity:
        print("\n=== Phase 2: confound weight intensity (20% decoys) ===")
        for num_agents in args.agent_counts:
            for w in args.confound_weights:
                rows.append(
                    run_condition(
                        num_agents, args.intensity_fraction, "confound",
                        decoy_ar1_rho=0.93,
                        decoy_confound_weight=w,
                        args=args,
                    )
                )

        print("\n=== Phase 3: AR(1) rho intensity (20% decoys) ===")
        for num_agents in args.agent_counts:
            for rho in args.ar1_rhos:
                rows.append(
                    run_condition(
                        num_agents, args.intensity_fraction, "ar1",
                        decoy_ar1_rho=rho,
                        decoy_confound_weight=1.0,
                        args=args,
                    )
                )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "rows": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(
        f"{'ag':>3} {'mode':>8} {'frac':>5} {'K*':>3} {'Kds':>3} {'MIfx':>5} {'MIvr':>5} {'MIds':>5} "
        f"{'base':>5} {'rFix':>5} {'rVar':>5} {'rDs':>5} {'dec%':>5}"
    )
    for r in rows:
        print(
            f"{r['num_agents']:>3} {r['decoy_mode']:>8} {r['decoy_fraction_target']:>5.0%} "
            f"{r['mi_variable']['best_k']:>3.0f} "
            f"{r['mi_downstream']['best_k']:>3.0f} "
            f"{r['mi_fixed']['recall']:>5.2f} {r['mi_variable']['recall']:>5.2f} "
            f"{r['mi_downstream']['recall']:>5.2f} "
            f"{r['baseline']['pre_recall_at_k']:>5.2f} "
            f"{r['refine_fixed_k']['pre_recall_at_k']:>5.2f} "
            f"{r['refine_variable_k']['pre_recall_at_k']:>5.2f} "
            f"{r['refine_downstream_k']['pre_recall_at_k']:>5.2f} "
            f"{100 * r['baseline']['mapping'].get('decoy_frac_mean', 0):>4.0f}%"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
