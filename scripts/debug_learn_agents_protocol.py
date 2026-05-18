#!/usr/bin/env python3
"""
Six-stage debug protocol for learn_agents / latent-UAD pipeline.

Stages isolate failure modes:
  1) Oracle (ground-truth clusters) — environment / strict UAD feasibility
  2) Raw MI clustering baseline — separability without latent model
  3) Slot persistence — temporal stability of learned assignments
  4) Interface / role structure — sensor / internal / action separation in slots
  5) MDL-regularized adaptation — anti-shrink local search vs epsilon-only
  6) Strict UAD threshold sweep — validator sensitivity
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agency_detect.config import DetectionConfig
from agency_detect.detection import AgentDetector, build_similarity_matrix, filter_weak_connections
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from agency_detect.markov_blanket import MarkovBlanketValidator
from learn_agents.learn_agents import (
    ModelConfig,
    TraceSimulationConfig,
    TrainConfig,
    encode_trace,
    oracle_uad_scores,
    simulate_known_agent_trace,
    train_model,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from evaluate_latent_candidates_with_uad import (
    adapt_candidate_set,
    build_affinity_matrix,
    build_trace_dicts,
    discretize_trace,
    evaluate_candidate_hits,
    generate_latent_candidates,
    jaccard,
    map_candidate_to_raw_vars,
    recall_at_k,
    strict_precision_recall_at_k,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run learn_agents debug protocol (6 stages).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--copies-per-role", type=int, default=2)
    p.add_argument("--decoy-vars", type=int, default=2)
    p.add_argument("--T", type=int, default=4000)
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--slot-dim", type=int, default=16)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--mdl-lambda", type=float, default=0.15)
    p.add_argument("--adapt-top-n", type=int, default=8)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def default_slots(num_agents: int) -> int:
    return 9 if num_agents <= 3 else 20


def validate_clusters(
    clusters: List[List[str]],
    var_names: Sequence[str],
    disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    tolerance: float,
) -> List[Dict[str, Any]]:
    cfg = DetectionConfig()
    cfg.VALIDATE_BLANKETS = True
    cfg.BLANKET_TOLERANCE = tolerance
    validator = MarkovBlanketValidator(cfg)
    out_list: List[Dict[str, Any]] = []
    for raw_vars in clusters:
        with redirect_stdout(io.StringIO()):
            res = validator.validate_cluster(list(raw_vars), list(var_names), disc, trace_dict)
        blanket = res["blanket_validation"]
        out_list.append(
            {
                "n_vars": len(raw_vars),
                "strict_valid": blanket["valid"],
                "violation": float(blanket["violation"]),
                "details": blanket["details"],
            }
        )
    return out_list


def stage1_oracle(
    trace: np.ndarray,
    metadata: Dict[str, object],
    var_names: Sequence[str],
    disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    tolerance: float,
) -> Dict[str, Any]:
    oracle_scores = oracle_uad_scores(trace, metadata)
    agent_clusters = metadata["agent_clusters"]
    oracle_raw = [[var_names[i] for i in agent_clusters[k]] for k in sorted(agent_clusters)]
    strict = validate_clusters(oracle_raw, var_names, disc, trace_dict, tolerance)
    sep_ratios = [v["separation_ratio"] for v in oracle_scores.values()]
    return {
        "oracle_uad_scores": {str(k): v for k, v in oracle_scores.items()},
        "separation_ratio_mean": float(np.mean(sep_ratios)),
        "separation_ratio_max": float(np.max(sep_ratios)),
        "strict_on_ground_truth": strict,
        "strict_pass_rate": float(np.mean([s["strict_valid"] for s in strict if s["strict_valid"] is not None])),
        "mean_violation": float(np.mean([s["violation"] for s in strict])),
    }


def _score_clusters_against_truth(
    labeled_clusters: Dict[Any, List[str]],
    agent_clusters: Dict[int, List[int]],
    var_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    cluster_hits: List[Dict[str, Any]] = []
    for lbl, variables in labeled_clusters.items():
        if lbl == "env":
            continue
        idxs = [var_to_idx[v] for v in variables if v in var_to_idx]
        best_agent, best_j = -1, 0.0
        for a, truth in agent_clusters.items():
            jj = jaccard(idxs, truth)
            if jj > best_j:
                best_j, best_agent = jj, int(a)
        cluster_hits.append(
            {"cluster_label": str(lbl), "n_vars": len(idxs), "best_agent": best_agent, "best_jaccard": float(best_j)}
        )
    return cluster_hits


def stage2_raw_clustering(
    trace: np.ndarray,
    metadata: Dict[str, object],
    var_names: Sequence[str],
    disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    num_agents: int,
    tolerance: float,
) -> Dict[str, Any]:
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}
    var_to_idx = {v: i for i, v in enumerate(var_names)}

    # Pre-validation agglomerative clustering (isolates MI separability from classifier).
    vars_ = list(var_names)
    data = np.array([[rec[v] for v in vars_] for rec in trace_dict], dtype=np.float64)
    var_variance = data.var(axis=0)
    active_idx = np.where(var_variance > 0.0)[0]
    vars_active = [vars_[i] for i in active_idx]
    data_active = data[:, active_idx]
    sim, dist = build_similarity_matrix(data_active)
    labels = AgglomerativeClustering(
        n_clusters=num_agents, metric="precomputed", linkage="complete"
    ).fit_predict(dist)
    initial_clusters: Dict[int, List[str]] = defaultdict(list)
    for v, lbl in zip(vars_active, labels):
        initial_clusters[int(lbl)].append(v)
    filtered_clusters, _env = filter_weak_connections(dict(initial_clusters), vars_active, sim)
    initial_hits = _score_clusters_against_truth(filtered_clusters, agent_clusters, var_to_idx)
    covered_initial = {c["best_agent"] for c in initial_hits if c["best_jaccard"] >= 0.30 and c["best_agent"] >= 0}

    cfg = DetectionConfig()
    cfg.N_AGENTS = num_agents
    cfg.VALIDATE_BLANKETS = True
    cfg.BLANKET_TOLERANCE = tolerance
    detector = AgentDetector(cfg)
    with redirect_stdout(io.StringIO()):
        clusters = detector.detect_agents(trace_dict)

    cluster_hits: List[Dict[str, Any]] = []
    for lbl, info in clusters.items():
        if lbl == "env":
            continue
        idxs = [var_to_idx[v] for v in info["variables"] if v in var_to_idx]
        best_agent, best_j = -1, 0.0
        for a, truth in agent_clusters.items():
            jj = jaccard(idxs, truth)
            if jj > best_j:
                best_j, best_agent = jj, int(a)
        blanket = info.get("blanket_validation", {})
        cluster_hits.append(
            {
                "cluster_label": str(lbl),
                "n_vars": len(idxs),
                "best_agent": best_agent,
                "best_jaccard": float(best_j),
                "strict_valid": blanket.get("valid"),
                "violation": float(blanket.get("violation", 0.0)),
            }
        )

    covered = {c["best_agent"] for c in cluster_hits if c["best_jaccard"] >= 0.30 and c["best_agent"] >= 0}
    return {
        "n_detected_clusters": len(cluster_hits),
        "cluster_hits": cluster_hits,
        "agent_recall_jaccard03": len(covered) / max(len(agent_clusters), 1),
        "mean_best_jaccard": float(np.mean([c["best_jaccard"] for c in cluster_hits])) if cluster_hits else 0.0,
        "strict_pass_rate": float(
            np.mean([c["strict_valid"] for c in cluster_hits if c["strict_valid"] is not None])
        )
        if cluster_hits
        else 0.0,
        "initial_cluster_hits": initial_hits,
        "initial_recall_jaccard03": len(covered_initial) / max(len(agent_clusters), 1),
        "initial_mean_best_jaccard": float(np.mean([c["best_jaccard"] for c in initial_hits])) if initial_hits else 0.0,
    }


def _match_slots(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Hungarian match slots between two [K, N] assignment matrices; return perm and mean matched cosine."""
    ka, kb = a.shape[0], b.shape[0]
    cost = np.zeros((ka, kb), dtype=np.float64)
    for i in range(ka):
        ni = np.linalg.norm(a[i]) + 1e-9
        for j in range(kb):
            nj = np.linalg.norm(b[j]) + 1e-9
            cost[i, j] = 1.0 - float(np.dot(a[i], b[j]) / (ni * nj))
    row, col = linear_sum_assignment(cost)
    matched_cos = 1.0 - cost[row, col]
    return np.stack([row, col], axis=1), float(np.mean(matched_cos))


def stage3_persistence(
    model,
    trace: np.ndarray,
    n_chunks: int = 3,
) -> Dict[str, Any]:
    T = trace.shape[0]
    chunk_len = T // n_chunks
    chunk_assigns: List[np.ndarray] = []
    for c in range(n_chunks):
        start = c * chunk_len
        end = T if c == n_chunks - 1 else (c + 1) * chunk_len
        if end - start < model.cfg.window + 2:
            continue
        latent_c = encode_trace(model, trace[start:end])
        chunk_assigns.append(latent_c["assign"].mean(axis=0))

    if len(chunk_assigns) < 2:
        return {"error": "trace too short for chunk persistence"}

    cross_cosines: List[float] = []
    for i in range(len(chunk_assigns) - 1):
        _, mean_cos = _match_slots(chunk_assigns[i], chunk_assigns[i + 1])
        cross_cosines.append(mean_cos)

    full = encode_trace(model, trace)
    assign = full["assign"]  # [T_win, K, N]
    top1_stable: List[float] = []
    for k in range(assign.shape[1]):
        winners = np.argmax(assign[:, k, :], axis=1)
        if len(winners) < 2:
            continue
        stable = float(np.mean(winners[1:] == winners[:-1]))
        top1_stable.append(stable)

    return {
        "n_chunks": len(chunk_assigns),
        "cross_chunk_mean_cosine": float(np.mean(cross_cosines)),
        "cross_chunk_cosines": cross_cosines,
        "top1_winner_persistence_mean": float(np.mean(top1_stable)) if top1_stable else 0.0,
        "top1_winner_persistence_per_slot": top1_stable,
    }


def stage4_interface(
    latent: Dict[str, np.ndarray],
    metadata: Dict[str, object],
    assign_threshold: float = 0.15,
) -> Dict[str, Any]:
    avg_assign = latent["assign"].mean(axis=0)
    var_role = list(metadata["var_role"])
    var_agent = np.asarray(metadata["var_agent"], dtype=int)
    K = avg_assign.shape[0]

    slot_stats: List[Dict[str, Any]] = []
    for k in range(K):
        w = avg_assign[k]
        w = w / (w.sum() + 1e-9)
        role_counts: Dict[str, float] = {}
        agent_counts: Dict[int, float] = {}
        for j, wt in enumerate(w):
            if wt < 1e-6:
                continue
            r = str(var_role[j])
            role_counts[r] = role_counts.get(r, 0.0) + float(wt)
            ag = int(var_agent[j])
            agent_counts[ag] = agent_counts.get(ag, 0.0) + float(wt)

        if role_counts:
            roles_sorted = sorted(role_counts.items(), key=lambda x: -x[1])
            role_purity = roles_sorted[0][1]
            dominant_role = roles_sorted[0][0]
        else:
            role_purity, dominant_role = 0.0, "none"

        non_decoy = {a: v for a, v in agent_counts.items() if a >= 0}
        if non_decoy:
            ag_best = max(non_decoy.items(), key=lambda x: x[1])
            agent_purity = float(ag_best[1])
            dominant_agent = int(ag_best[0])
        else:
            agent_purity, dominant_agent = 0.0, -1

        high = np.where(w >= assign_threshold)[0]
        has_sia = all(
            any("sensor" in str(var_role[j]) for j in high)
            and any("internal" in str(var_role[j]) for j in high)
            and any("action" in str(var_role[j]) for j in high)
            for _ in [0]
        )
        slot_stats.append(
            {
                "slot": k,
                "mass": float(w.sum()),
                "dominant_role": dominant_role,
                "role_purity": float(role_purity),
                "dominant_agent": dominant_agent,
                "agent_purity": float(agent_purity),
                "has_sensor_internal_action": bool(has_sia),
                "n_vars_above_threshold": int(len(high)),
            }
        )

    active = [s for s in slot_stats if s["mass"] > 0.05]
    return {
        "n_slots": K,
        "n_active_slots": len(active),
        "role_purity_mean": float(np.mean([s["role_purity"] for s in active])) if active else 0.0,
        "agent_purity_mean": float(np.mean([s["agent_purity"] for s in active])) if active else 0.0,
        "sia_complete_slots": int(sum(s["has_sensor_internal_action"] for s in active)),
        "slot_stats": slot_stats,
    }


def adapt_candidate_set_mdl(
    initial_idxs: Sequence[int],
    all_var_names: Sequence[str],
    trace_disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    validator: MarkovBlanketValidator,
    affinity: np.ndarray,
    search_steps: int,
    frontier_size: int,
    min_improvement: float,
    mdl_lambda: float,
    uad_verbose: bool = False,
) -> Dict[str, Any]:
    """Local search with J = violation + lambda * log((N+1)/|C|)."""
    n_vars = len(all_var_names)
    all_idx_set = set(range(n_vars))
    cache: Dict[frozenset, Dict[str, Any]] = {}

    def mdl_penalty(size: int) -> float:
        return float(mdl_lambda * np.log((n_vars + 1) / max(size, 1)))

    def evaluate_idx_set(idx_set: set[int]) -> Dict[str, Any]:
        key = frozenset(idx_set)
        if key in cache:
            return cache[key]
        raw_vars = [all_var_names[i] for i in sorted(idx_set)]
        if uad_verbose:
            out = validator.validate_cluster(raw_vars, all_var_names, trace_disc, trace_dict)
        else:
            with redirect_stdout(io.StringIO()):
                out = validator.validate_cluster(raw_vars, all_var_names, trace_disc, trace_dict)
        blanket = out["blanket_validation"]
        violation = float(blanket["violation"])
        rec = {
            "idx_set": set(idx_set),
            "raw_vars": raw_vars,
            "objective": violation + mdl_penalty(len(idx_set)),
            "strict_valid": blanket["valid"],
            "strict_violation": violation,
            "mdl_penalty": mdl_penalty(len(idx_set)),
            "classification": out["classification"],
        }
        cache[key] = rec
        return rec

    current = set(int(i) for i in initial_idxs)
    if not current:
        return {"final": None, "trajectory": []}
    cur_eval = evaluate_idx_set(current)
    trajectory: List[Dict[str, Any]] = [
        {"step": 0, "action": "init", "objective": cur_eval["objective"], "size": len(current), "violation": cur_eval["strict_violation"]}
    ]

    for step in range(1, search_steps + 1):
        cur_list = sorted(current)
        outside = sorted(all_idx_set - current)
        add_scores = [(float(np.mean(affinity[v, cur_list])), v) for v in outside] if cur_list else [(0.0, v) for v in outside]
        add_frontier = [v for _, v in sorted(add_scores, reverse=True)[:frontier_size]]
        rem_frontier = []
        if len(current) > 1:
            rem_scores = []
            for v in cur_list:
                others = [u for u in cur_list if u != v]
                s = float(np.mean(affinity[v, others])) if others else 0.0
                rem_scores.append((s, v))
            rem_frontier = [v for _, v in sorted(rem_scores)[:frontier_size]]

        candidates: List[tuple] = []
        for v in add_frontier:
            ns = set(current)
            ns.add(v)
            candidates.append(("add", v, evaluate_idx_set(ns)))
        if len(current) > 1:
            for v in rem_frontier:
                ns = set(current)
                ns.remove(v)
                if ns:
                    candidates.append(("remove", v, evaluate_idx_set(ns)))

        if not candidates:
            break
        move_type, move_var, move_eval = min(candidates, key=lambda x: x[2]["objective"])
        improvement = cur_eval["objective"] - move_eval["objective"]
        if improvement <= min_improvement:
            trajectory.append({"step": step, "action": "stop", "size": len(current)})
            break
        current = set(move_eval["idx_set"])
        cur_eval = move_eval
        trajectory.append(
            {
                "step": step,
                "action": move_type,
                "size": len(current),
                "objective": cur_eval["objective"],
                "violation": cur_eval["strict_violation"],
            }
        )

    return {"final": cur_eval, "trajectory": trajectory}


def stage5_adapt_compare(
    candidates: List[Dict[str, Any]],
    var_names: Sequence[str],
    var_to_idx: Dict[str, int],
    disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    affinity: np.ndarray,
    validator: MarkovBlanketValidator,
    adapt_top_n: int,
    mdl_lambda: float,
) -> Dict[str, Any]:
    eps_sizes: List[int] = []
    mdl_sizes: List[int] = []
    for rank, c in enumerate(candidates[:adapt_top_n], start=1):
        initial = [var_to_idx[v] for v in c["raw_vars"] if v in var_to_idx]
        eps_out = adapt_candidate_set(
            initial_idxs=initial,
            all_var_names=var_names,
            trace_disc=disc,
            trace_dict=trace_dict,
            validator=validator,
            affinity=affinity,
            search_steps=8,
            frontier_size=8,
            min_improvement=1e-3,
            uad_verbose=False,
        )
        mdl_out = adapt_candidate_set_mdl(
            initial_idxs=initial,
            all_var_names=var_names,
            trace_disc=disc,
            trace_dict=trace_dict,
            validator=validator,
            affinity=affinity,
            search_steps=8,
            frontier_size=8,
            min_improvement=1e-3,
            mdl_lambda=mdl_lambda,
            uad_verbose=False,
        )
        if eps_out["final"]:
            eps_sizes.append(len(eps_out["final"]["idx_set"]))
        if mdl_out["final"]:
            mdl_sizes.append(len(mdl_out["final"]["idx_set"]))

    return {
        "adapt_top_n": adapt_top_n,
        "mdl_lambda": mdl_lambda,
        "epsilon_only_size_mean": float(np.mean(eps_sizes)) if eps_sizes else 0.0,
        "epsilon_only_size_median": float(np.median(eps_sizes)) if eps_sizes else 0.0,
        "mdl_size_mean": float(np.mean(mdl_sizes)) if mdl_sizes else 0.0,
        "mdl_size_median": float(np.median(mdl_sizes)) if mdl_sizes else 0.0,
    }


def stage6_threshold_sweep(
    cluster_sets: List[List[str]],
    var_names: Sequence[str],
    disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    thresholds: Sequence[float],
    label: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for tol in thresholds:
        vals = validate_clusters(cluster_sets, var_names, disc, trace_dict, tol)
        rows.append(
            {
                "tolerance": tol,
                "pass_rate": float(np.mean([v["strict_valid"] for v in vals])),
                "mean_violation": float(np.mean([v["violation"] for v in vals])),
                "per_cluster": vals,
            }
        )
    return {"label": label, "n_clusters": len(cluster_sets), "sweep": rows}


def run_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    num_agents = args.num_agents
    num_slots = args.num_slots or default_slots(num_agents)

    sim_cfg = TraceSimulationConfig(
        seed=args.seed,
        T=args.T,
        num_agents=num_agents,
        copies_per_role=args.copies_per_role,
        decoy_vars=args.decoy_vars,
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
    var_names = list(metadata["var_names"])
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}
    agent_ids = sorted(agent_clusters.keys())
    default_tol = DetectionConfig.BLANKET_TOLERANCE

    disc = discretize_trace(trace, bins=8)
    trace_dict = build_trace_dicts(disc, var_names)
    affinity = build_affinity_matrix(disc)

    print(f"\n{'='*60}\nDebug protocol: {num_agents} agents | seed={args.seed} | T={args.T} | slots={num_slots}\n{'='*60}")

    report: Dict[str, Any] = {"config": asdict(sim_cfg), "num_slots": num_slots, "epochs": args.epochs}

    print("\n[Stage 1] Oracle ground-truth clusters")
    s1 = stage1_oracle(trace, metadata, var_names, disc, trace_dict, default_tol)
    report["stage1_oracle"] = s1
    print(
        f"  sep_ratio_mean={s1['separation_ratio_mean']:.3f} "
        f"strict_pass={s1['strict_pass_rate']:.2f} mean_viol={s1['mean_violation']:.3f}"
    )

    print("\n[Stage 2] Raw MI clustering baseline")
    s2 = stage2_raw_clustering(trace, metadata, var_names, disc, trace_dict, num_agents, default_tol)
    report["stage2_raw_clustering"] = s2
    print(
        f"  post-val recall@j0.3={s2['agent_recall_jaccard03']:.3f} | "
        f"initial recall@j0.3={s2['initial_recall_jaccard03']:.3f} "
        f"initial_mean_j={s2['initial_mean_best_jaccard']:.3f}"
    )

    print("\n[Train] Latent model for stages 3–6")
    model_cfg = ModelConfig(num_vars=trace.shape[1], window=args.window, num_slots=num_slots, slot_dim=args.slot_dim)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, use_agency_regularizer=False, device=args.device)
    model, history = train_model(trace, model_cfg, train_cfg)
    latent = encode_trace(model, trace)
    report["train_final_loss"] = float(history["loss"][-1])

    print("\n[Stage 3] Slot persistence")
    s3 = stage3_persistence(model, trace)
    report["stage3_persistence"] = s3
    print(f"  cross_chunk_cosine={s3.get('cross_chunk_mean_cosine', 0):.3f} top1_persist={s3.get('top1_winner_persistence_mean', 0):.3f}")

    print("\n[Stage 4] Interface / role structure")
    s4 = stage4_interface(latent, metadata)
    report["stage4_interface"] = s4
    print(f"  agent_purity_mean={s4['agent_purity_mean']:.3f} sia_slots={s4['sia_complete_slots']}/{s4['n_active_slots']}")

    print("\n[Stage 5] Epsilon-only vs MDL adaptation (top-15 candidates)")
    candidates = generate_latent_candidates(latent, max_pairs=40)
    avg_assign = latent["assign"].mean(axis=0)
    for c in candidates:
        c["raw_vars"] = map_candidate_to_raw_vars(c["slots"], avg_assign, var_names, 0.20, 4)
    cfg = DetectionConfig()
    cfg.VALIDATE_BLANKETS = True
    validator = MarkovBlanketValidator(cfg)
    s5 = stage5_adapt_compare(
        candidates, var_names, var_to_idx, disc, trace_dict, affinity, validator, args.adapt_top_n, args.mdl_lambda
    )
    report["stage5_adapt"] = s5
    print(f"  eps_size_mean={s5['epsilon_only_size_mean']:.1f} mdl_size_mean={s5['mdl_size_mean']:.1f}")

    oracle_sets = [[var_names[i] for i in agent_clusters[a]] for a in agent_ids]
    learned_sets = [c["raw_vars"] for c in candidates[: min(8, len(candidates))]]

    print("\n[Stage 6] UAD threshold sweep")
    s6_oracle = stage6_threshold_sweep(oracle_sets, var_names, disc, trace_dict, args.thresholds, "oracle")
    s6_learned = stage6_threshold_sweep(learned_sets, var_names, disc, trace_dict, args.thresholds, "learned_top8")
    report["stage6_threshold_sweep"] = {"oracle": s6_oracle, "learned_top8": s6_learned}
    for row in s6_oracle["sweep"]:
        print(f"  oracle tol={row['tolerance']:.2f} pass={row['pass_rate']:.2f}")

    evaluate_candidate_hits(candidates, agent_clusters, var_to_idx, 0.30)
    report["latent_pre_uad_recall_at_30"] = recall_at_k(candidates, agent_ids, 30)

    return report


def interpret_report(report: Dict[str, Any]) -> str:
    """Heuristic failure-mode label from stage outcomes."""
    s1 = report["stage1_oracle"]
    s2 = report["stage2_raw_clustering"]
    s3 = report["stage3_persistence"]
    s4 = report["stage4_interface"]
    s5 = report["stage5_adapt"]

    modes: List[str] = []
    if s1["separation_ratio_mean"] > 0.35:
        modes.append("ENVIRONMENT: weak Gaussian epsilon-blanket separability")
    else:
        modes.append("ENVIRONMENT: Gaussian oracle separability OK")

    if s1["strict_pass_rate"] >= 0.5:
        modes.append("VALIDATOR: statistical role classifier passes oracle strict UAD")
    else:
        modes.append("VALIDATOR: statistical role classifier fails on ground truth")

    if s2["initial_recall_jaccard03"] < 0.5:
        modes.append("RAW_SEPARABILITY: MI clustering misses agents")
    else:
        modes.append("RAW_SEPARABILITY: MI clustering finds agents (pre-validation)")
    if s2["initial_recall_jaccard03"] >= 0.5 and s2["agent_recall_jaccard03"] < 0.5:
        modes.append("RAW_PIPELINE: post-validation discards all clusters")

    persist = s3.get("cross_chunk_mean_cosine", 0.0)
    if persist < 0.5:
        modes.append("REPRESENTATION: slots unstable across time chunks")
    else:
        modes.append("REPRESENTATION: moderate slot persistence")

    if s4["agent_purity_mean"] < 0.5:
        modes.append("MAPPING: slots mix multiple agents")
    else:
        modes.append("MAPPING: slots align with single agents")

    if s5["epsilon_only_size_mean"] < 4 and s5["mdl_size_mean"] > s5["epsilon_only_size_mean"] + 1:
        modes.append("SEARCH: epsilon-only shrink; MDL mitigates")
    elif s5["epsilon_only_size_mean"] < 4:
        modes.append("SEARCH: adaptation collapses to tiny sets")

    recall = report.get("latent_pre_uad_recall_at_30", 0.0)
    modes.append(f"LATENT_RECALL@30={recall:.3f}")

    return " | ".join(modes)


def main() -> None:
    args = parse_args()
    report = run_protocol(args)
    report["interpretation"] = interpret_report(report)
    print(f"\n=== Interpretation ===\n{report['interpretation']}\n")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
