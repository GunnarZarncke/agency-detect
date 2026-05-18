#!/usr/bin/env python3
"""
Four-step latent-candidate evaluation pipeline using strict UAD validation.

Pipeline:
1) Generate latent candidates from learned slots.
2) Map latent candidates back to raw variables via assignment matrix.
3) Run strict UAD validation on each candidate (agency_detect MarkovBlanketValidator).
4) Report candidate-stage metrics (Recall@K, post-UAD precision/recall).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agency_detect.config import DetectionConfig
from agency_detect.markov_blanket import MarkovBlanketValidator
from learn_agents.learn_agents import (
    ModelConfig,
    TraceSimulationConfig,
    TrainConfig,
    encode_trace,
    simulate_known_agent_trace,
    train_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate latent candidates with strict UAD.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-agents", type=int, default=8)
    p.add_argument("--copies-per-role", type=int, default=2)
    p.add_argument("--decoy-vars", type=int, default=0)
    p.add_argument("--T", type=int, default=6000)
    p.add_argument("--num-slots", type=int, default=20)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--slot-dim", type=int, default=16)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--top-k", nargs="+", type=int, default=[5, 10, 20, 30])
    p.add_argument("--max-pairs", type=int, default=40)
    p.add_argument("--assign-threshold", type=float, default=0.20)
    p.add_argument("--min-vars-per-candidate", type=int, default=4)
    p.add_argument("--discretization-bins", type=int, default=8)
    p.add_argument("--jaccard-hit-threshold", type=float, default=0.30)
    p.add_argument("--inspect-top-k", type=int, default=20)
    p.add_argument("--adapt-blankets", action="store_true", default=True)
    p.add_argument(
        "--no-adapt-blankets",
        action="store_false",
        dest="adapt_blankets",
        help="Skip epsilon/MDL local search on candidates.",
    )
    p.add_argument(
        "--adapt-objective",
        choices=["epsilon", "mdl"],
        default="mdl",
        help="Local search objective: violation only (epsilon) or violation + MDL penalty (mdl).",
    )
    p.add_argument("--mdl-lambda", type=float, default=0.15)
    p.add_argument("--adapt-top-n", type=int, default=30)
    p.add_argument("--search-steps", type=int, default=8)
    p.add_argument("--frontier-size", type=int, default=8)
    p.add_argument("--min-improvement", type=float, default=1e-3)
    p.add_argument("--uad-verbose", action="store_true", default=False)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def discretize_trace(trace: np.ndarray, bins: int) -> np.ndarray:
    """Discretize each variable independently into integer bins."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    T, N = trace.shape
    out = np.zeros((T, N), dtype=np.int64)
    quantiles = np.linspace(0, 1, bins + 1)
    for j in range(N):
        edges = np.quantile(trace[:, j], quantiles)
        # Ensure strict monotonicity to avoid empty/duplicate bin edges.
        edges = np.maximum.accumulate(edges)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        out[:, j] = np.clip(np.digitize(trace[:, j], edges[1:-1], right=False), 0, bins - 1)
    return out


def build_trace_dicts(trace_disc: np.ndarray, var_names: Sequence[str]) -> List[Dict[str, int]]:
    return [{name: int(trace_disc[t, i]) for i, name in enumerate(var_names)} for t in range(trace_disc.shape[0])]


def build_affinity_matrix(trace_disc: np.ndarray) -> np.ndarray:
    """Pairwise discrete MI matrix over raw variables."""
    n_vars = trace_disc.shape[1]
    aff = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars):
        aff[i, i] = 1.0
    for i in range(n_vars):
        xi = trace_disc[:, i]
        for j in range(i + 1, n_vars):
            m = mutual_info_score(xi, trace_disc[:, j])
            aff[i, j] = aff[j, i] = float(m)
    return aff


def generate_latent_candidates(latent: Dict[str, np.ndarray], max_pairs: int) -> List[Dict[str, Any]]:
    """
    Candidate generation from latent slots:
    - all single-slot candidates
    - top adjacency pair candidates
    """
    assign = latent["assign"]  # [T, K, N]
    adjacency = latent["adjacency"]  # [K, K]
    avg_assign = assign.mean(axis=0)  # [K, N]
    K = avg_assign.shape[0]

    candidates: List[Dict[str, Any]] = []
    for k in range(K):
        slot_mass = float(avg_assign[k].sum())
        slot_peak = float(avg_assign[k].max())
        candidates.append(
            {
                "slots": [int(k)],
                "proposal_score": slot_peak + 0.1 * slot_mass,
                "kind": "single",
            }
        )

    pair_scores = []
    for i in range(K):
        for j in range(i + 1, K):
            pair_score = float(adjacency[i, j] + adjacency[j, i])
            pair_scores.append((pair_score, i, j))
    pair_scores.sort(reverse=True)
    for s, i, j in pair_scores[:max_pairs]:
        mass = float(avg_assign[[i, j]].sum())
        peak = float(avg_assign[[i, j]].mean(axis=0).max())
        candidates.append(
            {
                "slots": [int(i), int(j)],
                "proposal_score": peak + 0.1 * mass + s,
                "kind": "pair",
            }
        )

    # Deduplicate by slot set.
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for c in candidates:
        key = tuple(sorted(c["slots"]))
        if key not in dedup or c["proposal_score"] > dedup[key]["proposal_score"]:
            dedup[key] = c
    return sorted(dedup.values(), key=lambda x: x["proposal_score"], reverse=True)


def map_candidate_to_raw_vars(
    candidate_slots: Sequence[int],
    avg_assign: np.ndarray,
    var_names: Sequence[str],
    threshold: float,
    min_vars: int,
) -> List[str]:
    var_score = avg_assign[list(candidate_slots)].mean(axis=0)
    chosen = np.where(var_score >= threshold)[0].tolist()
    if len(chosen) < min_vars:
        chosen = np.argsort(var_score)[-min_vars:].tolist()
    return [var_names[i] for i in sorted(set(chosen))]


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def evaluate_candidate_hits(
    candidates: List[Dict[str, Any]],
    agent_clusters: Dict[int, List[int]],
    var_to_idx: Dict[str, int],
    hit_thresh: float,
) -> None:
    for c in candidates:
        idxs = [var_to_idx[v] for v in c["raw_vars"] if v in var_to_idx]
        best_agent = -1
        best_j = 0.0
        for a, truth in agent_clusters.items():
            jj = jaccard(idxs, truth)
            if jj > best_j:
                best_j = jj
                best_agent = int(a)
        c["best_agent"] = best_agent
        c["best_jaccard"] = float(best_j)
        c["is_hit"] = bool(best_j >= hit_thresh)


def recall_at_k(candidates: List[Dict[str, Any]], agent_ids: Sequence[int], K: int) -> float:
    top = candidates[: min(K, len(candidates))]
    covered = set(c["best_agent"] for c in top if c["is_hit"] and c["best_agent"] >= 0)
    return len(covered) / max(len(agent_ids), 1)


def strict_precision_recall_at_k(candidates: List[Dict[str, Any]], agent_ids: Sequence[int], K: int) -> Dict[str, float]:
    top = candidates[: min(K, len(candidates))]
    strict = [c for c in top if c["strict_valid"] is True]
    if not strict:
        return {"precision": 0.0, "recall": 0.0, "strict_count": 0.0}
    tp = [c for c in strict if c["is_hit"]]
    precision = len(tp) / len(strict)
    covered = set(c["best_agent"] for c in tp if c["best_agent"] >= 0)
    recall = len(covered) / max(len(agent_ids), 1)
    return {"precision": float(precision), "recall": float(recall), "strict_count": float(len(strict))}


def adapt_candidate_set(
    initial_idxs: Sequence[int],
    all_var_names: Sequence[str],
    trace_disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    validator: MarkovBlanketValidator,
    affinity: np.ndarray,
    search_steps: int,
    frontier_size: int,
    min_improvement: float,
    uad_verbose: bool,
    objective: str = "epsilon",
    mdl_lambda: float = 0.15,
) -> Dict[str, Any]:
    """
    Local search over add/remove operations.
    objective='epsilon': minimize blanket violation only.
    objective='mdl': violation + lambda * log((N+1)/|C|) to discourage shrink-to-tiny sets.
    """
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
        obj = violation + (mdl_penalty(len(idx_set)) if objective == "mdl" else 0.0)
        rec = {
            "idx_set": set(idx_set),
            "raw_vars": raw_vars,
            "objective": obj,
            "strict_valid": blanket["valid"],
            "strict_violation": violation,
            "strict_details": blanket["details"],
            "classification": out["classification"],
        }
        cache[key] = rec
        return rec

    current = set(int(i) for i in initial_idxs)
    if not current:
        return {"final": None, "trajectory": []}
    cur_eval = evaluate_idx_set(current)
    best_eval = cur_eval
    trajectory: List[Dict[str, Any]] = [
        {
            "step": 0,
            "action": "init",
            "objective": cur_eval["objective"],
            "size": len(current),
            "strict_valid": cur_eval["strict_valid"],
        }
    ]

    for step in range(1, search_steps + 1):
        # Build frontier based on mean affinity to current set.
        cur_list = sorted(current)
        outside = sorted(all_idx_set - current)
        if cur_list:
            add_scores = []
            for v in outside:
                s = float(np.mean(affinity[v, cur_list]))
                add_scores.append((s, v))
            add_frontier = [v for _, v in sorted(add_scores, reverse=True)[:frontier_size]]

            rem_scores = []
            for v in cur_list:
                others = [u for u in cur_list if u != v]
                if others:
                    s = float(np.mean(affinity[v, others]))
                else:
                    s = 0.0
                rem_scores.append((s, v))
            # Weakly connected members first for removal.
            rem_frontier = [v for _, v in sorted(rem_scores)[:frontier_size]]
        else:
            add_frontier = outside[:frontier_size]
            rem_frontier = []

        candidates: List[tuple[str, int | None, Dict[str, Any]]] = []
        for v in add_frontier:
            new_set = set(current)
            new_set.add(v)
            candidates.append(("add", v, evaluate_idx_set(new_set)))
        if len(current) > 1:
            for v in rem_frontier:
                new_set = set(current)
                new_set.remove(v)
                if len(new_set) >= 1:
                    candidates.append(("remove", v, evaluate_idx_set(new_set)))

        if not candidates:
            break

        move_type, move_var, move_eval = min(candidates, key=lambda x: x[2]["objective"])
        improvement = cur_eval["objective"] - move_eval["objective"]
        if improvement <= min_improvement:
            trajectory.append(
                {
                    "step": step,
                    "action": "stop_no_improvement",
                    "objective": cur_eval["objective"],
                    "size": len(current),
                    "strict_valid": cur_eval["strict_valid"],
                }
            )
            break

        current = set(move_eval["idx_set"])
        cur_eval = move_eval
        if cur_eval["objective"] < best_eval["objective"]:
            best_eval = cur_eval
        trajectory.append(
            {
                "step": step,
                "action": move_type,
                "var_idx": int(move_var) if move_var is not None else None,
                "objective": cur_eval["objective"],
                "size": len(current),
                "strict_valid": cur_eval["strict_valid"],
                "improvement": float(improvement),
            }
        )

    return {"final": best_eval, "trajectory": trajectory}


def summarize_candidate_mapping(
    candidates: List[Dict[str, Any]],
    var_to_idx: Dict[str, int],
    var_agent: np.ndarray,
    inspect_top_k: int,
) -> Dict[str, Any]:
    top = candidates[: min(inspect_top_k, len(candidates))]
    if not top:
        return {"count": 0}

    sizes: List[int] = []
    majority_fracs: List[float] = []
    unique_agent_counts: List[int] = []
    decoy_fracs: List[float] = []
    pairwise_overlaps: List[float] = []

    idx_sets: List[set] = []
    for c in top:
        idxs = [var_to_idx[v] for v in c["raw_vars"] if v in var_to_idx]
        idx_set = set(idxs)
        idx_sets.append(idx_set)
        sizes.append(len(idx_set))
        if len(idx_set) == 0:
            majority_fracs.append(0.0)
            unique_agent_counts.append(0)
            decoy_fracs.append(0.0)
            c["map_majority_agent"] = -1
            c["map_majority_frac"] = 0.0
            c["map_unique_agents"] = 0
            c["map_decoy_frac"] = 0.0
            continue

        agents = np.array([int(var_agent[i]) for i in idx_set], dtype=int)
        non_decoy = agents[agents >= 0]
        decoy_frac = float(np.mean(agents < 0))
        decoy_fracs.append(decoy_frac)

        if len(non_decoy) == 0:
            maj_agent = -1
            maj_frac = 0.0
            uniq_agents = 0
        else:
            vals, counts = np.unique(non_decoy, return_counts=True)
            j = int(np.argmax(counts))
            maj_agent = int(vals[j])
            maj_frac = float(counts[j] / len(non_decoy))
            uniq_agents = int(len(vals))
        majority_fracs.append(maj_frac)
        unique_agent_counts.append(uniq_agents)
        c["map_majority_agent"] = maj_agent
        c["map_majority_frac"] = maj_frac
        c["map_unique_agents"] = uniq_agents
        c["map_decoy_frac"] = decoy_frac

    for i in range(len(idx_sets)):
        for j in range(i + 1, len(idx_sets)):
            a, b = idx_sets[i], idx_sets[j]
            pairwise_overlaps.append(jaccard(list(a), list(b)))

    return {
        "count": len(top),
        "size_mean": float(np.mean(sizes)),
        "size_median": float(np.median(sizes)),
        "majority_frac_mean": float(np.mean(majority_fracs)),
        "unique_agents_mean": float(np.mean(unique_agent_counts)),
        "decoy_frac_mean": float(np.mean(decoy_fracs)),
        "pairwise_overlap_mean": float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 0.0,
    }


def main() -> None:
    args = parse_args()

    sim_cfg = TraceSimulationConfig(
        seed=args.seed,
        T=args.T,
        num_agents=args.num_agents,
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

    model_cfg = ModelConfig(
        num_vars=trace.shape[1],
        window=args.window,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_agency_regularizer=False,
        device=args.device,
    )

    print("=== Step 1: Train latent model and generate candidates ===")
    model, history = train_model(trace, model_cfg, train_cfg)
    latent = encode_trace(model, trace)
    candidates = generate_latent_candidates(latent, max_pairs=args.max_pairs)
    avg_assign = latent["assign"].mean(axis=0)
    print(f"Generated {len(candidates)} latent candidates")

    print("=== Step 2: Map candidates to raw variables ===")
    for c in candidates:
        c["raw_vars"] = map_candidate_to_raw_vars(
            c["slots"], avg_assign, var_names, args.assign_threshold, args.min_vars_per_candidate
        )
        c["num_raw_vars"] = len(c["raw_vars"])

    print("=== Step 3: Strict UAD validation on candidates ===")
    disc = discretize_trace(trace, bins=args.discretization_bins)
    trace_dict = build_trace_dicts(disc, var_names)
    affinity = build_affinity_matrix(disc)
    detect_cfg = DetectionConfig()
    detect_cfg.VALIDATE_BLANKETS = True
    validator = MarkovBlanketValidator(detect_cfg)

    # Adapt top-N candidates via epsilon-driven local search.
    for rank, c in enumerate(candidates, start=1):
        initial_idxs = [var_to_idx[v] for v in c["raw_vars"] if v in var_to_idx]
        c["raw_vars_initial"] = list(c["raw_vars"])
        if args.adapt_blankets and rank <= args.adapt_top_n:
            out = adapt_candidate_set(
                initial_idxs=initial_idxs,
                all_var_names=var_names,
                trace_disc=disc,
                trace_dict=trace_dict,
                validator=validator,
                affinity=affinity,
                search_steps=args.search_steps,
                frontier_size=args.frontier_size,
                min_improvement=args.min_improvement,
                uad_verbose=args.uad_verbose,
                objective=args.adapt_objective,
                mdl_lambda=args.mdl_lambda,
            )
            final_eval = out["final"]
            if final_eval is not None:
                c["raw_vars"] = final_eval["raw_vars"]
                c["strict_valid"] = final_eval["strict_valid"]
                c["strict_violation"] = final_eval["strict_violation"]
                c["strict_details"] = final_eval["strict_details"]
                c["classification"] = final_eval["classification"]
                c["adapt_trajectory"] = out["trajectory"]
                c["adapted"] = True
            else:
                c["adapted"] = False
        else:
            if args.uad_verbose:
                result = validator.validate_cluster(c["raw_vars"], var_names, disc, trace_dict)
            else:
                with redirect_stdout(io.StringIO()):
                    result = validator.validate_cluster(c["raw_vars"], var_names, disc, trace_dict)
            blanket = result["blanket_validation"]
            c["strict_valid"] = blanket["valid"]
            c["strict_violation"] = float(blanket["violation"])
            c["strict_details"] = blanket["details"]
            c["classification"] = result["classification"]
            c["adapted"] = False

    # Rank by proposal score, then strict preference.
    candidates.sort(
        key=lambda x: (
            x["proposal_score"],
            1.0 if x["strict_valid"] is True else 0.0,
            -x["strict_violation"],
        ),
        reverse=True,
    )

    evaluate_candidate_hits(
        candidates,
        agent_clusters=agent_clusters,
        var_to_idx=var_to_idx,
        hit_thresh=args.jaccard_hit_threshold,
    )
    mapping_diag = summarize_candidate_mapping(
        candidates,
        var_to_idx=var_to_idx,
        var_agent=np.asarray(metadata["var_agent"], dtype=int),
        inspect_top_k=args.inspect_top_k,
    )

    print("=== Step 4: Candidate-stage metrics ===")
    top_k = sorted(set(args.top_k))
    metrics = {
        "seed": args.seed,
        "num_candidates": len(candidates),
        "train_final_loss": float(history["loss"][-1]),
        "pre_uad_recall_at_k": {},
        "post_uad_precision_recall_at_k": {},
    }
    for K in top_k:
        pre = recall_at_k(candidates, agent_ids, K)
        post = strict_precision_recall_at_k(candidates, agent_ids, K)
        metrics["pre_uad_recall_at_k"][str(K)] = pre
        metrics["post_uad_precision_recall_at_k"][str(K)] = post
        print(
            f"K={K:>3d} | pre_recall={pre:.3f} | "
            f"post_precision={post['precision']:.3f} | post_recall={post['recall']:.3f} | "
            f"strict_count={int(post['strict_count'])}"
        )
    print(
        "Mapping diagnostics (top "
        f"{mapping_diag.get('count', 0)}): "
        f"size_mean={mapping_diag.get('size_mean', 0.0):.2f}, "
        f"size_median={mapping_diag.get('size_median', 0.0):.2f}, "
        f"majority_frac_mean={mapping_diag.get('majority_frac_mean', 0.0):.3f}, "
        f"unique_agents_mean={mapping_diag.get('unique_agents_mean', 0.0):.3f}, "
        f"overlap_mean={mapping_diag.get('pairwise_overlap_mean', 0.0):.3f}"
    )

    top_candidates = candidates[: max(top_k)]
    report = {
        "config": {
            "sim_cfg": asdict(sim_cfg),
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "max_pairs": args.max_pairs,
            "adapt_blankets": args.adapt_blankets,
            "adapt_objective": args.adapt_objective,
            "mdl_lambda": args.mdl_lambda,
            "adapt_top_n": args.adapt_top_n,
            "search_steps": args.search_steps,
            "frontier_size": args.frontier_size,
            "min_improvement": args.min_improvement,
            "assign_threshold": args.assign_threshold,
            "min_vars_per_candidate": args.min_vars_per_candidate,
            "jaccard_hit_threshold": args.jaccard_hit_threshold,
            "discretization_bins": args.discretization_bins,
        },
        "metrics": metrics,
        "mapping_diagnostics": mapping_diag,
        "top_candidates": top_candidates,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
