"""E9a serial spotlight peel loop."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Set

import numpy as np

from learn_agents.learn_agents import (
    ModelConfig,
    TraceSimulationConfig,
    TrainConfig,
    simulate_known_agent_trace,
    train_model,
)

from .config import SpotlightConfig
from .metrics import PassMetrics, best_agent_match, cumulative_agent_recall
from .proposal import rank_cluster_candidates
from .refine import refine_to_cluster
from .agency_gate import build_candidate_log, select_cluster_for_pass
from .diagnostics import compute_agent_diagnostics, mi_partition_agent_overlap
from .validation import build_candidate, validate_candidate_uad


def _sim_config(cfg: SpotlightConfig) -> TraceSimulationConfig:
    return TraceSimulationConfig(
        seed=cfg.seed,
        T=cfg.T,
        num_agents=cfg.num_agents,
        copies_per_role=cfg.copies_per_role,
        decoy_vars=cfg.decoy_vars,
        decoy_mode=cfg.decoy_mode,
        agent_variant_mode=cfg.agent_variant_mode,
        agent_variant_delay=cfg.agent_variant_delay,
        process_noise=cfg.process_noise,
        observation_noise=cfg.observation_noise,
        interaction_strength=cfg.interaction_strength,
        confound_strength=cfg.confound_strength,
        leakage_strength=cfg.leakage_strength,
        mixing_strength=cfg.mixing_strength,
        local_env_strength=cfg.local_env_strength,
        env_vars_per_agent=cfg.env_vars_per_agent,
        env_action_coupling=cfg.env_action_coupling,
        env_to_sensor_strength=cfg.env_to_sensor_strength,
        env_ar1_rho=cfg.env_ar1_rho,
        env_ar1_sigma=cfg.env_ar1_sigma,
        env_copies_per_var=cfg.env_copies_per_var,
        world_vars=cfg.world_vars,
        world_to_sensor_strength=cfg.world_to_sensor_strength,
        world_ar1_rho=cfg.world_ar1_rho,
        world_ar1_sigma=cfg.world_ar1_sigma,
        episodic=cfg.episodic,
        episode_len=cfg.episode_len,
        episode_gap=cfg.episode_gap,
        shared_period=cfg.shared_period,
        shared_periodic_strength=cfg.shared_periodic_strength,
        innovation_dist=cfg.innovation_dist,
        innovation_strength=cfg.innovation_strength,
        innovation_df=cfg.innovation_df,
    )


def run_spotlight_peel(cfg: SpotlightConfig) -> Dict[str, Any]:
    timing: Dict[str, float] = {
        "simulate_sec": 0.0,
        "proposal_sec": 0.0,
        "pretrain_sec": 0.0,
        "refine_sec": 0.0,
        "uad_sec": 0.0,
        "diagnostics_sec": 0.0,
        "peel_total_sec": 0.0,
    }
    peel_started = time.perf_counter()

    t0 = time.perf_counter()
    sim = simulate_known_agent_trace(_sim_config(cfg))
    timing["simulate_sec"] = time.perf_counter() - t0
    trace = sim.trace
    metadata = sim.metadata
    var_names = list(metadata["var_names"])
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}
    agent_ids = sorted(agent_clusters.keys())

    peeled: Set[int] = set()
    admitted_agents: Set[int] = set()
    passes: List[PassMetrics] = []
    pass_candidate_logs: List[List[Dict[str, Any]]] = []

    for pass_idx in range(cfg.max_passes):
        t0 = time.perf_counter()
        candidates, part = rank_cluster_candidates(trace, cfg, peeled, var_names=var_names)
        timing["proposal_sec"] += time.perf_counter() - t0
        pass_candidate_logs.append(
            build_candidate_log(candidates, agent_clusters, var_names, trace, cfg)
        )
        cluster, skipped_agency, agency_sig, peeled_agency_skip = select_cluster_for_pass(
            candidates, var_names, trace, cfg
        )
        if peeled_agency_skip and skipped_agency:
            for entry in skipped_agency:
                if entry["n_actions"] == 0 and entry["size"] >= 6:
                    peeled.update(entry["var_indices"])

        if cluster is None:
            if cfg.peel_on_agency_skip and skipped_agency:
                for entry in skipped_agency:
                    if entry["n_actions"] == 0 and entry["size"] >= 6:
                        peeled.update(entry["var_indices"])
                        peeled_agency_skip = True
            if peeled_agency_skip:
                passes.append(
                    PassMetrics(
                        pass_index=pass_idx,
                        proposal_mi_k=cfg.proposal_mi_k,
                        n_clusters_scored=len(candidates),
                        selected_cluster_id=skipped_agency[0]["cluster_id"] if skipped_agency else -1,
                        cluster_size=0,
                        cluster_var_indices=[],
                        cluster_score=0.0,
                        precursor_passed=False,
                        precursor_persistence=0.0,
                        precursor_contingency=0.0,
                        precursor_richness=0.0,
                        pretrain_final_loss=0.0,
                        refine_final_align=0.0,
                        candidate_var_count=0,
                        candidate_mode=cfg.candidate_mode,
                        peeled_var_count=len(peeled),
                        cumulative_recall=cumulative_agent_recall(admitted_agents, agent_ids),
                        stop_reason="agency_peel_only",
                        extra={"skipped_agency": skipped_agency},
                    )
                )
                if cfg.verbose:
                    print(
                        f"pass {pass_idx + 1}: peeled {len(skipped_agency)} non-agency clusters, continuing"
                    )
                continue
            stop_reason = "agency_skipped" if skipped_agency else "no_cluster"
            passes.append(
                PassMetrics(
                    pass_index=pass_idx,
                    proposal_mi_k=cfg.proposal_mi_k,
                    n_clusters_scored=len(candidates),
                    selected_cluster_id=skipped_agency[0]["cluster_id"] if skipped_agency else -1,
                    cluster_size=0,
                    cluster_var_indices=[],
                    cluster_score=0.0,
                    precursor_passed=False,
                    precursor_persistence=0.0,
                    precursor_contingency=0.0,
                    precursor_richness=0.0,
                    pretrain_final_loss=0.0,
                    refine_final_align=0.0,
                    candidate_var_count=0,
                    candidate_mode=cfg.candidate_mode,
                    peeled_var_count=len(peeled),
                    cumulative_recall=cumulative_agent_recall(admitted_agents, agent_ids),
                    stop_reason=stop_reason,
                    extra={"skipped_agency": skipped_agency} if skipped_agency else {},
                )
            )
            if cfg.verbose and skipped_agency:
                print(
                    f"pass {pass_idx + 1}: STOP all {len(skipped_agency)} candidates failed agency gate"
                )
            break

        if cfg.require_precursor_pass and not cluster.precursor_passed:
            if cfg.peel_on_precursor_skip:
                peeled.update(cluster.var_indices)
            passes.append(
                PassMetrics(
                    pass_index=pass_idx,
                    proposal_mi_k=cfg.proposal_mi_k,
                    n_clusters_scored=len(np.unique(part.labels[part.labels >= 0])),
                    selected_cluster_id=cluster.cluster_id,
                    cluster_size=len(cluster.var_indices),
                    cluster_var_indices=cluster.var_indices,
                    cluster_score=cluster.score,
                    precursor_passed=False,
                    precursor_persistence=cluster.persistence,
                    precursor_contingency=cluster.contingency,
                    precursor_richness=cluster.richness,
                    pretrain_final_loss=0.0,
                    refine_final_align=0.0,
                    candidate_var_count=0,
                    candidate_mode=cfg.candidate_mode,
                    peeled_var_count=len(peeled),
                    cumulative_recall=cumulative_agent_recall(admitted_agents, agent_ids),
                    stop_reason="precursor_skipped",
                )
            )
            if cfg.verbose:
                print(
                    f"pass {pass_idx + 1}: SKIP precursor fail cluster={cluster.cluster_id} "
                    f"vars={cluster.var_indices[:6]}... peeled={len(peeled)}"
                )
            if cfg.stop_if_precursor_fails:
                break
            continue

        model_cfg = ModelConfig(
            num_vars=trace.shape[1],
            window=cfg.window,
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
        )
        train_cfg = TrainConfig(
            epochs=cfg.pretrain_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.pretrain_lr,
            device=cfg.device,
            use_agency_regularizer=False,
        )
        t0 = time.perf_counter()
        model, history = train_model(trace, model_cfg, train_cfg)
        timing["pretrain_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        model, refine_meta = refine_to_cluster(model, trace, cluster.var_indices, cfg)
        timing["refine_sec"] += time.perf_counter() - t0

        candidate = build_candidate(model, trace, var_names, cluster.var_indices, cfg)
        var_indices = candidate["var_indices"]
        best_agent, best_j = best_agent_match(var_indices, agent_clusters)
        is_hit = best_j >= cfg.jaccard_hit_threshold

        uad_valid = None
        uad_violation = None
        uad_details = None
        if cfg.validate_uad and candidate["raw_vars"]:
            t0 = time.perf_counter()
            uad = validate_candidate_uad(candidate["raw_vars"], var_names, trace, cfg)
            timing["uad_sec"] += time.perf_counter() - t0
            uad_valid = uad["strict_valid"]
            uad_violation = uad["strict_violation"]
            uad_details = uad["strict_details"]

        admitted = True
        if cfg.require_precursor_pass and not cluster.precursor_passed:
            admitted = False
        if cfg.require_uad_pass and uad_valid is not True:
            admitted = False
        if cfg.require_jaccard_hit and not is_hit:
            admitted = False

        record_admitted = is_hit and best_agent >= 0
        if record_admitted:
            admitted_agents.add(best_agent)

        if cfg.peel_selected_always or record_admitted:
            peeled.update(cluster.var_indices)

        stop_reason = "admitted" if record_admitted else ("low_jaccard" if not is_hit else "not_admitted")

        refine_hist = refine_meta.get("history", {})
        align_hist = refine_hist.get("align", [0.0])

        pm = PassMetrics(
            pass_index=pass_idx,
            proposal_mi_k=cfg.proposal_mi_k,
            n_clusters_scored=len(np.unique(part.labels[part.labels >= 0])),
            selected_cluster_id=cluster.cluster_id,
            cluster_size=len(cluster.var_indices),
            cluster_var_indices=cluster.var_indices,
            cluster_score=cluster.score,
            precursor_passed=cluster.precursor_passed,
            precursor_persistence=cluster.persistence,
            precursor_contingency=cluster.contingency,
            precursor_richness=cluster.richness,
            pretrain_final_loss=float(history["loss"][-1]),
            refine_final_align=float(align_hist[-1]) if align_hist else 0.0,
            candidate_var_count=len(var_indices),
            candidate_mode=cfg.candidate_mode,
            uad_valid=uad_valid,
            uad_violation=uad_violation,
            best_agent_id=best_agent,
            best_jaccard=best_j,
            is_hit=is_hit,
            admitted=record_admitted,
            peeled_var_count=len(peeled),
            cumulative_recall=cumulative_agent_recall(admitted_agents, agent_ids),
            stop_reason=stop_reason,
            extra={
                "uad_details": uad_details,
                "refine_cluster_to_slot": refine_meta.get("cluster_to_slot"),
                "candidate_raw_vars": candidate["raw_vars"],
                "agency_signature": agency_sig,
                "skipped_agency": skipped_agency,
            },
        )
        passes.append(pm)

        if cfg.verbose:
            print(
                f"pass {pass_idx + 1}/{cfg.max_passes}: cluster={cluster.cluster_id} "
                f"size={len(cluster.var_indices)} score={cluster.score:.3f} "
                f"prec={'Y' if cluster.precursor_passed else 'N'} "
                f"J={best_j:.2f} agent={best_agent} uad={uad_valid} "
                f"cum_recall={pm.cumulative_recall:.2f}"
            )

    t0 = time.perf_counter()
    agent_diag = compute_agent_diagnostics(
        agent_clusters=agent_clusters,
        admitted_agents=admitted_agents,
        peeled=peeled,
        passes=passes,
        pass_candidate_logs=pass_candidate_logs,
        var_names=var_names,
        trace=trace,
        cfg=cfg,
    )
    mi_residual = mi_partition_agent_overlap(
        trace=trace,
        cfg=cfg,
        peeled=peeled,
        agent_clusters=agent_clusters,
        var_names=var_names,
    )
    timing["diagnostics_sec"] = time.perf_counter() - t0
    timing["peel_total_sec"] = time.perf_counter() - peel_started

    summary = {
        "cumulative_recall": cumulative_agent_recall(admitted_agents, agent_ids),
        "pass1_jaccard": passes[0].best_jaccard if passes else 0.0,
        "pass1_hit": passes[0].is_hit if passes else False,
        "n_passes": len(passes),
        "n_admitted": len(admitted_agents),
        "admitted_agent_ids": sorted(admitted_agents),
        "peeled_var_count": len(peeled),
        "missed_agent_ids": agent_diag["missed_agent_ids"],
        "agency_gate_mode": cfg.effective_agency_gate_mode(),
    }

    return {
        "config": cfg.to_dict(),
        "sim_metadata": {
            "num_vars": trace.shape[1],
            "var_names": var_names,
            "agent_clusters": agent_clusters,
            "decoy_fraction_actual": float(metadata.get("decoy_fraction", 0.0)),
        },
        "passes": [p.to_dict() for p in passes],
        "agent_diagnostics": agent_diag,
        "mi_residual": mi_residual,
        "timing": timing,
        "summary": summary,
    }
