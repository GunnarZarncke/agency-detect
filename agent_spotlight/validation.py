"""Candidate extraction and strict UAD validation for one spotlight pass."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import mutual_info_score

from agency_detect.config import DetectionConfig
from agency_detect.markov_blanket import MarkovBlanketValidator, classify_variables
from learn_agents.learn_agents import encode_trace

from .config import SpotlightConfig


def agency_signature(
    raw_vars: List[str],
    var_names: Sequence[str],
    trace: np.ndarray,
    cfg: SpotlightConfig,
) -> Dict[str, Any]:
    """Data-only S/A/I role check (no var-name heuristics)."""
    disc = discretize_trace(trace, bins=cfg.discretization_bins)
    with redirect_stdout(io.StringIO()):
        classification = classify_variables(raw_vars, list(var_names), disc, trace)
    n_s = len(classification["S"])
    n_a = len(classification["A"])
    n_i = len(classification["I"])
    passed = n_s >= 1 and n_a >= 1 and n_i >= 1
    return {
        "passed": passed,
        "n_sensors": n_s,
        "n_actions": n_a,
        "n_internals": n_i,
        "classification": classification,
    }


def agency_role_penalty(sig: Dict[str, Any], cfg: SpotlightConfig) -> float:
    penalty = 0.0
    if sig["n_actions"] == 0:
        penalty += cfg.agency_penalty_no_actions
    if sig["n_sensors"] == 0:
        penalty += cfg.agency_penalty_no_sensors
    if sig["n_internals"] == 0:
        penalty += cfg.agency_penalty_no_internals
    return penalty * cfg.agency_score_penalty_weight


def agency_gate_passes(sig: Dict[str, Any], mode: str) -> bool:
    if mode in ("off", "soft", "score_penalty"):
        return True
    if mode == "strict":
        return bool(sig["passed"])
    if mode == "actions_only":
        return sig["n_actions"] >= 1
    return True


def agency_signature_for_indices(
    var_indices: Sequence[int],
    var_names: Sequence[str],
    trace: np.ndarray,
    cfg: SpotlightConfig,
) -> Dict[str, Any]:
    raw_vars = [var_names[int(i)] for i in var_indices]
    return agency_signature(raw_vars, var_names, trace, cfg)


def discretize_trace(trace: np.ndarray, bins: int) -> np.ndarray:
    t_len, n_vars = trace.shape
    out = np.zeros((t_len, n_vars), dtype=np.int64)
    quantiles = np.linspace(0, 1, bins + 1)
    for j in range(n_vars):
        edges = np.quantile(trace[:, j], quantiles)
        edges = np.maximum.accumulate(edges)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        out[:, j] = np.clip(np.digitize(trace[:, j], edges[1:-1], right=False), 0, bins - 1)
    return out


def build_trace_dicts(trace_disc: np.ndarray, var_names: Sequence[str]) -> List[Dict[str, int]]:
    return [
        {name: int(trace_disc[t, i]) for i, name in enumerate(var_names)}
        for t in range(trace_disc.shape[0])
    ]


def map_slot_to_raw_vars(
    avg_assign: np.ndarray,
    slot_index: int,
    var_names: Sequence[str],
    threshold: float,
    min_vars: int,
) -> List[str]:
    var_score = avg_assign[int(slot_index)]
    chosen = np.where(var_score >= threshold)[0].tolist()
    if len(chosen) < min_vars:
        chosen = np.argsort(var_score)[-min_vars:].tolist()
    return [var_names[i] for i in sorted(set(chosen))]


def mi_cluster_to_raw_vars(
    cluster_var_indices: List[int],
    var_names: Sequence[str],
) -> List[str]:
    return [var_names[i] for i in sorted(cluster_var_indices)]


def build_candidate(
    model,
    trace: np.ndarray,
    var_names: Sequence[str],
    cluster_var_indices: List[int],
    cfg: SpotlightConfig,
) -> Dict[str, Any]:
    if cfg.candidate_mode == "mi_cluster":
        raw_vars = mi_cluster_to_raw_vars(cluster_var_indices, var_names)
        return {
            "mode": "mi_cluster",
            "raw_vars": raw_vars,
            "var_indices": list(cluster_var_indices),
            "slot_index": None,
        }

    latent = encode_trace(model, trace)
    avg_assign = latent["assign"].mean(axis=0)
    raw_vars = map_slot_to_raw_vars(
        avg_assign,
        cfg.spotlight_slot_index,
        var_names,
        cfg.assign_threshold,
        cfg.min_vars_per_candidate,
    )
    var_indices = [var_names.index(v) for v in raw_vars if v in var_names]
    return {
        "mode": "spotlight_slot",
        "raw_vars": raw_vars,
        "var_indices": var_indices,
        "slot_index": cfg.spotlight_slot_index,
        "slot_peak_assign": float(avg_assign[cfg.spotlight_slot_index].max()),
    }


def validate_candidate_uad(
    raw_vars: List[str],
    var_names: Sequence[str],
    trace: np.ndarray,
    cfg: SpotlightConfig,
) -> Dict[str, Any]:
    disc = discretize_trace(trace, bins=cfg.discretization_bins)
    trace_dict = build_trace_dicts(disc, var_names)
    detect_cfg = DetectionConfig()
    detect_cfg.BLANKET_TOLERANCE = cfg.uad_tolerance
    detect_cfg.VALIDATE_BLANKETS = True
    validator = MarkovBlanketValidator(detect_cfg)
    with redirect_stdout(io.StringIO()):
        result = validator.validate_cluster(raw_vars, var_names, disc, trace_dict)
    blanket = result["blanket_validation"]
    return {
        "strict_valid": bool(blanket["valid"]),
        "strict_violation": float(blanket["violation"]),
        "strict_details": blanket["details"],
        "classification": result["classification"],
    }
