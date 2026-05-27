"""Configuration for serial spotlight / peel-off agent discovery (E9+)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class SpotlightConfig:
    """
    E9a defaults target the E8 setting: 8 agents, ~20% decoys, seed=1, T=4000.
    All fields are exposed for ablation sweeps.
    """

    # --- Simulation ---
    seed: int = 1
    T: int = 4000
    num_agents: int = 8
    copies_per_role: int = 2
    decoy_vars: int = 0
    decoy_mode: str = "noise"
    process_noise: float = 0.02
    observation_noise: float = 0.01
    interaction_strength: float = 0.10
    confound_strength: float = 0.0
    leakage_strength: float = 0.01
    mixing_strength: float = 0.02
    local_env_strength: float = 1.8
    env_vars_per_agent: int = 0
    env_action_coupling: float = 0.0
    env_to_sensor_strength: float = 0.08
    env_ar1_rho: float = 0.94
    env_ar1_sigma: float = 0.12
    env_copies_per_var: int = 1
    world_vars: int = 12
    world_to_sensor_strength: float = 0.08
    world_ar1_rho: float = 0.96
    world_ar1_sigma: float = 0.10

    # --- Peel loop ---
    max_passes: int = 8
    peel_mode: Literal["mask_zero", "none"] = "mask_zero"
    stop_if_no_cluster: bool = True
    stop_if_precursor_fails: bool = False

    # --- MI proposal (pick ONE cluster per pass) ---
    proposal_mi_k: int = 16
    proposal_background_factorize: bool = True
    proposal_background_components: int = 1
    mi_bins: int = 8
    mi_max_lag: int = 3
    min_cluster_size: int = 2
    cluster_score: Literal["precursor", "precursor_x_size"] = "precursor"
    within_mi_weight: float = 0.30
    tiny_cluster_penalty: float = 0.25

    # --- Precursor gates ---
    precursor_persistence_floor: float = 0.12
    precursor_contingency_floor: float = 0.015
    require_precursor_pass: bool = False

    # --- Latent model (small capacity per pass) ---
    num_slots: int = 3
    slot_dim: int = 16
    window: int = 16
    pretrain_epochs: int = 50
    refine_epochs: int = 40
    batch_size: int = 128
    pretrain_lr: float = 3e-4
    refine_lr: float = 1e-4
    lambda_align: float = 4.0
    lambda_sparse: float = 1e-3
    target_smoothing: float = 0.05
    device: Optional[str] = None

    # --- Candidate mapping ---
    candidate_mode: Literal["spotlight_slot", "mi_cluster"] = "mi_cluster"
    assign_threshold: float = 0.20
    min_vars_per_candidate: int = 4
    spotlight_slot_index: int = 0

    # --- UAD validation ---
    validate_uad: bool = True
    uad_tolerance: float = 1.0
    discretization_bins: int = 8
    require_uad_pass: bool = False

    # Data-only agency gate before training (see agency_gate_mode).
    require_agency_signature: bool = False
    agency_gate_mode: Literal["off", "strict", "soft", "actions_only", "score_penalty"] = "off"
    agency_score_penalty_weight: float = 1.0
    agency_penalty_no_actions: float = 0.85
    agency_penalty_no_sensors: float = 0.35
    agency_penalty_no_internals: float = 0.15

    def effective_agency_gate_mode(self) -> str:
        if self.require_agency_signature and self.agency_gate_mode == "off":
            return "strict"
        return self.agency_gate_mode

    # --- Admission (eval / optional filter) ---
    jaccard_hit_threshold: float = 0.30
    require_jaccard_hit: bool = False

    # If True, mask selected cluster vars after every pass (greedy progress even on miss).
    peel_selected_always: bool = False
    # If True, peel vars even when precursor gate skips training (usually decoy blobs).
    peel_on_precursor_skip: bool = False
    # Peel large passive clusters when all candidates fail agency (exogenous world blobs).
    peel_on_agency_skip: bool = True
    # Eval/oracle only: peel ground-truth agent vars (uses agent_clusters metadata).
    peel_full_agent_on_hit: bool = False

    # --- Logging ---
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
