"""Small configuration surface for hierarchical spotlight fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class HierarchicalConfig:
    """Only parameters used by the E12 chunk-fusion experiment."""

    input_json: str = "results/spotlight/e11/spotlight_e11_rich_agents_cpr3_k24_p16.json"
    output_json: str = "results/hierarchical/e12/hierarchical_e12_rich_agents.json"
    output_dot: str = ""
    output_png: str = ""
    timestamp_outputs: bool = True
    render_fusion_hints: bool = False
    max_rendered_fusion_edges: int = 12

    # Node selection from spotlight passes.
    include_low_jaccard_nodes: bool = False
    min_node_size: int = 2

    # Data-only edge criteria.
    mi_bins: int = 8
    mi_max_lag: int = 3
    min_cross_mi: float = 0.70
    require_union_uad: bool = False
    max_union_uad_violation: float = 1.5
    require_union_precursor: bool = False

    # Evaluation only (uses simulator ground truth).
    hit_jaccard: float = 0.30
    clean_min_agent_recall: float = 0.30
    clean_max_extra_fraction: float = 0.50
    clean_max_world_vars: int = 0
    clean_max_decoy_vars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

