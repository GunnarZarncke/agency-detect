"""Critical outcome labels and threat masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CriticalOutcome:
    name: str
    index: int
    direction: str = "lower_is_better"  # lower_is_better | higher_is_better

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "CriticalOutcome":
        return cls(
            name=str(row["name"]),
            index=int(row["index"]),
            direction=str(row.get("direction", "lower_is_better")),
        )


def parse_critical_outcomes(metadata: Mapping[str, object]) -> List[CriticalOutcome]:
    raw = metadata.get("critical_outcomes") or []
    return [CriticalOutcome.from_mapping(r) for r in raw]


def control_indices(metadata: Mapping[str, object]) -> List[int]:
    """Exogenous world channels for partialing (not agent bodies or target outcome)."""
    idx: List[int] = []
    for key in ("world_var_indices", "world_all_var_indices"):
        vals = metadata.get(key) or []
        idx.extend(int(i) for i in vals)
    seen = set()
    out: List[int] = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def threat_mask(
    outcome: np.ndarray,
    *,
    direction: str,
    bad_quantile: float = 0.80,
) -> np.ndarray:
    q = float(np.quantile(outcome, bad_quantile))
    if direction == "lower_is_better":
        return outcome >= q
    return outcome <= np.quantile(outcome, 1.0 - bad_quantile)


def attach_physics_critical_outcome(result, *, policy: str):
    """Label pole angle as critical outcome for CartPole eval (in-place metadata)."""
    meta = result.metadata
    role_indices = meta["role_indices"]
    var_names = list(meta["var_names"])
    pole_idx = int(role_indices[(0, "internal")][1])
    meta["critical_outcomes"] = [
        {
            "name": var_names[pole_idx],
            "index": pole_idx,
            "direction": "lower_is_better",
        }
    ]
    meta["world_var_indices"] = []
    meta["resource_var_indices"] = []
    meta["outcome_influence_ground_truth"] = {
        "0": policy in ("balance", "track"),
    }
    return result
