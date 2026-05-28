"""Fuse spotlight chunks into higher-level agent hypotheses."""

from __future__ import annotations

import io
import json
import subprocess
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from agency_detect.config import DetectionConfig
from agency_detect.detection import build_similarity_matrix
from agency_detect.markov_blanket import MarkovBlanketValidator
from learn_agents.learn_agents import TraceSimulationConfig, precursor_cluster_stats, simulate_known_agent_trace

from .config import HierarchicalConfig


@dataclass
class ChunkNode:
    node_id: int
    pass_index: int
    var_indices: List[int]
    score: float
    source_best_agent: int
    source_best_jaccard: float


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _purity_match(
    var_indices: Sequence[int],
    agent_id: int,
    agent_clusters: Dict[int, List[int]],
    metadata: Dict[str, Any],
    cfg: HierarchicalConfig,
) -> Dict[str, Any]:
    selected = set(var_indices)
    truth = set(agent_clusters[agent_id])
    hit = len(selected & truth)
    agent_recall = hit / len(truth) if truth else 0.0
    jaccard = _jaccard(list(selected), list(truth))

    all_agent_vars = {v for values in agent_clusters.values() for v in values}
    world_vars = set(metadata.get("world_all_var_indices", metadata.get("world_var_indices", [])))
    selected_agent_vars = selected & all_agent_vars
    extra_agent_vars = selected_agent_vars - truth
    world_count = len(selected & world_vars)
    decoy_count = len(selected - all_agent_vars - world_vars)
    extra_count = len(selected - truth)
    extra_fraction = extra_count / max(len(selected), 1)

    clean = (
        agent_recall >= cfg.clean_min_agent_recall
        and extra_fraction <= cfg.clean_max_extra_fraction
        and world_count <= cfg.clean_max_world_vars
        and decoy_count <= cfg.clean_max_decoy_vars
    )
    return {
        "clean": clean,
        "jaccard": jaccard,
        "agent_recall": agent_recall,
        "hit_vars": hit,
        "extra_vars": extra_count,
        "extra_fraction": extra_fraction,
        "extra_agent_vars": len(extra_agent_vars),
        "world_vars": world_count,
        "decoy_vars": decoy_count,
    }


def _component_sets(n_nodes: int, edges: Iterable[Tuple[int, int]]) -> List[Set[int]]:
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    comps: Dict[int, Set[int]] = {}
    for i in range(n_nodes):
        comps.setdefault(find(i), set()).add(i)
    return list(comps.values())


def _discretize(trace: np.ndarray, bins: int) -> np.ndarray:
    out = np.zeros(trace.shape, dtype=np.int64)
    quantiles = np.linspace(0, 1, bins + 1)
    for j in range(trace.shape[1]):
        edges = np.quantile(trace[:, j], quantiles)
        edges = np.maximum.accumulate(edges)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        out[:, j] = np.clip(np.digitize(trace[:, j], edges[1:-1]), 0, bins - 1)
    return out


def _trace_dicts(disc: np.ndarray, var_names: Sequence[str]) -> List[Dict[str, int]]:
    return [{name: int(disc[t, i]) for i, name in enumerate(var_names)} for t in range(disc.shape[0])]


def _uad_valid(
    var_indices: Sequence[int],
    var_names: Sequence[str],
    disc: np.ndarray,
    trace_dict: List[Dict[str, int]],
    tolerance: float,
) -> Tuple[bool, float]:
    raw_vars = [var_names[i] for i in sorted(set(var_indices))]
    detect_cfg = DetectionConfig()
    detect_cfg.BLANKET_TOLERANCE = tolerance
    detect_cfg.VALIDATE_BLANKETS = True
    validator = MarkovBlanketValidator(detect_cfg)
    with redirect_stdout(io.StringIO()):
        result = validator.validate_cluster(raw_vars, list(var_names), disc, trace_dict)
    blanket = result["blanket_validation"]
    return bool(blanket["valid"]), float(blanket["violation"])


def _precursor_passed(trace: np.ndarray, var_indices: Sequence[int], cfg: HierarchicalConfig) -> bool:
    labels = np.full(trace.shape[1], -1, dtype=np.int64)
    labels[list(var_indices)] = 0
    stats = precursor_cluster_stats(trace, labels, bins=cfg.mi_bins, max_lag=cfg.mi_max_lag)
    return bool(stats and stats[0].passed)


def _mean_cross_mi(trace: np.ndarray, a: Sequence[int], b: Sequence[int], cfg: HierarchicalConfig) -> float:
    if not a or not b:
        return 0.0
    disc = _discretize(trace[:, list(a) + list(b)], cfg.mi_bins).astype(np.float64)
    sim, _ = build_similarity_matrix(disc, max_lag=cfg.mi_max_lag)
    n_a = len(a)
    cross = sim[:n_a, n_a:]
    return float(np.mean(cross)) if cross.size else 0.0


def _load_nodes(report: Dict[str, Any], cfg: HierarchicalConfig) -> List[ChunkNode]:
    nodes: List[ChunkNode] = []
    for p in report["passes"]:
        var_indices = list(p.get("cluster_var_indices", []))
        if len(var_indices) < cfg.min_node_size:
            continue
        if not cfg.include_low_jaccard_nodes and not p.get("is_hit", False):
            continue
        nodes.append(
            ChunkNode(
                node_id=len(nodes),
                pass_index=int(p["pass_index"]),
                var_indices=var_indices,
                score=float(p["cluster_score"]),
                source_best_agent=int(p.get("best_agent_id", -1)),
                source_best_jaccard=float(p.get("best_jaccard", 0.0)),
            )
        )
    return nodes


def _sim_from_report(report: Dict[str, Any]):
    sim_keys = TraceSimulationConfig.__dataclass_fields__.keys()
    sim_kwargs = {k: v for k, v in report["config"].items() if k in sim_keys}
    return simulate_known_agent_trace(TraceSimulationConfig(**sim_kwargs))


def run_hierarchical_fusion(cfg: HierarchicalConfig) -> Dict[str, Any]:
    report = json.loads(Path(cfg.input_json).read_text(encoding="utf-8"))
    sim = _sim_from_report(report)
    trace = sim.trace
    metadata = sim.metadata
    var_names = list(metadata["var_names"])
    agent_clusters = {int(k): list(v) for k, v in metadata["agent_clusters"].items()}
    nodes = _load_nodes(report, cfg)

    disc = _discretize(trace, cfg.mi_bins)
    trace_dict = _trace_dicts(disc, var_names)

    edges: List[Dict[str, Any]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            left, right = nodes[i], nodes[j]
            union_vars = sorted(set(left.var_indices) | set(right.var_indices))
            cross_mi = _mean_cross_mi(trace, left.var_indices, right.var_indices, cfg)
            if cross_mi < cfg.min_cross_mi:
                continue
            uad_ok, violation = _uad_valid(
                union_vars, var_names, disc, trace_dict, cfg.max_union_uad_violation
            )
            if cfg.require_union_uad and not uad_ok:
                continue
            precursor_ok = _precursor_passed(trace, union_vars, cfg)
            if cfg.require_union_precursor and not precursor_ok:
                continue
            edges.append(
                {
                    "source": i,
                    "target": j,
                    "cross_mi": cross_mi,
                    "union_size": len(union_vars),
                    "union_uad_valid": uad_ok,
                    "union_uad_violation": violation,
                    "union_precursor_passed": precursor_ok,
                }
            )

    components = []
    for comp_id, node_ids in enumerate(_component_sets(len(nodes), [(e["source"], e["target"]) for e in edges])):
        union_vars = sorted({v for node_id in node_ids for v in nodes[node_id].var_indices})
        best_agent, best_j = -1, 0.0
        best_clean_agent, best_clean_recall = -1, 0.0
        clean_matches = {}
        for agent_id, truth in agent_clusters.items():
            jj = _jaccard(union_vars, truth)
            if jj > best_j:
                best_agent, best_j = agent_id, jj
            purity = _purity_match(union_vars, agent_id, agent_clusters, metadata, cfg)
            clean_matches[str(agent_id)] = purity
            if purity["clean"] and purity["agent_recall"] > best_clean_recall:
                best_clean_agent = agent_id
                best_clean_recall = purity["agent_recall"]
        components.append(
            {
                "component_id": comp_id,
                "node_ids": sorted(node_ids),
                "var_indices": union_vars,
                "size": len(union_vars),
                "best_agent_id": best_agent,
                "best_jaccard": best_j,
                "is_agent_hit": best_j >= cfg.hit_jaccard,
                "best_clean_agent_id": best_clean_agent,
                "best_clean_agent_recall": best_clean_recall,
                "is_clean_agent_hit": best_clean_agent >= 0,
                "clean_matches": clean_matches,
            }
        )

    covered_agents = set()
    clean_node_agents = set()
    clean_component_agents = set()
    best_by_agent: Dict[int, float] = {}
    best_clean_by_agent: Dict[int, float] = {}
    for agent_id, truth in agent_clusters.items():
        best = 0.0
        best_clean = 0.0
        for node in nodes:
            best = max(best, _jaccard(node.var_indices, truth))
            purity = _purity_match(node.var_indices, agent_id, agent_clusters, metadata, cfg)
            if purity["clean"]:
                clean_node_agents.add(agent_id)
                best_clean = max(best_clean, purity["agent_recall"])
        for comp in components:
            best = max(best, _jaccard(comp["var_indices"], truth))
            purity = comp["clean_matches"][str(agent_id)]
            if purity["clean"]:
                clean_component_agents.add(agent_id)
                best_clean = max(best_clean, purity["agent_recall"])
        best_by_agent[agent_id] = best
        best_clean_by_agent[agent_id] = best_clean
        if best >= cfg.hit_jaccard:
            covered_agents.add(agent_id)

    summary = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_components": len(components),
        "agent_graph_recall": len(covered_agents) / len(agent_clusters) if agent_clusters else 0.0,
        "clean_node_recall": len(clean_node_agents) / len(agent_clusters) if agent_clusters else 0.0,
        "clean_component_recall": len(clean_component_agents) / len(agent_clusters) if agent_clusters else 0.0,
        "clean_graph_recall": len(clean_node_agents | clean_component_agents) / len(agent_clusters) if agent_clusters else 0.0,
        "covered_agent_ids": sorted(covered_agents),
        "clean_node_agent_ids": sorted(clean_node_agents),
        "clean_component_agent_ids": sorted(clean_component_agents),
        "clean_graph_agent_ids": sorted(clean_node_agents | clean_component_agents),
        "best_jaccard_by_agent": {str(k): v for k, v in sorted(best_by_agent.items())},
        "best_clean_recall_by_agent": {str(k): v for k, v in sorted(best_clean_by_agent.items())},
    }

    return {
        "config": cfg.to_dict(),
        "source_summary": report.get("summary", {}),
        "nodes": [node.__dict__ for node in nodes],
        "edges": edges,
        "components": components,
        "summary": summary,
    }


def _quote_dot(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _html_label(lines: Sequence[str]) -> str:
    escaped = [
        str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for line in lines
    ]
    return "<" + "<BR/>".join(escaped) + ">"


def write_graphviz(result: Dict[str, Any], dot_path: str, png_path: str = "") -> Dict[str, str]:
    """Write a readable component forest and optionally render PNG."""
    dot = Path(dot_path)
    dot.parent.mkdir(parents=True, exist_ok=True)

    covered = set(result["summary"].get("covered_agent_ids", []))
    components = result.get("components", [])
    nodes_by_id = {int(node["node_id"]): node for node in result.get("nodes", [])}

    lines = [
        "digraph hierarchical_spotlight {",
        "  graph [rankdir=LR, splines=ortho, nodesep=0.45, ranksep=0.95];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10];",
        "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.7];",
    ]

    for comp in components:
        comp_id = int(comp["component_id"])
        agent = int(comp.get("best_agent_id", -1))
        color = "#c7e9ff" if agent in covered else "#eeeeee"
        label = _html_label(
            [
                f"component {comp_id}",
                f"best agent {agent}",
                f"J={float(comp.get('best_jaccard', 0.0)):.2f}",
                f"clean agent {comp.get('best_clean_agent_id', -1)}",
                f"nodes={len(comp.get('node_ids', []))}",
                f"vars={comp.get('size', 0)}",
            ]
        )
        lines.append(
            f"  c{comp_id} [label={label}, fillcolor={_quote_dot(color)}, shape=folder];"
        )
        for node_id in sorted(int(n) for n in comp.get("node_ids", [])):
            node = nodes_by_id[node_id]
            node_agent = int(node.get("source_best_agent", -1))
            node_color = "#d6f5d6" if node_agent in covered else "#eeeeee"
            node_label = _html_label(
                [
                    f"chunk {node_id}",
                    f"pass {node['pass_index']}",
                    f"agent {node_agent}",
                    f"J={float(node.get('source_best_jaccard', 0.0)):.2f}",
                    f"vars={len(node.get('var_indices', []))}",
                ]
            )
            lines.append(
                f"  n{node_id} [label={node_label}, fillcolor={_quote_dot(node_color)}];"
            )
            lines.append(f"  c{comp_id} -> n{node_id};")

    if result.get("config", {}).get("render_fusion_hints", False):
        max_edges = int(result.get("config", {}).get("max_rendered_fusion_edges", 12))
        strong_edges = sorted(
            result.get("edges", []), key=lambda e: float(e.get("cross_mi", 0.0)), reverse=True
        )[: max(0, max_edges)]
        for edge in strong_edges:
            source = int(edge["source"])
            target = int(edge["target"])
            if source not in nodes_by_id or target not in nodes_by_id:
                continue
            label = f"MI {float(edge['cross_mi']):.2f}"
            lines.append(
                f"  n{source} -> n{target} [style=dashed, color=\"#888888\", "
                f"constraint=false, label={_quote_dot(label)}];"
            )

    summary = result["summary"]
    graph_label = _html_label(
        [
            "E12 hierarchical spotlight",
            f"recall={float(summary['agent_graph_recall']):.3f}",
            f"clean={float(summary['clean_graph_recall']):.3f}",
            f"nodes={summary['n_nodes']} edges={summary['n_edges']} components={summary['n_components']}",
        ]
    )
    lines.append(f"  labelloc=t; label={graph_label};")
    lines.append("}")
    dot.write_text("\n".join(lines) + "\n", encoding="utf-8")

    written = {"dot": str(dot)}
    if png_path:
        png = Path(png_path)
        png.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["dot", "-Tpng", str(dot), "-o", str(png)], check=True)
            written["png"] = str(png)
        except (OSError, subprocess.CalledProcessError) as exc:
            written["png_error"] = str(exc)
    return written

