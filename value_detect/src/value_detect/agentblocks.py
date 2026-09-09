"""V2-3: fused-agents + fused-environment (SJ's test), two architectures.

Pipeline: detect agents (Gunnar's adaptive detector, untouched) → fuse each agent into
a compressed block (top-K macro-states, lost mass reported) → score each variable
against the block-level elements. Orphans (variables detection drops — the goal, in the
V1 world) stay as singleton elements. Design decisions: docs/V2_3_OPTIONS_MEMO.md and
DECISIONS.md 2026-08-11.

Architectures (both run as co-equal tests):
* ``keyring`` — the candidate's own-agent block stands in the conditioning: intake is a
  ring-conditioned chain sum; each outbound flow takes max(ring-on, ring-off) per target
  (mediation-safe); the own block is also a TARGET (a goal's grip on its own agent's
  action is real outbound).
* ``menu`` — blocks are a best-key menu: pairwise-over-elements intake; outbound per
  target takes the best single element key (or none).
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .directed import ALPHA, _cmi_labeled, _labels
from .floors import _dense_pair

ARCHITECTURES = ("keyring", "menu")
VIABLE_LOST_MASS = 0.10  # a block counts toward selection coverage only if the
                         # registered top-K compression keeps >=90% of its mass


def _viable_coverage(frame: pd.DataFrame, agents: List[List[str]], budget: int = 64):
    """Coverage counted only over compression-viable blocks (lost mass < 10%)."""
    cov = 0
    for block in agents:
        _, lost = compress_block(frame, block, budget)
        if lost < VIABLE_LOST_MASS:
            cov += len(block)
    return cov


def detect_blocks(frame: pd.DataFrame) -> Dict[str, object]:
    """Run Gunnar's adaptive detector; return {'agents': [[cols]], 'env': [cols],
    'orphans': [cols]} (his prints captured, not shown)."""
    from agency_detect.config import DetectionConfig
    from agency_detect.detection import AgentDetector

    trace = frame.to_dict("records")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        detector = AgentDetector(DetectionConfig)
        results = detector.adaptive_detect_agents(trace)
    agents, env = [], []
    for label, info in (results or {}).items():
        if label == "env":
            env = list(info["variables"])
        else:
            agents.append(list(info["variables"]))
    assigned = set(env) | {v for block in agents for v in block}
    orphans = [c for c in frame.columns if c not in assigned]
    return {"agents": agents, "env": env, "orphans": orphans, "raw_log": buf.getvalue()}


def compress_block(frame: pd.DataFrame, cols: List[str], budget: int = 64) -> Tuple[np.ndarray, float]:
    """Joint labels of ``cols`` compressed to at most ``budget`` states (top budget−1 by
    frequency + one 'other'). Returns (labels, lost_mass)."""
    joint = _labels(frame[cols].to_numpy() if len(cols) > 1 else frame[cols[0]].to_numpy())
    counts = np.bincount(joint)
    if len(counts) <= budget:
        return joint, 0.0
    keep = np.argsort(counts)[::-1][: budget - 1]
    remap = np.full(len(counts), budget - 1, dtype=np.int64)
    remap[keep] = np.arange(budget - 1)
    lost = 1.0 - counts[keep].sum() / counts.sum()
    return remap[joint], float(lost)


class BlockScorer:
    """Block-level scorer with the ConventionScorer interface (works with null_floors).

    ``partition`` may be supplied directly (unit tests / reuse across seeds); otherwise
    :func:`detect_blocks` runs once per frame. Nulls shuffle only the candidate column —
    the partition and the other elements stay fixed, as the addendum prescribes.
    """

    def __init__(self, frame: pd.DataFrame, architecture: str, env_var: Optional[str] = "E",
                 lag: int = 1, alpha: float = ALPHA, budget: int = 64,
                 partition: Optional[Dict[str, object]] = None):
        assert architecture in ARCHITECTURES, architecture
        self.frame = frame
        self.architecture = architecture
        self.cols: List[str] = list(frame.columns)
        self.env_var = env_var if (env_var in self.cols if env_var else False) else None
        self.lag = lag
        self.alpha = alpha
        self.budget = budget
        self.n = len(frame)
        self.labels: Dict[str, np.ndarray] = {c: _labels(frame[c].to_numpy()) for c in self.cols}
        self.partition = partition or detect_blocks(frame)
        self.lost_mass: Dict[str, float] = {}
        self._member_of: Dict[str, Optional[int]] = {}
        for i, block in enumerate(self.partition["agents"]):
            for v in block:
                self._member_of[v] = i
        for v in self.cols:
            self._member_of.setdefault(v, None)
        self._elem_cache: Dict[tuple, np.ndarray] = {}
        self._pair_cache: Dict[tuple, np.ndarray] = {}

    # ---- element construction ----

    def _fused(self, cols_key: tuple) -> np.ndarray:
        if cols_key not in self._elem_cache:
            lab, lost = compress_block(self.frame, list(cols_key), self.budget)
            self._elem_cache[cols_key] = lab
            self.lost_mass["+".join(cols_key)] = lost
        return self._elem_cache[cols_key]

    def elements_for(self, v: str) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """(own_ring_labels or None, list of other element label arrays), all excluding v."""
        own = None
        others: List[np.ndarray] = []
        my_block = self._member_of.get(v)
        for i, block in enumerate(self.partition["agents"]):
            cols = tuple(c for c in block if c != v)
            if not cols:
                continue
            lab = self._fused(cols)
            if i == my_block:
                own = lab
            else:
                others.append(lab)
        # Environment members are ALWAYS singleton elements, never fused (SJ,
        # 2026-08-11): agent-blocks respect a discovered coherence; the environment,
        # by the theory's own lights, has none to respect — and a consistent rule
        # beats a data-dependent one. Env members (like orphans) carry no ring.
        for c in self.partition["env"]:
            if c != v:
                others.append(self.labels[c])
        for o in self.partition["orphans"]:
            if o != v:
                others.append(self.labels[o])
        return own, others

    # ---- estimates ----

    def _te(self, xl: np.ndarray, yl: np.ndarray, zl_extra: Optional[np.ndarray] = None) -> float:
        lag = self.lag
        z = yl[:-lag] if zl_extra is None else _dense_pair(yl[:-lag], zl_extra[:-lag])
        return _cmi_labeled(yl[lag:], xl[:-lag], z, self.alpha)

    def score_variable(self, v: str, vlab_override: Optional[np.ndarray] = None) -> Dict[str, float]:
        vl = self.labels[v] if vlab_override is None else vlab_override
        lag = self.lag
        own, others = self.elements_for(v)
        targets = ([own] if own is not None else []) + others

        if self.architecture == "keyring":
            push_in = 0.0
            if own is not None:
                push_in += _cmi_labeled(vl[lag:], own[:-lag], vl[:-lag], self.alpha)
            for e in others:
                z = vl[:-lag] if own is None else _dense_pair(vl[:-lag], own[:-lag])
                push_in += _cmi_labeled(vl[lag:], e[:-lag], z, self.alpha)
            out_sys = 0.0
            for e in targets:
                plain = self._te(vl, e)
                ringed = plain if (own is None or e is own) else self._te(vl, e, own)
                out_sys += max(plain, ringed)
            if self.env_var is None or v == self.env_var:
                out_env = float("nan")
            else:
                el = self.labels[self.env_var]
                plain = self._te(vl, el)
                out_env = plain if own is None else max(plain, self._te(vl, el, own))
        else:  # menu
            push_in = sum(_cmi_labeled(vl[lag:], e[:-lag], vl[:-lag], self.alpha) for e in targets)
            out_sys = 0.0
            for e in targets:
                best = self._te(vl, e)
                for k in targets:
                    if k is e:
                        continue
                    best = max(best, self._te(vl, e, k))
                out_sys += best
            if self.env_var is None or v == self.env_var:
                out_env = float("nan")
            else:
                el = self.labels[self.env_var]
                best = self._te(vl, el)
                for k in targets:
                    best = max(best, self._te(vl, el, k))
                out_env = best

        return {"push_in": push_in, "out_sys": out_sys, "out_env": out_env}

    def score_all(self) -> pd.DataFrame:
        rows = []
        for v in self.cols:
            s = self.score_variable(v)
            out, inn, oe = s["out_sys"], s["push_in"], s["out_env"]
            denom = out + inn
            pol = (out - inn) / denom if denom > 0 else 0.0
            pol_env = ((oe - inn) / (oe + inn)) if np.isfinite(oe) and (oe + inn) > 0 else (
                float("nan") if not np.isfinite(oe) else 0.0)
            rows.append({"variable": v, "push_in": inn, "out_sys": out, "out_env": oe,
                         "raw_sys": out - inn,
                         "raw_env": (oe - inn) if np.isfinite(oe) else float("nan"),
                         "polarity_sys": pol, "polarity_env": pol_env,
                         "total_flow": out + inn})
        return pd.DataFrame(rows).set_index("variable").sort_values("polarity_sys", ascending=False)


def detect_blocks_swept(frame: pd.DataFrame, dials: Optional[List[int]] = None) -> Dict[str, object]:
    """Gunnar's plain detector at every dial; keep the dial whose blanket-VALID agents
    cover the most variables (ties → fewer clusters). His machinery + his E7/P1
    selection philosophy (downstream validity chooses the cluster count)."""
    from agency_detect.config import DetectionConfig
    from agency_detect.detection import AgentDetector

    trace = frame.to_dict("records")
    n_vars = len(frame.columns)
    dials = dials or list(range(2, max(3, min(n_vars, 17))))
    if n_vars > 12:  # wide worlds: hoist the similarity matrix (identical behaviour)
        return _swept_fast(frame, dials)
    best = None
    original = DetectionConfig.N_AGENTS
    log_parts = []
    try:
        for dial in dials:
            DetectionConfig.N_AGENTS = dial
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    results = AgentDetector(DetectionConfig).detect_agents(trace)
            except Exception as e:  # a bad dial must not sink the sweep
                log_parts.append(f"dial {dial}: error {e}")
                continue
            agents, env = [], []
            for label, info in (results or {}).items():
                if label == "env":
                    env = list(info["variables"])
                elif info["blanket_validation"]["valid"] is not False:
                    agents.append(list(info["variables"]))
            coverage = sum(len(b) for b in agents)
            log_parts.append(f"dial {dial}: {len(agents)} valid agents, coverage {coverage}/{n_vars}")
            key = (coverage, -len(agents))
            if best is None or key > best[0]:
                best = (key, agents, env, dial)
    finally:
        DetectionConfig.N_AGENTS = original
    _, agents, env, dial = best
    assigned = set(env) | {v for b in agents for v in b}
    orphans = [c for c in frame.columns if c not in assigned]
    return {"agents": agents, "env": env, "orphans": orphans,
            "dial": dial, "raw_log": "\n".join(log_parts)}


def _swept_fast(frame: pd.DataFrame, dials: List[int]) -> Dict[str, object]:
    """Same sweep as :func:`detect_blocks_swept`, hoisting the similarity matrix (the
    invariant, expensive step) out of the dial loop. Piecewise calls to Gunnar's own
    functions in his own order — identical behaviour, ~50× cheaper on wide worlds."""
    from collections import defaultdict

    from agency_detect.config import DetectionConfig
    from agency_detect.detection import build_similarity_matrix, filter_weak_connections
    from agency_detect.markov_blanket import MarkovBlanketValidator
    from sklearn.cluster import AgglomerativeClustering

    cols = list(frame.columns)
    data = frame.to_numpy()
    trace = frame.to_dict("records")
    active = [i for i in range(len(cols)) if data[:, i].var() > 0]
    vars_active = [cols[i] for i in active]
    sim, dist = build_similarity_matrix(data[:, active], DetectionConfig.MAX_LAG)
    validator = MarkovBlanketValidator(DetectionConfig)

    best = None
    log_parts = []
    n_vars = len(cols)
    for dial in dials:
        if dial >= len(vars_active):
            continue
        labels = AgglomerativeClustering(n_clusters=dial, metric="precomputed",
                                         linkage="complete").fit_predict(dist)
        clusters = defaultdict(list)
        for v, lbl in zip(vars_active, labels):
            clusters[lbl].append(v)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            filtered, env_bucket = filter_weak_connections(clusters, vars_active, sim,
                                                           DetectionConfig.WEAK_THRESHOLD)
            agents, env = [], list(env_bucket)
            for lbl, variables in filtered.items():
                if not variables:
                    continue
                res = validator.validate_cluster(variables, cols, data, trace)
                if res["blanket_validation"]["valid"] is False:
                    env.extend(variables)
                else:
                    agents.append(list(variables))
        coverage = _viable_coverage(frame, agents)
        raw_cov = sum(len(b) for b in agents)
        log_parts.append(f"dial {dial}: {len(agents)} valid agents, viable coverage {coverage}/{n_vars} (raw {raw_cov})")
        key = (coverage, len(agents))  # ties -> MORE clusters (E12 anti-lump precedent)
        if best is None or key > best[0]:
            best = (key, agents, env, dial)
    _, agents, env, dial = best
    assigned = set(env) | {v for b in agents for v in b}
    orphans = [c for c in cols if c not in assigned]
    return {"agents": agents, "env": env, "orphans": orphans,
            "dial": dial, "raw_log": "\n".join(log_parts)}
