#!/usr/bin/env python3
"""
Scaling scaffold for handle-aware UAD.

This expands the minimal handle-UAD toy into a small benchmark that varies:
  * number of passive alias decoys competing with the true sensor/action handles
  * passive sample count
  * active intervention budget / rounds
  * candidate cap

Core point tested:
  Plain passive UAD is vulnerable to clean observational aliases.
  Handle-UAD can reject aliases only by spending handle tests.
  Therefore the relevant scaling variable is not just T samples, but also
  the number of plausible false handles and the cost/fidelity of operations.

The script is intentionally small and transparent. It is not a universal UAD
implementation; it is a scaling-oriented first step.
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

NO_OP = "none"
OP_TYPES = ["sensor_flip", "action_block", "goal_flip"]


@dataclass(frozen=True)
class VariableSchema:
    n_alias: int
    n_distractor: int
    n_random: int
    names: Tuple[str, ...]
    idx: Dict[str, int]
    s_aliases: Tuple[int, ...]
    a_aliases: Tuple[int, ...]
    distractors: Tuple[int, ...]
    randoms: Tuple[int, ...]

    @staticmethod
    def make(n_alias: int, n_distractor: int, n_random: int) -> "VariableSchema":
        names: List[str] = ["B", "S", "A", "E", "G"]
        names += [f"S_alias_{i}" for i in range(n_alias)]
        names += [f"A_alias_{i}" for i in range(n_alias)]
        names += [f"D_{i}" for i in range(n_distractor)]
        names += [f"W_{i}" for i in range(n_random)]
        idx = {name: i for i, name in enumerate(names)}
        s_aliases = tuple(idx[f"S_alias_{i}"] for i in range(n_alias))
        a_aliases = tuple(idx[f"A_alias_{i}"] for i in range(n_alias))
        distractors = tuple(idx[f"D_{i}"] for i in range(n_distractor))
        randoms = tuple(idx[f"W_{i}"] for i in range(n_random))
        return VariableSchema(n_alias, n_distractor, n_random, tuple(names), idx, s_aliases, a_aliases, distractors, randoms)

    @property
    def true_loop(self) -> Tuple[int, int, int, int, int]:
        return (self.idx["B"], self.idx["S"], self.idx["A"], self.idx["G"], self.idx["E"])

    def pretty_tuple(self, tup: Tuple[int, int, int, int, int]) -> str:
        b, s, a, g, e = tup
        return f"B={self.names[b]}, S={self.names[s]}, A={self.names[a]}, G={self.names[g]}, E={self.names[e]}"


@dataclass(frozen=True)
class Candidate:
    b: int
    s: int
    a: int
    g: int
    e: int

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.b, self.s, self.a, self.g, self.e)

    def pretty(self, schema: VariableSchema) -> str:
        return schema.pretty_tuple(self.as_tuple())


@dataclass
class DataSet:
    x: np.ndarray
    x_next: np.ndarray
    op_type: np.ndarray
    op_handle: np.ndarray
    op_names: Tuple[str, ...]

    def append(self, other: "DataSet") -> "DataSet":
        assert self.op_names == other.op_names
        return DataSet(
            x=np.vstack([self.x, other.x]),
            x_next=np.vstack([self.x_next, other.x_next]),
            op_type=np.concatenate([self.op_type, other.op_type]),
            op_handle=np.concatenate([self.op_handle, other.op_handle]),
            op_names=self.op_names,
        )

    def subset(self, mask: np.ndarray) -> "DataSet":
        return DataSet(self.x[mask], self.x_next[mask], self.op_type[mask], self.op_handle[mask], self.op_names)

    def mask_op(self, op_name: str, handle: Optional[int] = None) -> np.ndarray:
        code = self.op_names.index(op_name)
        mask = self.op_type == code
        if handle is not None:
            mask = mask & (self.op_handle == handle)
        return mask

    def mask_none_or(self, op_name: str, handle: int) -> np.ndarray:
        return self.mask_op(NO_OP) | self.mask_op(op_name, handle)


class ScalingHandleWorld:
    def __init__(
        self,
        schema: VariableSchema,
        seed: int,
        sensor_noise: float = 0.05,
        belief_noise: float = 0.03,
        action_noise: float = 0.04,
        env_noise: float = 0.03,
        alias_noise: float = 0.0,
        handle_obs_noise: float = 0.08,
        distractor_noise: float = 0.06,
        goal_flip_rate: float = 0.015,
    ) -> None:
        self.schema = schema
        self.rng = np.random.default_rng(seed)
        self.sensor_noise = sensor_noise
        self.belief_noise = belief_noise
        self.action_noise = action_noise
        self.env_noise = env_noise
        self.alias_noise = alias_noise
        self.handle_obs_noise = handle_obs_noise
        self.distractor_noise = distractor_noise
        self.goal_flip_rate = goal_flip_rate
        self.reset()

    def bern(self, p: float) -> int:
        return int(self.rng.random() < p)

    def reset(self) -> None:
        self.B = int(self.rng.integers(0, 2))
        self.E = int(self.rng.integers(0, 2))
        self.G = int(self.rng.integers(0, 2))
        self.D = self.rng.integers(0, 2, size=self.schema.n_distractor).astype(np.int8)
        self.W = self.rng.integers(0, 2, size=self.schema.n_random).astype(np.int8)

    def _observe(self, B: int, E: int, G: int, D: np.ndarray, W: np.ndarray, op: Tuple[str, int]) -> Tuple[np.ndarray, int, int]:
        op_type, handle = op
        S_line = E ^ self.bern(self.sensor_noise)
        S_eff = 1 - S_line if (op_type == "sensor_flip" and handle == self.schema.idx["S"]) else S_line
        G_eff = 1 - G if (op_type == "goal_flip" and handle == self.schema.idx["G"]) else G
        A_line = (B ^ G_eff) ^ self.bern(self.action_noise)

        vals = np.zeros(len(self.schema.names), dtype=np.int8)
        vals[self.schema.idx["B"]] = B
        vals[self.schema.idx["S"]] = S_eff ^ self.bern(self.handle_obs_noise)
        vals[self.schema.idx["A"]] = A_line ^ self.bern(self.handle_obs_noise)
        vals[self.schema.idx["E"]] = E
        vals[self.schema.idx["G"]] = G_eff
        for j in self.schema.s_aliases:
            vals[j] = S_eff ^ self.bern(self.alias_noise)
        for j in self.schema.a_aliases:
            vals[j] = A_line ^ self.bern(self.alias_noise)
        for k, j in enumerate(self.schema.distractors):
            vals[j] = int(D[k])
        for k, j in enumerate(self.schema.randoms):
            vals[j] = int(W[k])
        return vals, S_eff, A_line

    def step(self, op: Tuple[str, int] = (NO_OP, -1)) -> Tuple[np.ndarray, np.ndarray]:
        x_t, S_eff, A_line = self._observe(self.B, self.E, self.G, self.D, self.W, op)
        op_type, handle = op
        effective_A = 0 if (op_type == "action_block" and handle == self.schema.idx["A"]) else A_line

        E_next = self.E ^ effective_A ^ self.bern(self.env_noise)
        B_next = S_eff ^ self.bern(self.belief_noise)
        G_next = self.G ^ self.bern(self.goal_flip_rate)
        D_next = np.array([E_next ^ self.bern(self.distractor_noise) for _ in self.schema.distractors], dtype=np.int8)
        W_next = self.rng.integers(0, 2, size=self.schema.n_random).astype(np.int8)
        x_next, _, _ = self._observe(B_next, E_next, G_next, D_next, W_next, (NO_OP, -1))
        self.B, self.E, self.G, self.D, self.W = B_next, E_next, G_next, D_next, W_next
        return x_t, x_next

    def rollout(self, n: int, op_policy: Sequence[Tuple[str, int]]) -> DataSet:
        xs: List[np.ndarray] = []
        xns: List[np.ndarray] = []
        op_codes: List[int] = []
        handles: List[int] = []
        op_names = (NO_OP, *OP_TYPES)
        for t in range(n):
            op = op_policy[t % len(op_policy)]
            x, xn = self.step(op)
            xs.append(x)
            xns.append(xn)
            op_codes.append(op_names.index(op[0]))
            handles.append(op[1])
        return DataSet(np.vstack(xs), np.vstack(xns), np.array(op_codes), np.array(handles), op_names)


def generate_passive(schema: VariableSchema, seed: int, n: int, **world_kwargs) -> DataSet:
    world = ScalingHandleWorld(schema, seed=seed, **world_kwargs)
    world.rollout(100, [(NO_OP, -1)])
    return world.rollout(n, [(NO_OP, -1)])


def generate_operation(schema: VariableSchema, seed: int, op_name: str, handle: int, n: int, **world_kwargs) -> DataSet:
    world = ScalingHandleWorld(schema, seed=seed, **world_kwargs)
    world.rollout(100, [(NO_OP, -1)])
    rng = np.random.default_rng(seed + 991)
    policy = [(op_name, handle) if rng.random() < 0.5 else (NO_OP, -1) for _ in range(n)]
    return world.rollout(n, policy)


# ---- binary information utilities ----

def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def encode_cols(*cols: np.ndarray) -> np.ndarray:
    mats = [_as2d(c) for c in cols if c is not None]
    mat = np.hstack(mats)
    # Binary columns: pack into integer. If any value is not binary, still safe if small ints.
    code = np.zeros(mat.shape[0], dtype=np.int64)
    base = 1
    for j in range(mat.shape[1]):
        col = mat[:, j].astype(np.int64)
        # Values are binary in this benchmark. Keep a slightly safer mixed-radix path.
        maxv = int(col.max()) if col.size else 0
        code += col * base
        base *= max(2, maxv + 1)
    return code


def entropy_code(code: np.ndarray) -> float:
    if code.size == 0:
        return 0.0
    counts = np.bincount(code.astype(np.int64))
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def entropy_cols(x: np.ndarray) -> float:
    return entropy_code(encode_cols(x))


def mi(x: np.ndarray, y: np.ndarray) -> float:
    val = entropy_cols(x) + entropy_cols(y) - entropy_cols(np.hstack([_as2d(x), _as2d(y)]))
    return max(0.0, float(val))


def cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xz = np.hstack([_as2d(x), _as2d(z)])
    yz = np.hstack([_as2d(y), _as2d(z)])
    xyz = np.hstack([_as2d(x), _as2d(y), _as2d(z)])
    val = entropy_cols(xz) + entropy_cols(yz) - entropy_cols(z) - entropy_cols(xyz)
    return max(0.0, float(val))


# ---- candidates and scoring ----

def candidate_pools(schema: VariableSchema) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    idx = schema.idx
    b_pool = [idx["B"], idx["S"]] + list(schema.s_aliases) + list(schema.distractors[:6]) + list(schema.randoms[:6])
    s_pool = [idx["S"], idx["E"]] + list(schema.s_aliases) + list(schema.distractors[:6]) + list(schema.randoms[:6])
    a_pool = [idx["A"]] + list(schema.a_aliases) + list(schema.distractors[:6]) + list(schema.randoms[:6])
    g_pool = [idx["G"], idx["E"]] + list(schema.distractors[:6]) + list(schema.randoms[:6])
    e_pool = [idx["E"], idx["S"]] + list(schema.distractors[:6]) + list(schema.randoms[:6])
    # Deduplicate while preserving order
    def dedup(xs: List[int]) -> List[int]:
        seen = set(); out = []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return dedup(b_pool), dedup(s_pool), dedup(a_pool), dedup(g_pool), dedup(e_pool)


def total_candidate_space(schema: VariableSchema) -> int:
    """Fast upper-bound count of the candidate product space.

    The exact distinct-role count would require iterating a large product for
    high alias counts. For scaling diagnostics we only need a comparable size
    proxy, so we report the product of role-pool sizes.
    """
    total = 1
    for pool in candidate_pools(schema):
        total *= len(pool)
    return int(total)


def enumerate_candidates(schema: VariableSchema, cap: int, seed: int) -> List[Candidate]:
    pools = candidate_pools(schema)
    true = Candidate(*schema.true_loop)
    upper = total_candidate_space(schema)
    # Enumerate exact candidates when the product is moderate. This avoids
    # rejection-sampling forever when cap exceeds the number of valid distinct-role
    # candidates in small worlds.
    if upper <= 250_000:
        cands = []
        for tup in itertools.product(*pools):
            if len(set(tup)) == 5:
                cands.append(Candidate(*tup))
        if len(cands) <= cap:
            return cands
        rng = np.random.default_rng(seed + 12345)
        true_tuple = true.as_tuple()
        valid = [c.as_tuple() for c in cands if c.as_tuple() != true_tuple]
        idx = rng.choice(len(valid), size=cap-1, replace=False)
        return [true] + [Candidate(*valid[i]) for i in idx]

    rng = np.random.default_rng(seed + 12345)
    cands = {true.as_tuple()}
    tries = 0
    max_tries = cap * 100
    while len(cands) < cap and tries < max_tries:
        tup = tuple(int(rng.choice(pool)) for pool in pools)
        if len(set(tup)) == 5:
            cands.add(tup)
        tries += 1
    return [Candidate(*t) for t in cands]


def passive_score(c: Candidate, data: DataSet) -> Dict[str, float]:
    X, Xn = data.x, data.x_next
    b, s, a, g, e = c.as_tuple()
    sensor = cmi(Xn[:, b], X[:, s], X[:, b])
    policy = mi(X[:, a], X[:, [b, g]])
    control = cmi(Xn[:, e], X[:, a], X[:, e])
    blanket = cmi(Xn[:, b], Xn[:, e], X[:, [s, a]])
    activity = sum(entropy_cols(X[:, j]) for j in [b, s, a, g, e]) / 5.0
    score = 1.15 * sensor + 1.0 * policy + 1.2 * control - 1.1 * blanket + 0.08 * activity
    return {"passive": score, "sensor": sensor, "policy": policy, "control": control, "blanket": blanket, "activity": activity}


def _combined_noop_and(data: DataSet, op_name: str, handle: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mask = data.mask_none_or(op_name, handle)
    if mask.sum() < 50 or data.mask_op(op_name, handle).sum() < 25:
        return None
    sub = data.subset(mask)
    active = (sub.op_type == sub.op_names.index(op_name)).astype(np.int8)
    return sub.x, sub.x_next, active


def interventional_score(c: Candidate, data: DataSet) -> Dict[str, float]:
    b, s, a, g, e = c.as_tuple()
    out = {"sensor_int": 0.0, "action_int": 0.0, "goal_int": 0.0}
    got = _combined_noop_and(data, "sensor_flip", s)
    if got is not None:
        X, Xn, active = got
        out["sensor_int"] = cmi(active, Xn[:, a], X[:, [e, g]]) - 0.020
    got_block = _combined_noop_and(data, "action_block", a)
    if got_block is not None:
        Xc, Xnc, active = got_block
        normal = active == 0
        blocked = active == 1
        if normal.sum() > 25 and blocked.sum() > 25:
            base_ctrl = cmi(Xnc[normal, e], Xc[normal, a], Xc[normal, e])
            block_ctrl = cmi(Xnc[blocked, e], Xc[blocked, a], Xc[blocked, e])
            out["action_int"] = (base_ctrl - block_ctrl) - 0.12
    got = _combined_noop_and(data, "goal_flip", g)
    if got is not None:
        X, Xn, active = got
        out["goal_int"] = cmi(active, X[:, a], X[:, [b, e]]) - 0.002
    out["interventional"] = 30.0 * out["sensor_int"] + 4.0 * out["action_int"] + 4.0 * out["goal_int"]
    return out


def score_candidates(schema: VariableSchema, cands: Sequence[Candidate], data: DataSet, use_interventions: bool) -> pd.DataFrame:
    no_op_data = data.subset(data.mask_op(NO_OP))
    rows = []
    true_loop = schema.true_loop
    for c in cands:
        ps = passive_score(c, no_op_data)
        ints = interventional_score(c, data) if use_interventions else {"sensor_int": 0.0, "action_int": 0.0, "goal_int": 0.0, "interventional": 0.0}
        total = ps["passive"] + ints["interventional"]
        rows.append({
            "b": c.b, "s": c.s, "a": c.a, "g": c.g, "e": c.e,
            "candidate": c.pretty(schema),
            "is_true": c.as_tuple() == true_loop,
            "score": total,
            **ps, **ints,
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def choose_next_operation(df: pd.DataFrame, already: set[Tuple[str, int]]) -> Tuple[str, int]:
    top = df.iloc[0]
    for role, op in [("a", "action_block"), ("s", "sensor_flip"), ("g", "goal_flip")]:
        h = int(top[role])
        if (op, h) not in already:
            return op, h
    # If the current top is fully tested, search next rows for untested top handles.
    for _, row in df.head(25).iterrows():
        for role, op in [("a", "action_block"), ("s", "sensor_flip"), ("g", "goal_flip")]:
            h = int(row[role])
            if (op, h) not in already:
                return op, h
    return "action_block", int(top["a"])


def rank_of_true(df: pd.DataFrame) -> int:
    arr = np.where(df["is_true"].to_numpy())[0]
    return int(arr[0] + 1) if len(arr) else -1


def run_progressive(
    schema: VariableSchema,
    seed: int,
    passive_n: int,
    intervention_batch: int,
    max_rounds: int,
    candidate_cap: int,
    world_kwargs: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    cands = enumerate_candidates(schema, cap=candidate_cap, seed=seed)
    total_space = total_candidate_space(schema)
    data = generate_passive(schema, seed=seed, n=passive_n, **world_kwargs)
    already: set[Tuple[str, int]] = set()
    history = []
    top0: Optional[pd.DataFrame] = None
    top_final: Optional[pd.DataFrame] = None
    for r in range(max_rounds + 1):
        t0 = time.perf_counter()
        df = score_candidates(schema, cands, data, use_interventions=(r > 0))
        score_time = time.perf_counter() - t0
        if r == 0:
            top0 = df.head(30).copy()
        top_final = df.head(30).copy()
        tr = rank_of_true(df)
        top = df.iloc[0]
        history.append({
            "seed": seed,
            "n_alias": schema.n_alias,
            "n_distractor": schema.n_distractor,
            "n_random": schema.n_random,
            "n_variables": len(schema.names),
            "candidate_cap": candidate_cap,
            "candidate_sampled": len(cands),
            "candidate_space": total_space,
            "passive_n": passive_n,
            "intervention_batch": intervention_batch,
            "round": r,
            "active_rows": len(data.x),
            "true_rank": tr,
            "top_true": bool(top["is_true"]),
            "top_candidate": top["candidate"],
            "top_score": float(top["score"]),
            "true_score": float(df.loc[df["is_true"], "score"].iloc[0]),
            "score_time_sec": score_time,
            "tested_handles": len(already),
        })
        if r == max_rounds:
            break
        op, handle = choose_next_operation(df, already)
        already.add((op, handle))
        new = generate_operation(schema, seed=seed + 10000 + 101 * r + 7 * handle, op_name=op, handle=handle, n=intervention_batch, **world_kwargs)
        data = data.append(new)
        history[-1]["operation"] = op
        history[-1]["handle"] = schema.names[handle]
    return pd.DataFrame(history), top0 if top0 is not None else pd.DataFrame(), top_final if top_final is not None else pd.DataFrame(), len(cands)


def run_scaling(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_hist = []
    exemplar_frames = []
    decoys_list = [int(x) for x in args.alias_counts.split(",") if x.strip()]
    passive_list = [int(x) for x in args.passive_ns.split(",") if x.strip()]
    seeds = list(range(args.seeds))
    world_kwargs = {
        "alias_noise": args.alias_noise,
        "handle_obs_noise": args.handle_obs_noise,
        "sensor_noise": args.sensor_noise,
        "action_noise": args.action_noise,
    }
    total_jobs = len(decoys_list) * len(passive_list) * len(seeds)
    job = 0
    for n_alias in decoys_list:
        # Keep distractors coupled to aliases so dimensionality and false candidate space both grow.
        n_distractor = max(args.min_distractors, n_alias)
        n_random = max(args.min_random, n_alias // 2)
        schema = VariableSchema.make(n_alias=n_alias, n_distractor=n_distractor, n_random=n_random)
        for passive_n in passive_list:
            for seed in seeds:
                job += 1
                if args.verbose:
                    print(f"[{job}/{total_jobs}] aliases={n_alias} passive_n={passive_n} seed={seed}", flush=True)
                hist, top0, topf, _ = run_progressive(
                    schema=schema,
                    seed=seed,
                    passive_n=passive_n,
                    intervention_batch=args.intervention_batch,
                    max_rounds=args.max_rounds,
                    candidate_cap=args.candidate_cap,
                    world_kwargs=world_kwargs,
                )
                if args.verbose:
                    print(f"    finished job {job}: final_rank={int(hist.iloc[-1]['true_rank'])}, success={bool(hist.iloc[-1]['top_true'])}", flush=True)
                all_hist.append(hist)
                if seed == 0 and passive_n == passive_list[0]:
                    tag = f"alias{n_alias}_passive{passive_n}"
                    top0.to_csv(outdir / f"top_passive_{tag}.csv", index=False)
                    topf.to_csv(outdir / f"top_active_round{args.max_rounds}_{tag}.csv", index=False)
                    exemplar_frames.append(hist)
    history = pd.concat(all_hist, ignore_index=True)
    history.to_csv(outdir / "scaling_history.csv", index=False)

    # Aggregate by alias count, passive sample count, round.
    summary = history.groupby(["n_alias", "passive_n", "round"], as_index=False).agg(
        success=("top_true", "mean"),
        median_rank=("true_rank", "median"),
        mean_rank=("true_rank", "mean"),
        median_score_time=("score_time_sec", "median"),
        candidate_space=("candidate_space", "first"),
        candidate_sampled=("candidate_sampled", "first"),
        n_variables=("n_variables", "first"),
        active_rows=("active_rows", "median"),
    )
    summary.to_csv(outdir / "scaling_summary.csv", index=False)

    # Final-round snapshot.
    final = summary[summary["round"] == args.max_rounds].copy()
    final.to_csv(outdir / "scaling_final_round_summary.csv", index=False)

    if plt is not None and not args.no_plots:
        # Success versus intervention budget for each alias count at largest passive_n.
        largest_passive = max(passive_list)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for n_alias in decoys_list:
            sub = summary[(summary["n_alias"] == n_alias) & (summary["passive_n"] == largest_passive)]
            ax.plot(sub["round"], sub["success"], marker="o", label=f"aliases={n_alias}")
        ax.set_xlabel("Active intervention rounds")
        ax.set_ylabel("Exact recovery rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Handle-UAD scaling with false handles (passive_n={largest_passive})")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "success_vs_rounds.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for n_alias in decoys_list:
            sub = summary[(summary["n_alias"] == n_alias) & (summary["passive_n"] == largest_passive)]
            ax.plot(sub["round"], sub["median_rank"], marker="o", label=f"aliases={n_alias}")
        ax.set_xlabel("Active intervention rounds")
        ax.set_ylabel("Median rank of true loop")
        ax.set_title(f"True loop rank under growing false-handle set")
        ax.invert_yaxis()
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "rank_vs_rounds.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        base_round = 0
        end_round = args.max_rounds
        for passive_n in passive_list:
            passive_sub = summary[(summary["passive_n"] == passive_n) & (summary["round"] == base_round)]
            active_sub = summary[(summary["passive_n"] == passive_n) & (summary["round"] == end_round)]
            ax.plot(passive_sub["n_alias"], passive_sub["success"], marker="o", linestyle="--", label=f"plain, T={passive_n}")
            ax.plot(active_sub["n_alias"], active_sub["success"], marker="o", label=f"active r={end_round}, T={passive_n}")
        ax.set_xlabel("Number of passive alias pairs")
        ax.set_ylabel("Exact recovery rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Passive samples alone do not remove false handles")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "success_vs_aliases.png", dpi=180)
        plt.close(fig)

    # README
    readme = f"""# Handle-UAD scaling benchmark\n\nGenerated by `uad_handles/scaling.py`.\n\nCore sweep:\n- alias counts: {decoys_list}\n- passive sample counts: {passive_list}\n- seeds: {args.seeds}\n- max active rounds: {args.max_rounds}\n- intervention batch: {args.intervention_batch}\n- candidate cap: {args.candidate_cap}\n\nImportant files:\n- `scaling_history.csv`: one row per seed/config/round.\n- `scaling_summary.csv`: aggregated success/rank/runtime by alias count, passive sample count, and round.\n- `scaling_final_round_summary.csv`: final active-round snapshot.\n- `success_vs_rounds.png`: recovery versus intervention budget.\n- `rank_vs_rounds.png`: median rank versus intervention budget.\n- `success_vs_aliases.png`: passive versus active recovery as false handles grow.\n\nInterpretation:\nPlain UAD is round 0: passive observational evidence only.\nHandle-UAD is round > 0: the system actively tests the current top candidate's claimed handles.\nFalse alias handles are observationally clean but operationally inert.\n"""
    (outdir / "README.md").write_text(readme)

    # Zip outputs.
    zip_path = outdir.with_suffix(".zip")
    pkg = Path(__file__).resolve().parent
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pkg / "scaling.py", arcname="uad_handles/scaling.py")
        for p in outdir.iterdir():
            if p.is_file():
                zf.write(p, arcname=p.name)
    return history, summary


def main() -> None:
    p = argparse.ArgumentParser()
    from uad_handles import default_outdir

    p.add_argument("--outdir", type=Path, default=default_outdir("scaling"))
    p.add_argument("--alias-counts", default="0,1,2,4,8")
    p.add_argument("--passive-ns", default="300,800")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--intervention-batch", type=int, default=80)
    p.add_argument("--candidate-cap", type=int, default=3000)
    p.add_argument("--min-distractors", type=int, default=2)
    p.add_argument("--min-random", type=int, default=2)
    p.add_argument("--alias-noise", type=float, default=0.0)
    p.add_argument("--handle-obs-noise", type=float, default=0.08)
    p.add_argument("--sensor-noise", type=float, default=0.05)
    p.add_argument("--action-noise", type=float, default=0.04)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    t0 = time.perf_counter()
    history, summary = run_scaling(args)
    elapsed = time.perf_counter() - t0
    print(f"Wrote scaling benchmark to {args.outdir}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("\nFinal-round summary:")
    print(summary[summary["round"] == args.max_rounds].to_string(index=False))
    print("\nRound-0 passive summary:")
    print(summary[summary["round"] == 0].to_string(index=False))


if __name__ == "__main__":
    main()
