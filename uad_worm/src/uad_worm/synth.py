"""Synthetic benchmark systems for M0 validation (README §5, §27).

Three generators with known structure, used to confirm the estimator + blanket contrast
before touching worm data:

1. ``controller_with_blanket`` — explicit Markov blanket; the true cut must score low and
   beat random partitions.
2. ``correlated_no_blanket`` — strong shared-latent correlation but NO mediating
   interface; no cut should beat random partitions (method must reject).
3. ``two_agents_one_memory`` — two coupled subsystems with one long-memory node
   (generator only; the memory assertion is deferred with M7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class SyntheticSystem:
    trace: np.ndarray  # (T, V)
    names: List[str]
    roles: Dict[str, List[int]]  # keys: internal, sensor, action, external
    memory_nodes: List[int] = field(default_factory=list)

    @property
    def interface(self) -> List[int]:
        return list(self.roles.get("sensor", [])) + list(self.roles.get("action", []))


def controller_with_blanket(T: int = 3000, seed: int = 0, noise: float = 0.3) -> SyntheticSystem:
    """Internal state I interacts with environment E only through sensor S and action A.

    Columns: I0, I1, S, A, E0, E1. One-step transitions of I and E are conditionally
    independent given (S_t, A_t), so the true cut has blanket loss ≈ 0.
    """
    rng = np.random.default_rng(seed)
    I0 = np.zeros(T); I1 = np.zeros(T)
    E0 = np.zeros(T); E1 = np.zeros(T)
    S = np.zeros(T); A = np.zeros(T)

    def n() -> float:
        return noise * rng.standard_normal()

    # Coefficient rationale. SELF = 0.6: each node's own decay rate; <1 so a node forgets
    # (sets the intrinsic timescale). READ = 0.2/0.5/0.3: how strongly S/A/I read their
    # inputs. The danger is the feedback loop I→A→E→S→I: its gain is the *product* of the
    # per-edge gains, so small READ values keep the closed-loop spectral radius < 1 (here
    # 0.755, verified by eigenvalues of the 4-state transition matrix). Larger values
    # (e.g. the original 0.7 SELF + 0.3 READ) push it > 1 and the trace explodes, making
    # every variable collinear and the CMI partition-invariant.
    SELF, READ_S, READ_A, READ_I = 0.6, 0.5, 0.3, 0.2
    for t in range(T):
        S[t] = READ_S * E0[t] + READ_S * E1[t] + n()  # sensor reads current environment
        A[t] = READ_A * (I0[t] + I1[t]) + n()         # action reflects internal state
        if t + 1 < T:
            E0[t + 1] = SELF * E0[t] + READ_I * A[t] + n()
            E1[t + 1] = SELF * E1[t] + READ_I * A[t] + n()
            I0[t + 1] = SELF * I0[t] + READ_I * S[t] + n()
            I1[t + 1] = SELF * I1[t] + READ_I * S[t] + n()

    trace = np.column_stack([I0, I1, S, A, E0, E1])
    roles = {"internal": [0, 1], "sensor": [2], "action": [3], "external": [4, 5]}
    names = ["I0", "I1", "S", "A", "E0", "E1"]
    return SyntheticSystem(trace=trace, names=names, roles=roles)


def correlated_no_blanket(T: int = 3000, seed: int = 0, n_vars: int = 6, noise: float = 0.5) -> SyntheticSystem:
    """All variables share a slow latent z; no variable mediates between any two others.

    Because the shared innovation of z is injected into every variable at t+1, every cut
    retains residual internal/external dependence given any interface — no partition is a
    blanket.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    # 0.8 = latent persistence: slow enough to give strong autocorrelation/cross-correlation
    # (the confound we must survive) while staying < 1 for stationarity.
    for t in range(1, T):
        z[t] = 0.8 * z[t - 1] + rng.standard_normal()
    # Each variable is the shared latent plus independent observation noise; no variable
    # mediates between others, so there is no Markov blanket to find.
    cols = [z + noise * rng.standard_normal(T) for _ in range(n_vars)]
    trace = np.column_stack(cols)
    half = n_vars // 2
    roles = {
        "internal": list(range(0, half - 1)) or [0],
        "sensor": [half - 1],
        "action": [half],
        "external": list(range(half + 1, n_vars)),
    }
    names = [f"x{i}" for i in range(n_vars)]
    return SyntheticSystem(trace=trace, names=names, roles=roles)


def two_agents_one_memory(T: int = 3000, seed: int = 0, noise: float = 0.3) -> SyntheticSystem:
    """Two blanket subsystems coupled through a shared environment; agent A holds a
    long-timescale memory node (high self-regression). Generator only for M0."""
    rng = np.random.default_rng(seed)
    # Agent A: IA0 (memory, slow), IA1; sensor SA, action AA.
    # Agent B: IB0, IB1; sensor SB, action AB. Shared environment E0, E1.
    IA0 = np.zeros(T); IA1 = np.zeros(T)
    IB0 = np.zeros(T); IB1 = np.zeros(T)
    E0 = np.zeros(T); E1 = np.zeros(T)
    SA = np.zeros(T); AA = np.zeros(T)
    SB = np.zeros(T); AB = np.zeros(T)

    def n() -> float:
        return noise * rng.standard_normal()

    # Same stability logic as controller_with_blanket (small READ gains keep the loop
    # bounded). One node, IA0, uses a near-unit self-coefficient 0.95 to plant a genuinely
    # long-timescale memory; the cross-agent env coupling is split 0.15 (own action) /
    # 0.05 (other agent's action) so the two agents are coupled but distinct. Spectral
    # radius is 0.954 — stable, dominated by the 0.95 memory mode (the long timescale we
    # want M7 to recover).
    MEM_SELF, SELF, READ_S, READ_A, READ_I = 0.95, 0.6, 0.7, 0.3, 0.2
    COUPLE_OWN, COUPLE_OTHER, ENV_SELF = 0.15, 0.05, 0.5
    for t in range(T):
        SA[t] = READ_S * E0[t] + n()
        SB[t] = READ_S * E1[t] + n()
        AA[t] = READ_A * (IA0[t] + IA1[t]) + n()
        AB[t] = READ_A * (IB0[t] + IB1[t]) + n()
        if t + 1 < T:
            E0[t + 1] = ENV_SELF * E0[t] + COUPLE_OWN * AA[t] + COUPLE_OTHER * AB[t] + n()
            E1[t + 1] = ENV_SELF * E1[t] + COUPLE_OWN * AB[t] + COUPLE_OTHER * AA[t] + n()
            IA0[t + 1] = MEM_SELF * IA0[t] + 0.05 * SA[t] + n()  # long-memory node
            IA1[t + 1] = SELF * IA1[t] + READ_I * SA[t] + n()
            IB0[t + 1] = SELF * IB0[t] + READ_I * SB[t] + n()
            IB1[t + 1] = SELF * IB1[t] + READ_I * SB[t] + n()

    trace = np.column_stack([IA0, IA1, SA, AA, IB0, IB1, SB, AB, E0, E1])
    names = ["IA0", "IA1", "SA", "AA", "IB0", "IB1", "SB", "AB", "E0", "E1"]
    roles = {
        "internal": [0, 1, 4, 5],
        "sensor": [2, 6],
        "action": [3, 7],
        "external": [8, 9],
    }
    return SyntheticSystem(trace=trace, names=names, roles=roles, memory_nodes=[0])
