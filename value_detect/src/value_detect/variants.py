"""Control-world variants (built as subclasses/wrappers in OUR package; Gunnar's files untouched).

* :class:`NoCoreHandleWorld` — the C1 control: the goal column still exists and flips by
  its own coin, but the decision no longer consults it (action = belief + noise only).
  A correct instrument must refuse to give G the value signature here.
* :func:`scramble_frame` — the C2 control: every column independently time-shuffled,
  destroying all temporal and cross-variable structure while preserving each column's
  marginal statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uad_handles.minimal import IDX, NO_OP, VAR_NAMES, SyntheticHandleWorld

from .world import BURN_IN, DEFAULT_NOISE, Trace


class NoCoreHandleWorld(SyntheticHandleWorld):
    """Same world, but the goal is disconnected from the policy (A_line = B ^ noise).

    Overrides only the within-step readout; everything else (transition, noise levels,
    the goal's own coin) is inherited unchanged, so the trace stays shape-comparable.
    """

    def _observe_from_state(self, B, E, G, D, W, *, op=(NO_OP, -1)):
        op_type, handle = op
        S_line = E ^ self.bern(self.sensor_noise)
        if op_type == "sensor_flip" and handle == IDX["S"]:
            S_eff = 1 - S_line
        else:
            S_eff = S_line

        if op_type == "goal_flip" and handle == IDX["G"]:
            G_eff = 1 - G
        else:
            G_eff = G

        # THE ONE CHANGE: G no longer enters the action line.
        A_line = B ^ self.bern(self.action_noise)
        S_obs = S_eff ^ self.bern(self.handle_obs_noise)
        A_obs = A_line ^ self.bern(self.handle_obs_noise)
        S_alias = S_eff ^ self.bern(self.alias_noise)
        A_alias = A_line ^ self.bern(self.alias_noise)
        return np.array([B, S_obs, A_obs, E, G_eff, S_alias, A_alias, D, W], dtype=np.int8)


def passive_trace_nocore(seed: int = 0, n_steps: int = 2000) -> Trace:
    """Passive rollout of the no-core world (burn-in matching generate_passive)."""
    world = NoCoreHandleWorld(seed=seed)
    world.rollout(BURN_IN, [(NO_OP, -1)])
    data = world.rollout(n_steps, [(NO_OP, -1)])
    frame = pd.DataFrame(np.asarray(data.x), columns=list(VAR_NAMES)).astype(int)
    return Trace(frame=frame, seed=seed, n_steps=n_steps, burn_in=BURN_IN, noise=dict(DEFAULT_NOISE))


def scramble_frame(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Independently permute each column in time (destroys all real structure)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in frame.columns:
        out[c] = rng.permutation(frame[c].to_numpy())
    return pd.DataFrame(out, columns=list(frame.columns))


def calibration_frame(seed: int = 0, n_steps: int = 20000, flip_rate: float = 0.005) -> pd.DataFrame:
    """V2-1 discriminating-pair world: the V1 loop with a SLOWER genuine value, plus two
    appended meter witnesses (causally inert, derived from the recorded trace — exactly
    the status of `goal_progress` in Gunnar's simulator):

    * G — the planted value, flipping at ``flip_rate`` (default 0.5%, three times slower
      than V1). MUST keep the signature.
    * M_slow — a slow moving meter: a saturating counter that steps up after a run of
      eight consecutive action-1s, down after eight action-0s (~0.8% change rate).
      MUST NOT get the signature.
    * F_frozen — a near-frozen variable: flips 0→1 permanently the first time the
      environment holds 1 for twelve consecutive steps. At most one transition ever —
      the degenerate case that stole C3. MUST NOT get the signature.
    """
    world = SyntheticHandleWorld(seed=seed, goal_flip_rate=flip_rate)
    world.rollout(BURN_IN, [(NO_OP, -1)])
    data = world.rollout(n_steps, [(NO_OP, -1)])
    frame = pd.DataFrame(np.asarray(data.x), columns=list(VAR_NAMES)).astype(int)

    A = frame["A"].to_numpy()
    E = frame["E"].to_numpy()
    n = len(frame)

    m = np.empty(n, dtype=int)
    m[0] = 3
    run_ones = run_zeros = 0
    for t in range(1, n):
        if A[t - 1] == 1:
            run_ones += 1
            run_zeros = 0
        else:
            run_zeros += 1
            run_ones = 0
        step = 1 if run_ones >= 8 else (-1 if run_zeros >= 8 else 0)
        if step:
            run_ones = run_zeros = 0
        m[t] = min(6, max(0, m[t - 1] + step))
    frame["M_slow"] = m

    f = np.zeros(n, dtype=int)
    run_e = 0
    frozen_from = None
    for t in range(n):
        run_e = run_e + 1 if E[t] == 1 else 0
        if run_e >= 12:
            frozen_from = t
            break
    if frozen_from is not None:
        f[frozen_from:] = 1
    frame["F_frozen"] = f
    return frame
