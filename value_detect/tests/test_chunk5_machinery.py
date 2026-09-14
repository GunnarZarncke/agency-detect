"""Tests for the Chunk 5 machinery: floors engine, variants, signature evaluation.

The critical one is the mirror test: the floor engine's scorer must reproduce the
published convention scorers exactly, or floors would not gate what we report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import value_detect as vd
from value_detect.criteria import signature_flags
from value_detect.floors import ConventionScorer, shift_null_floors


def _small_world_frame(seed=0, n=4000):
    return vd.passive_trace(seed=seed, n_steps=n).frame


def test_convention_scorer_mirrors_published_scorers():
    frame = _small_world_frame()
    for convention, ref_fn in [
        ("pairwise", vd.score_trace),
        ("fused", vd.score_trace_fused),
        ("fused_bestkey", vd.score_trace_fused_bestkey),
    ]:
        ref = ref_fn(frame, env_var="E")
        got = ConventionScorer(frame, convention, env_var="E").score_all()
        for v in frame.columns:
            for col in ("push_in", "out_sys", "out_env", "polarity_sys"):
                a, b = ref.loc[v, col], got.loc[v, col]
                if np.isnan(a) and np.isnan(b):
                    continue
                assert abs(a - b) < 1e-9, (convention, v, col, a, b)


def test_nocore_world_disconnects_goal():
    # In the no-core world the action line equals belief (up to 4% action noise),
    # regardless of the goal.
    tr = vd.passive_trace_nocore(seed=1, n_steps=6000)
    f = tr.frame
    disagree = np.mean(f["A_alias"].to_numpy() != f["B"].to_numpy())
    assert disagree < 0.07, disagree  # ~4% designed
    # And G still flips at its own designed rate.
    g = f["G"].to_numpy()
    rate = np.mean(g[1:] != g[:-1])
    assert 0.003 < rate < 0.05


def test_scramble_kills_directed_flow():
    frame = _small_world_frame(seed=2, n=6000)
    scr = vd.scramble_frame(frame, seed=3)
    # The strongest genuine flow in the world dies under scrambling.
    strong = vd.transfer_entropy(frame["A_alias"].to_numpy(), frame["E"].to_numpy())
    dead = vd.transfer_entropy(scr["A_alias"].to_numpy(), scr["E"].to_numpy())
    assert strong > 0.2
    assert dead < 0.01


def test_floors_gate_null_variable_and_pass_real_signal():
    frame = _small_world_frame(seed=4, n=6000)
    scorer = ConventionScorer(frame, "pairwise", env_var="E")
    scores = scorer.score_all()
    floors = shift_null_floors(scorer, n_shifts=40, seed=5)
    # W's drive must NOT clear its own floor; the action wire's drive must clear its own.
    assert not scores.loc["W", "out_sys"] > floors.loc["W", "out_sys_p95"]
    assert scores.loc["A_alias", "out_sys"] > floors.loc["A_alias", "out_sys_p95"]
    # Percentile ordering sane.
    assert floors.loc["W", "out_sys_p995"] >= floors.loc["W", "out_sys_p95"]


def test_signature_logic():
    scores = pd.DataFrame({
        "push_in": [0.001, 0.5, 0.05],
        "out_sys": [0.5, 0.6, 0.5],
        "out_env": [0.2, 0.1, 0.1],
        "polarity_sys": [0.99, 0.09, 0.82],
        "total_flow": [0.501, 1.1, 0.55],
    }, index=["G_like", "mediator", "leaky_value"])
    floors = pd.DataFrame({
        "push_in_p95": [0.004, 0.004, 0.004],
        "out_sys_p95": [0.004, 0.004, 0.004],
        "out_env_p95": [0.004, 0.004, 0.004],
        "total_flow_p95": [0.008, 0.008, 0.008],
    }, index=["G_like", "mediator", "leaky_value"])
    sig = signature_flags(scores, floors)
    assert bool(sig.loc["G_like", "signature"])          # intake at floor
    assert not bool(sig.loc["mediator", "signature"])    # intake above floor, ratio 1.2x
    assert bool(sig.loc["leaky_value", "signature"])     # intake above floor but 10x ratio


# ---------- V2-1: floor statistics, surrogate nulls, z-gates, calibration world ----------

def test_null_floors_emit_statistics_and_both_samplers_work():
    frame = _small_world_frame(seed=6, n=4000)
    scorer = ConventionScorer(frame, "pairwise", env_var="E")
    for sampler in ("roll", "transition"):
        fl = vd.null_floors(scorer, n_shifts=15, seed=7, sampler=sampler, variables=["G", "W"])
        for col in ("out_sys_p95", "out_sys_mean", "out_sys_sd", "total_flow_sd"):
            assert col in fl.columns, (sampler, col)
        assert (fl["out_sys_sd"] >= 0).all()


def test_transition_surrogate_preserves_dynamics_and_kills_dependence():
    frame = _small_world_frame(seed=8, n=6000)
    scorer = ConventionScorer(frame, "pairwise", env_var="E")
    from value_detect.floors import _transition_sampler
    import numpy as _np
    rng = _np.random.default_rng(9)
    gen = _transition_sampler(scorer, "G", rng)
    surr = next(gen)
    real = scorer.labels["G"]
    rate_real = _np.mean(real[1:] != real[:-1])
    rate_surr = _np.mean(surr[1:] != surr[:-1])
    # Dynamics preserved in distribution (change rate within a factor ~2 for slow G).
    assert 0.3 * rate_real < rate_surr < 3.0 * rate_real + 0.01
    # Real cross-dependence destroyed: the surrogate of A_alias must not drive E.
    genA = _transition_sampler(scorer, "A_alias", _np.random.default_rng(10))
    surrA = next(genA)
    assert vd.transfer_entropy(surrA, frame["E"].to_numpy()) < 0.01


def test_z_gate_blocks_hair_thin_residuals():
    scores = pd.DataFrame({
        "push_in": [0.0455], "out_sys": [0.257], "out_env": [0.0],
        "polarity_sys": [0.7], "total_flow": [0.303],
    }, index=["frozen_meter"])
    floors = pd.DataFrame({
        "push_in_p95": [0.045], "out_sys_p95": [0.249], "out_env_p95": [0.01],
        "total_flow_p95": [0.295],
        "push_in_mean": [0.040], "push_in_sd": [0.003],
        "out_sys_mean": [0.240], "out_sys_sd": [0.006],   # score is +2.8 sd
        "total_flow_mean": [0.280], "total_flow_sd": [0.008],
    }, index=["frozen_meter"])
    from value_detect.criteria import signature_flags
    v1_style = signature_flags(scores, floors)                 # v1 rules: steals
    v2_style = signature_flags(scores, floors, z_min=3.0)      # z-gate: blocked
    assert bool(v1_style.loc["frozen_meter", "signature"])
    assert not bool(v2_style.loc["frozen_meter", "signature"])


def test_calibration_world_ground_truth_shapes():
    frame = vd.calibration_frame(seed=3, n_steps=12000)
    g = frame["G"].to_numpy()
    m = frame["M_slow"].to_numpy()
    f = frame["F_frozen"].to_numpy()
    g_rate = np.mean(g[1:] != g[:-1])
    m_rate = np.mean(m[1:] != m[:-1])
    f_transitions = int(np.sum(f[1:] != f[:-1]))
    assert 0.001 < g_rate < 0.02, g_rate          # slower genuine value, still alive
    assert 0.001 < m_rate < 0.05, m_rate          # slow moving meter
    assert f_transitions <= 1, f_transitions      # near-frozen: at most one flip


# ---------- V2-3: agent-block machinery ----------

def _xor3_frame(seed=21, n=20000):
    rng = np.random.default_rng(seed)
    B1 = rng.integers(0, 2, n); B2 = rng.integers(0, 2, n)
    G = np.empty(n, dtype=int); A = np.empty(n, dtype=int); E = np.empty(n, dtype=int)
    G[0], A[0], E[0] = rng.integers(0, 2, 3)
    for t in range(n - 1):
        A[t + 1] = B1[t] ^ B2[t] ^ G[t] ^ (rng.random() < 0.04)
        E[t + 1] = E[t] ^ A[t] ^ (rng.random() < 0.03)
        G[t + 1] = G[t] ^ (rng.random() < 0.015)
    return pd.DataFrame({"B1": B1, "B2": B2, "G": G, "A": A, "E": E})


def test_swept_detection_recovers_body_and_orphans_G():
    # Gunnar's adaptive-as-is fragments small worlds (starts at his 8-agent config dial
    # and stops at first success); the swept selector — his detector at every dial, the
    # dial chosen by his own blanket-validity coverage (his E7/P1 selection philosophy)
    # — recovers the sensible partition.
    frame = vd.passive_trace(seed=0, n_steps=8000).frame
    part = vd.detect_blocks_swept(frame)
    body = max(part["agents"], key=len) if part["agents"] else []
    # The coupled body must be recovered as one valid agent, and G must NOT be inside
    # it (env-bucket vs orphan assignment is run-length-dependent and both are benign:
    # candidates are always scored individually; assignment only shifts G's ring).
    assert {"B", "S", "A", "E"}.issubset(set(body))
    assert "G" not in body
    assert "G" in part["orphans"] + part["env"]


def test_compress_block_budget_and_lost_mass():
    frame = vd.passive_trace(seed=1, n_steps=6000).frame
    lab, lost = vd.compress_block(frame, ["B", "S", "A", "E", "S_alias", "A_alias", "D"], budget=16)
    assert lab.max() < 16
    assert 0.0 <= lost < 0.5


def test_keyring_cracks_parity_for_a_member_goal():
    # Manual partition: G is a MEMBER of its agent {B1, B2, G, A}; both beliefs sit in
    # its ring, so the within-block parity decrypts; own-block-as-target carries G's
    # grip on A; max(ring, no-ring) keeps mediation safety.
    frame = _xor3_frame()
    part = {"agents": [["B1", "B2", "G", "A"]], "env": ["E"], "orphans": []}
    from value_detect.agentblocks import BlockScorer
    t = BlockScorer(frame, "keyring", env_var="E", partition=part).score_all()
    # Mechanism asserts: parity cracked; G near-pure driver; G above its agent-mates
    # that have any intake path (A). B1/B2 are iid coins in this toy (no inputs), so
    # they are genuinely uncaused drivers too — global rank is not asserted here; the
    # properly-audited deep-synergy world (V2-4) owns ecology-level claims.
    assert t.loc["G", "out_sys"] > 0.3, t.loc["G", "out_sys"]
    assert t.loc["G", "push_in"] < 0.03
    assert t.loc["G", "polarity_sys"] > 0.9
    assert t.loc["G", "polarity_sys"] > t.loc["A", "polarity_sys"]


def test_menu_architecture_also_cracks_parity_via_own_block_key():
    frame = _xor3_frame(seed=22)
    part = {"agents": [["B1", "B2", "G", "A"]], "env": ["E"], "orphans": []}
    from value_detect.agentblocks import BlockScorer
    t = BlockScorer(frame, "menu", env_var="E", partition=part).score_all()
    assert t.loc["G", "out_sys"] > 0.25, t.loc["G", "out_sys"]
    assert t.loc["G", "polarity_sys"] > 0.9
    assert t.loc["G", "polarity_sys"] > t.loc["A", "polarity_sys"]


def test_block_scorer_works_with_null_floors():
    frame = vd.passive_trace(seed=2, n_steps=5000).frame
    part = {"agents": [["B", "S", "A", "E", "S_alias", "A_alias", "D"]], "env": ["W"], "orphans": ["G"]}
    from value_detect.agentblocks import BlockScorer
    scorer = BlockScorer(frame, "keyring", env_var="E", partition=part)
    fl = vd.null_floors(scorer, n_shifts=10, seed=3, variables=["G", "W"])
    assert "out_sys_p95" in fl.columns and "out_sys_sd" in fl.columns
    t = scorer.score_all()
    # The orphan goal is scored against the body block: parity-free V1 world, G on top.
    assert list(t.index)[0] == "G"


# ---------- V2-4: world mechanics ----------

def test_colony_mechanics_and_coupling():
    f = vd.colony_frame(seed=1, n_steps=8000, coupling=0.1)
    assert f.shape[1] == 49
    for i in (0, 3, 7):
        g = f[f"G{i}"].to_numpy()
        r = np.mean(g[1:] != g[:-1])
        assert 0.005 < r < 0.03, (i, r)
    # Ring coupling is real: neighbour 0's action informs patch 1's next step
    # (conditioned on patch 1's own past and its own action).
    from value_detect.directed import cmi
    import numpy as _np
    E1, A0, A1 = f["E1"].to_numpy(), f["A0"].to_numpy(), f["A1"].to_numpy()
    z = _np.column_stack([E1[:-1], A1[:-1]])
    coupled = cmi(E1[1:], A0[:-1], z)
    assert coupled > 0.005, coupled
    unc = vd.colony_frame(seed=1, n_steps=8000, coupling=0.0)
    E1u, A0u, A1u = unc["E1"].to_numpy(), unc["A0"].to_numpy(), unc["A1"].to_numpy()
    zu = _np.column_stack([E1u[:-1], A1u[:-1]])
    assert cmi(E1u[1:], A0u[:-1], zu) < 0.004


def test_deep_synergy_mechanics():
    f = vd.deep_synergy_frame(seed=2, n_steps=8000)
    # Parity agent: action disagrees with B1^B2^G at ~ the action+obs noise rate.
    pred = f["B1_P"].to_numpy() ^ f["B2_P"].to_numpy() ^ f["G_P"].to_numpy()
    dis = np.mean(f["A_P"].to_numpy() != pred)
    assert 0.05 < dis < 0.16, dis
    # Majority agent likewise.
    tot = f["B1_M"].to_numpy() + f["B2_M"].to_numpy() + f["G_M"].to_numpy()
    predm = (tot >= 2).astype(int)
    dism = np.mean(f["A_M"].to_numpy() != predm)
    assert 0.05 < dism < 0.16, dism
    # Beliefs are DRIVEN here (the toy-world lesson): each has real intake from its channel.
    b1 = f["B1_P"].to_numpy(); e1 = f["E1_P"].to_numpy()
    assert np.mean(b1[1:] != e1[:-1]) < 0.15
    # Channel independence (the 2026-08-12 fix): E1^E2 must move fast, or the
    # three-way parity silently collapses to "goal + slow drift".
    e2 = f["E2_P"].to_numpy()
    x = e1 ^ e2
    assert np.mean(x[1:] != x[:-1]) > 0.35, np.mean(x[1:] != x[:-1])


def test_slow_meter_world_shapes():
    f = vd.slow_meter_frame(seed=3, n_steps=12000)
    g = f["G"].to_numpy()
    assert 0.001 < np.mean(g[1:] != g[:-1]) < 0.02
    rates = {m: np.mean(f[m].to_numpy()[1:] != f[m].to_numpy()[:-1]) for m in ("M_fast", "M8", "M32")}
    assert rates["M_fast"] > 0.5                     # fast integrator moves constantly
    # Two distinct slow timescales (realized ~1.25% and ~0.23%; window size does not
    # order them — streak-reset vs windowed-majority mechanics differ) + near-frozen.
    assert 0.0001 < rates["M8"] < 0.05
    assert 0.0001 < rates["M32"] < 0.05
    assert abs(rates["M8"] - rates["M32"]) > 0.003   # genuinely different scales
    assert int(np.sum(np.abs(np.diff(f["F_sat"].to_numpy())))) <= 3   # near-frozen


def test_v24_nocore_variants_disconnect_goals():
    f = vd.colony_frame(seed=5, n_steps=6000, disconnect_goals=True)
    pred = f["B2"].to_numpy() ^ f["G2"].to_numpy()
    base = f["B2"].to_numpy()
    # With goals disconnected, A tracks B alone (not B^G).
    assert np.mean(f["A2"].to_numpy() != base) < np.mean(f["A2"].to_numpy() != pred)
    d = vd.deep_synergy_frame(seed=5, n_steps=6000, disconnect_goals=True)
    predP = d["B1_P"].to_numpy() ^ d["B2_P"].to_numpy()
    assert np.mean(d["A_P"].to_numpy() != predP) < 0.16
    s = vd.slow_meter_frame(seed=5, n_steps=6000, disconnect_goal=True)
    assert np.mean(s["A_alias"].to_numpy() != s["B"].to_numpy()) < 0.07


# ---------- V3: puppets, aliases, the yardstick ----------

def test_puppet_colonies_mechanics():
    fast = vd.colony_frame(seed=7, n_steps=8000, puppet=("fast", 3))
    g3, e3 = fast["G3"].to_numpy(), fast["E3"].to_numpy()
    # Emergent tempo ~22%: the captured goal copies the same patch the belief copies,
    # so the two partially XOR-cancel in the action and the puppet CALMS its own patch.
    assert np.mean(g3[1:] != g3[:-1]) > 0.15                     # busy copier tempo
    assert np.mean(g3[1:] != e3[:-1]) < 0.06                     # copies own patch (3%)
    slow = vd.colony_frame(seed=7, n_steps=12000, puppet=("slow", 3))
    g3, e3 = slow["G3"].to_numpy(), slow["E3"].to_numpy()
    rate = np.mean(g3[1:] != g3[:-1])
    assert 0.005 < rate < 0.03, rate                             # a true goal's tempo
    # EVERY slow-puppet flip follows a 6-run of its patch (the trigger, verified).
    flips = np.where(g3[1:] != g3[:-1])[0]
    for t in flips[:50]:
        assert t >= 6 and len(set(e3[t - 6:t])) == 1, t
    # Wiring intact in both: A3 still composes belief-and-goal-slot.
    a3, b3 = slow["A3"].to_numpy(), slow["B3"].to_numpy()
    g3s = slow["G3"].to_numpy()
    assert np.mean(a3 != (b3 ^ g3s)) < 0.13
    # True-goal agents unaffected.
    g0 = slow["G0"].to_numpy()
    assert 0.005 < np.mean(g0[1:] != g0[:-1]) < 0.03


def test_alias_colony_mechanics():
    base = vd.colony_frame(seed=8, n_steps=6000)
    al = vd.alias_colony_frame(seed=8, n_steps=6000, alias_noise=0.01)
    # Dynamics untouched: base columns bit-identical.
    for c in base.columns:
        assert (al[c].to_numpy() == base[c].to_numpy()).all(), c
    for i in (2, 5):
        g, ga = al[f"G{i}"].to_numpy(), al[f"G{i}_alias"].to_numpy()
        assert 0.003 < np.mean(g != ga) < 0.03                   # ~1% designed
    z = vd.alias_colony_frame(seed=8, n_steps=6000, alias_noise=0.0)
    assert (z["G2_alias"].to_numpy() == z["G2"].to_numpy()).all()  # the theorem case


def test_yardstick_separates_goal_puppet_alias():
    # True goal: levers behaviour, unlevered by its patch.
    s = vd.yardstick_scores(seed=3, target="G0", n_steps=2000, n_perms=200)
    v = vd.yardstick_verdict(s)
    assert v["role_effect"] and not v["autonomy_effect"] and v["interventional_value"]
    # Fast captured puppet: levers behaviour AND is levered -> refused.
    s = vd.yardstick_scores(seed=3, target="G3", n_steps=2000, n_perms=200,
                            puppet=("fast", 3))
    v = vd.yardstick_verdict(s)
    assert v["role_effect"] and v["autonomy_effect"] and not v["interventional_value"]
    # Alias: a recording levers nothing -> refused.
    s = vd.yardstick_scores(seed=3, target="G2_alias", n_steps=2000, n_perms=200,
                            alias_noise=0.0)
    v = vd.yardstick_verdict(s)
    assert not v["role_effect"] and not v["interventional_value"]
