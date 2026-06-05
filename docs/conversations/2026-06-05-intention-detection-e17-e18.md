# Conversation Summary: Intention Detection — E17, E18, E19

Date: 2026-06-05

Compact record of starting the intention-detection derivative line. Metrics in
[`docs/EXPERIMENTS.md`](../EXPERIMENTS.md#e17--option-d-homeostatic-regulation-probe-2026-06-05).

## Initial Problem

Agency discovery (E0–E16) answers **where is the agent**; the UAD paper’s fourth
pillar — **goal/intention inference** — was never implemented. Two complementary
framings emerged from the UAD / EIS papers and the deployment-pipeline auditor:

1. **Option D:** homeostatic regulation (disturbance rejection, suppressed internal).
2. **Option A (EIS):** compression gain — goal-rational prior over actions explains
   behavior better than a purely mechanistic baseline (`Δℒ > 0`).
3. **Outcome influence (deployment-pipeline style):** label critical outcomes (CPU,
   RAM, pole angle) and test whether each agent **defends or steers** them after
   controlling for exogenous world state.

## Key Decisions

- **Ship Option D first** as the smallest falsifiable probe; scope is explicitly
  *maintenance/regulation*, not pursuit or navigation.
- **`physics_cartpole_track`** added as intentional contrast to balance (same S/A/I,
  active internal around `θ_ref=0.12`, not suppressed).
- **Do not z-score traces** used for regulation or outcome-influence eval
  (`normalize=False` / `normalize_trace=False`); per-column z-score destroys
  variance ratios and defense ORs.
- **Active-internal gate (E17):** zero regulation score when
  `Var(internal)/Var(sensor) > 0.012` so pursuit/track is not misclassified as
  homeostasis.
- **E18 pivot:** implement **outcome-influence** first (partial lagged influence +
  defense OR + selectivity), not full EIS compression gain on spotlight clusters.
  Oracle S/A/I clusters / sim metadata suffice for v1.
- **E19:** real-machine dataset with confounded shell processes (stressor schedule,
  bystander correlates with W but does not burn CPU) to stress-test E18 off-simulator.

## E17 — Option D (shipped)

Regulation probe: flatness × compensation on paired S/A/I; flags homeostatic agents
only when internal variance is suppressed relative to sensor drive.

**Result:** balance CartPole flagged 100% (mean R≈0.66); track, telemetry, rock, grid
not flagged. Validates negative controls and positive homeostatic control.

## E18 — Outcome influence (shipped)

New package `intention_detect/`:

- Partial influence \(A \to \Delta O \mid W\)
- Defense OR + bootstrap CI + selectivity ratio
- Dual-path flag (resource defense vs control/influence)

Sim extensions: `resource.cpu` / `resource.memory`, optional `self_preserving_agent`.
Physics eval via `attach_physics_critical_outcome`.

**Result:** pooled AUROC **0.941** (n=40); agent-level accuracy 14/15 reactive,
13/15 self-preserving, 5/5 CartPole balance/track.

E17 and E18 are complementary: E17 = internal homeostasis; E18 = outcome-directed action.

## E19 — Real-machine harness (in progress)

**Motivation:** E18 worked on sim and physics; next falsifier is real CPU/RAM traces
with operator-style confounds (shared stressor schedule, correlated non-causal logger).

**Layout:**

| Process | Role | Ground truth |
|---------|------|--------------|
| background stressor | exogenous W (2-core busy-loop, ~45% duty) | world control |
| cpu_regulator | homeostatic CPU → setpoint | influencer |
| deadline_burster | goal bursts, ignores load | influencer |
| mem_grabber | pushes RAM | influencer |
| fixed_worker | regular, no influence | negative |
| bystander | reads stressor schedule, disk I/O only | negative (confound) |

**Package:** `data_collect/` + `scripts/intention/run_machine_dataset.py`.

**Run budget:** up to **4 CPU cores**, **30 min** (T=1800, dt=1s). Requires
`.venv` with `psutil` (added to `requirements-dev.txt`).

**Status at commit:** harness built and smoke-tested (30 s); full 30-min collection
launched in background. Scoring needs T≥80 (same as E17/E18).

## Follow-Up

1. Finish E19 30-min run; record AUROC / per-agent accuracy in EXPERIMENTS.md.
2. Optional: EIS compression gain (original Option A) on oracle clusters.
3. Wire E18 into spotlight post-UAD admit; bridge deployment-pipeline traces.
