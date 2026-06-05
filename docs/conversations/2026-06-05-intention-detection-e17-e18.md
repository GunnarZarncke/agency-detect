# Conversation Summary: Intention Detection — E17, E18, E19

Date: 2026-06-05

Compact record of the intention-detection derivative line. Metrics in
[`docs/EXPERIMENTS.md`](../EXPERIMENTS.md#e17--option-d-homeostatic-regulation-probe-2026-06-05).

## Initial Problem

Agency discovery (E0–E16) answers **where is the agent**; goal/intention inference
was never implemented. Three framings:

1. **Option D:** homeostatic regulation (disturbance rejection).
2. **Option A (EIS):** compression gain on oracle S/A/I clusters (deferred).
3. **Outcome influence:** label critical outcomes and test whether each agent
   defends or **drives** them after controlling for exogenous world.

## Key Decisions

- Ship **Option D (E17)** first; then **outcome-influence (E18)** on sim/physics.
- **Do not z-score** regulation/outcome eval traces.
- **E19:** real-machine dataset with confounds (stressor + schedule-correlated bystander).
- Two outcome scenarios: **global** (system CPU/RAM — the ops worry) vs **owned**
  (per-process baseline — boring but useful control).
- Random-filled RAM allocations, multi-core multi-second burns, 1 Hz sampling.

## E17 / E18 (shipped)

- E17: balance CartPole flagged 100%; track/telemetry/rock/grid not — homeostasis only.
- E18: pooled AUROC **0.941** on sim + physics; complementary to E17.

## E19 — Real machine

### v1 run (T=1800, first harness)

**2/5 accuracy, AUROC 0.333.** Not a training or data-quantity failure (detector is
statistical; T=1800 ample). Diagnosis:

- Agent CPU/RAM effects **below 1 Hz noise floor** (1-core ≈12.5%, 400 MB ≈1.6% of 24 GB).
- Sub-second bursts aliased; stressor dominated global CPU.
- **Defense-only flag logic** missed RAM *drivers* (`infl<0`); bystander false-positive on flat RAM noise.
- Scoring script bug (str/int agent keys); per-agent flag used max-combined outcome only.

### v2 harness + detector fixes

**Harness:** 2.5 GB random-filled RAM (Δ~3 GB global), 2-core full-tick bursts 4–10+ s,
per-process owned channels, `--scenario global|owned|both`.

**Detector (`evaluate.py`):**

- **Driver path:** `|influence| ≥ 0.30` flags regardless of sign.
- **Any-outcome:** flag if any critical outcome flags (not only highest combined).
- E18 regression: AUROC still **0.941**.

**150 s smoke:** global AUROC **1.0**, agent acc **4/5** (regulator miss; no FP on bystander/fixed).

**20-min v2 run:** launched after fixes; results pending in EXPERIMENTS.md.

## Follow-Up

1. Record 20-min v2 results; tune cpu_regulator if still missed.
2. Optional EIS (Option A) on oracle clusters.
3. Wire E18 into spotlight post-UAD; bridge deployment-pipeline traces.
