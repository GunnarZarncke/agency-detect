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
- Two outcome scenarios: **global** (system CPU/RAM) vs **owned** (per-process baseline).
- Random-filled RAM, multi-core multi-second burns, 1 Hz sampling.
- **Segmentation (E19c):** auto window + activity calibration for long episodic traces.

## E17 / E18 (shipped)

- E17: balance CartPole flagged 100%; track/telemetry/rock/grid not.
- E18: pooled AUROC **0.941** on sim + physics.
- Follow-ups: driver path (`|infl|≥0.30`), any-outcome aggregation, auto segmentation — sim AUROC unchanged.

## E19 — Real machine

| Phase | Result |
|-------|--------|
| **E19a v1** | 2/5 — SNR/aliasing/defense-only semantics |
| **E19b v2** | Smoke global 4/5 AUROC 1.0; 20-min global 3/5, owned 4/5 |
| **E19c segment** | Re-score 20-min: global AUROC 0.33→**0.67**, acc still 3/5 |

**Why longer runs hurt (full-trace):** episodic agents active 12–52% of ticks; pooling idle mass dilutes partial influence — not a data-quantity problem.

**What works:** mem_grabber on global RAM; confound negatives (fixed_worker, bystander on strict segment rules); owned per-process attribution.

**Still open:** global CPU for regulator/burster (best segment `|infl|≈0.07` &lt; 0.25).

## Follow-Up

1. Segment-relative influence floor or activity-only windows for CPU bursts.
2. Continuous stressor control channel.
3. Optional EIS (Option A); wire E18 into spotlight post-UAD.
