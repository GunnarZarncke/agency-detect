# Conversation Summary: Amortized Agency Detection — Baseline (E13)

Date: 2026-06-01

Compact record of the pivot toward transient-agent detection via amortization,
and the MI window breaking-point baseline that anchors it. Detailed metrics live
in
[`learn_agents/EXPERIMENTS.md`](../../learn_agents/EXPERIMENTS.md#e13--amortized-agency-detection-mi-window-breaking-point-baseline-2026-06-01).

## Initial Problem

After E12b confirmed fixed-coordinate complex agents are reliably discovered, the
open question became *where to push next*. The driving target is the **short
duration** of real transient agents (the "10-second proto-agent" demo): an agent
that exists only briefly. Every current estimator — CMI blanket validation,
lagged-MI clustering, IRL goal inference (UAD paper §III.C) — is a property of a
stationary window, so it needs many samples and averages a transient agent away.
Goal inference is the most sample-hungry rung, so short duration hurts teleology
most.

## Key Decisions

- **Amortization is the chosen route to short-duration detection:** train one
  agency detector over a pool of many varied (short- and long-lived) agents, then
  apply to new traces without relearning each agent (the human analogy).
- **Held-out-of-kinds** is the transfer test (train on some agent kinds, evaluate
  on unseen kinds), not just held-out seeds.
- **Baseline before model:** locate where the existing MI method breaks vs window
  length `W` first; that curve quantifies the bar the learned model must beat.
- **Common comparison artifact:** every method (MI, Siamese, set/slot) emits the
  same permutation-invariant `N×N` same-agent affinity → identical downstream
  clustering. Slot index is never compared across worlds.
- Interventions for passive-failure cases are handled in a separate project; not
  in scope here.

## Experiment Progression

### MI window breaking-point baseline

`scripts/amortized/baseline_window_breaking_point.py` slices the first `W` steps
of a trace and runs the repo's real proposal step (`mi_cluster_variable_labels`)
on agent variables only, scoring ARI / Jaccard vs ground-truth agent ids. Easy→
hard kinds, 5 seeds, `W ∈ {2000…60}`.

Result: ARI = 1.0 down to ~`W=250`, sharp collapse between `W=250` and `W=125`
(ARI ~0.73), near-chance (~0.5) at `W=60`. The breaking point ~`W≈125` is
**independent of agent kind** — the limit is statistical power, not complexity.

## Current State

- The amortized line is opened (E13) and reprioritized ahead of moving agents.
- Quantified target band for the learned model: `W ∈ [60, 250]`.
- Baseline script + artifacts under `results/amortized/`.

## Follow-Up Ideas

- Build the learned same-agent affinity model: **Siamese pairwise** floor, then a
  **context-aware Set-Transformer / slot-attention** model (pairwise scoring
  ignores the conditional nature of Markov-blanket agency, so it has a
  generalization ceiling).
- Fold decoy/world rejection back in (baseline currently isolates agent vars).
- Add episodic short-episode worlds (on/off gaps) on top of the pure window axis.
- Later line: moving/non-stationary agents with identity invariant over local
  charts.
