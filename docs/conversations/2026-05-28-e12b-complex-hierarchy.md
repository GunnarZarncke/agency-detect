# Conversation Summary: E12b Complex Fixed-Agent Hierarchy

Date: 2026-05-28

Compact record of the E12b stress test after E12 rich-agent fusion. Detailed
metrics live in
[`docs/EXPERIMENTS.md`](../EXPERIMENTS.md#e12b--complex-fixed-agent-hierarchy-sample-2026-05-28).

## Initial Problem

E12 fusion worked on rich fixed-coordinate agents, but success might depend on
near-redundant role copies that fuse too easily. Before moving/non-stationary
agents, we needed a fixed-coordinate benchmark where within-agent channels are
heterogeneous and weak inter-agent coupling is present.

## Key Decisions

- Added `agent_variant_mode="complex"`: every role channel is a transformed view,
  not a lockstep copy of a base signal.
- E12b sweeps weak ring coupling via `interaction_strength` and
  `mixing_strength` while keeping hierarchy rules fixed (`min_cross_mi=0.70`).
- Rejected pass counts below agent/chunk needs; 16+ passes required for valid
  hierarchy testing.
- Added parallel sweep driver with MPS device support and per-stage timing
  (proposal on CPU dominates; pretrain/refine use GPU).

## Experiment Progression

### E12b sample design

Six cases varying `proposal_mi_k`, `max_passes`, `interaction_strength`, and
`mixing_strength`. Each run: spotlight peel → hierarchical fusion → graph metrics.

### Invalid early run

A 6-pass fast run capped below meaningful chunk coverage (6/8 agents). Useful
only for pipeline timing, not hierarchy evaluation.

### Valid six-case parallel run

All six cases at 16–20 passes with `--fast --device mps --jobs 6`:

- spotlight, graph, and clean recall **1.0** on every case;
- 8–9 components, 3–7 fusion edges;
- no collapse into one giant component even at `interaction_strength=0.10`.

## Current State

- Complex fixed-coordinate agents are discoverable as chunks and fuse into
  agent-level components under weak neighbor coupling.
- Sweep infrastructure supports `--device`, `--jobs`, `--run-id`, and stage
  timing for future benchmarks.
- Next hard case remains moving/non-stationary agents with identity invariant
  over local charts.

## Follow-Up Ideas

- Speed up or parallelize MI proposal (largest CPU stage).
- Run one full-scale case (T=4000, full train epochs) as a non-smoke confirmation.
- Begin E13 moving-agent simulator with egocentric/local chart identity.
