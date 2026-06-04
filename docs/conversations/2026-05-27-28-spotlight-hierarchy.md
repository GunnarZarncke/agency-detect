# Conversation Summary: Spotlight To Hierarchical Agents

Date range: 2026-05-27 to 2026-05-28

This is a compact record of reasoning, decisions, and actions from the long
spotlight/hierarchy development conversation. Detailed experimental results live
in [`docs/EXPERIMENTS.md`](../EXPERIMENTS.md).

## Initial Problem

The earlier global latent-candidate pipeline had a persistent issue: raw MI often
contained agent structure, but the learned global slot representation and
slot-to-raw mapping mixed several agents before strict UAD validation saw a
candidate. The working hypothesis became: do not discover all agents in one
global slot model; instead, discover one high-signal local unit at a time.

## Key Decisions

- Created `agent_spotlight/` for serial one-cluster-at-a-time discovery.
- Treated UAD as validation/falsification, not as the first proposal mechanism.
- Removed name-based S/A/I reasoning from the discovery path; metadata remains
  evaluation-only.
- Replaced action-coupled per-agent env niches with an exogenous shared world
  because action-driven env variables behave like an extended agent loop.
- Left strict agency gates off by default: they identify passive blobs but reject
  legitimate partial agent chunks.
- Removed `peel_full_agent_on_hit` because it used ground-truth `agent_clusters`
  and was therefore an oracle-only cheat.

## Experiment Progression

### E9 Serial Spotlight

Early spotlight failed because binary precursor scoring made failing clusters tie
and decoy blobs won by arbitrary cluster id. Continuous precursor + within-MI
scoring fixed proposal ranking.

The next failure was slot mapping: a good MI proposal became a 47-variable mixed
candidate when mapped through `spotlight_slot`. Switching to `mi_cluster`
candidates bypassed the slot-to-raw bottleneck and recovered 5/8 agents.

### Exogenous World Benchmark

The old `env{k}` variables were partly driven by agent actions. This blurred the
line between environment and agent body. Shared exogenous world variables fixed
the ontology: agents read world state but do not drive it. Spotlight then reached
0.875 recall on the 8-agent benchmark and matched E8 global recall while keeping
cleaner per-pass structure.

### E10 Diagnosis

The remaining 1/8 miss was not failure to identify the missing agent. Diagnostics
showed a clean MI chunk for the missing agent existed early, but cluster-only
peel orphaned its variables before selection. Lowering thresholds or peeling
ground-truth agent variables would have hidden the issue. A small recovery sweep
found `proposal_mi_k=24` recovered 8/8 with data-only peel.

### E11 Rich Fixed Agents

Scaling by more small agents was not the next hard problem. Instead, we added
heterogeneous fixed-coordinate agents: delayed variants and nonlinear add/min/max
role composites. Spotlight recovered all agents with more passes, but each pass
usually found a role/subrole chunk (`J≈3/9`) rather than a whole agent.

### E12 Hierarchical Fusion

The rich-agent result suggested a hierarchy:

```text
raw variables -> local chunks -> fused sub-agent graph -> larger agent hypotheses
```

`hierarchical_spotlight/` starts this second-stage layer. Chunks are graph nodes;
edges are data-only compatibility links. Initial permissive edges over-merged all
chunks into one component, so the default threshold was tightened to produce
agent-like components. Metrics now distinguish:

- permissive graph coverage;
- purity-aware clean graph coverage.

## Current State

- `SpotlightConfig` defaults to the 8/8 fixed-coordinate exogenous benchmark
  (`proposal_mi_k=24`, agency gate off).
- Rich fixed agents are recoverable as chunks with enough passes.
- Hierarchical fusion gives a first graph-level route from chunks to larger
  agent hypotheses.
- Main next research direction: moving/non-stationary agents, where identity
  must be invariant over local charts rather than fixed raw variables.

## Follow-Up Ideas

- Add data-only chunk growth from refine assignment or MI neighborhood expansion.
- Make hierarchical fusion stricter and more structured than pairwise MI
  thresholding.
- Introduce moving/egocentric agents and validate UAD in an aligned local frame.
- Keep `docs/CHANGELOG.md` as the brief chronological index and
  `docs/EXPERIMENTS.md` as the canonical experiment narrative.

