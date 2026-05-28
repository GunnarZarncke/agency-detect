# Changelog

Chronological, high-level record of repo and experiment milestones.

For detailed hypotheses, failures, reasoning, and tables, see
[`learn_agents/EXPERIMENTS.md`](../learn_agents/EXPERIMENTS.md). For compact
conversation-level decisions and rationale, see [`docs/conversations/README.md`](conversations/README.md).

---

## 2026-05-26 — Validation And Estimator Cleanup

- Verified existing adaptive/threshold/MI tests and committed the go/no-go sweep script.
- Removed stale debug/scratch files and kept the repo focused on `agency_detect`, `learn_agents`, and experiment scripts.
- Added CMI estimator research under `scripts/research/cmi/`; conclusion: k-NN CMI was poorly matched to discretized traces, while Laplace-smoothed discrete plug-in CMI is the better production path.
- Recalibration note: current `DetectionConfig.BLANKET_TOLERANCE=1.0` is stricter than the old k-NN-era guidance and should be recalibrated on oracle clusters.

---

## 2026-05-27 — Serial Spotlight Line (E9-E11)

See [`learn_agents/EXPERIMENTS.md#e9--serial-spotlight-agent_spotlight`](../learn_agents/EXPERIMENTS.md#e9--serial-spotlight-agent_spotlight).

- E8 showed that global latent slots mixed agents before UAD validation could help; E9 introduced serial spotlight: propose one MI cluster, refine locally, validate, peel, repeat.
- E9a isolated two bottlenecks: binary precursor scoring picked decoys, then `spotlight_slot` expanded good MI proposals into huge mixed candidates.
- E9b switched to `mi_cluster` candidates and recovered 5/8 agents, confirming that slot-to-raw mapping was the main blocker.
- E9c/E9d corrected the simulator ontology: action-coupled `env{k}` behaved like extended agent body, so shared exogenous `world.shared*` became the cleaner world benchmark.
- E10 diagnosed the remaining 7/8 miss as partial-peel orphaning, not failure to identify the agent. K=24 became the data-only 8/8 default.
- E11 added rich fixed-coordinate agents with delayed and nonlinear role variants. Spotlight still recovered all agents with more passes, but discoveries became role/subrole chunks rather than whole agents.

Key packages and scripts:

- `agent_spotlight/`
- `scripts/spotlight/run_spotlight_e9a.py`
- `scripts/spotlight/run_spotlight_sweeps.py`
- `scripts/spotlight/run_spotlight_recovery_sweep.py`

---

## 2026-05-28 — Hierarchical Spotlight (E12)

See [`hierarchical_spotlight/README.md`](../hierarchical_spotlight/README.md) and [`learn_agents/EXPERIMENTS.md#e12--hierarchical-chunk-fusion-2026-05-28`](../learn_agents/EXPERIMENTS.md#e12--hierarchical-chunk-fusion-2026-05-28).

- Added `hierarchical_spotlight/` as a second-stage experiment: spotlight chunks become graph nodes; compatible chunks are fused into higher-level agent hypotheses.
- Added timestamped JSON/DOT/PNG Graphviz output under `results/hierarchical/e12/`.
- Split graph metrics into permissive graph coverage and purity-aware clean coverage.
- Tightened default fusion threshold (`min_cross_mi=0.70`) after the initial permissive graph over-merged all chunks into one component.
- Reorganized experiment scripts and artifacts into family subfolders:
  - `scripts/learn_agents/`, `scripts/decoys/`, `scripts/spotlight/`
  - `results/learn_agents/`, `results/decoys/`, `results/spotlight/`, `results/hierarchical/`

Current direction: moving/non-stationary agents will require identity as an invariant over local charts, rather than a fixed raw variable subset.

