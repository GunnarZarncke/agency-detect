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

### E12b — complex fixed-agent hierarchy

See [`learn_agents/EXPERIMENTS.md#e12b--complex-fixed-agent-hierarchy-sample-2026-05-28`](../learn_agents/EXPERIMENTS.md#e12b--complex-fixed-agent-hierarchy-sample-2026-05-28).

- Added `agent_variant_mode="complex"` so role channels are heterogeneous views, not lockstep copies.
- Added `scripts/hierarchical/run_hierarchical_e12b_sweep.py` with parallel case execution, MPS device selection, and per-stage timing.
- Six-case sample at 16–20 passes: all runs reached spotlight/graph/clean recall **1.0** with 8–9 components (no giant-component collapse) up to `interaction_strength=0.10`.

Current direction: moving/non-stationary agents will require identity as an invariant over local charts, rather than a fixed raw variable subset.

---

## 2026-06-01 — Amortized Agency Detection: Baseline (E13)

See [`learn_agents/EXPERIMENTS.md#e13--amortized-agency-detection-mi-window-breaking-point-baseline-2026-06-01`](../learn_agents/EXPERIMENTS.md#e13--amortized-agency-detection-mi-window-breaking-point-baseline-2026-06-01) and [`docs/conversations/2026-06-01-amortized-agency-baseline.md`](conversations/2026-06-01-amortized-agency-baseline.md).

- Reprioritized toward **short-duration / transient agents**: the goal is one agency detector trained across a pool of varied agents, applied to new traces without relearning each agent. Moving/non-stationary agents are deferred to a later line.
- Added `scripts/amortized/baseline_window_breaking_point.py`: measures where the existing MI proposal step (`mi_cluster_variable_labels`) loses agent separation as the observation window `W` shrinks, across an easy→hard kind spectrum.
- Result: MI recovery is perfect to ~`W=250`, collapses between `W=250` and `W=125`, near-chance by `W=60` — and the breaking point (~`W≈125`) is **independent of agent kind**. For short windows the bottleneck is sample count, not complexity.
- Sets the quantified target band `W ∈ [60, 250]` for the planned learned same-agent affinity model (Siamese floor → context-aware Set-Transformer / slot attention), evaluated on held-out kinds.

### E13b — pooled Siamese + slot affinity

See [`learn_agents/EXPERIMENTS.md#e13b--pooled-siamese--slot-affinity-2026-06-01`](../learn_agents/EXPERIMENTS.md#e13b--pooled-siamese--slot-affinity-2026-06-01).

- Added `amortized_agency/` package and `scripts/amortized/run_pooled_experiment.py`.
- Trained Siamese (pairwise) and slot-attention (co-assignment) on train kinds; held-out `hard8_complex`.
- Siamese reaches near-parity with MI at `W=125` on held-out kind (ARI 0.63 vs 0.68); beats MI on train kind at `W=60`. Simple slot co-assignment failed (~chance ARI).

### E13c — slot upgrades, train-long / detect-short

See [`learn_agents/EXPERIMENTS.md#e13c--slot-upgrades--train-long--detect-short-2026-06-01`](../learn_agents/EXPERIMENTS.md#e13c--slot-upgrades--train-long--detect-short-2026-06-01).

- Default training windows `{500,1000}`; eval at `{250,125,60}`.
- Slot fixes: correct attention axis, profile-cosine affinity, contrastive/cohesion/recon losses.
- Slot improved from ARI ~0.01 to ~0.14–0.18 at short W (not yet competitive); Siamese still stronger on held-out.

### E13d — stable slot objective + context-aware model

See [`learn_agents/EXPERIMENTS.md#e13d--slot-objective-fixes--context-aware-model-2026-06-01`](../learn_agents/EXPERIMENTS.md#e13d--slot-objective-fixes--context-aware-model-2026-06-01).

- Applied slot fixes #1–5 (canonical slot-competition softmax, unified train/inference affinity, corrected sharpness, vectorized contrastive, optional sampled slots); dropped conflicting reconstruction. Objective now converges stably with no early stopping.
- Diagnostics proved two causes for slot failure: the slot readout cannot express same-agent membership, and the per-channel encoder is relational-blind. BCE-only floors at ln 2 (chance); cross-channel pairwise overfits training (ARI 0.80).
- Added `amortized_agency/context_model.py`: cross-channel attention encoder + direct pairwise affinity. On held-out `hard8_complex` it is the best learned method (W=250 ARI 0.66 vs Siamese 0.25), degrades gracefully at long W, but MI still leads the short-window band (W=125: MI 0.68 vs context 0.54).

### E13e — method-trend sweep across test-time parameters

See [`learn_agents/EXPERIMENTS.md#e13e--method-trend-sweep-across-test-time-parameters-2026-06-01`](../learn_agents/EXPERIMENTS.md#e13e--method-trend-sweep-across-test-time-parameters-2026-06-01).

- Added `scripts/amortized/run_method_sweep.py`: train learned models once, then vary one test-time parameter at a time (window, observation+process noise, agent count) on complex agents. `simulate_episode` gained a config-`overrides` hook.
- **Crossover at W≈70**: MI is monotone in window (chance at W=30, ~0.96 at W=400) and overtakes the learned models above it, but **below ~W=70 the amortized models win** (W=30: context/Siamese ≈0.4 vs MI 0.13) — the transient regime the project targets.
- **MI and context are noise-robust** (flat across a 16× noise increase); **Siamese is noise-fragile** (0.59→0.39). Context exploits extra samples (rises to 0.76 at W=400); Siamese is flat in W.
- **Agent count is a training-distribution effect**: learned models match/beat MI at in-distribution n=3, but fall to ~0.46 at n=8/12 (past the 3/5-agent pool) while MI holds 0.69–0.79. Clear next lever: broaden the pool over agent count.
- **Compute scaling** (`run_compute_scaling.py`): MI is ~O(N²) per trace (47.8 s at N=216 vs ~39 ms for context); learned models are ~linear in N and W (~10–60 ms). MI is flat in W but does not scale for many-channel / many-trace diagnostics; amortization pays once at train time.

### E13f — reference-regime benchmark protocol

- Primary amortized target is **gap-to-MI at W≥250** (E13 MI ceiling), not beating weak MI at short W. Added `amortized_agency/benchmark.py`, `run_reference_benchmark.py`, and `--mode reference|trends|both` on `run_method_sweep.py`.

### E13g — learned-only sweeps and model scales

- Routine benchmarks skip per-trace MI; `benchmark.MI_REFERENCE_ARI` supplies frozen gaps (hard8 W=250 → 0.964).
- Added `model_presets.py` (`base`/`large`/`xl` context encoders) and `run_learned_sweep.py` to grid scale × train worlds × epochs (with train/eval/infer timing).
- `--run-mi` opt-in on reference/pooled/sweep scripts; default is learned-only.
- **Full sweep (12 configs, ~3.6 h CPU):** best `xl` + 40 worlds + 60 epochs → held-out hard8 W=250 ARI **0.810**, gap **0.154**, infer **148 ms** (MI_ref 0.964). `large` 80/60 close (gap 0.159, 44 ms infer). Results seed-sensitive across runs.

### E14 — telemetry extensions (sim); ClusterData rejected

See [`learn_agents/EXPERIMENTS.md#e14--telemetry-extensions-sim-real-data-anchor-rejected-2026-06-03`](../learn_agents/EXPERIMENTS.md#e14--telemetry-extensions-sim-real-data-anchor-rejected-2026-06-03).

- Sim: periodic driver + heavy-tailed bursts in `TraceSimulationConfig`; MI detectability matrix passes for extensions (`telemetry_extension_detectability.py`).
- **Google ClusterData 2019** trialed as real anchor; rejected (job usage metrics are comoving processes, not blanket-valid agents). Adapter/downloader removed.

### E15 — external POMDP trace families

See [`learn_agents/EXPERIMENTS.md#e15--external-pomdp-trace-families-2026-06-04`](../learn_agents/EXPERIMENTS.md#e15--external-pomdp-trace-families-2026-06-04).

- Added trace loggers: partial CartPole (`physics_pomdp.py`), 5×5 RockSample (`rock_sample.py`), 3×3/5×5 multi-agent grid POMDP (`grid_pomdp.py`) via `external_traces.py`.
- Detectability: `external_pomdp_detectability.py` — physics and RockSample MI ARI 1.0 at available T; grids mixed (stepping stone toward Melting Pot + learned/UAD).
- Roadmap staged: Melting Pot (structured obs) next; D4RL/robotics deferred.

### Agent detectability summary table

- Added `scripts/learn_agents/agent_ari_table.py`: live MI ARI + timing; learned+UAD from cached spotlight JSON only (no batch re-run).
- Documented full family table in `learn_agents/EXPERIMENTS.md` (pool, E14 telemetry, E15 external POMDPs).

