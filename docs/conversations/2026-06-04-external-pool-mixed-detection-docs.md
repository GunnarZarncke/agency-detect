# Conversation Summary: E16 transfer, mixed detection, documentation (2026-06-04)

Date: 2026-06-04

Session bridging E15 external traces to amortized transfer benchmarks, diagnosing
weak physics baselines, running heterogeneous mixed detection, and reorganizing
repo documentation. Metrics and tables:
[`docs/EXPERIMENTS.md`](../EXPERIMENTS.md#e14e16--2026-06-04-summary-telemetry-externals-detectability-transfer).

## Initial Problem

After E15 external POMDP loggers (CartPole, RockSample, grid) and a full MI +
learned+UAD agent table, three gaps remained:

1. **Amortized context transfer** — does a sim-trained encoder generalize to
   external traces at `W=250`?
2. **Physics “success”** — transfer ARI 1.0 on CartPole/RockSample looked strong
   but might be trivial (single agent, decoys stripped from encoder input).
3. **Documentation** — experiment log lived under `learn_agents/` though it
   covered spotlight, hierarchy, and amortized lines; interpretation doc was
   named `learn_agents_debug_findings.md`.

Grid transfer failure was explicitly **deferred** after a simplest ablation;
episodic telemetry limitations were not reopened.

## Key Decisions

- **Treat single-agent physics/rock ctx=1.0 as weak evidence** — n=1 clustering
  is degenerate; amortized training uses agent columns only (env decoys in full
  trace but not in encoder windows).
- **Extended pool train** — `EXTENDED_TRAIN_KINDS` = sim easy3/med5 + physics +
  rock + grid3; grid5 held out; Melting Pot scaffold only.
- **CartPole is not a good co-movement-MI probe** — three identical poles lump;
  heterogeneous mix (pole + rock + grid) is the right test design; pole failures
  are structural (transient under random policy; regulated-flat under balance),
  not fixed by padding alone.
- **Grid ctx gap is not primarily training duration** — 2× epochs moved grid3
  context ~0.03 → ~0.08 while MI stayed ~0.82; next levers are pool mix / encoder
  scale, not more epochs alone.
- **Docs:** canonical run log → `docs/EXPERIMENTS.md`; interpretation →
  `docs/FINDINGS.md`; list **pre–E0** work (`agency_detect/`, examples, CMI
  research) at top of EXPERIMENTS; redirect stubs at old paths.
- **Pause** with prioritized open-experiment table in EXPERIMENTS (grid transfer
  first when resuming).

## Experiment Progression

### E16b — dataset vs sim baseline

`run_dataset_vs_baseline.py`: train on sim only, then extended pool; eval sim +
externals @ `W=250`.

- Sim-only train: grid3/grid5 context ~0; physics/rock ctx 1.0.
- Extended train (+ grid3 in pool): grid3 ctx **~0.12**, grid5 ~0; sim rows slightly
  worse. ~12 min/run CPU.

### Grid training-duration ablation

`ablate_grid_context_epochs.py`: default vs 2× context/siamese epochs on extended
pool. Grid3 context **0.027 → 0.084**; MI unchanged. ~33 min total.

### Harder physics and mixed detection

- `physics_cartpole_x3` — three parallel CartPoles; MI ≈ 0 (symmetric dynamics).
- `mixed_detection.py` + `merge_agent_traces` — one trace: CartPole(1) + Rock(1) +
  grid3(2).
  - Truncate-to-min-T: MI ~0.05 (pole dead most of window).
  - Pad to T=250 + **balance** policy on pole: overall MI ~0.6; rock + grid
    per-agent ARI 1.0; **pole per-agent ARI 0** (channels merge with rock or
    split across clusters — good-regulator / low-DOF issue).

### Agent table and spotlight (earlier in session)

Long `agent_ari_table.py` run filled live spotlight for telemetry families;
externals MI-only. Results archived under `results/learn_agents/` (gitignored).

## Current State

- **E16 wiring** committed (`2190af7`): registry, extended kinds, transfer
  scripts, mixed detection, safe_results, tests.
- **Grid transfer** remains the main amortized failure; duration ablation ruled
  out as sole cause.
- **Mixed detection** supports “different plants separable”; CartPole needs
  role/blanket path or dropping from MI-style tests, not longer identical poles.
- **Docs** committed (`2200a51`, `777bce9`): `docs/EXPERIMENTS.md`, `docs/FINDINGS.md`,
  pre–E0 lineage table, open-experiment priority list, redirects from
  `learn_agents/EXPERIMENTS.md` and `learn_agents_debug_findings.md`.
- **Melting Pot** not installed/run; ClusterData rejected earlier (E14).

## Follow-Up Ideas (paused; see EXPERIMENTS open table)

1. Grid context — oversample grid in pool; xl encoder sweep.
2. Melting Pot install + obs map + extended benchmark smoke.
3. Mixed detection with **telemetry sim + externals** in one trace (user’s original
   intent vs externals-only mix).
4. Spotlight on external traces; episodic telemetry fair comparison.
5. Do not pursue three-alike CartPoles for separation; prefer persistent-dynamics
   agents (grid, foraging, Melting Pot).
