# learn_agents / agency_detect — updates log

Chronological summary of **result artifacts** (by file creation time) and **verification runs**.
Full experiment design: [`learn_agents/EXPERIMENTS.md`](../learn_agents/EXPERIMENTS.md).

---

## Verification run (2026-05-26)

### Tests (all pass)

| Script | Outcome |
|--------|---------|
| `tests/test_mi_k_selection.py` | ok — downstream K, background factorization, precursor gates |
| `tests/test_adaptive.py` | 2/2 valid agents with adaptive detection |
| `tests/test_threshold.py` | N=2 passes; N=3 also passes (threshold may be lenient) |
| `tests/test_validation_fix.py` | Over-clustered run completes; validation executes |
| `tests/test_mi_analysis.py` | Same-type agents cluster by functional MI (A↔C corr 0.93) |

### Scripts

| Script | Outcome |
|--------|---------|
| `scripts/learn_agents/learn_agents_go_nogo_sweep.py` (smoke: 3 agents, 1 seed, 16 slots, 25 epochs) | `var_acc=0.67`, slot purity 0.45 — confirms slots mix agents at scale; motivates candidate framing |

Artifact: `results/learn_agents/go_nogo/learn_agents_go_nogo_commit_smoke.jsonl` (gitignored).

---

## Result artifacts (chronological)

### 2026-05-08 — Early latent + candidate pipeline

| File | Interpretation |
|------|----------------|
| `learn_agents_go_nogo_smoke.jsonl` | First slot sweep; recovery far below usable classifier |
| `learn_agents_go_nogo_8agents.jsonl` | 8-agent go/no-go; best `var_acc` ~0.47 → **reframe as candidate proposer** |
| `candidate_uad_smoke*.json` | Four-step pipeline wired |
| `candidate_uad_eval_*_seed1*.json` | 3-agent OK-ish; 8-agent weak pre/post UAD |
| `candidate_uad_*_richer.json` | More candidates ≠ better ranking |
| `candidate_uad_*_reverted.json` | Baseline before adapt degeneracy analysis |
| `candidate_uad_*_adapt.json` | ε-only adapt shrinks sets; precision/recall tradeoff |

### 2026-05-18 — Debug protocol

| File | Interpretation |
|------|----------------|
| `debug_protocol_3agents.json` | Strict UAD rejected all clusters → looked like env failure |
| `debug_protocol_8agents.json` | Same false negative |

### 2026-05-19 — Classifier fix + MI refine

| File | Interpretation |
|------|----------------|
| `debug_protocol_*_v2.json` | Name-hint classifier fixed oracle; not valid for real data |
| `debug_protocol_*_v3.json` | MI-only S/A/I; env OK; 8-agent raw MI perfect, latent R@30 ~0.25 |
| `candidate_uad_eval_*_v2/v3_*.json` | MDL adapt helps size; strict UAD still misleading alone |
| `candidate_uad_eval_8agents_baseline.json` | Baseline slots without refine |
| `candidate_uad_eval_8agents_mi_refine.json` | **MI refine → R@30 1.0** on clean 8 agents |
| `candidate_uad_eval_3agents_mi_refine.json` | Refine no help when MI partition already broken (decoys) |

### 2026-05-22 — Agent-count sweep (clean)

| File | Interpretation |
|------|----------------|
| `agent_count_sweep.json` | Baseline breaks ≥7 agents; MI refine fixes through ~12; explains earlier 8 vs 3 confusion (decoys) |

### 2026-05-26 — Decoys + validation loop

| File | Interpretation |
|------|----------------|
| `agent_count_sweep_decoy20pct.json` | MI breaks ≥2 agents; refine capped by MI |
| `agent_count_sweep_decoy70pct.json` | Multi-agent collapse |
| `decoy_ablation_smoke.json` | Decoy type ablation script smoke |
| `decoy_ablation_core.json` | 24 conditions MDL K; 8 noise 20% MI recall 0 |
| `decoy_ablation_sweep.json` | 48 conditions (types + intensity); full grid |
| `candidate_uad_eval_8agents_decoy20_downstream.json` | **E8:** downstream K=30, MI recall 1.0 in smoke but **R@30=0.25** end-to-end |
| `decoy_ablation_core_downstream.json` | MIds=1.0 on hard cases; **rDs≈0.38** — partition→refine gap confirmed |

---

## Conclusions (current)

1. **Environment has structure**; raw MI often sufficient for partition (clean regime).
2. **Latent slots** stable but mix agents without MI init.
3. **MI refine** fixes clean multi-agent; fails when MI partition wrong or K over-segments.
4. **Downstream K + background** fixes MI recall on decoys but **not** slot→candidate mapping.
5. **Next direction:** serial spotlight / peel-off (one agent at a time), not global N-slot competition.

---

## Repo cleanup (2026-05-26)

- Committed: `scripts/learn_agents/learn_agents_go_nogo_sweep.py`, agency_detect tests
- Committed: CMI research → `scripts/research/cmi/` (see README there)
- Removed: stale factory debug scripts, scratch `t*.dot`/`t.png`, `loop-hub-value-graph.py`, orphan `src/parameters.py*`

---

## CMI estimator analysis (2026-05-26)

**Scripts + findings:** [`scripts/research/cmi/README.md`](../scripts/research/cmi/README.md)

Ran all four scripts; logs in `results/cmi/*.log` (gitignored).

| Script | Runtime | Verdict |
|--------|---------|---------|
| `nats_inflation_analysis.py` | ~1s | k-NN inflates CMI on continuous data; discrete independent vars → ~0; high cardinality can spike to ~23 nats |
| `cmi_scaling_analysis.py` | ~48s | Memory coupling realistically 2–8 nats; recommends **5.0 nats** blanket threshold for k-NN era |
| `discrete_cmi_alternatives.py` | ~1s | **Smoothed plug-in (α=0.1)** best for discrete; chi-square as validation; independent → ~0.7 nats (smoothed) vs ~1.2 (plug-in) |
| `discrete_cmi_evaluation.py` | ~4s | Discrete beats k-NN on sample size, dimension, memory tests; suggests threshold **3.0–5.0 nats** with smoothed plug-in |

### Interpretation

1. **k-NN CMI was the wrong tool** for discretized agent traces — erratic or zero on discrete data, inflated on continuous.
2. **Production already migrated:** `agency_detect/markov_blanket.py` uses **Laplace-smoothed discrete plug-in** (`CMI_SMOOTHING_ALPHA=0.1`).
3. **Threshold mismatch:** scripts calibrate to **5.0 nats** (k-NN legacy); current `DetectionConfig.BLANKET_TOLERANCE = **1.0**` (stricter, post-fix). Strict UAD on learn_agents simulator uses tolerance 1.0 in evaluate script.
4. **Actionable:** Re-calibrate `BLANKET_TOLERANCE` empirically on learn_agents oracle clusters with the discrete estimator (not k-NN scripts' 5.0 recommendation). Chi-square independence test could complement CMI as a precursor gate.

### Sample discrete CMI baselines (from `discrete_cmi_evaluation.py`)

| Variable type | Smoothed CMI (typical) |
|---------------|------------------------|
| Independent actions | ~0.002 nats |
| Independent sensors | ~0.15 nats |
| Memory + 3D conditioning | ~0.67 nats |
| Invalid / mixed cluster | ~0.6–1.1 nats (test scenarios) |

With tolerance **1.0**, well-formed small clusters should pass; strong spurious coupling still fails — aligns with strict UAD role as falsification, not discovery.

---

## E9 spotlight serial discovery (2026-05-27)

**Design:** `agent_spotlight/` — one MI cluster per pass, peel on hit. Full log: [`learn_agents/EXPERIMENTS.md`](../learn_agents/EXPERIMENTS.md) §E9.

| Run | Mode | Cumulative recall | Pass-1 J | Admitted |
|-----|------|-------------------|----------|----------|
| E8 baseline | global K=30 + 24 slots | **0.25** | — | — |
| E9a v1 | spotlight, bad scoring | 0.00 | 0.00 | 0/8 |
| E9a v2 | spotlight + fixed proposal | 0.00 | 0.13 | 0/8 |
| **E9b** | **`mi_cluster` candidate** | **0.625** | **0.50** | **5/8** |

Artifacts: `results/spotlight/e9/spotlight_peel_e8_decoy20_v2.json`, `results/spotlight/e9/spotlight_mi_cluster_e8_decoy20.json`.

**Findings:** (1) Binary precursor scoring tied decoy blobs at 0.02 — fixed with continuous signal + K=16. (2) `spotlight_slot` bloated 12-var MI clusters to 47 vars — bypass with `mi_cluster`. (3) Remaining misses: agents 4, 5, 7 — K=16 partial/cross-agent clusters at J=0.25 stuck below hit threshold.

### E9c env-realism adaptions (2026-05-27)

Sequential adaptions with E9b (`mi_cluster`): `scripts/spotlight/run_spotlight_env_adaptions.py`

| Step | Recall | Pass-1 J | vs E9b |
|------|--------|----------|--------|
| 1. Rebalance A-A coupling | 0.625 | **1.00** | tie recall, cleaner pass-1 |
| 2. + per-agent env niches | 0.125 | 0.67 | **regression** (MI splits agent/env) |
| 3. + shared world | 0.500 | 0.67 | partial recovery |

Sim support: `local_env_strength`, `env_vars_per_agent`, `world_vars` in `TraceSimulationConfig`.

### E9d — exogenous world (2026-05-27)

Replaced action-coupled env niches with shared exogenous `world.shared*` (agents read, do not drive). Spotlight + adapt1: **recall 0.875**, pass-1 J=1.0.

### E10 — diagnosis + sweeps (2026-05-27)

**Script:** `scripts/spotlight/run_spotlight_sweeps.py`  
**Artifacts:** `results/spotlight/e10/spotlight_e10_diagnose.json`, `results/spotlight/e10/spotlight_e10_sweeps.json`

| Result | Detail |
|--------|--------|
| **8-agent benchmark** | **0.875 recall** (7/8), pass-1 J=1.0 — matches E8 global on exogenous sim |
| **1/8 miss** | Agent 7: `partial_peel_orphan` — MI had J=1.0 cluster at pass 0 but cluster-only peel left 2/6 vars |
| **16 agents** | 1.0 recall with enough passes |
| **Decoys ≥12** | Recall collapses to 0.25 |
| **World 12 vars** | Default; 0.875 recall |
| **Agency gates** | **Off by default** — no gain; strict costs 0.125 recall |

### E10b recovery sweep

**Script:** `scripts/spotlight/run_spotlight_recovery_sweep.py`  
**Artifact:** `results/spotlight/e10/spotlight_recovery_sweep.json`

Small one-at-a-time sweep found the simplest 100% recall setup: raise `proposal_mi_k` from 16 to **24**. This recovers **8/8 agents** with exogenous world, agency gate off, and cluster-only peel. Lower coupling/noise generally did not help; reducing `world_vars` to 6 also reached 8/8 but changes the benchmark.

**Status:** Serial spotlight + exogenous world benchmark is the working default at `proposal_mi_k=24`. Next: data-only peel expansion via refine for harder decoy and moving-agent worlds.

### E11 rich fixed agents

**Change:** `agent_variant_mode=rich` adds heterogeneous per-role variables: delayed sensor/internal/action variants and nonlinear add/min/max-style composites, rather than purely redundant copies.

| Run | Setting | Recall | Notes |
|-----|---------|--------|-------|
| `spotlight_e11_rich_agents_cpr3_k32.json` | 8 agents, 3 vars/role, K=32, 8 passes | 0.625 | finds role chunks; 8 passes insufficient |
| `spotlight_e11_rich_agents_cpr3_k24_p16.json` | 8 agents, 3 vars/role, K=24, 16 passes | **1.000** | all agents recovered from data-only cluster peel |
| `spotlight_e11_rich_agents_cpr3_k32_p16.json` | same, K=32, 16 passes | 0.875 | too fine; repeated role chunks |

Interpretation: richer agents are still recoverable locally, but an agent now appears as multiple agency-bearing role chunks. Scaling complexity therefore needs more passes or a data-only cluster-growth/refine expansion step.
