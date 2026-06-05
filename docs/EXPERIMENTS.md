# Experiment log (agency-detect)

Chronological notebook for the **numbered experiment line** (E0 onward). Records **why** each run was done, **settings**, **results**, and **artifact paths**.

**Documentation map**

| Doc | Role |
|-----|------|
| **This file** (`docs/EXPERIMENTS.md`) | Run log from E0 (May 2026); later sections add E9 spotlight, E12 hierarchy, E13 amortized, E14–E16 externals, **E17–E19 intention detection** |
| [`FINDINGS.md`](FINDINGS.md) | Cross-cutting interpretation (mainly E0–E8 latent-UAD / decoys) |
| [`CHANGELOG.md`](CHANGELOG.md) | Short milestones |
| [`conversations/`](conversations/README.md) | Session summaries (why, not full tables) |
| [`results/README.md`](../results/README.md) | Artifact paths by experiment id |

Moved here from `learn_agents/EXPERIMENTS.md` (2026-06-04). Experiment ids (E0, E9, …) are unchanged.

**Work before this log** — not given E-numbers here; lives in other packages and docs:

| Stage | Location | What |
|-------|----------|------|
| **Core UAD** | `agency_detect/` | `AgentDetector`, lagged-MI clustering, Markov-blanket validation (`markov_blanket.py`, `detection.py`); decoupled solar-panel / factory traces (`agents.py`) |
| **Quickstart** | `examples/basic_detection.py`, `detect.py` | End-to-end detection on decoupled multi-agent sim |
| **Design notes** | `dev.md`, root `README.md` | Original two-domain → multi-agent narrative; API examples |
| **Tests** | `tests/test_mi_*.py`, `test_validation_fix.py`, `test_threshold.py` | Detector and estimator regression tests |
| **CMI research** | `scripts/research/cmi/` | k-NN vs discrete plug-in study (May 2026); production path uses Laplace discrete CMI |
| **Papers** | `docs/papers/` | Theory write-ups (not experiment runs) |

**This log starts at E0** when the **telemetry simulator** in `learn_agents/learn_agents.py` became the main research vehicle (latent slots + strict UAD). Packages added later are documented in later sections: `agent_spotlight/` (E9), `hierarchical_spotlight/` (E12), `amortized_agency/` (E13).


**Default simulator (clean regime)** unless noted:

| Parameter | Value |
|-----------|-------|
| `seed` | 1 |
| `T` | 4000 (6000 in some early runs) |
| `copies_per_role` | 2 → 6 vars/agent |
| `process_noise` | 0.02 |
| `observation_noise` | 0.01 |
| `interaction_strength` | 0.45 |
| `confound_strength` | 0.0 |
| `decoy_vars` | 0 unless noted |
| `episodic` | false |
| Train epochs | 50 (75 in early 8-agent runs) |
| `num_slots` | `max(4, 3 × num_agents)` |
| Metric | **Recall@30** = fraction of true agents covered in top-30 latent candidates (Jaccard ≥ 0.3 hit) |

---

## E0 — Go/no-go parameter sweep (May 2026)

**Why:** Find slot count / epoch settings where latent model recovers agent structure; compare to chance `var_acc`.

**Script:** `scripts/learn_agents/learn_agents_go_nogo_sweep.py`

**Settings:** 8 agents, low noise, no decoys; sweep `num_slots`, seeds, epoch checkpoints.

**Results:** Best `var_acc` ~0.47 multi-seed — far below chance at 8 agents (0.125) but not usable as final classifier. Motivated reframing as **candidate proposer** not identifier.

**Artifacts:** `results/learn_agents/go_nogo/learn_agents_go_nogo_8agents.jsonl`, `results/learn_agents/go_nogo/learn_agents_go_nogo_smoke.jsonl`

---

## E1 — Four-step candidate + strict UAD pipeline (May 2026)

**Why:** Evaluate latent slots as **candidate generator** filtered by strict Markov-blanket validation; report Recall@K and post-UAD precision/recall.

**Script:** `scripts/learn_agents/evaluate_latent_candidates_with_uad.py`

**Key runs:**

| Run | Agents | Decoys | Adapt | Pre R@30 | Post R@30 | Notes |
|-----|--------|--------|-------|----------|-----------|-------|
| seed1 | 3 | 2 | MDL | moderate | low | Early pipeline |
| seed1 | 8 | 0 | MDL | ~0.38 | ~0 | Many strict survivors, poor hits |
| reverted | 3/8 | as above | off | 0.67 / 0.38 | — | Before adapt degeneracy analysis |
| adapt | 3/8 | as above | ε-only | — | shrink to ~2–3 vars/set | Motivated MDL anti-shrink |

**Artifacts:** `results/learn_agents/candidate_uad/candidate_uad_eval_*agents_seed1*.json`, `*_reverted.json`, `*_adapt.json`

---

## E2 — Six-stage debug protocol (May 2026)

**Why:** Isolate failure mode: environment vs representation vs validation vs search.

**Script:** `scripts/learn_agents/debug_learn_agents_protocol.py`

**Stages:** (1) oracle ε-blanket, (2) raw MI clustering, (3) slot persistence, (4) interface/roles [metadata only], (5) ε vs MDL adapt, (6) UAD threshold sweep.

### E2a — Initial runs (v1)

**Finding:** Strict UAD rejected **all** ground-truth clusters (`"no internal variables"`). Looked like environment failure.

**Artifacts:** `results/learn_agents/debug_protocol/debug_protocol_3agents.json`, `results/learn_agents/debug_protocol/debug_protocol_8agents.json`

### E2b — After name-hint classifier fix (v2)

**Why:** Test whether oracle passes when `internal` in variable name used for S/A/I.

**Finding:** Oracle and raw MI passed; confirmed **classifier bug**, not missing structure. User rejected name hints as cheating for real data.

**Artifacts:** `results/learn_agents/debug_protocol/debug_protocol_*_v2.json`

### E2c — Statistical-only classification (v3)

**Why:** Names only for scoring; MI-only S/A/I assignment.

| | 3 agents (2 decoys) | 8 agents (0 decoys) |
|--|---------------------|---------------------|
| Oracle sep. ratio | ~0.04 | ~0.11 |
| Raw MI pre-val recall | 33% | **100%** |
| Latent R@30 | 1.00 | 0.25–0.38 |
| Slot persistence | ~0.99 | ~0.99 |
| Agent purity (slots) | ~0.43 | ~0.31 |

**Conclusion:** Environment OK; 8-agent bottleneck = latent mapping; raw MI already perfect at 8.

**Artifacts:** `results/learn_agents/debug_protocol/debug_protocol_*_v3.json`

---

## E3 — MDL adaptation in candidate search (May 2026)

**Why:** ε-only local search shrank candidates to tiny sets (mean size ~2–3) with no complexity penalty.

**Change:** `J = violation + λ·log((N+1)/|C|)` in `adapt_candidate_set` (`--adapt-objective mdl`, default λ=0.15).

**8 agents, no adapt vs MDL adapt (v2, name-free):**

| | Pre R@30 | Post prec @30 |
|--|----------|---------------|
| no adapt | 0.38 | 0.60 |
| MDL adapt | 0.50 | 0.28 |

MDL increases size and recall; precision drops (more strict false positives).

**Artifacts:** `results/learn_agents/candidate_uad/candidate_uad_eval_8agents_v2_no_adapt.json`, `*_v2_mdl.json`

---

## E4 — MI-guided latent refinement (May 2026)

**Why:** Raw MI solves 8-agent partition; slot model alone does not. **Coarse-to-fine:** MI partition → KL alignment loss on slot assignments.

**Code:** `refine_model_with_mi()` in `learn_agents.py`

**8 agents, clean, 50 pretrain + 25 refine, no adapt:**

| | Pre R@30 | Post recall @30 | Post prec @30 |
|--|----------|-----------------|---------------|
| Baseline slots | 0.25 | 0.00 | 0.00 |
| + MI refine (fixed K=8) | **1.00** | **1.00** | **0.86** |

**3 agents, 2 decoys:** refine did not help (R@30 ~0.33) — MI partition already broken.

**Artifacts:** `results/learn_agents/candidate_uad/candidate_uad_eval_8agents_mi_refine.json`, `*_8agents_baseline.json`, `*_3agents_mi_refine.json`

---

## E5 — Agent-count sweep, 0% decoys (May 2026)

**Why:** Explain “8 agents better than 3” — test 1–12 agents under **comparable** clean settings.

**Script:** `scripts/learn_agents/learn_agents_agent_count_sweep.py`

**Settings:** `decoy_vars=0`, 50 train + 25 refine, seed=1.

| Agents | MI recall | Baseline R@30 | Refine R@30 |
|--------|-----------|---------------|-------------|
| 1–4 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 0.60 | 0.80 |
| 6 | 1.00 | 0.50 | 1.00 |
| 7 | 1.00 | 0.00 | 0.86 |
| 8 | 1.00 | 0.25 | **1.00** |
| 9 | 1.00 | 0.11 | 0.89 |
| 10 | 1.00 | 0.10 | 0.50 |
| 11 | 1.00 | 0.09 | 0.82 |
| 12 | 1.00 | 0.08 | 0.67 |

**Breaking points:** Baseline R@30 &lt; 0.5 from **≥7 agents**; MI refine useful from **≥5 agents**; softens at 10–12 (slot pressure).

**Conclusion:** Earlier 8&gt;3 comparison mixed **decoy counts** (3 agents + decoys ≈ 25% decoys).

**Artifact:** `results/learn_agents/agent_count/agent_count_sweep.json`

---

## E6 — Agent-count sweep, 20% and 70% decoys (May 2026)

**Why:** Decoys strongly hurt; test fractional decoy load (`decoy/(agent+decoy)`).

**Formula:** `decoy_vars = round(f × agent_vars / (1−f))`, `agent_vars = 3 × copies_per_role × num_agents`.

### E6a — 20% decoys

**Artifact:** `results/learn_agents/agent_count/agent_count_sweep_decoy20pct.json`

| Agents | MI recall | Baseline R@30 | Refine R@30 |
|--------|-----------|---------------|-------------|
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 0.50 | 1.00 | 1.00 |
| 3 | **0.33** | 1.00 | **0.33** |
| 8 | 0.38 | 0.12 | 0.38 |

**Breaking points:** MI &lt; 1.0 from ≥2 agents; refine capped by MI from ≥3 agents.

### E6b — 70% decoys

**Artifact:** `results/learn_agents/agent_count/agent_count_sweep_decoy70pct.json`

Multi-agent discovery **collapses** (MI recall 0, R@30 ≈ 0 from ≥2 agents). Only 1-agent trivial case survives.

---

## E7 — Decoy type & intensity ablation (May 2026, in progress)

**Why:** Decoys are not interchangeable — confound/AR(1) “shout” in MI space; test **type** and **intensity** separately. Remove fixed-K assumption (original UAD did not assume agent count).

**Simulator additions:**

| Field | Purpose |
|-------|---------|
| `decoy_mode` | `noise` \| `confound` \| `ar1` \| `mixed` |
| `decoy_ar1_rho` | AR(1) persistence |
| `decoy_confound_weight` | scale on global confound in decoys |

**MI clustering:** `mi_partition_search()` — sweep K with MDL score (λ=0.02 default); no assumed `num_agents`. Compare **fixed K = num_agents** (legacy) vs **variable K**.

**Script:** `scripts/decoys/decoy_ablation_sweep.py`

### E7 partial results (log: `/tmp/decoy_ablation.log`)

MI partition recall — **fixed_K / variable_K**:

| Agents | Mode | Decoy % | fixed_R | var_R | K* |
|--------|------|---------|---------|-------|-----|
| 3 | noise | 0 | 1.00 | 1.00 | 9 |
| 3 | noise | 10–20 | 0.33 | **1.00** | 10 |
| 3 | noise | 50 | 0.33 | 0.33 | 3 |
| 8 | noise | 0 | 1.00 | 1.00 | 8 |
| 8 | noise | 10 | 0.38 | **1.00** | 13 |
| 8 | noise | 20–50 | 0.00 | 0.00 | 2 |
| 3 | confound | 0 | 1.00 | 1.00 | 9 |
| 3 | confound | 10–20 | 0.67 | **1.00** | 10 |

**Early conclusions:**

1. **Fixed K = num_agents was hurting** — variable K fixes many noise/confound cases at low–moderate decoy load.
2. **Decoy type ordering (expected):** confound/AR(1) &gt; noise for MI damage.
3. **MDL K-selection still fails** at 8 agents + 20% noise (K→2) — needs **step-2 K pick** by downstream test (planned).
4. Decoys **steal MI clusters** and **pollute candidates** (~30% decoy vars in top sets at 20%); rarely pass as true agents.

**Artifacts:** `results/decoys/ablation/decoy_ablation_smoke.json`; full run → `results/decoys/ablation/decoy_ablation_sweep.json` (in progress); **core subset** → `results/decoys/ablation/decoy_ablation_core.json` ✓

### E7 core subset (24 conditions, May 26)

Settings: agents 3 & 8, decoy fractions 0/20/50%, modes noise/confound/ar1/mixed, `--skip-intensity`, **MDL K only** (pre–P1/P3).

| Highlight | MIfx | MIvr | rVar |
|-----------|------|------|------|
| 8 agents, noise 20% | 0.00 | **0.00** | 0.12 |
| 8 agents, confound 20% | 0.88 | **1.00** | 0.38 |
| 8 agents, mixed 20% | 0.38 | 0.00 | 0.00 |
| 3 agents, noise 20% | 0.33 | **1.00** | 1.00 |

Variable-K MI (`MIvr`) rescues many 3-agent and confound cases; **8-agent noise/mixed @ 20% still collapses** under MDL (K→2). Refine tracks MI ceiling (`rVar` ≤ `MIvr` in most rows).

Full ablation (`decoy_ablation_sweep.json`) complete (48 rows, MDL only). **Core re-run with downstream K** → `decoy_ablation_core_downstream.json` ✓ (~56 min).

### E7b — Core ablation with downstream K (May 26)

Compare **MIds** (MI recall, downstream K + background) vs **rDs** (refine with downstream K):

| Condition | MIvr (MDL K) | MIds | rVar | rDs |
|-----------|--------------|------|------|-----|
| 8 noise 0% | 1.00 | 1.00 | 1.00 | **1.00** |
| 8 noise 20% | 0.00 | **1.00** | 0.00 | **0.38** |
| 8 confound 20% | 1.00 | 1.00 | 0.50 | 0.38 |
| 8 mixed 20% | 0.00 | **1.00** | 0.00 | 0.38 |
| 3 noise 20% | 1.00 | 1.00 | 1.00 | **1.00** |

Downstream K fixes **MI partition** on hard 8-agent decoy cases (K=30 on noise/mixed 20%) but **refine R@30 stays ~0.38** — consistent with E8 end-to-end (0.25). Clean regime unchanged (rDs=1.0).

**Reproduce:**

```bash
.venv/bin/python scripts/decoys/decoy_ablation_sweep.py \
  --output-json results/decoys/ablation/decoy_ablation_sweep.json
```

---

## Planned (not yet run)

| ID | Experiment | Why |
|----|------------|-----|
| P4 | Decoy-only cluster UAD audit | Do decoy clusters pass blanket tests? |

---

## P1–P3 implementation (May 2026)

### P1 — Step-2 K selection (`mi_k_selection`)

**Problem:** MDL alone collapses to K=2 at 8 agents + 20% noise decoys (MI recall 0).

**Fix:** After building the MI similarity matrix, score each K with **downstream** metrics (no ground truth):
- precursor pass rate + mean persistence/contingency per cluster
- slot-alignment strength (when model available during refine)

Modes: `mdl` | `downstream` (default) | `hybrid`.

**Smoke (8 agents, 20% noise, seed=1, T=2000):**

| Selector | K | MI recall |
|----------|---|-----------|
| MDL | 2 | 0.00 |
| downstream + background | 30 | **1.00** |

### P2 — Background factorization

`factorize_background()` removes rank-1 temporal PCA before MI clustering (`RefineConfig.mi_background_factorize=True` default).

Targets global-confound / shared-driver decoys that dominate pairwise MI.

### P3 — Precursor gates

Per-cluster scores before UAD:
- **Persistence:** lag-1 autocorrelation of cluster mean trace
- **Contingency:** max lagged MI(cluster → rest)

Used in K selection scoring and optional candidate filtering (`--precursor-gate-candidates`, default on in evaluate script).

**Code:** `precursor_cluster_stats()`, `precursor_passes_var_indices()` in `learn_agents.py`.

```bash
.venv/bin/python scripts/learn_agents/evaluate_latent_candidates_with_uad.py \
  --num-agents 8 --mi-refine --mi-k-selection downstream \
  --decoy-vars 12 --T 4000 --epochs 50 --no-adapt-blankets
```

### E8 — End-to-end validation loop (8 agents, 20% decoys, downstream K)

**Artifact:** `results/learn_agents/candidate_uad/candidate_uad_eval_8agents_decoy20_downstream.json`

Settings: seed=1, T=4000, 50 pretrain + 25 refine, 12 decoys (20%), downstream K + background + precursor gates.

| Metric | Old (MDL, core ablation) | End-to-end downstream |
|--------|--------------------------|------------------------|
| MI K | 2 | **30** |
| MI recall (smoke) | 0.00 | 1.00 |
| Pre-UAD Recall@30 | 0.00–0.12 | **0.25** |
| Post-UAD recall@30 | 0.00 | 0.25 |
| Post-UAD precision@30 | — | 0.61 |

**Conclusion:** Downstream K fixes MI partition recall in isolation, but **end-to-end candidate recall stays low** (2/8 agents). Validates partition→refine gap: slots align to K=30 clusters but candidate mapping still mixes agents (`unique_agents_mean=2.75` in top 20). Precursor gate dropped 0/64 — too permissive or clusters pass individually.

---

## E9 — Serial spotlight (`agent_spotlight/`)

**Architecture:** separate package for one-agent-at-a-time discovery. See [`agent_spotlight/README.md`](../agent_spotlight/README.md) (repo root).

### Reasoning arc: why the spotlight line exists

E8 showed a specific failure mode: raw MI could often find meaningful structure, but the learned global slot model and slot→raw mapping did not preserve agent boundaries. Increasing candidate counts or using stricter UAD after the fact did not fix that, because the error happened earlier: a global model with many slots mixed several agents before validation ever saw a candidate.

That led to the E9 hypothesis:

> Stop asking one global latent model to explain all agents at once. Instead, propose one high-signal MI cluster, refine a small model only around that local target, validate it, peel it, and repeat.

The early E9 failures were useful because each isolated one layer of the pipeline:

- **E9a v1:** proposal scoring itself was wrong. Binary precursor scoring made all failing clusters tie, so tiny decoy clusters won by arbitrary cluster id.
- **E9a v2:** proposal improved, but `spotlight_slot` reintroduced the old slot→raw bottleneck by expanding a 12-var MI proposal into a 47-var candidate.
- **E9b:** using the MI proposal itself as the candidate bypassed that mapping error and recovered 5/8 agents, confirming that the serial peel idea was viable.

The next question was not "can we overfit the toy sim?" but "which aspects of the sim are real obstacles versus artifacts of a bad world model?" That motivated the E9c/E9d environmental changes and the later agency-gate tests.

**E9a (implemented):** MI at `proposal_mi_k=8` → pick best cluster by precursor → 3-slot pretrain+refine → one candidate → peel → repeat.

```bash
.venv/bin/python scripts/spotlight/run_spotlight_e9a.py \
  --output-json results/spotlight/e9/spotlight_peel_e8_decoy20.json
```

**Artifact:** `results/spotlight/e9/spotlight_peel_e8_decoy20.json`. Compare cumulative recall vs E8 R@30=0.25.

### E9a results (8 agents, 12 noise decoys, E8 setting)

| Metric | E8 (global) | E9a (spotlight) |
|--------|-------------|-----------------|
| Cumulative recall | 0.25 | **0.00** |
| Pass-1 Jaccard | — | 0.00 |
| Agents admitted | — | 0 / 8 |
| Passes executed | — | 4 (stopped: no vars left) |

**Run notes:**

1. First background job exited after pass 1 (`stop_if_precursor_fails=True`, default before fix): decoy cluster `[49,53,57]` failed precursor → loop halted (~9 s).
2. Rerun with `stop_if_precursor_fails=False` (~23 s): passes 1–2 peeled decoy pairs; pass 3 selected **all 48 agent vars** as one cluster but still failed precursor (contingency 0.0075 < floor 0.015) → skipped train/refine due to `require_precursor_pass=True`; peeled everything; pass 4 had no cluster.

**Conclusion:** Peel-on-precursor-fail avoids early halt, but **`require_precursor_pass` + precursor scoring on decoy-first MI order** never reaches train/refine. Next knobs: relax precursor for proposal-only (train anyway), score clusters by size/purity not precursor alone, or peel only on admitted hits (not `peel_selected_always` on skips).

### E9a fix — proposal ranking (2026-05-27)

**Root cause:** `_score_single_cluster` used `precursor_partition_score` on a single cluster. Any cluster failing precursor floors got the same score (`0.02`), so `max()` tie-broke on lowest `cluster_id` — always tiny decoy blobs at K=8.

**Changes (`agent_spotlight/`):**
- Continuous score: `max(persistence,0) + contingency + within_mi_weight * within_MI`
- `tiny_cluster_penalty` down-weights size≤2 decoy pairs
- `proposal_mi_k`: 8 → **16** (K=8 merges all agents into one blob)
- `require_precursor_pass`: **False** (precursor ranks, does not block train)
- Peel only on **hits** (`peel_selected_always=False`, `peel_on_precursor_skip=False`)

**Smoke (5 epochs):** pass 1 selects agent cluster (J=0.13, not decoys); full run pending.

### E9a v2 full run (`spotlight_peel_e8_decoy20_v2.json`)

| Metric | E8 (global) | E9a v1 | E9a v2 (fixed proposal) |
|--------|-------------|--------|---------------------------|
| Cumulative recall | 0.25 | 0.00 | **0.00** |
| Pass-1 Jaccard | — | 0.00 | **0.13** |
| Agents admitted | — | 0 | **0 / 8** |
| Runtime | — | ~23 s | **~281 s** |

**What improved:** Proposal no longer picks decoys first. Pass 1 MI cluster = 12 vars (agents 0+4), precursor pass, UAD pass.

**New bottleneck:** `spotlight_slot` candidate maps slot 0 → **47 vars** (multi-agent + decoys) despite 12-var MI target; Jaccard ~0.13 every pass. No hits → nothing peeled → same cluster repeated 8×.

**Next:** E9b `candidate_mode=mi_cluster` (use proposal cluster directly, skip slot mapping).

### E9b predictions vs results (`spotlight_mi_cluster_e8_decoy20.json`)

**Predictions (before run):**

| Prediction | Rationale |
|------------|-----------|
| Pass-1 Jaccard **~0.50** (not 0.13) | E9a v2 pass-1 MI cluster = 12 vars spanning 2 agents; `mi_cluster` uses those vars directly → J = 6/12 |
| Cumulative recall **0.50–0.75** | Hits at J≥0.3 should peel clusters and advance; 8 passes, K=16 often merges 2 agents/cluster |
| Beats E8 (**0.25**) if peel works | Slot→candidate mapping was the E9a v2 bottleneck; bypassing it should unlock serial progress |
| UAD mostly passes | E9a already passed UAD on bloated candidates; tighter 12-var sets may fail on 2-agent merges |
| Agents 4, 5, 7 at risk | Late passes may hit partial clusters (J=0.25 = 4-var overlap on 6-var agent) below hit threshold |

**Results:**

| Metric | E8 | E9a v2 (slot) | E9b (mi_cluster) | Predicted |
|--------|-----|---------------|------------------|-----------|
| Cumulative recall | 0.25 | 0.00 | **0.625** | 0.50–0.75 ✓ |
| Pass-1 Jaccard | — | 0.13 | **0.50** | ~0.50 ✓ |
| Agents admitted | — | 0 | **5 / 8** | — |
| Runtime | — | ~281 s | ~253 s | — |

**Pass log:** agents **0, 1, 2, 3, 6** admitted (passes 1–5); passes 6–8 stuck at J=0.25 (agent 5 partial cluster, below 0.3 threshold), no further peel.

**Conclusion:** Prediction confirmed — slot mapping was the blocker. E9b **2.5× E8** recall. Remaining gap: K=16 merges agents → J=0.25–0.67 per pass; need purer single-agent clusters or lower hit threshold / split merged clusters.

### E9c — realism adaptions 1→3 (sequential, E9b / `mi_cluster`)

**Script:** `scripts/spotlight/run_spotlight_env_adaptions.py`  
**Summary:** `results/spotlight/e9/spotlight_env_adaptions_summary.json`

**Why these adaptions:** The E9b setting still had strong ring coupling and decoy-style background variables. That made it unclear whether misses were caused by the discovery method or by an unrealistic simulator where "environment" variables were actually downstream of agent actions. We therefore changed one aspect at a time:

1. weaken A-A coupling so agents are less merged by the ring;
2. add local environment variables to make sensors less trivially tied to internals/actions;
3. add shared world structure to test whether common exogenous causes create false agent clusters.

| Condition | Change | Recall | Pass-1 J | Admitted | N vars |
|-----------|--------|--------|----------|----------|--------|
| E9b baseline | ring-heavy | **0.625** | 0.50 | 5/8 | 60 |
| **Adapt 1** | weak A-A (`interaction=0.1`, `mix=0.02`, `local_env×1.8`) | **0.625** | **1.00** | 5/8 `[0,4,5,6,7]` | 60 |
| **Adapt 2** | + per-agent `env{k}.*` (3/agent, decoys off) | **0.125** | 0.67 | 1/8 | 72 |
| **Adapt 3** | + `world.*` (4 shared) | **0.500** | 0.67 | 4/8 | 76 |

**Findings:**
1. **Adapt 1 alone** matches baseline recall but **pass-1 is a clean single-agent hit** (J=1.0) — rebalancing A-A vs private env removes early hybrid merges without extra vars.
2. **Adapt 2 regresses:** MI clusters `agent{k}` separately from `env{k}`; proposal picks env-dominated or partial clusters → peel stalls (only agent 2 admitted).
3. **Adapt 3 partial recovery (0.5):** shared world gives global structure that partially re-stabilizes clustering vs env-only niches, still below adapt 1.

**Next for env-rich sim:** score/propose **agent+attached-env** jointly (precursor on agent vars only, or hard-link env indices from metadata at eval time); or peel env with admitted agent.

### E9d — exogenous world redesign (2026-05-27)

**Change:** Replaced action-coupled `env{k}` niches with true exogenous world:
- `world.shared*` — shared AR(1), all agents read weakly (`world_to_sensor_strength=0.08`)
- `world.local{k}.*` — optional local exogenous patches (`env_action_coupling=0` default)
- Agents do **not** drive world (`env_action_coupling=0`)

**Verify:** `scripts/spotlight/verify_exogenous_world.py` — PASS shared-only (0 mixed MI clusters, pass-1 agent-only).

**Spotlight (adapt1 + shared world, E9b):** recall **0.875** (7/8), pass-1 J=**1.0** vs old coupled-env adapt2 **0.125**.

**Interpretation:** This was an ontology correction. The old per-agent `env{k}` was not really "environment" because it was partly driven by the agent's action. In UAD terms it behaved like an extended body/loop, so penalizing the detector for grouping it with agent variables was conceptually wrong. The exogenous shared world lets us test a cleaner distinction: agents may read world state, but they do not drive it. World-only clusters can then be identified by low agency signature and passive dynamics, not by names or metadata.

### E9e — exogenous benchmark defaults + agency gate (2026-05-27)

**Changes:**
- `SpotlightConfig` defaults → exogenous world (`world_vars=12`, `decoy_vars=0`, `mi_cluster`, adapt1 strengths)
- Data-only **agency signature gate** (S+A+I on cluster vars before train)
- **Peel full agent** on Jaccard hit; peel non-agency clusters on skip

**Artifacts:** `results/spotlight/e10/spotlight_exogenous_baseline.json`, `results/spotlight/e8/e8_exogenous_benchmark.json`

| Method | Cumulative recall | Pass-1 J | Notes |
|--------|-------------------|----------|-------|
| E8 global (`--exogenous-benchmark`, K=30) | **0.875** | — | post-UAD recall@30 |
| E9e spotlight (exogenous defaults) | **0.875** | **1.00** | 7/8 agents; `--require-agency-signature` → 0.750 |

**Why the agency gate stayed off:** The gate was meant to distinguish passive world/decoy clusters from agent clusters without using names. It did that, but strict S+A+I before training also rejected partial agent chunks. Since the benchmark already had clean pass-1 behavior and the gate did not improve recall, the better lesson was to treat agency-role evidence as a ranking/refinement signal, not a hard admission gate.

### E10 — miss diagnosis + sweeps (2026-05-27)

**Script:** `scripts/spotlight/run_spotlight_sweeps.py`  
**Artifacts:** `results/spotlight/e10/spotlight_e10_diagnose.json`, `results/spotlight/e10/spotlight_e10_sweeps.json`

**Why 7/8 agents (not cheating):** We identify the **right agent** but often not **all six variables**.

1. **MI at K=16** returns ~6-var clusters that are usually a *subset* of one agent, or a partial merge with a neighbor (ring coupling). Jaccard ≥ 0.3 admits at 2/6 overlap.
2. **Peel masks `cluster.var_indices` only** — the proposed MI cluster, not the full agent. Overlapping peels from earlier passes orphan 1–2 vars (agent 7: 4/6 peeled before it was ever trained).
3. We intentionally do **not** peel ground-truth agent vars; peel remains data-only (`cluster.var_indices`).

**Production path:** expand peel set from refine alignment or grow cluster within MI partition (data-only).

**Why diagnose before changing the method:** At 0.875 recall, a naive fix would have been to lower the Jaccard threshold or peel the full ground-truth agent on a hit. Both would hide the real failure. The diagnostics showed the detector had already proposed the missing agent cleanly early in the run; the miss came from serial ordering and cluster-only peel orphaning variables before that agent was selected. That made "oracle full-agent peel" an invalid but useful control, and pointed to data-only peel expansion/stitching as the real next method improvement.

| Sweep | Setting | Recall | Missed |
|-------|---------|--------|--------|
| Agents | 8 / 12 / 16 | 0.875 / 0.917 / **1.000** | [7] / [5] / [] |
| Decoys | 0 / 6 / 12 / 18 | 0.875 / 0.750 / **0.25** / 0.25 | decoys≥12 collapses |
| World | 0 / 6 / 12 / 18 | **1.0** / **1.0** / 0.875 / 0.875 | 12 shared = default |
| Agency gate | **off (default)** | **0.875** | [7] |
| | score_penalty / actions_only / soft | 0.875 | same recall, no gain |
| | strict | 0.750 | −0.125 |

### E10b — 100% recall setup (2026-05-27)

**Script:** `scripts/spotlight/run_spotlight_recovery_sweep.py`  
**Artifact:** `results/spotlight/e10/spotlight_recovery_sweep.json`

Small one-at-a-time sweep over coupling/noise/world/K found that lower coupling/noise does **not** fix the orphan issue; most such tweaks remain at 0.875 or regress. The clean data-only fix is finer MI proposal:

| Setting | Recall | Pass-1 J | Missed |
|---------|--------|----------|--------|
| baseline (`K=16`, world=12) | 0.875 | **1.000** | [7] |
| `world_vars=6` | **1.000** | 0.667 | [] |
| `proposal_mi_k=24` | **1.000** | 0.667 | [] |
| `proposal_mi_k=32` | **1.000** | 0.667 | [] |
| weak-world + `K=24` | **1.000** | 0.667 | [] |

**Default locked:** exogenous benchmark, `mi_cluster`, `proposal_mi_k=24`, agency gate **off**, cluster-only peel. This recovers **8/8** on the 8-agent benchmark without using ground-truth agent clusters. Gates optional for ablation; strict S+A+I not recommended.

**Why K=24 helped:** Lowering noise or coupling did not consistently fix the orphan issue, which suggested the signal was already present. Raising proposal K made the MI partition fine enough that later serial passes could isolate remaining agent chunks before they were destroyed by earlier peels. The trade-off is visible in pass-1 J: K=24 gives less "whole-agent" first clusters than K=16, but better eventual coverage.

### E11 — richer fixed-coordinate agents (2026-05-27)

**Simulator change:** `TraceSimulationConfig.agent_variant_mode="rich"` keeps fixed agent variables but makes per-role observations non-redundant:
- delayed/early-ish role variants (`agent_variant_delay=2`)
- additive sensor/internal/action composites
- nonlinear min/max-style role transforms

This is a complexity step before moving-agent invariants: the agent still occupies fixed variables, but its observable variables are no longer simple noisy copies.

**Why this scaling step:** Adding more small agents was not the next hard problem: the method already scaled surprisingly well to more independent 6-var agents. The more important question was whether it could handle a single agent whose observed variables are heterogeneous parts of one dynamical system. The rich variant therefore keeps the coordinate system fixed but makes each agent internally non-trivial before we attempt moving/non-stationary agents.

| Run | Vars/agent | K | Passes | Recall | Pass-1 J | Notes |
|-----|------------|---|--------|--------|----------|-------|
| `spotlight_e11_rich_agents_cpr3_k32.json` | 9 | 32 | 8 | 0.625 | 0.333 | finds 3-var role chunks; too few passes |
| `spotlight_e11_rich_agents_cpr3_k24_p16.json` | 9 | 24 | 16 | **1.000** | 0.333 | all 8 agents recovered |
| `spotlight_e11_rich_agents_cpr3_k32_p16.json` | 9 | 32 | 16 | 0.875 | 0.333 | too fine; repeats chunks |

**Interpretation:** richer agents are discoverable, but the unit of discovery becomes a role/subrole chunk (J≈3/9) rather than a whole agent. The next pipeline improvement is data-only stitching/growth of chunks into agent-level hypotheses.

### E12 — hierarchical chunk fusion (2026-05-28)

**Package:** `hierarchical_spotlight/`  
**Input:** `results/spotlight/e11/spotlight_e11_rich_agents_cpr3_k24_p16.json`

**Why hierarchy:** E11 changed the meaning of a successful local discovery. In the simple simulator, one spotlight chunk often was a whole agent. In the rich simulator, each chunk was more like an agency-bearing subcomponent: a sensor-like, internal-like, or action-like chart of the larger agent. That suggests a natural hierarchy:

```text
raw variables -> local chunks -> fused sub-agent graph -> larger agent hypotheses
```

E12 treats each admitted spotlight pass as a graph node and adds data-only fusion edges using lagged cross-MI (with optional UAD/precursor checks on the union). The evaluation then asks two separate questions:

- **Graph coverage:** does every true agent appear somewhere in the graph?
- **Clean coverage:** does every true agent appear in a node/component with limited extra agent/world/decoy contamination?

The first permissive run used `min_cross_mi=0.05` and produced one giant component. That still had graph coverage, but it was not a meaningful hierarchy. Raising the default to `min_cross_mi=0.70` produced 8 components, each corresponding to one plausible agent-level grouping, while preserving `clean_graph_recall=1.0`.

**Interpretation:** This confirms the richer-agent result should be read hierarchically: spotlight finds local agency charts; fusion begins to assemble those charts into larger agent hypotheses. The next hard case is non-stationary/moving agents, where "same agent" will mean an invariant identity across changing observed variables rather than a fixed variable set.

### E12b — complex fixed-agent hierarchy sample (2026-05-28)

**Script:** `scripts/hierarchical/run_hierarchical_e12b_sweep.py`

**Why this before moving agents:** E12 should not succeed merely because observed role variables are lockstep copies. E12b keeps fixed raw coordinates but uses `agent_variant_mode="complex"` so every role channel is a heterogeneous delayed, nonlinear, or cross-role view of the same latent agent. It also samples weak ring coupling between agents to test whether fusion separates same-agent cohesion from low-bandwidth neighbor interaction.

**Simulator change:** `TraceSimulationConfig.agent_variant_mode="complex"` applies heterogeneous transforms to all role channels (including `r=0`), not just delayed variants of a base copy.

Six-case sample (16–20 passes, MPS, `--fast` smoke settings: T=1600, 10/8 train epochs):

```bash
.venv/bin/python scripts/hierarchical/run_hierarchical_e12b_sweep.py --fast --device mps --jobs 6 \
  --summary-json results/hierarchical/e12b/e12b_complex_sample_summary.json
```

An early 6-pass attempt was invalid for hierarchy testing (fewer passes than agents/chunks). Valid E12b runs need 16+ passes so multiple heterogeneous chunks per agent appear before fusion.

| Run | interaction | mixing | spot recall | graph recall | clean recall | components | edges |
|-----|-------------|--------|-------------|--------------|--------------|------------|-------|
| `complex_k24_p16_i002_m000` | 0.02 | 0.00 | **1.000** | **1.000** | **1.000** | 8 | 7 |
| `complex_k24_p16_i005_m000` | 0.05 | 0.00 | **1.000** | **1.000** | **1.000** | 9 | 6 |
| `complex_k24_p16_i010_m000` | 0.10 | 0.00 | **1.000** | **1.000** | **1.000** | 8 | 6 |
| `complex_k32_p16_i005_m000` | 0.05 | 0.00 | **1.000** | **1.000** | **1.000** | 8 | 4 |
| `complex_k24_p20_i005_m002` | 0.05 | 0.02 | **1.000** | **1.000** | **1.000** | 9 | 6 |
| `complex_k32_p20_i010_m002` | 0.10 | 0.02 | **1.000** | **1.000** | **1.000** | 8 | 3 |

**Artifact:** `results/hierarchical/e12b/e12b_complex_sample_summary.json` (includes per-stage timing: proposal dominates spotlight; pretrain/refine use MPS).

**Interpretation:** Under heterogeneous within-agent channels and weak inter-agent coupling up to `interaction_strength=0.10`, spotlight still recovers all agents as role chunks and hierarchical fusion assembles them into agent-level components without collapsing into one giant graph. E12b supports treating the hierarchy as stable on fixed coordinates before attempting moving/non-stationary agents.

---

## E13 — amortized agency detection: MI window breaking-point baseline (2026-06-01)

**Script:** `scripts/amortized/baseline_window_breaking_point.py`
**Artifacts:** `results/amortized/baseline_window_breaking_point.{json,png}`

**Why:** The driving target is now **short-duration / transient agents** (an agent that only exists for a brief window). All current estimators — CMI blanket validation, lagged-MI clustering, IRL goal inference — are properties of a *stationary window*, so they need many samples and average a transient agent out of existence. The proposed escape is **amortization**: train one agency detector across a pool of many varied (short- and long-lived) agents, then apply it to new traces without relearning each agent — the way a human recognizes "an agent" without re-deriving the concept per encounter.

Before building any learned model, this experiment **locates the breaking point of the existing MI method** as the observation window `W` shrinks. That curve is the baseline the amortized detector must beat at short `W`, and the criterion is now quantified instead of asserted.

**Hypothesis:** MI-based agent recovery is strong at long `W` and collapses below some window length; if the collapse is driven by sample count rather than agent complexity, the breaking `W` should be roughly **independent of agent kind**.

**Method:** For an easy→hard spectrum of kinds, simulate one non-stationary-free (`episodic=False`) world per seed at `T=2000`, slice the first `W` steps, and run the repo's real proposal step (`mi_cluster_variable_labels`: per-column quantile discretization → lagged MI → agglomerative clustering) restricted to agent variables (`var_agent >= 0`, isolating duration from decoy/world rejection). Score agent separation by adjusted Rand index (ARI) and mean best-Jaccard against ground-truth agent ids. 5 seeds/kind; `W ∈ {2000, 1000, 500, 250, 125, 60}`.

| Kind | W≥500 | W=250 | W=125 | W=60 |
|------|-------|-------|-------|------|
| `easy3_redundant` (3 agents) | **1.00** | 1.00 | 0.77 | 0.47 |
| `med5_rich` (5 agents) | **1.00** | 1.00 | 0.73 | 0.60 |
| `hard8_complex` (8 agents) | **1.00** | 0.96 | 0.73 | 0.52 |

(ARI mean over 5 seeds; full mean±std and Jaccard in the JSON.)

**Interpretation:** MI recovers agents perfectly down to ~`W=250`, then collapses sharply between `W=250` and `W=125`, reaching near-chance by `W=60`. The breaking point is **~`W≈125` regardless of agent kind** — easy 3-agent and hard 8-agent-complex break at essentially the same window. So for short durations the bottleneck is **statistical power (sample count), not agent complexity**, exactly as the stationary-window argument predicts.

**Consequence for the amortized line (revised after E13e compute/trend sweeps):**

- **Primary benchmark:** match MI in the **reference regime** `W ≥ 250` where MI is the trusted ceiling (ARI ~1.0). Report **`gap_to_mi = MI − learned`** on all kinds (train + held-out). Sweeps and hyperparameter search should minimize this gap first.
- **Secondary benchmark:** the short-window band `W ∈ [60, 250]` where MI collapses — amortization may beat a weak MI baseline here (useful for transient agents), but that is not a substitute for reference accuracy.
- Scripts: `run_reference_benchmark.py` (canonical reference table), `run_method_sweep.py --mode reference|trends`, constants in `amortized_agency/benchmark.py`.

**Next step:** a learned same-agent affinity model. All methods (MI, learned) emit the same permutation-invariant `N×N` affinity → identical downstream clustering, so comparisons stay apples-to-apples. Start with a **Siamese pairwise** encoder as the floor, then a **context-aware Set-Transformer / slot-attention** model that conditions on the whole channel set (Markov-blanket agency is conditional, so pairwise scoring has a generalization ceiling). Slot/affinity outputs are reduced to the same co-assignment matrix so slot index is never compared across worlds.

### E13b — pooled Siamese + slot affinity (2026-06-01)

**Package:** `amortized_agency/`
**Script:** `scripts/amortized/run_pooled_experiment.py`
**Artifacts:** `results/amortized/pooled_experiment_summary.json`, `siamese_model.pt`, `slot_model.pt`

**Why:** Test whether amortization works in principle: train on a pool of varied short windows from **train kinds** (`easy3_redundant`, `med5_rich`), evaluate on **held-out kind** (`hard8_complex`) without retraining. Both learned models emit the same `N×N` co-assignment affinity as MI; downstream clustering is identical.

**Training:** 120 episodes (60/kind), windows sampled from `{60, 125, 250}`, agent variables only. Siamese: pairwise BCE, 30 epochs. Slot: co-assignment BCE, 16 slots, 25 epochs. Channel encoder is a shared 1D conv (no variable-id embedding) for cross-world generalization.

**Held-out `hard8_complex` (5 seeds):**

| W | MI ARI | Siamese ARI | Slot ARI |
|---|--------|-------------|----------|
| 250 | 0.877 | 0.356 | 0.012 |
| 125 | 0.677 | **0.630** | 0.012 |
| 60 | 0.510 | 0.468 | 0.016 |

(Jaccard at W=125: MI 0.653, Siamese **0.661**.)

**Interpretation:**

1. **Amortization is partially viable.** Siamese trained only on easy/medium kinds reaches **near-parity with MI on held-out complex agents at W=125** (ARI 0.63 vs 0.68; Jaccard slightly higher). On train kind `med5_rich` at W=60, Siamese **beats** MI (ARI 0.56 vs 0.55). The model learns transferable same-agent structure without seeing complex 8-agent worlds.

2. **Siamese does not yet beat MI at W=60 on held-out kind** (0.47 vs 0.51). The target band improvement is real but incomplete — more pool diversity, context-aware models, or longer training may be needed.

3. **Simple slot co-assignment failed** (ARI ~0.01). Flat co-assignment matrices did not yield usable cluster structure under agglomerative clustering; likely needs contrastive slot binding loss, sharper attention, or direct slot→agent supervision. Context-aware binding remains the right architecture direction but this implementation is not yet competitive.

4. **Long-window regression:** Siamese is weak at W=250 on held-out (0.36) while MI stays strong — pairwise training at short/mixed windows may hurt long-window inference. Multi-scale encoders or explicit long-window episodes in the pool may fix this.

**Command:**

```bash
.venv/bin/python scripts/amortized/run_pooled_experiment.py --device cpu \
  --train-worlds 60 --siamese-epochs 30 --slot-epochs 25
```

### E13c — slot upgrades + train-long / detect-short (2026-06-01)

**Package:** `amortized_agency/slot_model.py` (upgraded)
**Default training:** windows `{500, 1000}`; evaluation windows `{250, 125, 60}`

**Design clarification:** *Train long, detect short* is valid. The conv encoder accepts variable-length windows, so the pool can expose rich temporal dynamics (500–1000 steps) while inference runs on short slices (60–125). That is separate from *cross-channel context at detection time*: a context-aware model sees **all N channels in the episode simultaneously** (self-attention / slot competition), implementing the conditional nature of blanket agency — not more timesteps at inference.

**Slot upgrades implemented:**

1. Correct slot-attention axis (softmax over variables, standard orientation).
2. Per-variable slot profile → cosine affinity at inference (replaces flat co-assignment).
3. Multi-term training: co-assign BCE + agent cohesion + profile contrastive + slot sharpness + reconstruction.
4. Inference temperature sharpening (`--slot-temp 0.35`).

**Held-out `hard8_complex` after upgrades (fast run, 20 slot epochs):**

| W | MI ARI | Siamese ARI | Slot ARI |
|---|--------|-------------|----------|
| 250 | 0.908 | 0.297 | 0.069 |
| 125 | 0.669 | 0.531 | **0.141** |
| 60 | 0.551 | 0.373 | **0.176** |

**Full run** (60 worlds, train W∈{500,1000}, 40 slot epochs): slot **regressed** (held-out W=125 ARI 0.029; W=60 ARI 0.074). Siamese held-out W=125 ARI 0.551. Longer slot training with current loss weights did not improve over the shorter run — likely loss imbalance / overfitting; hyperparameter sweep needed.

**Interpretation:**

- Slot went from ~chance (ARI 0.01) to **partial signal** (0.14–0.18 at short W) after architectural fixes — direction is right, not yet competitive with MI (0.51–0.68) or Siamese (0.37–0.55).
- **Siamese remains the stronger pooled baseline** on held-out kind at W=125.
- **Train-long / detect-short** is now the default; Siamese still weak at W=250 on held-out when trained only on long windows — the encoder may need explicit multi-scale or short-window finetuning episodes in the pool.

**Command (upgraded slot defaults):**

```bash
.venv/bin/python scripts/amortized/run_pooled_experiment.py --device cpu \
  --train-windows 500,1000 --train-worlds 60 --slot-epochs 20 --siamese-epochs 25
```

### E13d — slot objective fixes + context-aware model (2026-06-01)

**Packages:** `amortized_agency/slot_model.py` (rewritten), `amortized_agency/context_model.py` (new)

**Why:** E13b/c slot attention sat at ~chance. A careful audit found a root-cause bug plus a representational ceiling. This experiment fixes the objective to be stable, then isolates and removes the real bottleneck.

**Slot fixes applied (1–5):**

1. Softmax restored to **over slots** (canonical competition); E13c had inverted it to over-variables, which made the BCE target unreachable (co-assignment values ~0.003 vs target 1).
2. Per-variable profile normalized over K; **same profile-dot affinity used for both training target and inference** (removed train/inference mismatch and the inference-only temperature hack).
3. Sharpness loss corrected to per-variable entropy over slots (commit each variable to one slot) instead of the previous term that drove each slot to a single variable.
4. Contrastive loss vectorized (supervised-contrastive, no per-variable Python loop).
5. Optional shared-Gaussian sampled slots for exchangeability/anti-collapse.

Reconstruction was **dropped**: its optimum (soft mixing to reconstruct per-variable detail) conflicts with hard clustering. The remaining terms share one global optimum (each agent in a distinct slot, one-hot variable assignment), so training is stable under arbitrarily many epochs — no early stopping.

**Diagnostic result (decisive):**

- After the fixes, **BCE-only with fixed slots still floors at loss = ln 2 ≈ 0.693 with train ARI ≈ chance** — i.e. it converges cleanly to "predict 0.5 for every pair." The objective is now well-posed and stable; the failure is representational, not optimization.
- Bypassing slots (direct pairwise cosine BCE) overfits training only weakly with the per-channel encoder (train ARI 0.22) but reaches **train ARI 0.80 with a cross-channel encoder**.

**Conclusion:** two separate causes. (a) The **slot-attention readout** is the wrong inductive bias here — routing N variables through K competing slots cannot express same-agent membership; direct pairwise affinity can. (b) The **per-channel, time-pooled encoder is relational-blind** — agent membership is cross-channel correlation, which it discards. Both are needed: cross-channel context **and** a pairwise (not slot) readout.

**`ContextualAffinityModel`** implements this: a cross-channel attention encoder (channels as tokens, attending over time tokens) feeding a direct pairwise Gram affinity, trained with the balanced co-assignment BCE.

**Held-out `hard8_complex` (5 seeds, train W∈{500,1000}):**

| W | MI | Siamese | Slot | **Context** |
|---|-----|---------|------|-------------|
| 250 | **0.877** | 0.253 | 0.009 | 0.656 |
| 125 | **0.677** | 0.523 | 0.018 | 0.542 |
| 60 | 0.510 | 0.453 | 0.014 | **0.461** |

Train-kind transfer is strong too: on `med5_rich` W=250, context **0.739** vs Siamese 0.214; on `easy3_redundant` W=60, context **0.659** vs MI 0.675 (parity).

**Interpretation:**

- **Context is now the clear best learned method**, beating Siamese at every held-out window and roughly matching it at W=60. The slot model is confirmed a dead end for this readout; it remains in the repo as a documented negative result.
- **MI is still the overall short-window leader** on held-out complex agents (0.68 vs 0.54 at W=125). Amortization has not yet beaten MI in the target band — but the gap closed substantially and the context model generalizes across kinds and window lengths far better than Siamese, while MI must be recomputed per trace.
- **Context degrades more gracefully than Siamese at long W** (0.656 vs 0.253 at W=250), so the train-long/detect-short regime no longer hurts it the way it hurt Siamese.

**Command:**

```bash
.venv/bin/python scripts/amortized/run_pooled_experiment.py --device cpu \
  --train-windows 500,1000 --train-worlds 40 --context-epochs 40 \
  --siamese-epochs 25 --slot-epochs 25
```

---

### E13e — method-trend sweep across test-time parameters (2026-06-01)

**Script:** `scripts/amortized/run_method_sweep.py` (slot dropped — confirmed chance in E13d)

**Why:** E13d gave the headline ranking but not the *gradients*. The question is which way each method bends as the test trace gets harder, and where the MI↔amortization crossover actually sits. Learned models are trained **once** (pool W∈{500,1000}, 40 worlds/kind, train kinds = easy3/med5) and then evaluated on complex agents while varying **one** test-time parameter at a time (4 seeds each).

**Trends (ARI, complex agents):**

| window | 30 | 45 | 60 | 90 | 125 | 175 | 250 | 400 |
|--------|----|----|----|----|----|----|----|----|
| MI | 0.13 | 0.37 | 0.47 | 0.63 | 0.79 | **0.92** | 0.87 | **0.96** |
| Siamese | 0.39 | 0.48 | **0.57** | 0.60 | 0.54 | 0.55 | 0.53 | 0.51 |
| Context | **0.43** | 0.38 | 0.45 | 0.52 | 0.54 | 0.47 | 0.61 | 0.76 |

| obs+proc noise | 0.02 | 0.04 | 0.08 | 0.16 | 0.32 |
|------|----|----|----|----|----|
| MI | **0.82** | **0.79** | **0.80** | **0.80** | **0.76** |
| Siamese | 0.59 | 0.54 | 0.50 | 0.48 | 0.39 |
| Context | 0.55 | 0.54 | 0.53 | 0.62 | 0.53 |

| num_agents | 3 | 5 | 8 | 12 |
|------|----|----|----|----|
| MI | 0.85 | 0.75 | **0.79** | **0.69** |
| Siamese | 0.88 | **0.76** | 0.54 | 0.46 |
| Context | **0.89** | 0.68 | 0.54 | 0.46 |

**Directions:**

- **MI is statistics-driven and monotone in W**: near chance at W=30 (0.13 with 8 agents), climbing to ~0.96 at W=400. It is also strikingly **noise-robust** (correlation structure survives) and degrades only gently with agent count.
- **The crossover is at W≈70**: below it the per-trace estimator runs out of samples and the amortized models (which carry a prior) **overtake MI** — at W=30, context/siamese ≈ 0.4 vs MI 0.13. This is exactly the transient-agent regime the project targets, and the first place amortization clearly wins.
- **Context is the best learned method everywhere except in-distribution low agent count**, and is **noise-robust** like MI (flat ~0.55 across a 16× noise increase) while **Siamese is noise-fragile** (0.59→0.39). Siamese is also flat in W — it never exploits extra samples; context does (rises to 0.76 at W=400).
- **Agent count is a training-distribution effect**: at n=3 (in distribution) the learned models slightly beat MI; at n=8/12 (extrapolation past the 3/5-agent pool) they fall to ~0.46 while MI holds 0.69–0.79. The learned drop is an artifact of the pool, not the method — the clear next lever is to train on more agent counts.

**Takeaway for amortization:** push to **very short windows** (its native edge), keep the cross-channel context encoder (noise-robust, sample-exploiting), and **broaden the training pool over agent count** so the n=8/12 deficit closes. MI remains the long-window / high-agent-count reference.

**Command:**

```bash
.venv/bin/python scripts/amortized/run_method_sweep.py --device cpu
```

**Compute scaling (`scripts/amortized/run_compute_scaling.py`, CPU, full trace→labels per trace):**

MI is the most *accurate* per-trace estimator, but accuracy is not free — it recomputes pairwise lagged MI on every trace. Measured inference time:

| N variables | 27 | 45 | 72 | 108 | 144 | 216 |
|------|----|----|----|----|----|----|
| MI | 0.69 s | 1.95 s | 5.60 s | 11.6 s | 21.2 s | 47.8 s |
| Siamese | 9 ms | 13 ms | 19 ms | 25 ms | 33 ms | 46 ms |
| Context | 10 ms | 13 ms | 19 ms | 23 ms | 30 ms | 39 ms |

| window W (N=72) | 60 | 125 | 250 | 500 | 1000 |
|------|----|----|----|----|----|
| MI | 6.0 s | 4.8 s | 5.0 s | 5.2 s | 5.9 s |
| Siamese | 6 ms | 9 ms | 16 ms | 30 ms | 64 ms |
| Context | 6 ms | 9 ms | 16 ms | 31 ms | 62 ms |

- **MI is ~O(N²) in variable count and ~flat in window length** (its cost is dominated by the pairwise lagged-MI loop, not by W). It is **66× slower than the learned models at N=27 and ~1200× slower at N=216** (47.8 s vs 39 ms).
- **The learned models scale ~linearly** in both N and W with a millisecond constant (the channel-attention N² term is negligible at these sizes). Even at W=1000 context is 62 ms vs MI's 5.9 s (~95×).
- This **reframes the accuracy result**: MI's edge in the W∈[125,1000] band costs two-to-three orders of magnitude more compute *per trace*, and it scales quadratically in exactly the dimension that explodes on real infrastructure (channel count). The amortized models pay their cost **once** in training, then run in milliseconds and are batchable/GPU-able. For the "active diagnostics over many transient agents" target, MI is the method that does **not** scale.

**Command:**

```bash
.venv/bin/python scripts/amortized/run_compute_scaling.py --device cpu --repeats 3
```

### E13f — reference-regime benchmark protocol (2026-06-02)

**Why:** Trend sweeps (E13e) mixed regimes where MI is already weak (W=30, W=125) with regimes where MI is the gold standard (W≥250). For amortization research the right target is: **use settings where MI recovers all agents reliably, then close the gap**; short-W wins are secondary.

**Reference regime (from E13 baseline):** `W ∈ {250, 500}` — MI ARI ~1.0 on `easy3`/`med5`, ~0.96 mean on `hard8_complex` at W=250 (5 seeds; not always 1.0 per seed). Eval must use **`T=2000` simulation then slice `[:W]`** (`benchmark.EVAL_T_STEPS`); shorter `T` changes the RNG trajectory so the first `W` steps differ (a bug in early reference runs used `T=500`, depressing MI to ~0.82).

**Artifacts / scripts:**

| Script | Role |
|--------|------|
| `run_reference_benchmark.py` | Primary: train once, eval all kinds × reference windows, emit `gap_context`, `gap_siamese` |
| `run_method_sweep.py --mode reference` | Same reference grid inside sweep harness |
| `run_method_sweep.py --mode trends` | Secondary parameter axes (complex8 only) |
| `benchmark.py` | `REFERENCE_WINDOWS`, `BREAKING_WINDOWS`, gap helpers |
| `run_compute_scaling.py` | Compute at W=250 anchor where MI is still trusted |

**Success criterion for amortized sweeps:** on held-out `hard8_complex` at W=250/500, drive `gap_context` toward 0 vs **frozen MI_ref=0.964** (W=250) while keeping inference ~O(N) ms-scale. Routine sweeps **skip live MI**; use `run_learned_sweep.py` to grid model scale (`base`/`large`/`xl`), pool size, and epochs.

**Commands:**

```bash
# Primary: learned-only sweeps (no per-trace MI)
.venv/bin/python scripts/amortized/run_learned_sweep.py --device cpu \
  --scales base,large,xl --train-worlds 40,80 --context-epochs 40,60

# Reference table with frozen gaps (add --run-mi only to re-validate)
.venv/bin/python scripts/amortized/run_reference_benchmark.py --device cpu
```

### E13g — learned-only sweeps, model scales, always measure runtime (2026-06-02)

**Lesson:** Every benchmark row should include **train / eval / inference wall time**. E13e showed MI is ~300–1200× slower per trace than context at comparable N; without timing, accuracy-only sweeps over-weight MI. Routine work **skips live MI** and uses frozen `MI_REFERENCE_ARI` (hard8 W=250 → **0.964**).

**Compute anchor (E13e, CPU, trace→labels):** MI ~5.6 s vs context ~19 ms at N=72, W=250; MI ~O(N²), context ~linear in N and W.

**Protocol:** `run_learned_sweep.py` grids `base` / `large` / `xl` context encoders × train worlds × epochs; reports `gap_context`, `train_seconds`, `eval_seconds`, `held_out_w250_infer_ms_median`.

**Full grid** (2026-06-03, CPU, 5 seeds, frozen MI_ref=0.964, no live MI). Sorted by held-out `hard8_complex` W=250 `gap_context`:

| scale | worlds | epochs | params | train (s) | infer (ms) | hard8 ARI | gap |
|-------|--------|--------|--------|-----------|------------|-----------|-----|
| **xl** | **40** | **60** | 2.4M | 1496 | 148 | **0.810** | **0.154** |
| large | 80 | 60 | 460k | 1023 | 44 | 0.805 | 0.159 |
| xl | 80 | 40 | 2.4M | 1967 | 146 | 0.745 | 0.219 |
| xl | 40 | 40 | 2.4M | 1000 | 145 | 0.743 | 0.221 |
| large | 40 | 40 | 460k | 673 | 95 | 0.719 | 0.245 |
| large | 80 | 40 | 460k | 684 | 43 | 0.717 | 0.247 |
| large | 40 | 60 | 460k | 985 | 43 | 0.704 | 0.260 |
| base | 40 | 60 | 83k | 426 | 41 | 0.682 | 0.282 |
| xl | 80 | 60 | 2.4M | 2895 | 141 | 0.675 | 0.289 |
| base | 80 | 60 | 83k | 830 | 39 | 0.669 | 0.295 |
| base | 80 | 40 | 83k | 554 | 41 | 0.655 | 0.309 |
| base | 40 | 40 | 83k | 281 | 39 | 0.613 | 0.351 |

**Interpretation:**

- **Best config (this run):** `xl`, 40 worlds/kind, 60 epochs — gap **0.154** (ARI 0.810 vs MI_ref 0.964), inference **148 ms/trace** vs MI ~5.6 s (E13e). `large` 80/60 is close (gap 0.159, **44 ms** infer).
- **Training is seed-sensitive:** an earlier partial read of this sweep showed `large`/80/40 at gap 0.097; the final 12-config pass did not reproduce that — treat best row as run-specific until multi-seed training sweeps.
- **XL edges large slightly on gap** here but at ~3× inference ms; **base** remains far from ceiling (gap ≥ 0.28).
- **Runtime lesson confirmed:** report train_s and infer_ms every row; skip MI in the loop; use frozen ceiling for gaps.

Artifacts: `results/amortized/learned_sweep_summary.json`, `learned_sweep_full.log` (gitignored).

**Command:**

```bash
.venv/bin/python scripts/amortized/run_learned_sweep.py --device cpu \
  --scales base,large,xl --train-worlds 40,80 --context-epochs 40,60
```

### E14 — telemetry extensions (sim); real-data anchor rejected (2026-06-03)

**Sim:** Added optional periodic shared driver and heavy-tailed sensor bursts to `TraceSimulationConfig` (defaults unchanged). MI detectability matrix on extensions (periodic, heavytail, regime, episodic, saturate) passes at W≥250 (all-combined ARI ~0.87–0.95). Script: `scripts/learn_agents/telemetry_extension_detectability.py`.

**Google ClusterData 2019 — considered, rejected:** Probed as a sim-to-real bridge. Borg `collection_id` + CPU/memory usage are **comoving workload telemetry**, not UAD agents: no sensor/action/internal roles, no blanket validation, MI ARI vs job ID only measures co-movement. **Removed** adapter/downloader code; not a viable anchor.

**Real-data directions (if pursued later):** Prefer traces with **explicit per-actor observation + action channels** and falsifiable blankets — e.g. multi-agent RL / sim logs (PettingZoo, Melting Pot, SMAC), robotics state–action datasets (D4RL-style separate proprioception vs motor command), or logged control stacks with plant/controller variable split. Infrastructure telemetry (ClusterData, S3E, generic metrics) ranks low for agency.

### E14–E16 — 2026-06-04 summary (telemetry, externals, detectability, transfer)

**E14 (sim):** Optional periodic driver + heavy-tailed innovations on telemetry sim; MI detectability passes at W≥250 (`telemetry_extension_detectability.py`). ClusterData rejected (no S/A/I agency).

**E15–E16 (externals + pool):** `external_traces.pack_trace` adds env decoys on the **full** trace; **MI and amortized eval cluster agent columns only** (decoys stripped in `worlds.episode_from_result`). Single-agent physics/rock with n=1 is a **degenerate** test (one cluster always correct); success there is weak evidence.

**Scripts / artifacts:**

| Script | Role | Artifact |
|--------|------|----------|
| `external_pomdp_detectability.py` | MI @ multiple W on externals | `e15_external_pomdp_detectability.json` |
| `agent_ari_table.py` | MI + learned+UAD on sim + externals | `agent_ari_table.json` (~2 h with live spotlight) |
| `run_dataset_vs_baseline.py` | Context transfer @ W=250 | `dataset_vs_baseline.json`, `dataset_vs_baseline_extended.json` |
| `check_extended_pool.py` | Smoke build, no training | — |
| `ablate_grid_context_epochs.py` | 2× train epochs ablation on grid3 | `ablate_grid_epochs_*.json` |

**Regimes (one table):** W=250 unless noted; eval seeds 0–2; amortized train defaults **40 worlds/kind**, **40 context / 25 siamese epochs** unless ablation.

| Family | Regime | n | decoys (full trace) | T_med | MI ARI | LU ARI | ctx train | Notes |
|--------|--------|--:|---:|---:|---:|---:|---:|-------|
| easy3_redundant | sim pool, episodic=False | 3 | 6 | 2000 | 1.00 | 0.18±0.26 | 0.90 / 0.76† | MI 0.8 s; LU live 29 s |
| med5_rich | sim pool | 5 | 8 | 2000 | 1.00 | 1.00 | 0.55 / 0.67† | LU = E11 8-agent proxy |
| hard8_complex | sim held-out | 8 | 8 | 2000 | 0.94 | 1.00 | 0.72 / 0.52† | LU cache ~5.2 min |
| telemetry_* | E14 extensions, 5 agents | 5 | 8 | 2000 | 0.80–1.00 | 0.35–0.65 | — | regime/all LU use **episodic=True** sim |
| physics_cartpole | 1 pole, partial obs | 1 | 4 | ~16–24 | 1.00 | — | 1.00 | trivial n=1; ~30 ms MI |
| physics_cartpole_x3 | 3 parallel poles + per-agent drive | 3 | 8 | ~28 | ~0 | — | 0.05–0.14 | MI merges poles; ctx non-trivial (not 1.0) |
| rock_sample_5x5 | 5×5 POMDP | 1 | 4 | ~81 | 1.00 | — | 1.00 | trivial n=1 |
| grid_pomdp_3x3 | 2 agents, ego 3×3 | 2 | 4 | 250 | 0.82±0.25 | — | ~0 / **0.12**† | MI ~230 ms; **main failure** |
| grid_pomdp_5x5 | 2 agents, 5×5 | 2 | 4 | 250 | 0.40±0.30 | — | ~0 / 0.01† | held-out size |

† **ctx train:** sim-only (`easy3+med5`) / **extended** (`+physics, rock, grid3`). Extended run ~12 min (`dataset_vs_baseline_extended.json`).

**Transfer readout (E16b):** Grids: MI separates agents partially; context encoder stays near chance even **in-distribution** (extended train: grid3 ctx **0.12**, MI **0.82**). Physics/rock ctx=1.0 is **weak evidence** (n=1, amortized path drops decoys). `physics_cartpole_x3`: MI≈0, ctx≈0.05–0.14 — genuinely hard.

**Grid failure — simplest ablation (2026-06-04):** `ablate_grid_context_epochs.py` doubles context/siamese epochs (40→80, 25→50); grid3 ctx **0.027 → 0.084** (MI unchanged 0.824). **Training duration alone does not explain the gap.** Next tests: oversample grid worlds in pool, then encoder scale (`run_learned_sweep` xl). Likely drivers: mixed pool dominated by sim statistics (redundant role copies) vs 14 egocentric channels; short-window slice on long sim vs full grid T=250.

**Extended train wall time:** ~12 min/run (200 episodes, 5 kinds, CPU); 2× ablation ~33 min total.

**Harder physics:** `physics_cartpole_x3` — three independent CartPoles, 8 env decoys on full trace (`physics_pomdp.roll_cartpole_multi`).

```bash
.venv/bin/python scripts/learn_agents/agent_ari_table.py --seeds 0 1 2 --run-spotlight-missing --force
.venv/bin/python scripts/learn_agents/external_pomdp_detectability.py --sources physics physics3 grid3 grid5 --seeds 0 1 2
.venv/bin/python scripts/amortized/run_dataset_vs_baseline.py --eval-seeds 3 --force
.venv/bin/python scripts/amortized/run_dataset_vs_baseline.py --extended-pool --run-mi --force
.venv/bin/python scripts/amortized/ablate_grid_context_epochs.py --force
.venv/bin/python scripts/amortized/check_extended_pool.py
```

**Mixed heterogeneous detection (2026-06-04):** `scripts/learn_agents/mixed_detection.py` — one trace, 4 agents: CartPole(1) + RockSample(1) + grid3(2), padded to T=250, 8 decoys, fixed K=4. Artifact `results/learn_agents/mixed_detection.json`. Tests whether MI separates **different** plants (not identical poles).

| Pole policy | overall MI ARI (5 seeds) | per-agent: pole / rock / grid×2 |
|-------------|--------------------------:|----------------------------------|
| random (short, dies ~20 steps) | 0.31 ± 0.34 | 0 / ~1 / ~1 (pole padded-dead) |
| balance (controller, full 250) | **0.60 ± 0.26** | **0** / 1 / 1 |

**Finding — heterogeneous plants ARE separable, but CartPole is a degenerate probe for co-movement MI.** Rock + both grid agents cluster cleanly (per-agent ARI 1.0). The pole stays at **0 regardless of episode length** for two compounding reasons: (1) **random** action → transient agent (absent in long windows); (2) **balance** controller → the regulated variables (angle, angular velocity) are driven near-flat, so the agent's own channels stop co-moving (good-regulator effect), and its few smooth channels **merge with the other smooth low-channel agent (rock)**. MI co-movement clustering needs an agent whose channels sustain a distinctive correlated dance; a regulated 1-DOF controller does not provide one. Oracle ε-blanket sep ratios for the balanced pole are ~0.89–0.96 (poor) — even the blanket/role test is low-evidence for this tiny system.

**Implication:** prefer agents with persistent, distinctive internal dynamics (grid / foraging / pursuit / Melting Pot); treat CartPole as an edge case, not a canonical agent. Controllers, if targeted, need the S→A→I role/blanket path, not co-movement MI.

**Open experiments (paused, 2026-06-04):** Pick up in this order when resuming the E14–E16 line.

| Priority | Experiment | Question / lever | Script / hook |
|----------|------------|------------------|---------------|
| 1 | **Grid context transfer** | Why ctx ≪ MI on grid3/5 despite extended pool + 2× epochs? | Oversample `grid_pomdp_3x3` in `generate_pool`; `run_learned_sweep.py` xl; optional per-kind window policy |
| 2 | **Melting Pot bridge** | Does structured multi-agent obs train/eval like grid? | `pip install dm-env meltingpot`; `verify_external_deps.py --melting-pot`; `check_extended_pool.py --melting-pot`; `run_reference_benchmark.py --extended-pool` |
| 3 | **Mixed detection — sim + external** | Does MI separate **telemetry sim agents** from grid/rock in one trace (original intent)? | Extend `mixed_detection.py` with `easy3`/`med5` slices + `merge_agent_traces` |
| 4 | **Mixed detection — persistent agents only** | Drop CartPole; rock + grid + new foraging/pursuit env | New logger + mixed builder |
| 5 | **Spotlight on externals** | Learned+UAD on E15 traces (not MI-only today) | Wire `agent_spotlight` to `SimulationResult` / external peel |
| 6 | **Episodic telemetry ablation** | Regime/all spotlight used `episodic=True`; fair compare at `episodic=False` | Re-run `agent_ari_table.py` with matched sim config |
| 7 | **Controller / blanket path** | Detect regulated CartPole via S→A→I lagged structure, not co-movement MI | `oracle_uad_scores` + strict UAD on role subsets; not `mi_cluster_variable_labels` |
| 8 | **Transient agents** | Short-lived agent among long-lived (E13 thesis) | Time-localized or change-point MI; episodic sim as testbed |
| 9 | **Location invariance** | Grid transfer vs ego-frame randomization | `grid_pomdp` random rotation/reflection per seed; add Kind + detectability row |
| 10 | **Amortized hygiene** | Frozen external MI refs; variable N across kinds in one batch | Extend `benchmark.MI_REFERENCE_ARI`; document batching limits |

**Not planned here:** three identical CartPoles (`physics_cartpole_x3` for separation); ClusterData-style infra telemetry; full episodic-limitations study until grid or Melting Pot bridge moves.

---

## E17 — Option D: homeostatic regulation probe (2026-06-05)

**Why:** Start the **intention-detection** derivative line with the smallest falsifiable test: does a data-only probe detect *disturbance-rejection regulation* (good-regulator / setpoint maintenance) without labels or known rewards? Option D scopes **homeostasis**, not pursuit or navigation — complementary to later EIS/compression-gain (Option A).

**Package / scripts:**

| Component | Path |
|-----------|------|
| Regulation probe | `learn_agents/regulation_probe.py` |
| CartPole track variant | `learn_agents/physics_pomdp.py` (`policy="track"`, `theta_ref=0.12`) |
| External kind | `physics_cartpole_track` in `external_registry.py`, `amortized_agency/kinds.py` |
| Eval script | `scripts/learn_agents/run_regulation_probe.py` |
| Tests | `tests/test_regulation_probe.py` |

**Probe (per agent, paired internal↔sensor by index):**

- **Flatness** `F = max(0, 1 − Var(internal)/Var(paired_sensor))`
- **Compensation** `K = max(0, −corr(a_{t−1}, s_t))`
- **Regulation** `R = F × K`, forced to **0** when `Var(internal)/Var(sensor) > 0.012` (actively maintained internal, not suppressed)
- **Flag** when `max_internal R ≥ 0.15` and trace `T ≥ 80`

Physics rollouts use `pack_trace(..., normalize=False)` so variance ratios are meaningful (per-column z-score was collapsing all `F` to 0).

**CartPole track:** tracks `θ_ref=0.12` rad (below env failure threshold ~0.209). Contrast pair with `policy="balance"` on the same S/A/I layout.

**Settings:** 5 seeds `{0…4}`; telemetry kinds use default sim (`episodic=False`, `T=2000`); physics `normalize=False`.

**Artifact:** `results/intention/e17_regulation_probe.json`

| Family | Mean max R | Flagged rate | Interpretation |
|--------|------------|--------------|----------------|
| **physics_cartpole_balance** | **0.660 ± 0.016** | **100%** | homeostatic positive control ✓ |
| physics_cartpole_track | 0.000 | 0% | intentional but **active** internal (ratio ~0.6–0.8) ✓ |
| physics_cartpole_random | 0.000 | 0% | short / unregulated ✓ |
| telemetry easy3 / med5 / hard8 | 0.000 | 0% | reactive, no setpoint — negative control ✓ |
| rock_sample_5x5 | 0.015 | 0% | navigation, not homeostatic ✓ |
| grid_pomdp_3x3 | 0.000 | 0% | random policy, not homeostatic ✓ |

**Balance detail (seed 0):** `pole_ang` paired with `pole_ang_v` — F≈0.997, K≈0.685, R≈0.683, active_ratio≈0.003.

**Track detail (seed 0):** same pairing — F≈0.98, K≈0.77 but active_ratio≈0.62 → R zeroed (not flagged).

**Conclusion:** Option D **works as scoped**: flags homeostatic regulation (balance), rejects reactive telemetry and pursuit-style track. It does **not** detect achievement/pursuit intentions — that is the gap for **Option A** (EIS compression gain / goal-conditioned action likelihood).

**Reproduce:**

```bash
.venv/bin/python scripts/learn_agents/run_regulation_probe.py --seeds 0 1 2 3 4
```

**Next:** E19 real-machine eval; optional EIS compression gain (Option A); wire E18 into spotlight.

---

## E18 — Outcome-influence detection on labeled critical variables (2026-06-05)

**Why:** Operational intention detection for ops-style traces: label critical outcomes (`resource.cpu`, pole angle, …) and test whether each agent cluster **defends or steers** those variables after controlling for exogenous world state. Complements E17 (internal homeostasis only).

**Package / scripts:**

| Component | Path |
|-----------|------|
| Outcome influence | `intention_detect/` (`outcomes.py`, `influence.py`, `defense.py`, `evaluate.py`) |
| Sim: resource channels + self-preserving agent | `learn_agents/learn_agents.py` (`resource_vars`, `self_preserving_agent`, `normalize_trace=False`) |
| Eval script | `scripts/intention/run_outcome_influence.py` |
| Tests | `tests/test_outcome_influence.py` |

**Labeled outcomes:** `metadata["critical_outcomes"]` — `{name, index, direction}`. Telemetry sim adds `resource.cpu` / `resource.memory`; CartPole uses `pole_ang` via `attach_physics_critical_outcome`.

**Scores per (agent, outcome):**

- **Partial influence:** signed corr\((a_{t-1}, \Delta o_t)\) given world controls
- **Defense OR:** mean \(|a|\) when outcome bad vs good (bootstrap 90% CI), after residualizing on world
- **Selectivity:** OR on high-|Δo| vs low-|Δo| windows (pipeline-style confound strip)
- **Flag:** strong defense path (`OR≥1.40`, `infl>0`) or strong-OR path (`OR≥1.65`); **or** control path (`|infl|≥0.25`, selectivity ≥0.78) for regulators/pursuit; **or** driver path (`|infl|≥0.30`, bypasses selectivity) for resource attackers (E19b)
- **Per agent:** flagged if **any** outcome flags (not only highest combined score)
- **Segmentation (E19c):** on long sparse traces (`segment_mode=auto`), score sliding + activity windows; segment flag only if partial influence materially beats full-trace (`intention_detect/segmentation.py`)

**Sim families (5 seeds, T=2000, raw trace):**

| Family | Ground-truth influencer | Agent-level accuracy |
|--------|---------------------------|----------------------|
| telemetry reactive | none | **14/15** |
| telemetry self-preserving (agent 0) | agent 0 | **13/15** |
| physics_cartpole balance | agent 0 | **5/5** |
| physics_cartpole track | agent 0 | **5/5** |
| physics_cartpole random | none | skipped (T&lt;80) |

**Pooled AUROC** (combined score, n=40 agent rows): **0.941**

**Artifact:** `results/intention/e18_outcome_influence.json`

**Reproduce:**

```bash
.venv/bin/python scripts/intention/run_outcome_influence.py --seeds 0 1 2 3 4
```

**Conclusion:** Labeling a few critical variables is enough to separate (1) reactive telemetry, (2) self-preserving resource defense, and (3) control/pursuit on pole angle — with one false positive on reactive telemetry across 15 agent-rows. E17 and E18 are complementary: E17 flags homeostatic suppression; E18 flags outcome-directed action.

---

## E19 — Real-machine outcome-influence dataset (2026-06-05)

**Why:** E18 passed on sim and physics; the next falsifier is **real CPU/RAM telemetry** with deployment-pipeline-style confounds: a shared exogenous stressor, agents that genuinely influence outcomes, and a bystander that correlates with the stressor schedule without causing CPU load.

**Package / scripts:**

| Component | Path |
|-----------|------|
| Worker roles | `data_collect/workers.py` (stressor, cpu_regulator, deadline_burster, mem_grabber, fixed_worker, bystander) |
| Orchestrator + recorder | `data_collect/run.py` |
| Pack to SimulationResult | `data_collect/pack_run.py` |
| CLI | `scripts/intention/run_machine_dataset.py` |
| Dependency | `psutil` in `requirements-dev.txt` (use project `.venv`) |

**Process design (5 agents + world stressor):**

| Process | S/A/I role | Influences critical outcome? |
|---------|------------|------------------------------|
| background stressor | world control (`world.stressor_active`) | exogenous W only |
| cpu_regulator (A0) | reads live CPU, burns/throttles to setpoint | yes (CPU) |
| deadline_burster (A1) | periodic CPU bursts | yes (CPU) |
| mem_grabber (A2) | allocates/releases RAM chunks | yes (RAM) |
| fixed_worker (B1) | steady low activity | no |
| bystander (B2) | mirrors stressor schedule via disk I/O | no (confound) |

**Critical outcomes (two scenarios via `--scenario global|owned|both`):**

| Scenario | Outcomes | Question |
|----------|----------|----------|
| **global** | `resource.cpu_percent`, `resource.ram_used_frac` | Does the agent move **system-wide** pressure beyond its own footprint? |
| **owned** | `owned.<agent>.cpu`, `owned.<agent>.rss_mb` per process | Boring baseline — does it move **its own** resources? |

Parent recorder logs global metrics + per-process owned CPU/RSS (`proc_usage.jsonl`). World controls: stressor flag + slow time phase. Traces packed with `normalize=False`.

**Run budget:** up to **4 CPU cores**; default **1200 s** (20 min) or **1800 s** (30 min), `dt=1 s`.

**Artifacts (local, gitignored):**

- Raw JSONL per run: `results/intention/machine_runs/<timestamp>/`
- Scored summary: `results/intention/e19_machine_dataset.json`
- Run log: `results/intention/e19_run.log`

### E19a — first real-machine run (v1 harness, T=1800)

Run `machine_runs/20260605T073554Z`. Collection succeeded; scoring initially crashed on a label-key bug (fixed).

| Agent | GT | Flagged | Notes |
|-------|-----|---------|-------|
| cpu_regulator | yes | no | weak lagged CPU coupling |
| deadline_burster | yes | no | sub-second bursts, OR below threshold |
| mem_grabber | yes | yes | spurious RAM OR on flat signal |
| fixed_worker | no | no | ✓ |
| bystander | no | yes | false positive on noisy RAM |

**Agent-level accuracy 2/5; pooled AUROC 0.333.**

**Root causes (not training or sample count — E18 is training-free, T=1800):**

1. **SNR:** 1-core bursts ≈12.5% on 8 cores, sampled at 1 Hz → `corr(action_lag, Δcpu) ≈ 0` for all agents; exogenous stressor dominated (`corr(stressor,cpu)=0.63`).
2. **Outcome not agent-controllable:** mem_grabber held ~400 MB on 24 GB → RAM range ~1.3 GB from OS noise; `corr(chunks, ram)≈−0.15`.
3. **Sub-second effects:** CPU bursts &lt;1 s within 1 s ticks → aliasing.
4. **Defense-only semantics:** flag logic required `infl>0` (defender); RAM *drivers* show `infl<0` and were ranked highly but not flagged.
5. **Per-agent aggregation bug:** agent flagged only if its single highest-`combined` outcome flagged, not if *any* outcome flagged.

### E19b — revised harness + detector fixes (2026-06-05)

**Harness changes (`data_collect/`):**

- Multi-core burns spanning **4–10+ ticks** (full-tick `burn_cores`: 1 in-process core + helper subprocesses for “beyond footprint”).
- **mem_grabber:** 2.5 GB target in **256 MB random-filled** numpy chunks (incompressible), held 10+ ticks → verified **+12 pp / ~2.9 GB** global RAM, `corr(held, ram)=+0.84`.
- **cpu_regulator / deadline_burster:** 2-core sustained bursts; regulator setpoint 55%.
- Parent **per-process owned CPU/RSS** in `proc_usage.jsonl`.
- CLI **`--scenario global|owned|both`**.

**E18 detector fixes (`intention_detect/evaluate.py`):**

- **Driver/attacker path:** large `|influence| ≥ 0.30` flags regardless of sign (process *driving* resource toward bad); E18 sim regression unchanged (AUROC **0.941**).
- **Any-outcome aggregation:** agent flagged if **any** critical outcome flags.

**150 s smoke (`machine_runs/20260605T142210Z`):**

| Scenario | AUROC | Agent acc | Notes |
|----------|-------|-----------|-------|
| global | **1.000** | **4/5** | mem_grabber ✓, deadline_burster ✓; cpu_regulator miss; fixed_worker & bystander ✓ |
| owned | 0.667 | 3/5 | baseline separation (boring per-process case) |

Physical signal confirmed: CPU mean 44.8% (0–90%), RAM Δ12 pp, mem held up to 2560 MB.

**20-min run (v2, both scenarios):** `machine_runs/20260605T143100Z`, T=1200.

| Scenario | AUROC | Agent acc | Per-agent |
|----------|-------|-----------|-----------|
| **global** | 0.333 | **3/5** | mem_grabber ✓; fixed_worker ✓; bystander ✓; cpu_regulator ✗; deadline_burster ✗ |
| **owned** | 0.667 | **4/5** | deadline_burster ✓; mem_grabber ✓; fixed_worker ✓; bystander ✓; cpu_regulator ✗ |

Longer run regressed vs 150 s smoke (global 4/5, AUROC 1.0): over 20 min OS churn and higher stressor duty dilute full-trace partial CPU influence for burster/regulator (`|infl|≈0.01` on CPU). RAM driver (mem_grabber) remains stable; confound controls hold (no bystander FP on full-trace score). **Owned scenario** separates better (4/5) — per-process attribution is easier than global beyond-footprint.

### E19c — auto segmentation for episodic long runs (2026-06-05)

**Why:** Full-trace linear partial influence **averages over idle ticks**; episodic agents (12–52% active) lose SNR as T grows even when burst windows remain informative.

**Package:** `intention_detect/segmentation.py`; CLI `--segment-mode auto|full|segmented` (default **auto**).

**Auto-calibration (no manual window size):**

| Knob | Rule |
|------|------|
| Enable | `T≥250` and sparse agent mix (min active duty &lt;22%, median &lt;40%), **or** `metadata["prefer_segment_scoring"]` (E19 packs set this) |
| Window `W` | `clip(T/6, 80, 300)`, step `W/2` |
| Activity threshold | `median(action) + coef·std(action)`; coef rises for sparser agents; constant agents → sliding windows only |
| Segment flag | Full-trace flag **or** segment flag when `\|infl_seg\| ≥ max(0.25, \|infl_full\|×1.1 + 0.02)` (blocks OR-only window false positives) |

**Re-score of E19b 20-min run** (`20260605T143100Z`) with `segment_mode=auto`:

| Scenario | Full-trace (E19b) | Segmented (E19c) |
|----------|-------------------|------------------|
| global AUROC / acc | 0.333 / **3/5** | **0.667** / **3/5** |
| owned AUROC / acc | 0.667 / **4/5** | (same run; owned benefits less — dense per-process channels) |

Segmentation **improves ranking** on global (AUROC 0.33→0.67) without re-collection; agent accuracy unchanged at 3/5 because `deadline_burster` best window still reaches only `|infl|≈0.07` (below 0.25 floor). E18 sim regression unchanged (AUROC **0.941**).

**Reproduce segmentation:**

```bash
.venv/bin/python scripts/intention/run_machine_dataset.py \
  --score-only results/intention/machine_runs/<timestamp> \
  --scenario both --segment-mode auto
```

**Open ends (E19):**

1. **Global CPU influencers** — regulator/burster: raise burst SNR or lower segment influence floor when `|infl_seg| ≫ |infl_full|` (relative gain).
2. **Activity-only windows** — score burst-aligned segments only (not all sliding windows) to reduce OR-driven segment noise.
3. **Richer world controls** — continuous stressor load channel, not just 0/1 flag.
4. **Owned vs global gap** — owned 4/5 vs global 3/5 confirms ops worry (beyond-footprint attribution) is strictly harder.
5. **Wire E18 into spotlight** post-UAD; optional EIS compression gain (Option A).

**Reproduce collection:**

```bash
python3 -m venv .venv && .venv/bin/pip install psutil numpy
.venv/bin/python scripts/intention/run_machine_dataset.py \
  --duration 1200 --dt 1 --max-cores 4 --stressor-cores 2 --scenario both
```

Score an existing run:

```bash
.venv/bin/python scripts/intention/run_machine_dataset.py \
  --score-only results/intention/machine_runs/<timestamp> --scenario both
```

**Conclusion so far:** Off-simulator failure was **measurement design, episodic aggregation, and detector semantics** — not lack of data or training. v2 harness + driver path + segmentation close most gaps for RAM drivers and confounds; **global CPU attribution** (regulator, burster) remains open.

---

## E20 — UAD on real C. elegans whole-brain data (WormWideWeb) (2026-06-06)

**Why:** apply Unsupervised Agent Discovery to real *C. elegans* calcium imaging (Atanas & Kim 2023). Tests whether the blanket criterion finds a statistically distinguishable agentive subsystem in a real, densely-coupled nervous system with no ground truth. Scoped plan + scaffold: [`uad_worm/README.md`](../uad_worm/README.md); planning/M0–M1 in CHANGELOG 2026-06-05.

**Package / scripts:**

| Component | Path |
|-----------|------|
| CMI + blanket loss + nulls + synthetic | `uad_worm/{cmi,blanket,nulls,synth}.py` |
| Ingestion (fetch/cache/provenance) | `uad_worm/data.py` |
| Preprocess + whiten (M2) | `uad_worm/preprocess.py` |
| Candidates: lagged-corr communities + anchor (M3) | `uad_worm/candidates.py` |
| Roles + E-reduced blanket loss + pooled LOAO (M4) | `uad_worm/score.py` |
| Random-class-set null + behavior gain (M5/M6) | `uad_worm/evaluate.py` |
| Runner / probe | `scripts/worm/run_discovery.py`, `scripts/worm/probe.py` |

**v1 method:** pool the same `neuron_class`-defined candidate across 8 NeuroPAL-Baseline animals (T≈1600 @ 0.6 s); whiten (temporal derivative) to reduce GCaMP autocorrelation; assign S/A/I roles by lagged influence to/from the rest of the brain; reduce the external set to PCs; score blanket loss `I(I_{t+1};E_{t+1}|S_t,A_t)` against a random-partition contrast; leave-one-animal-out is the headline.

**Headline result (negative, robust).** Locomotor command-circuit anchor (`AIB AVA AVB AVD AVE PVC RIB RIM`): pooled `pass_rate 0/8`, **leave-one-animal-out 0/8**, combined p≈0.36. It is marginally below biologically-matched random *class sets* (random-class-set null z=−1.35, p≈0.10) but **not** below random same-size *neuron* partitions (median p≈0.5). Unsupervised recurrent candidate is worse (z=+2.56). Behavior-prediction gain small (anchor +0.013, recurrent +0.065).

**Probe (`scripts/worm/probe.py`) — why 0/8:** ruled out the obvious knobs.

| Probe | Result | Verdict |
|-------|--------|---------|
| P1 null reference (all-neuron vs labeled-only) | median p 0.41 vs 0.37 | not a null-reference artifact |
| P2 representation (whitened vs raw) | raw median p 0.58 (worse) | whitened is better but still 0/8 |
| P3 external rank (ext_dim 4→20) | pass 0→1/8, median p stays ≈0.5 | more E-PCs sharpen slightly, not decisive |

The command-circuit blanket loss sits **in the middle** of the random same-size partition distribution across every configuration ⇒ a **genuine** negative, not a tuning artifact: at the class level, lag-1, Gaussian-CMI, PC-reduced, the command circuit is not a distinguishable Markov blanket in this cohort.

**Exploration (post-M6, `scripts/worm/explore.py`).** Probed the negative on three axes.

- **X1 timescale — ruled out.** Lag sweep 1→5 (0.6→3.0 s) leaves the anchor at median p≈0.36–0.43; a slower conditioning lag does not reveal a blanket.
- **Added a second axis — internal autonomy `I(C_{t+1};C_t | E_t)`** (self-prediction *beyond* the environment) to separate a real coupled subsystem from disconnected low-loss noise. **Methodological correction:** a naive self-prediction R² is invalid — it rewards *redundancy* (a shared-latent block out-scores a true controller; verified on synthetic). Conditioning on the (PC-reduced) environment fixes it (synthetic agent ≈0.70 vs redundant block ≈0.01). Agent signature = **low blanket loss AND high internal autonomy**.
- **X2 anchor in the (autonomy, loss) plane.** Heterogeneous: the command circuit lands in the agent corner in only **3/8** animals; median autonomy_p≈0.33 (below chance). Consistent with the 0/8 LOAO — no generalizable agent.
- **X3 per-animal unsupervised communities.** Lagged-corr communities are internally coupled by construction (autonomy_p≈1.0) but **leak** (loss_p 0.56–0.80); only **1/4** has a borderline agent-corner module (AIM/ASE/RMD, loss_p≈0.47).

**Interpretation:** the 2D plane shows *why* — in this cohort, coupling and encapsulation are **anti-located** (coupled subsystems leak; low-loss sets aren't internally driven), so the agent corner is sparsely and only weakly populated. The negative is structural, not a tuning/timescale artifact. Weak per-animal candidates (command circuit in 3 animals; AIM/ASE/RMD) do not generalize.

**Open ends (E20):** (1) nonlinearity — Gaussian CMI may miss nonlinear dependence (kNN/kernel CMI). (2) stricter agent-corner thresholds + per-animal candidate stability across seeds. (3) larger cohort / Heat-vs-Baseline contrast. (4) M7 memory localization (deferred).

**Reproduce:**

```bash
PYTHONPATH=. .venv/bin/python scripts/worm/run_discovery.py --max-animals 8 --n-perm 100
PYTHONPATH=. .venv/bin/python scripts/worm/probe.py
PYTHONPATH=. .venv/bin/python scripts/worm/explore.py
PYTHONPATH=. .venv/bin/python scripts/worm/export_assignments.py --max-animals 8
```

---

## Code map

| Component | Path |
|-----------|------|
| Simulator + refine | `learn_agents/learn_agents.py` |
| Debug protocol | `scripts/learn_agents/debug_learn_agents_protocol.py` |
| Candidate eval | `scripts/learn_agents/evaluate_latent_candidates_with_uad.py` |
| Agent-count sweep | `scripts/learn_agents/learn_agents_agent_count_sweep.py` |
| E14 telemetry extensions + detectability | `scripts/learn_agents/telemetry_extension_detectability.py` |
| E15 external POMDP loggers + detectability | `learn_agents/external_traces.py`, `physics_pomdp.py`, `rock_sample.py`, `grid_pomdp.py`, `scripts/learn_agents/external_pomdp_detectability.py` |
| MI vs learned+UAD summary table | `scripts/learn_agents/agent_ari_table.py` |
| Decoy ablation | `scripts/decoys/decoy_ablation_sweep.py` |
| E12 hierarchical fusion | `hierarchical_spotlight/` |
| E12b complex hierarchy sweep | `scripts/hierarchical/run_hierarchical_e12b_sweep.py` |
| E13 amortized agency | `amortized_agency/`, `scripts/amortized/` |
| E16 extended pool + Melting Pot | `learn_agents/external_registry.py`, `melting_pot.py`, `amortized_agency/kinds.py`, `scripts/amortized/check_extended_pool.py` |
| E16b dataset vs sim baseline | `scripts/amortized/run_dataset_vs_baseline.py`, `ablate_grid_context_epochs.py` |
| E17 regulation probe (Option D) | `learn_agents/regulation_probe.py`, `physics_pomdp.py` (track policy), `scripts/learn_agents/run_regulation_probe.py` |
| E18 outcome influence | `intention_detect/`, `intention_detect/segmentation.py`, `scripts/intention/run_outcome_influence.py` |
| E19 real-machine dataset | `data_collect/`, `scripts/intention/run_machine_dataset.py` |
| Multi-agent physics | `learn_agents/physics_pomdp.py` (`roll_cartpole_multi`) |
| Strict UAD / MI roles | `agency_detect/markov_blanket.py` |

