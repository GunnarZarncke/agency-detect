# learn_agents experiment log

Chronological record of **why** each experiment was run, **settings**, **results**, and **artifact paths**.
Analysis and recommendations: [`docs/learn_agents_debug_findings.md`](../docs/learn_agents_debug_findings.md).

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

**Architecture:** separate package for one-agent-at-a-time discovery. See [`agent_spotlight/README.md`](../agent_spotlight/README.md).

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

## Code map

| Component | Path |
|-----------|------|
| Simulator + refine | `learn_agents/learn_agents.py` |
| Debug protocol | `scripts/learn_agents/debug_learn_agents_protocol.py` |
| Candidate eval | `scripts/learn_agents/evaluate_latent_candidates_with_uad.py` |
| Agent-count sweep | `scripts/learn_agents/learn_agents_agent_count_sweep.py` |
| Decoy ablation | `scripts/decoys/decoy_ablation_sweep.py` |
| E12 hierarchical fusion | `hierarchical_spotlight/` |
| E12b complex hierarchy sweep | `scripts/hierarchical/run_hierarchical_e12b_sweep.py` |
| Strict UAD / MI roles | `agency_detect/markov_blanket.py` |

---

## Git commits (experiment-related)

| Commit | Summary |
|--------|---------|
| `7e86139` | Debug findings doc, debug protocol, name-free classifier |
| `a577ddc` | MI-guided latent refinement |
| `8ec7069` | Agent-count sweep 1–12 |
| `50554e8` | Decoy fraction sweeps 20%/70% |
| `5cec8b4` | Decoy types/intensities, variable-K MI |
