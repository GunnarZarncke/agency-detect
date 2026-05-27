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

**Script:** `scripts/learn_agents_go_nogo_sweep.py`

**Settings:** 8 agents, low noise, no decoys; sweep `num_slots`, seeds, epoch checkpoints.

**Results:** Best `var_acc` ~0.47 multi-seed — far below chance at 8 agents (0.125) but not usable as final classifier. Motivated reframing as **candidate proposer** not identifier.

**Artifacts:** `results/learn_agents_go_nogo_8agents.jsonl`, `results/learn_agents_go_nogo_smoke.jsonl`

---

## E1 — Four-step candidate + strict UAD pipeline (May 2026)

**Why:** Evaluate latent slots as **candidate generator** filtered by strict Markov-blanket validation; report Recall@K and post-UAD precision/recall.

**Script:** `scripts/evaluate_latent_candidates_with_uad.py`

**Key runs:**

| Run | Agents | Decoys | Adapt | Pre R@30 | Post R@30 | Notes |
|-----|--------|--------|-------|----------|-----------|-------|
| seed1 | 3 | 2 | MDL | moderate | low | Early pipeline |
| seed1 | 8 | 0 | MDL | ~0.38 | ~0 | Many strict survivors, poor hits |
| reverted | 3/8 | as above | off | 0.67 / 0.38 | — | Before adapt degeneracy analysis |
| adapt | 3/8 | as above | ε-only | — | shrink to ~2–3 vars/set | Motivated MDL anti-shrink |

**Artifacts:** `results/candidate_uad_eval_*agents_seed1*.json`, `*_reverted.json`, `*_adapt.json`

---

## E2 — Six-stage debug protocol (May 2026)

**Why:** Isolate failure mode: environment vs representation vs validation vs search.

**Script:** `scripts/debug_learn_agents_protocol.py`

**Stages:** (1) oracle ε-blanket, (2) raw MI clustering, (3) slot persistence, (4) interface/roles [metadata only], (5) ε vs MDL adapt, (6) UAD threshold sweep.

### E2a — Initial runs (v1)

**Finding:** Strict UAD rejected **all** ground-truth clusters (`"no internal variables"`). Looked like environment failure.

**Artifacts:** `results/debug_protocol_3agents.json`, `results/debug_protocol_8agents.json`

### E2b — After name-hint classifier fix (v2)

**Why:** Test whether oracle passes when `internal` in variable name used for S/A/I.

**Finding:** Oracle and raw MI passed; confirmed **classifier bug**, not missing structure. User rejected name hints as cheating for real data.

**Artifacts:** `results/debug_protocol_*_v2.json`

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

**Artifacts:** `results/debug_protocol_*_v3.json`

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

**Artifacts:** `results/candidate_uad_eval_8agents_v2_no_adapt.json`, `*_v2_mdl.json`

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

**Artifacts:** `results/candidate_uad_eval_8agents_mi_refine.json`, `*_8agents_baseline.json`, `*_3agents_mi_refine.json`

---

## E5 — Agent-count sweep, 0% decoys (May 2026)

**Why:** Explain “8 agents better than 3” — test 1–12 agents under **comparable** clean settings.

**Script:** `scripts/learn_agents_agent_count_sweep.py`

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

**Artifact:** `results/agent_count_sweep.json`

---

## E6 — Agent-count sweep, 20% and 70% decoys (May 2026)

**Why:** Decoys strongly hurt; test fractional decoy load (`decoy/(agent+decoy)`).

**Formula:** `decoy_vars = round(f × agent_vars / (1−f))`, `agent_vars = 3 × copies_per_role × num_agents`.

### E6a — 20% decoys

**Artifact:** `results/agent_count_sweep_decoy20pct.json`

| Agents | MI recall | Baseline R@30 | Refine R@30 |
|--------|-----------|---------------|-------------|
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 0.50 | 1.00 | 1.00 |
| 3 | **0.33** | 1.00 | **0.33** |
| 8 | 0.38 | 0.12 | 0.38 |

**Breaking points:** MI &lt; 1.0 from ≥2 agents; refine capped by MI from ≥3 agents.

### E6b — 70% decoys

**Artifact:** `results/agent_count_sweep_decoy70pct.json`

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

**Script:** `scripts/decoy_ablation_sweep.py`

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

**Artifacts:** `results/decoy_ablation_smoke.json`; full run → `results/decoy_ablation_sweep.json` (in progress); **core subset** → `results/decoy_ablation_core.json` ✓

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
.venv/bin/python scripts/decoy_ablation_sweep.py \
  --output-json results/decoy_ablation_sweep.json
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
.venv/bin/python scripts/evaluate_latent_candidates_with_uad.py \
  --num-agents 8 --mi-refine --mi-k-selection downstream \
  --decoy-vars 12 --T 4000 --epochs 50 --no-adapt-blankets
```

### E8 — End-to-end validation loop (8 agents, 20% decoys, downstream K)

**Artifact:** `results/candidate_uad_eval_8agents_decoy20_downstream.json`

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

## Code map

| Component | Path |
|-----------|------|
| Simulator + refine | `learn_agents/learn_agents.py` |
| Debug protocol | `scripts/debug_learn_agents_protocol.py` |
| Candidate eval | `scripts/evaluate_latent_candidates_with_uad.py` |
| Agent-count sweep | `scripts/learn_agents_agent_count_sweep.py` |
| Decoy ablation | `scripts/decoy_ablation_sweep.py` |
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
