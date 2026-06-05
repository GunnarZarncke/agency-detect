# Research findings (interpretation)

**Run log:** [`EXPERIMENTS.md`](EXPERIMENTS.md) (E0 onward)  
**Artifacts:** [`results/README.md`](../results/README.md)  
**Pre–E0 core detector:** `agency_detect/` (see EXPERIMENTS.md § work before this log)

This file is the **interpretation layer** for the **latent-UAD / telemetry-sim line (E0–E8)**—environment vs representation vs validation, decoys, failure modes. For E9+ (spotlight, hierarchy, amortized, externals), conclusions live in EXPERIMENTS sections and [`conversations/`](conversations/README.md). Renamed from `learn_agents_debug_findings.md` (2026-06-04) because it is repo-wide methodology, not a `learn_agents/` package doc.

---

## Goal

Determine whether failures in latent agent discovery come from (1) the simulated environment, (2) the learned representation, or (3) the candidate / validation pipeline.

The latent model is a **candidate proposer**, not a final agent identifier. Primary metrics: **Recall@K** (true agents in top candidates) and post-UAD precision/recall.

---

## Methodological constraints

1. **No variable names for S/A/I classification** — names/metadata only for scoring (`agency_detect/markov_blanket.py` uses MI-only roles).
2. **No fixed agent count for MI init** (from E7) — `mi_partition_search()` sweeps K via MDL (λ=0.02); legacy fixed K = `num_agents` available via `--mi-fixed-k-agents`.
3. **Strict UAD is falsification**, not sole success criterion — high `strict_count` with low precision observed at 8 agents.
4. **The random-partition contrast needs a densely-coupled background to be informative** (E20). Low blanket loss `I(I_{t+1};E_{t+1}|S,A)≈0` is achieved by *both* a true autonomous subsystem *and* a fully disconnected/isolated noise variable. The contrast (candidate vs random same-size cuts) only discriminates when most surrounding variables are mutually coupled, so random cuts leak (high loss) while the blanket stays low — the regime of a real brain. In a substrate with many independent variables, random cuts also score low and the contrast collapses (observed directly on synthetic toys: an isolated decoy scores as "blanket-like" as the agent). Practical consequence: a blanket score must be paired with a **predictive-coupling / internal-dynamics term** so disconnected sets cannot pass on low CMI alone.

---

## Consolidated conclusions

### Environment vs algorithm

- Gaussian oracle ε-blankets are strong (sep. ratio ~0.04–0.11 at 3/8 agents).
- Raw lagged-MI clustering recovers **all 8 agents** (J=1.0) with 0% decoys before any neural model.
- Failures are **not** missing simulator structure.

### Representation (latent slots)

- Slots are **temporally stable** (cross-chunk cosine ~0.99) but **mix agents** (purity ~0.26–0.43 at 8 agents).
- Baseline slot R@30 drops below 0.5 at **≥7 agents** (clean regime).
- **MI refine** transfers MI partition into slots → R@30 **1.0 at 8 agents** when MI recall is 1.0.

### Validation & search

- Initial strict UAD failures were partly **role classifier** issues (later fixed without names).
- ε-only candidate adapt **shrinks** sets; **MDL penalty** helps size/recall, hurts precision.
- Post-UAD precision ~0.16 at 8 agents with many false strict survivors.

### Decoys (E5–E7)

| Regime | MI breaks | Refine helps? | Mechanism |
|--------|-----------|---------------|-----------|
| 0% decoys | never (1–12 agents) | yes → ~12 agents | slot learning without init |
| 20% decoys | ≥2 agents | only if MI OK | MI partition corruption |
| 70% decoys | ≥2 agents | no | SNR / decoy-dominated trace |

Decoys **steal MI clusters** and **pollute candidates** (~30% decoy vars at 20%); they do not usually **pass as agents** but break partition discovery. **Confound/AR(1) decoys** correlate and are “shouty” in MI; **i.i.d. noise** is milder. **Variable K** fixes many cases where fixed K = num_agents failed.

**Why 8 looked better than 3 early on:** incomparable runs (8 agents clean vs 3 agents + ~25% decoys).

---

## Failure-mode diagram

```
Environment OK (oracle separability)
        │
        ├─ 0% decoys: MI perfect → refine fixes slots up to ~12 agents
        │
        ├─ 20% decoys: MI breaks first → refine capped by MI
        │              latent baseline can still show high R@30 while MI broken
        │
        └─ 70% decoys: total collapse (multi-agent)
```

---

## Open work (see also EXPERIMENTS.md — open experiments, E14–E16)

1. ~~**Step-2 K selection**~~ — implemented (`mi_k_selection=downstream` default).
2. ~~**Background factorization**~~ — rank-1 PCA before MI (default on in refine).
3. ~~**Precursor gates**~~ — persistence + contingency; used in K scoring and candidate filter.
4. Decoy mechanism ablation completion (E7 full sweep with `refine_downstream_k` column).
5. Decoy-only cluster UAD audit (P4).

---

## Reproduce (common commands)

```bash
# Debug protocol
.venv/bin/python scripts/learn_agents/debug_learn_agents_protocol.py \
  --num-agents 8 --num-slots 24 --epochs 50 --T 4000 --seed 1 \
  --output-json results/learn_agents/debug_protocol/debug_protocol_8agents_v3.json

# Candidate eval + MI refine (variable K default)
.venv/bin/python scripts/learn_agents/evaluate_latent_candidates_with_uad.py \
  --num-agents 8 --mi-refine --no-adapt-blankets \
  --output-json results/learn_agents/candidate_uad/candidate_uad_eval_8agents_mi_refine.json

# Agent-count sweep
.venv/bin/python scripts/learn_agents/learn_agents_agent_count_sweep.py \
  --min-agents 1 --max-agents 12 --output-json results/learn_agents/agent_count/agent_count_sweep.json

# Decoy ablation
.venv/bin/python scripts/decoys/decoy_ablation_sweep.py \
  --output-json results/decoys/ablation/decoy_ablation_sweep.json
```

---

## Code map

| Component | Path |
|-----------|------|
| Simulator, MI search, refine | `learn_agents/learn_agents.py` |
| Debug protocol | `scripts/learn_agents/debug_learn_agents_protocol.py` |
| Candidate eval | `scripts/learn_agents/evaluate_latent_candidates_with_uad.py` |
| Agent-count sweep | `scripts/learn_agents/learn_agents_agent_count_sweep.py` |
| Decoy ablation | `scripts/decoys/decoy_ablation_sweep.py` |
