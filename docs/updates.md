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
| `scripts/learn_agents_go_nogo_sweep.py` (smoke: 3 agents, 1 seed, 16 slots, 25 epochs) | `var_acc=0.67`, slot purity 0.45 — confirms slots mix agents at scale; motivates candidate framing |

Artifact: `results/learn_agents_go_nogo_commit_smoke.jsonl` (gitignored).

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

- Committed: `scripts/learn_agents_go_nogo_sweep.py`, agency_detect tests
- Removed: stale factory debug scripts, scratch `t*.dot`/`t.png`, `loop-hub-value-graph.py`, orphan `src/parameters.py*`
- Kept untracked: CMI research scripts (`scripts/discrete_cmi_*.py`, etc.) — separate from learn_agents line
