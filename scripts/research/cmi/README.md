# Discrete CMI research scripts

Analysis that motivated replacing k-NN conditional MI with a **Laplace-smoothed plug-in** estimator in `agency_detect/markov_blanket.py`.

**Run from repo root:**

```bash
.venv/bin/python scripts/research/cmi/nats_inflation_analysis.py
.venv/bin/python scripts/research/cmi/cmi_scaling_analysis.py
.venv/bin/python scripts/research/cmi/discrete_cmi_alternatives.py
.venv/bin/python scripts/research/cmi/discrete_cmi_evaluation.py
```

Logs (gitignored): `results/cmi/*.log`

---

## Findings (2026-05-26)

### k-NN CMI problems

| Issue | Behavior |
|-------|----------|
| Independent continuous vars | **1.6–2.4 nats** bias (should be 0) |
| Independent discrete (3–8 values) | ~0 nats |
| High cardinality (100 values) | Spikes to **~23 nats** |
| Memory / sample size | Erratic; often returns 0 on discrete memory tests |

**Conclusion:** k-NN is unsuitable for discretized agent traces.

### Discrete smoothed plug-in (recommended)

| Variable type | Smoothed CMI (typical, α=0.1) |
|---------------|-------------------------------|
| Independent actions | ~0.002 nats |
| Independent sensors | ~0.15 nats |
| Memory + 3D conditioning | ~0.67 nats |
| Correlated (test) | ~1.0 nats |

Independent vs correlated separation is clean. Chi-square independence test agrees on synthetic data.

### Threshold calibration

| Source | Recommended tolerance | Context |
|--------|----------------------|---------|
| These scripts (k-NN era) | **5.0 nats** | When estimator was broken k-NN |
| These scripts (discrete) | **3.0–5.0 nats** | Smoothed plug-in, factory sim |
| **Production today** | **1.0 nats** | `DetectionConfig.BLANKET_TOLERANCE` after discrete migration |

Use **1.0** for strict falsification on learn_agents oracle clusters; do not blindly adopt 5.0 from k-NN-era scripts.

### Production mapping

| Script insight | Code |
|----------------|------|
| Smoothed plug-in | `conditional_mutual_info_discrete()` in `markov_blanket.py` |
| α = 0.1 | `DetectionConfig.CMI_SMOOTHING_ALPHA` |
| Blanket test | `MarkovBlanketValidator.validate_cluster()` |

### Open follow-ups

- Re-calibrate `BLANKET_TOLERANCE` on learn_agents oracle ε-blankets with current discrete estimator
- Chi-square conditional independence as optional precursor gate (see `discrete_cmi_alternatives.py`)
