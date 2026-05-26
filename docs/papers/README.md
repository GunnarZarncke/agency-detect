# Papers

LaTeX sources and built PDFs for each paper. Rebuild with the corresponding `build.sh` script.

| Paper | Source | PDF | Build |
|-------|--------|-----|-------|
| Foundations of Unsupervised Agent Discovery | [unsupervised-agent-discovery.tex](unsupervised-agent-discovery/unsupervised-agent-discovery.tex) | [PDF](unsupervised-agent-discovery/unsupervised-agent-discovery.pdf) | `unsupervised-agent-discovery/build.sh` |
| Formalization of Acausal Trade atop UAD | [acausal_trade_uad_formalization.tex](acausal-trade-uad-formalization/acausal_trade_uad_formalization.tex) | [PDF](acausal-trade-uad-formalization/acausal_trade_uad_formalization.pdf) | `acausal-trade-uad-formalization/build.sh` |
| Attractor Basins of Cooperation, Privacy, and Parasite Persistence | [attractor-basins.tex](attractor-basins/attractor-basins.tex) | [PDF](attractor-basins/attractor-basins.pdf) | `attractor-basins/build.sh` |
| Empirical Detection of Free-Energy Loops and Attractor Basins | [applications_uad_loops.tex](applications-uad-loops/applications_uad_loops.tex) | — | `applications-uad-loops/build.sh` |
| Bitwise Intelligence | [bitwise_iq.tex](bitwise-iq/bitwise_iq.tex) | [PDF](bitwise-iq/bitwise_iq.pdf) | `bitwise-iq/build.sh` |
| Stratification of Free-Energy Loops | [free_energy_loops.tex](free-energy-loops/free_energy_loops.tex) | [PDF](free-energy-loops/free_energy_loops.pdf) | `free-energy-loops/build.sh` |
| Loop-Hub-Value Model (LHV) | [loop-hub-value-model.tex](loop-hub-value-model/loop-hub-value-model.tex) | — | `loop-hub-value-model/build.sh` |
| Loop-Hub-Value Model v2 | [loop-hub-value-model2.tex](loop-hub-value-model2/loop-hub-value-model2.tex) | [PDF](loop-hub-value-model2/loop-hub-value-model2.pdf) | `loop-hub-value-model2/build.sh` |
| Construction Without Understanding | [construction_without_understanding.tex](construction-without-understanding/construction_without_understanding.tex) | [PDF](construction-without-understanding/construction_without_understanding.pdf) | `construction-without-understanding/build.sh` |
| The Endogenized Intentional Stance | [endogenized-intentional-stance.tex](endogenized-intentional-stance/endogenized-intentional-stance.tex) | [PDF](endogenized-intentional-stance/endogenized-intentional-stance.pdf) | `endogenized-intentional-stance/build.sh` |
| Preference-Conditioned Capability | [preference-capability.tex](preference-capability/preference-capability.tex) | [PDF](preference-capability/preference-capability.pdf) | `preference-capability/build.sh` |
| Prior and Related Work on UAD | [uad_literature_review.tex](uad-literature-review/uad_literature_review.tex) | [PDF](uad-literature-review/uad_literature_review.pdf) | `uad-literature-review/build.sh` |

Build from the repo root:

```bash
docs/papers/acausal-trade-uad-formalization/build.sh
docs/papers/applications-uad-loops/build.sh
docs/papers/attractor-basins/build.sh
docs/papers/bitwise-iq/build.sh
docs/papers/construction-without-understanding/build.sh
docs/papers/endogenized-intentional-stance/build.sh
docs/papers/free-energy-loops/build.sh
docs/papers/loop-hub-value-model/build.sh
docs/papers/loop-hub-value-model2/build.sh
docs/papers/preference-capability/build.sh
docs/papers/uad-literature-review/build.sh
docs/papers/unsupervised-agent-discovery/build.sh
```

LaTeX build artifacts (`*.aux`, `*.log`, etc.) are gitignored. Paper PDFs are tracked in git; regenerate after source changes.
