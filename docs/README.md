# Documentation

This folder separates project-level interpretation from runnable code and
paper sources.

| Path | Purpose |
|------|---------|
| `EXPERIMENTS.md` | Chronological experiment log: why each run exists, settings, results, artifact paths |
| `FINDINGS.md` | Cross-cutting interpretation from early UAD/decoy work |
| `CHANGELOG.md` | Short project milestones |
| `papers/` | LaTeX sources, PDFs, build scripts, and the paper dependency graph |
| `conversations/` | Session summaries and historical context |

Experiment-specific instructions live next to the relevant package when they are
closer to code than prose:

| Package README | Topic |
|----------------|-------|
| `../agent_spotlight/README.md` | Serial one-agent-at-a-time discovery |
| `../amortized_agency/README.md` | Amortized agency detection |
| `../hierarchical_spotlight/README.md` | Hierarchical chunk fusion |
| `../uad_handles/README.md` | Handle-aware UAD toy benchmarks for the access-uad paper |
| `../uad_worm/README.md` | WormWideWeb / C. elegans UAD application |

Run artifacts are indexed separately in `../results/README.md`.
