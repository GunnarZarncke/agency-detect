# Project Instructions

This repository is a research codebase for unsupervised agent discovery. Prefer preserving experimental clarity over broad architectural changes.

## Experiment Discipline

- Before changing experiments, identify the relevant stage: `agency_detect`, `learn_agents`, `agent_spotlight`, or `hierarchical_spotlight`.
- Keep generated artifacts under the appropriate `results/<family>/<experiment>/` location; do not overwrite canonical artifacts unless asked.
- For experiment runs, report the exact command and key metrics needed to compare results.

## Documentation

- Detailed experiment narrative: `docs/EXPERIMENTS.md`
- Chronological milestones: `docs/CHANGELOG.md`
- Conversation summaries (reasoning and decisions): see
  [`docs/conversations/README.md`](docs/conversations/README.md). Write a new
  summary there after significant sessions; use the template and example in that
  folder.
