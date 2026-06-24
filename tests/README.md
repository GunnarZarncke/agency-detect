# Tests

Pytest tests are grouped by research line while keeping one top-level `tests/`
entry point.

| Folder | Covers |
|--------|--------|
| `core/` | Core UAD estimators, thresholds, adaptive validation, MI/K-selection |
| `spotlight/` | Serial spotlight discovery and richer/exogenous simulator variants |
| `amortized/` | Amortized agency dataset/pool checks |
| `intention/` | Regulation, outcome-influence, and segmentation probes |
| `worm/` | WormWideWeb ingestion, preprocessing, CMI, blanket scoring, evaluation |

Run all tests:

```bash
PYTHONPATH=. pytest
```

Run one line:

```bash
PYTHONPATH=. pytest tests/worm
PYTHONPATH=. pytest tests/intention
```
