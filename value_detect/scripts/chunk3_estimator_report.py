#!/usr/bin/env python3
"""Chunk 3 — plain-English report on the measuring tools.

Runs the estimator on cases whose answer is known by hand and prints the measured
number next to the true one, so SJ can confirm the ruler reads correctly without
reading code. Values are in "nats"; one bit = 0.693.
"""

from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np

import value_detect as vd

LN2 = log(2.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "chunk3"


def _delayed_copy(seed, T, p=0.0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, T)
    Y = np.empty(T, dtype=int)
    Y[0] = rng.integers(0, 2)
    Y[1:] = X[:-1] ^ (rng.random(T - 1) < p).astype(int)
    return X, Y


def _independent(seed, T):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, T), rng.integers(0, 2, T)


def _feedforward(seed, N, n=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(N, n))
    Y = np.empty((N, n), dtype=int)
    Y[:, 0] = rng.integers(0, 2, size=N)
    Y[:, 1:] = X[:, :-1]
    return X, Y


def _feedback(seed, N, n=3):
    rng = np.random.default_rng(seed)
    X = np.empty((N, n), dtype=int)
    Y = np.empty((N, n), dtype=int)
    X[:, 0] = rng.integers(0, 2, size=N)
    Y[:, 0] = rng.integers(0, 2, size=N)
    for t in range(1, n):
        X[:, t] = Y[:, t - 1]
        Y[:, t] = X[:, t - 1]
    return X, Y


def main() -> None:
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    L = []

    def row(name, expected, got):
        L.append((name, expected, got))

    # Copy / noisy copy.
    X, Y = _delayed_copy(1, 50000, 0.0)
    row("A variable that copies another, one step later -> should read one bit", LN2, vd.transfer_entropy(X, Y))
    row("The same, measured backwards -> should read nothing", 0.0, vd.transfer_entropy(Y, X))
    for p in (0.05, 0.15, 0.30):
        X, Y = _delayed_copy(2, 80000, p)
        row(f"A noisy copy ({int(p*100)}% errors) -> a partial reading", LN2 - vd.binary_entropy(p), vd.transfer_entropy(X, Y))

    # Independence & bias.
    row("Two unrelated coin-flips -> should read nothing", 0.0, vd.transfer_entropy(*_independent(3, 50000)))
    row("Bias on unrelated pair, little data (2k steps) -> small, positive", 0.0, vd.transfer_entropy(*_independent(5, 2000)))
    row("Bias on unrelated pair, more data (80k steps) -> smaller", 0.0, vd.transfer_entropy(*_independent(5, 80000)))

    # Mediation.
    rng = np.random.default_rng(4)
    T = 60000
    Xm = rng.integers(0, 2, T)
    M = Xm.copy()
    Ym = np.empty(T, dtype=int)
    Ym[0] = rng.integers(0, 2)
    Ym[1:] = M[:-1]
    row("Flow through a middle-man, measured directly -> one bit", LN2, vd.transfer_entropy(Xm, Ym))
    row("The same, but conditioning on the middle-man -> collapses to nothing", 0.0, vd.transfer_entropy(Xm, Ym, cond=M))

    # Conservation audit.
    ff = vd.conservation_check(*_feedforward(6, 200000), alpha=1e-6)
    fb = vd.conservation_check(*_feedback(7, 200000), alpha=1e-6)

    # Build memo.
    lines = ["# Chunk 3 — the measuring tools, checked against known answers\n"]
    lines.append("All numbers in 'nats'; one bit = 0.693. 'Expected' is worked out by hand.\n")
    lines.append("| Check | Expected | Measured |")
    lines.append("|---|---|---|")
    for name, exp, got in L:
        lines.append(f"| {name} | {exp:.4f} | {got:.4f} |")
    lines.append("")
    lines.append("## The built-in audit (does inflow + outflow add up to the total?)")
    lines.append("A mathematical law says the total link between two variables must equal the flow one")
    lines.append("way plus the flow the other way. If our tools obey it, they are internally consistent.\n")
    lines.append("| Case | Total link | Forward flow | Backward flow | Forward+Backward | Left-over |")
    lines.append("|---|---|---|---|---|---|")
    for label, r in (("One-way (X drives Y)", ff), ("Two-way (mutual)", fb)):
        lines.append(
            f"| {label} | {r['total_mi']:.3f} | {r['forward_di']:.3f} | {r['reverse_di_delayed']:.3f} "
            f"| {r['sum_directed']:.3f} | {r['residual']:.3f} |"
        )
    lines.append("")
    lines.append("A left-over near zero means the books balance exactly.")

    memo = "\n".join(lines)
    (outdir / "SUMMARY.md").write_text(memo + "\n")
    print(memo)
    print(f"\nWritten to: {outdir/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
