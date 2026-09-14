#!/usr/bin/env python3
"""Chunk 4 — first scored run, as a 2×3 grid of co-equal tests (SJ, 2026-08-10).

Two world configurations × three instrument conventions, all reported equally:

  worlds:       (a) as-is (lookalikes present)   (b) lookalike columns dropped
  conventions:  pairwise · fused mega-state (both directions fused) · fused + best-key

Run lengths per convention: pairwise and fused+best-key at 20k steps; the fused
mega-state convention at ~2M steps, because its fused outbound tracks ~131k state
combinations in the as-is world and would be smoothing artifact at 20k (DECISIONS.md
2026-08-10). Same seed — the 2M trace's first 20k steps are identical to the 20k trace.

Where the conventions disagree, the disagreement is a finding about how the tool
works (see results/chunk4/DIAGNOSIS.md for the mechanism), never something to
average away. Also runs Gunnar's own detector on the trace as the boundary sanity
check (convention-independent).

No success criteria are evaluated here (Chunk 5, after per-convention noise floors).
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import value_detect as vd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "chunk4"

# §4 prediction table (design document), plain-English positions.
PREDICTIONS = {
    "G": "High drive, near-zero intake — the headline",
    "B": "High intake, moderate drive",
    "A": "High drive AND high intake — the mediator",
    "E": "Driven and driving — mid-map",
    "S": "High intake",
    "S_alias": "High intake, like S",
    "A_alias": "May show apparent drive (honest passive limitation)",
    "D": "Pure intake (timing quirk: observe and report)",
    "W": "Near the origin (nothing in, nothing out)",
}


def two_axis_map(tables: dict, outpath: Path) -> str:
    """Grid of intake-vs-output panels, one per analysis, with an origin inset each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(tables)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, t) in zip(axes, tables.items()):
        x = t["push_in"]
        y = t["out_sys"]
        ax.scatter(x, y, s=45, color="#2c3e50", zorder=3)
        for name in t.index:
            color = "#c0392b" if name == "G" else ("#2980b9" if name == "B" else "#2c3e50")
            ax.annotate(
                name,
                (x[name], y[name]),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=10,
                color=color,
                fontweight="bold" if name in ("G", "B") else "normal",
            )
        lim = max(float(x.max()), float(y.max())) * 1.12 + 0.02
        ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="#bbbbbb", zorder=1)
        ax.set_xlim(-0.02 * lim, lim)
        ax.set_ylim(-0.02 * lim, lim)
        ax.set_xlabel("Intake (push-in, nats)")
        ax.set_ylabel("Output (push-out, system flavour, nats)")
        ax.set_title(title, fontsize=11)
        ax.text(0.97 * lim, 0.99 * lim, "drive pole ↑", ha="right", va="top", fontsize=9, color="#888888")
        ax.text(0.99 * lim, 0.03 * lim, "intake pole →", ha="right", va="bottom", fontsize=9, color="#888888")

        # Inset: magnify the origin, where the quiet variables (G, W) live under the
        # pairwise convention. Skipped automatically when nobody is near the origin
        # (under the fused+best-key convention G moves up into the main field).
        zoom = max(t.loc[v, ["push_in", "out_sys"]].max() for v in ("G", "W") if v in t.index) * 1.5 + 1e-4
        if zoom < 0.2 * lim:
            axins = ax.inset_axes([0.40, 0.52, 0.30, 0.34])
            near = t[(t["push_in"] <= zoom) & (t["out_sys"] <= zoom)]
            axins.scatter(near["push_in"], near["out_sys"], s=30, color="#2c3e50", zorder=3)
            for name in near.index:
                color = "#c0392b" if name == "G" else "#2c3e50"
                axins.annotate(name, (near.loc[name, "push_in"], near.loc[name, "out_sys"]),
                               textcoords="offset points", xytext=(6, 3), fontsize=9,
                               color=color, fontweight="bold" if name == "G" else "normal")
            axins.plot([0, zoom], [0, zoom], ls="--", lw=0.8, color="#bbbbbb", zorder=1)
            axins.set_xlim(-0.04 * zoom, zoom)
            axins.set_ylim(-0.04 * zoom, zoom)
            axins.tick_params(labelsize=7)
            axins.set_title("origin, magnified", fontsize=8, color="#666666")
            ax.indicate_inset_zoom(axins, edgecolor="#999999")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return str(outpath)


def boundary_sanity_check(frame: pd.DataFrame, outdir: Path, n_agents_options=(2, 3)) -> str:
    """Run Gunnar's AgentDetector on the trace; return a plain report.

    His detector prints its full working; that raw output is saved as an artifact
    (boundary_check_raw.txt) so nothing is summarized away.
    """
    from agency_detect.config import DetectionConfig
    from agency_detect.detection import AgentDetector

    trace = frame.to_dict("records")
    chunks = []
    raw_parts = []
    original_n = DetectionConfig.N_AGENTS
    try:
        for n in n_agents_options:
            DetectionConfig.N_AGENTS = n
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                detector = AgentDetector(DetectionConfig)
                results = detector.detect_agents(trace)
            raw_parts.append(f"===== N_AGENTS = {n} =====\n{buf.getvalue()}")
            summary = []
            for label, info in results.items():
                if label == "env":
                    summary.append(f"  environment bucket: {info['variables']}")
                else:
                    v = info["blanket_validation"]
                    summary.append(
                        f"  cluster {label}: {info['variables']} "
                        f"(blanket valid={v['valid']}, leakage={v['violation']:.3f} vs tolerance 1.0)"
                    )
            chunks.append(f"With the dial set to {n} groups:\n" + "\n".join(summary))
    finally:
        DetectionConfig.N_AGENTS = original_n
    (outdir / "boundary_check_raw.txt").write_text("\n\n".join(raw_parts))
    return "\n\n".join(chunks)


def fmt_table(t: pd.DataFrame) -> str:
    show = t[["push_in", "out_sys", "out_env", "polarity_sys", "polarity_env", "raw_sys", "total_flow"]].copy()
    lines = ["| Variable | Intake | Output (system) | Output (to env) | Polarity | Polarity (env) | Raw diff | Total flow |", "|---|---|---|---|---|---|---|---|"]
    for name, r in show.iterrows():
        def f(v, digits=4):
            return "n/a" if not np.isfinite(v) else f"{v:.{digits}f}"
        lines.append(
            f"| **{name}** | {f(r['push_in'])} | {f(r['out_sys'])} | {f(r['out_env'])} "
            f"| {f(r['polarity_sys'], 3)} | {f(r['polarity_env'], 3)} | {f(r['raw_sys'])} | {f(r['total_flow'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--n-steps-fused", type=int, default=2_000_000)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    check = vd.verify_world_defaults()
    if not check["ok"]:
        raise SystemExit(f"World drifted: {check['mismatches']}")

    trace = vd.passive_trace(seed=args.seed, n_steps=args.n_steps)
    frame_a = trace.frame
    frame_b = vd.drop_aliases(frame_a)
    trace_long = vd.passive_trace(seed=args.seed, n_steps=args.n_steps_fused)
    frame_a_long = trace_long.frame
    frame_b_long = vd.drop_aliases(frame_a_long)

    # The 2×3: all six analyses, co-equal.
    k = args.n_steps // 1000
    m = args.n_steps_fused / 1_000_000
    analyses = {
        f"(a) as-is — pairwise ({k}k)": vd.score_trace(frame_a, env_var="E"),
        f"(b) lookalike-free — pairwise ({k}k)": vd.score_trace(frame_b, env_var="E"),
        f"(a) as-is — fused mega-state ({m:g}M)": vd.score_trace_fused(frame_a_long, env_var="E"),
        f"(b) lookalike-free — fused mega-state ({m:g}M)": vd.score_trace_fused(frame_b_long, env_var="E"),
        f"(a) as-is — fused + best-key ({k}k)": vd.score_trace_fused_bestkey(frame_a, env_var="E"),
        f"(b) lookalike-free — fused + best-key ({k}k)": vd.score_trace_fused_bestkey(frame_b, env_var="E"),
    }
    slugs = dict(zip(analyses.keys(), [
        "asis_pairwise", "noalias_pairwise",
        "asis_fused", "noalias_fused",
        "asis_fused_bestkey", "noalias_fused_bestkey",
    ]))
    for title, t in analyses.items():
        t.to_csv(outdir / f"scores_{slugs[title]}_seed{args.seed}.csv")

    map_path = two_axis_map(analyses, outdir / f"two_axis_map_seed{args.seed}.png")
    boundary = boundary_sanity_check(frame_a, outdir)

    # Memo.
    L = []
    L.append(f"# Chunk 4 — first scored run, 2×3 grid of co-equal tests (seed {args.seed})\n")
    L.append(f"Run lengths: pairwise and fused+best-key at {args.n_steps:,} steps; fused mega-state at")
    L.append(f"{args.n_steps_fused:,} steps (its fused outbound needs the data; same seed, same world).")
    L.append("Two world configurations × three instrument conventions; all six reported equally.")
    L.append("Polarity = fraction of a variable's traffic that is outbound (+1 pure driver, −1 pure")
    L.append("absorber). Convention disagreements are findings (mechanism: DIAGNOSIS.md).")
    L.append("No pass/fail here; Chunk 5 adds per-convention noise floors and the criteria.\n")

    for title, t in analyses.items():
        L.append(f"## {title}\n")
        L.append(fmt_table(t))
        L.append("")

    L.append("## Predicted vs observed — all six analyses (rank by polarity; polarity in brackets)\n")
    order = list(analyses.keys())
    short = ["(a) pair", "(b) pair", "(a) fused", "(b) fused", "(a) fus+key", "(b) fus+key"]
    header = "| Variable | §4 predicted position | " + " | ".join(short) + " |"
    L.append(header)
    L.append("|---|---|" + "---|" * len(order))
    ranks = {title: {v: i + 1 for i, v in enumerate(analyses[title].index)} for title in order}
    for v, pred in PREDICTIONS.items():
        cells = []
        for title in order:
            t = analyses[title]
            if v in t.index:
                cells.append(f"#{ranks[title][v]} ({t.loc[v, 'polarity_sys']:+.2f})")
            else:
                cells.append("—")
        L.append(f"| **{v}** | {pred} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Boundary sanity check — Gunnar's own detector on this trace\n")
    L.append("```")
    L.append(boundary)
    L.append("```")
    L.append("")
    L.append(f"Two-axis maps: `{Path(map_path).name}`; full detector output: `boundary_check_raw.txt`.")

    memo = "\n".join(L)
    (outdir / f"SUMMARY_seed{args.seed}.md").write_text(memo + "\n")
    print(memo)
    print(f"\nArtifacts written to: {outdir}")


if __name__ == "__main__":
    main()
