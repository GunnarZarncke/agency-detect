"""E20 exploration (post-M6): why no blanket, and is there one anywhere?

Three probes, do-one-thing-first:
  X1 timescale  — does a slower lag move the anchor off median p≈0.5?
  X2 joint axes — anchor in the (internal-autonomy, blanket-loss) plane vs random sets:
                  is it a coupled-but-leaky subsystem or nothing?
  X3 per-animal — unsupervised: score every lagged-corr community per animal on both axes;
                  does ANY land in the agent corner (low loss + high autonomy) beyond random?

Run: PYTHONPATH=. .venv/bin/python scripts/worm/explore.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from uad_worm.candidates import (
    ANCHOR_CLASSES,
    agglomerative_communities,
    classes_of,
    classes_to_indices,
)
from uad_worm.cohort import load_neuropal_cohort
from uad_worm.evaluate import joint_null
from uad_worm.score import score_members

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "worm"
REP = "whitened"
EXT_DIM = 6
BEST_EXEMPLAR = "2023-01-09-28"
N_RANDOM_CONTROLS = 20
SCATTER_N_PERM = 80
ANCHOR_NAME = "locomotor_command"


def x1_lag_sweep(cohort):
    print("X1 — timescale (anchor, whitened, ext_dim=6):")
    rows = []
    for lag in (1, 2, 3, 5):
        ps, losses = [], []
        for _, proc in cohort:
            trace = proc.representation(REP)
            members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
            if len(members) < 3:
                continue
            res = score_members(trace, members, ext_dim=EXT_DIM, lag=lag, n_perm=120, seed=0)
            if res:
                ps.append(res.pvalue)
                losses.append(res.loss)
        row = {"lag": lag, "median_p": float(np.median(ps)), "mean_loss": float(np.mean(losses))}
        rows.append(row)
        print(f"  lag={lag} ({lag*0.6:.1f}s): median_p={row['median_p']:.3f} "
              f"mean_loss={row['mean_loss']:.4f}")
    return rows


def x2_joint_anchor(cohort):
    print("\nX2 — anchor in (autonomy, loss) plane (n_perm=120):")
    rows = []
    for ds, proc in cohort:
        trace = proc.representation(REP)
        members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
        if len(members) < 3:
            continue
        jn = joint_null(trace, members, ext_dim=EXT_DIM, n_perm=120, seed=0)
        if jn is None:
            continue
        rows.append({"dataset_id": ds.dataset_id, "animal_id": ds.animal_id,
                     "loss_p": jn.loss_p, "autonomy_p": jn.autonomy_p,
                     "agent_corner": jn.agent_corner})
        print(f"  {ds.dataset_id}: loss_p={jn.loss_p:.2f} autonomy_p={jn.autonomy_p:.2f} "
              f"{'AGENT-CORNER' if jn.agent_corner else ''}")
    med_lp = float(np.median([r['loss_p'] for r in rows]))
    med_ap = float(np.median([r['autonomy_p'] for r in rows]))
    print(f"  median loss_p={med_lp:.2f} median autonomy_p={med_ap:.2f}  "
          f"(agent corner = low loss_p + high autonomy_p)")
    return rows, med_lp, med_ap


def x3_per_animal(cohort):
    print("\nX3 — per-animal unsupervised communities in (coupling, loss) plane:")
    rows = []
    hits = 0
    for ds, proc in cohort:
        trace = proc.representation(REP)
        comms = agglomerative_communities(trace, min_size=3, max_size=30)
        best = None
        for members in comms:
            jn = joint_null(trace, members, ext_dim=EXT_DIM, n_perm=80, seed=0)
            if jn is None:
                continue
            # rank by agent-ness: low loss_p and high autonomy_p
            score = jn.autonomy_p - jn.loss_p
            if best is None or score > best[0]:
                cls = sorted(classes_of(members, proc.neuron_class))
                best = (score, jn, cls, len(members))
        if best is None:
            continue
        _, jn, cls, n = best
        if jn.agent_corner:
            hits += 1
        rows.append({"dataset_id": ds.dataset_id, "n_communities": len(comms),
                     "best_n_members": n, "best_loss_p": jn.loss_p,
                     "best_autonomy_p": jn.autonomy_p, "agent_corner": jn.agent_corner,
                     "best_classes": cls})
        print(f"  {ds.dataset_id}: {len(comms)} comms, best(size={n}) "
              f"loss_p={jn.loss_p:.2f} autonomy_p={jn.autonomy_p:.2f} "
              f"{'AGENT-CORNER' if jn.agent_corner else ''} classes={cls[:6]}")
    print(f"  animals with an agent-corner community: {hits}/{len(rows)}")
    return rows, hits


def build_scatter_rows(cohort, anchor_rows):
    """All tested modules + random controls + command-circuit anchors (X2)."""
    rows = []
    for ds, proc in cohort:
        trace = proc.representation(REP)
        anchor_members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
        anchor_size = len(anchor_members)
        rng = np.random.default_rng(0)

        comms = agglomerative_communities(trace, min_size=3, max_size=30)
        for i, members in enumerate(comms):
            jn = joint_null(
                trace, members, ext_dim=EXT_DIM, n_perm=SCATTER_N_PERM, seed=0
            )
            if jn is None:
                continue
            cls = sorted(classes_of(members, proc.neuron_class))
            rows.append({
                "animal": ds.animal_id,
                "candidate_type": "community",
                "candidate_name": f"community_{i}_{'_'.join(cls[:4])}",
                "loss_p": jn.loss_p,
                "autonomy_p": jn.autonomy_p,
                "is_best_exemplar": False,
            })

        if anchor_size >= 2:
            v = trace.shape[1]
            for k in range(N_RANDOM_CONTROLS):
                rm = rng.permutation(v)[:anchor_size].tolist()
                jn = joint_null(
                    trace, rm, ext_dim=EXT_DIM, n_perm=SCATTER_N_PERM, seed=k + 1
                )
                if jn is None:
                    continue
                rows.append({
                    "animal": ds.animal_id,
                    "candidate_type": "random_control",
                    "candidate_name": f"random_{k}",
                    "loss_p": jn.loss_p,
                    "autonomy_p": jn.autonomy_p,
                    "is_best_exemplar": False,
                })

    for ar in anchor_rows:
        rows.append({
            "animal": ar["animal_id"],
            "candidate_type": "command_anchor",
            "candidate_name": ANCHOR_NAME,
            "loss_p": ar["loss_p"],
            "autonomy_p": ar["autonomy_p"],
            "is_best_exemplar": ar["animal_id"] == BEST_EXEMPLAR,
        })
    return rows


def build_raw_metric_rows(cohort):
    """Raw metric plane with p-values for the same agent-corner criterion as the p plot."""
    rows = []
    for ds, proc in cohort:
        trace = proc.representation(REP)
        anchor_members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
        anchor_size = len(anchor_members)
        rng = np.random.default_rng(0)

        if anchor_size >= 2:
            v = trace.shape[1]
            for k in range(N_RANDOM_CONTROLS):
                rm = rng.permutation(v)[:anchor_size].tolist()
                jn = joint_null(
                    trace, rm, ext_dim=EXT_DIM, n_perm=SCATTER_N_PERM, seed=k + 1
                )
                if jn is None:
                    continue
                rows.append({
                    "animal": ds.animal_id,
                    "candidate_type": "random_control",
                    "candidate_name": f"random_{k}",
                    "blanket_loss": jn.loss,
                    "internal_autonomy": jn.autonomy,
                    "loss_p": jn.loss_p,
                    "autonomy_p": jn.autonomy_p,
                    "is_best_exemplar": False,
                })

            jn = joint_null(
                trace, anchor_members, ext_dim=EXT_DIM, n_perm=SCATTER_N_PERM, seed=0
            )
            if jn is not None:
                rows.append({
                    "animal": ds.animal_id,
                    "candidate_type": "command_anchor",
                    "candidate_name": ANCHOR_NAME,
                    "blanket_loss": jn.loss,
                    "internal_autonomy": jn.autonomy,
                    "loss_p": jn.loss_p,
                    "autonomy_p": jn.autonomy_p,
                    "is_best_exemplar": ds.animal_id == BEST_EXEMPLAR,
                })
    return rows


def write_raw_scatter_csv(rows, path: Path) -> None:
    fields = [
        "animal", "candidate_type", "candidate_name",
        "blanket_loss", "internal_autonomy", "loss_p", "autonomy_p", "is_best_exemplar",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({**row, "is_best_exemplar": str(row["is_best_exemplar"]).lower()})


def write_scatter_csv(rows, path: Path) -> None:
    fields = [
        "animal", "candidate_type", "candidate_name",
        "loss_p", "autonomy_p", "is_best_exemplar",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({**row, "is_best_exemplar": str(row["is_best_exemplar"]).lower()})


def _agent_corner(row) -> bool:
    return row["loss_p"] < 0.5 and row["autonomy_p"] > 0.5


def scatter_calibration(rows) -> dict:
    """How command-circuit anchors compare to the matched null reference."""
    nulls = [r for r in rows if r["candidate_type"] == "random_control"]
    anchors = [r for r in rows if r["candidate_type"] == "command_anchor"]
    by_animal_null: dict[str, list] = {}
    for r in nulls:
        by_animal_null.setdefault(r["animal"], []).append(r)

    exemplar = next((r for r in anchors if r["is_best_exemplar"]), None)
    exemplar_null_corner = 0
    exemplar_null_n = 0
    if exemplar is not None:
        animal_nulls = by_animal_null.get(exemplar["animal"], [])
        exemplar_null_n = len(animal_nulls)
        exemplar_null_corner = sum(_agent_corner(n) for n in animal_nulls)

    return {
        "n_null": len(nulls),
        "n_anchor": len(anchors),
        "null_corner_rate": sum(_agent_corner(r) for r in nulls) / max(len(nulls), 1),
        "anchor_corner_rate": sum(_agent_corner(r) for r in anchors) / max(len(anchors), 1),
        "n_null_corner": sum(_agent_corner(r) for r in nulls),
        "n_anchor_corner": sum(_agent_corner(r) for r in anchors),
        "exemplar_null_corner": exemplar_null_corner,
        "exemplar_null_n": exemplar_null_n,
    }


def plot_autonomy_loss_scatter(rows, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return False

    def xy(r):
        return 1.0 - r["loss_p"], r["autonomy_p"]

    nulls = [r for r in rows if r["candidate_type"] == "random_control"]
    anchors = [r for r in rows if r["candidate_type"] == "command_anchor"]
    exemplar = next((r for r in anchors if r["is_best_exemplar"]), None)
    cal = scatter_calibration(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if nulls:
        nx, ny = zip(*[xy(r) for r in nulls])
        ax.scatter(
            nx, ny, c="#d4d4d4", s=22, alpha=0.55, edgecolors="none",
            label=f"matched null ({cal['n_null']} random sets)", zorder=1,
        )
    if anchors:
        ax_x, ax_y = zip(*[xy(r) for r in anchors])
        ax.scatter(
            ax_x, ax_y, c="#2563eb", s=70, alpha=0.9, edgecolors="white",
            linewidths=0.6, label="command circuit (8 worms)", zorder=3,
        )
    if exemplar:
        ex, ey = xy(exemplar)
        ax.scatter(
            [ex], [ey], marker="*", s=320, c="#eab308", edgecolors="#713f12",
            linewidths=0.8, label=f"best exemplar ({BEST_EXEMPLAR})", zorder=4,
        )

    ax.add_patch(Rectangle(
        (0.5, 0.5), 0.5, 0.5, facecolor="#22c55e", alpha=0.10, edgecolor="none", zorder=0,
    ))
    ax.axvline(0.5, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0.5, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.text(0.75, 0.97, "candidate functional\nsubsystem", ha="center", va="top",
            fontsize=9, color="#166534", alpha=0.85)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Separation from rest of brain  →  (1 − blanket-loss p)")
    ax.set_ylabel("Internal autonomy  (autonomy p)")
    ax.set_title("Agent signature: separable + internally driven (post-M6)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_autonomy_loss_scatter_exemplar(rows, path: Path) -> bool:
    """P-value plane: null cloud + gold star on the single best exemplar only."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return False

    def xy(r):
        return 1.0 - r["loss_p"], r["autonomy_p"]

    nulls = [r for r in rows if r["candidate_type"] == "random_control"]
    exemplar = next(
        (r for r in rows if r["candidate_type"] == "command_anchor" and r["is_best_exemplar"]),
        None,
    )
    cal = scatter_calibration(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if nulls:
        nx, ny = zip(*[xy(r) for r in nulls])
        ax.scatter(
            nx, ny, c="#d4d4d4", s=22, alpha=0.55, edgecolors="none",
            label=f"matched null ({cal['n_null']} random sets)", zorder=1,
        )
    if exemplar:
        ex, ey = xy(exemplar)
        ax.scatter(
            [ex], [ey], marker="*", s=320, c="#eab308", edgecolors="#713f12",
            linewidths=0.8, label=f"best exemplar ({BEST_EXEMPLAR})", zorder=4,
        )

    ax.add_patch(Rectangle(
        (0.5, 0.5), 0.5, 0.5, facecolor="#22c55e", alpha=0.10, edgecolor="none", zorder=0,
    ))
    ax.axvline(0.5, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0.5, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.text(0.75, 0.97, "candidate functional\nsubsystem", ha="center", va="top",
            fontsize=9, color="#166534", alpha=0.85)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Separation from rest of brain  →  (1 − blanket-loss p)")
    ax.set_ylabel("Internal autonomy  (autonomy p)")
    ax.set_title("Agent signature: separable + internally driven (post-M6)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_autonomy_loss_raw_scatter(rows, path: Path) -> bool:
    """Raw metrics: low blanket loss (right) + high internal autonomy (top)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return False

    nulls = [r for r in rows if r["candidate_type"] == "random_control"]
    anchors = [r for r in rows if r["candidate_type"] == "command_anchor"]
    exemplar = next((r for r in anchors if r["is_best_exemplar"]), None)
    if not nulls:
        return False

    null_loss = np.asarray([r["blanket_loss"] for r in nulls])
    null_aut = np.asarray([r["internal_autonomy"] for r in nulls])
    # Median-null splits: visual guide aligned with loss_p/autonomy_p ≈ 0.5 on each axis.
    loss_hi = float(np.percentile(null_loss, 50))
    aut_lo = float(np.percentile(null_aut, 50))
    def xy(r):
        return r["blanket_loss"], r["internal_autonomy"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    nx, ny = zip(*[xy(r) for r in nulls])
    ax.scatter(
        nx, ny, c="#d4d4d4", s=28, alpha=0.6, edgecolors="none",
        label=f"matched null ({len(nulls)} random sets)", zorder=1,
    )
    n_anchors = len(anchors)
    if anchors:
        ax_x, ax_y = zip(*[xy(r) for r in anchors])
        ax.scatter(
            ax_x, ax_y, c="#2563eb", s=70, alpha=0.9, edgecolors="white",
            linewidths=0.6, label=f"command circuit ({n_anchors} worms)", zorder=3,
        )
    if exemplar:
        ex, ey = xy(exemplar)
        ax.scatter(
            [ex], [ey], marker="*", s=320, c="#eab308", edgecolors="#713f12",
            linewidths=0.8, label=f"best exemplar ({BEST_EXEMPLAR})", zorder=5,
        )

    x0, x1 = float(null_loss.min()), float(null_loss.max())
    y0, y1 = float(null_aut.min()), float(null_aut.max())
    pad_x = 0.08 * (x1 - x0 + 1e-9)
    pad_y = 0.08 * (y1 - y0 + 1e-9)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)

    ax.add_patch(Rectangle(
        (ax.get_xlim()[0], aut_lo), loss_hi - ax.get_xlim()[0], ax.get_ylim()[1] - aut_lo,
        facecolor="#22c55e", alpha=0.10, edgecolor="none", zorder=0,
    ))
    ax.axvline(loss_hi, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(aut_lo, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.text(
        (ax.get_xlim()[0] + loss_hi) / 2, ax.get_ylim()[1] - 0.03 * (y1 - y0),
        "candidate functional\nsubsystem", ha="center", va="top",
        fontsize=9, color="#166534", alpha=0.85,
    )

    ax.invert_xaxis()
    ax.set_xlabel("Blanket loss  ←  lower = more separable from rest of brain")
    ax.set_ylabel("Internal autonomy  →  higher = more self-driven")
    ax.set_title("Agent signature: separable + internally driven (post-M6)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_autonomy_loss_raw_scatter_exemplar(rows, path: Path) -> bool:
    """Raw-metric plane: null cloud + gold star on the single best exemplar only."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return False

    nulls = [r for r in rows if r["candidate_type"] == "random_control"]
    exemplar = next(
        (r for r in rows if r["candidate_type"] == "command_anchor" and r["is_best_exemplar"]),
        None,
    )
    if not nulls:
        return False

    null_loss = np.asarray([r["blanket_loss"] for r in nulls])
    null_aut = np.asarray([r["internal_autonomy"] for r in nulls])
    loss_hi = float(np.percentile(null_loss, 50))
    aut_lo = float(np.percentile(null_aut, 50))

    def xy(r):
        return r["blanket_loss"], r["internal_autonomy"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    nx, ny = zip(*[xy(r) for r in nulls])
    ax.scatter(
        nx, ny, c="#d4d4d4", s=28, alpha=0.6, edgecolors="none",
        label=f"matched null ({len(nulls)} random sets)", zorder=1,
    )
    if exemplar:
        ex, ey = xy(exemplar)
        ax.scatter(
            [ex], [ey], marker="*", s=320, c="#eab308", edgecolors="#713f12",
            linewidths=0.8, label=f"best exemplar ({BEST_EXEMPLAR})", zorder=5,
        )

    x0, x1 = float(null_loss.min()), float(null_loss.max())
    y0, y1 = float(null_aut.min()), float(null_aut.max())
    pad_x = 0.08 * (x1 - x0 + 1e-9)
    pad_y = 0.08 * (y1 - y0 + 1e-9)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)

    ax.add_patch(Rectangle(
        (ax.get_xlim()[0], aut_lo), loss_hi - ax.get_xlim()[0], ax.get_ylim()[1] - aut_lo,
        facecolor="#22c55e", alpha=0.10, edgecolor="none", zorder=0,
    ))
    ax.axvline(loss_hi, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(aut_lo, color="#64748b", ls="--", lw=0.8, alpha=0.5)
    ax.text(
        (ax.get_xlim()[0] + loss_hi) / 2, ax.get_ylim()[1] - 0.03 * (y1 - y0),
        "candidate functional\nsubsystem", ha="center", va="top",
        fontsize=9, color="#166534", alpha=0.85,
    )

    ax.invert_xaxis()
    ax.set_xlabel("Blanket loss  ←  lower = more separable from rest of brain")
    ax.set_ylabel("Internal autonomy  →  higher = more self-driven")
    ax.set_title("Agent signature: separable + internally driven (post-M6)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _default_results_dir(max_animals, baseline_only: bool, quality_filter: bool) -> Path:
    if quality_filter and max_animals is None and not baseline_only:
        return RESULTS_DIR / "cohort_qfilt"
    if max_animals == 8 and baseline_only:
        return RESULTS_DIR
    if max_animals is None and not baseline_only:
        return RESULTS_DIR / "cohort56"
    label = f"n{max_animals}" if max_animals is not None else "all"
    return RESULTS_DIR / ("baseline_" if baseline_only else "") + label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-animals", type=int, default=None,
                    help="cap cohort size (default: all loadable NeuroPAL-labeled)")
    ap.add_argument("--baseline-only", action="store_true",
                    help="restrict to NeuroPAL-Baseline runs (excludes Heat)")
    ap.add_argument("--quality-filter", action="store_true",
                    help="keep only T>=1500, n_labeled>=70, anchor classes>=5")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="output directory (default: results/worm/ or cohort56/)")
    args = ap.parse_args()

    results_dir = args.results_dir or _default_results_dir(
        args.max_animals, args.baseline_only, args.quality_filter
    )
    print("Loading cohort ...")
    cohort, skipped, quality_filtered = load_neuropal_cohort(
        max_animals=args.max_animals,
        baseline_only=args.baseline_only,
        quality_filter=args.quality_filter,
        write_provenance=False,
    )
    tag = "NeuroPAL-Baseline" if args.baseline_only else "NeuroPAL-labeled"
    filt = " + quality filter" if args.quality_filter else ""
    print(f"cohort: {len(cohort)} {tag} animals{filt}"
          + (f" ({len(skipped)} load-fail)" if skipped else "")
          + (f" ({len(quality_filtered)} quality-filtered)" if quality_filtered else "")
          + "\n")
    x1 = x1_lag_sweep(cohort)
    x2_rows, med_lp, med_ap = x2_joint_anchor(cohort)
    x3_rows, hits = x3_per_animal(cohort)

    print("\nScatter — autonomy vs separation plane:")
    scatter_rows = build_scatter_rows(cohort, x2_rows)
    n_bg = sum(1 for r in scatter_rows if r["candidate_type"] != "command_anchor")
    n_anchor = sum(1 for r in scatter_rows if r["candidate_type"] == "command_anchor")
    print(f"  {n_bg} background points + {n_anchor} command-circuit anchors")
    cal = scatter_calibration(scatter_rows)
    print(
        f"  agent corner: anchors {cal['n_anchor_corner']}/{cal['n_anchor']} "
        f"({100 * cal['anchor_corner_rate']:.0f}%) vs null "
        f"{cal['n_null_corner']}/{cal['n_null']} ({100 * cal['null_corner_rate']:.0f}%)"
    )
    print(
        f"  {BEST_EXEMPLAR}: {cal['exemplar_null_corner']}/{cal['exemplar_null_n']} "
        f"per-animal nulls also in corner"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "autonomy_loss_scatter.csv"
    write_scatter_csv(scatter_rows, csv_path)
    print(f"  Wrote {csv_path}")

    png_path = results_dir / "autonomy_loss_scatter.png"
    if plot_autonomy_loss_scatter(scatter_rows, png_path):
        print(f"  Wrote {png_path}")
    else:
        print("  (matplotlib unavailable — skipped scatter plot)")

    exemplar_png = results_dir / "autonomy_loss_scatter_exemplar.png"
    if plot_autonomy_loss_scatter_exemplar(scatter_rows, exemplar_png):
        print(f"  Wrote {exemplar_png}")

    print("\nRaw-metric scatter (intuitive null cloud):")
    raw_rows = build_raw_metric_rows(cohort)
    raw_csv = results_dir / "autonomy_loss_raw_scatter.csv"
    write_raw_scatter_csv(raw_rows, raw_csv)
    print(f"  Wrote {raw_csv} ({len(raw_rows)} points)")
    raw_png = results_dir / "autonomy_loss_scatter_raw.png"
    if plot_autonomy_loss_raw_scatter(raw_rows, raw_png):
        print(f"  Wrote {raw_png}")
    else:
        print("  (matplotlib unavailable — skipped raw scatter plot)")

    raw_exemplar_png = results_dir / "autonomy_loss_scatter_raw_exemplar.png"
    if plot_autonomy_loss_raw_scatter_exemplar(raw_rows, raw_exemplar_png):
        print(f"  Wrote {raw_exemplar_png}")

    out = results_dir / "exploration.json"
    out.write_text(json.dumps({
        "cohort": [ds.dataset_id for ds, _ in cohort],
        "skipped": skipped,
        "quality_filtered": quality_filtered,
        "config": {
            "max_animals": args.max_animals,
            "baseline_only": args.baseline_only,
            "quality_filter": args.quality_filter,
        },
        "x1_lag_sweep": x1,
        "x2_anchor_joint": {"per_animal": x2_rows, "median_loss_p": med_lp,
                            "median_autonomy_p": med_ap},
        "x3_per_animal_communities": {"per_animal": x3_rows, "agent_corner_hits": hits},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
