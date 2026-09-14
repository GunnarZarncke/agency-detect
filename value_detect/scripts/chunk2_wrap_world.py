#!/usr/bin/env python3
"""Chunk 2 — wrap the world.

Generate a passive trace from Gunnar's read-only handle-world, record it, and write
the plain-English audit artifacts SJ signs off on before any measurement:

  * a story printout of the loop,
  * a goal-flip report (the rare value-core events),
  * a change-frequency chart (image + text),
  * a mechanism-agreement table (observed vs designed noise),
  * a short plain-English summary memo.

Everything is regenerable from this one command; outputs go to results/chunk2/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import value_detect as vd

# Project root is two levels up from this script (value_detect/scripts/..).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "chunk2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk 2: wrap the handle-world and audit it in plain English.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--story-steps", type=int, default=25)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 0. Verify the live world still matches the noise defaults we read from source.
    check = vd.verify_world_defaults()
    if not check["ok"]:
        raise SystemExit(f"World defaults drifted from the design notes: {check['mismatches']}")

    # 1. Generate and record the trace.
    trace = vd.passive_trace(seed=args.seed, n_steps=args.n_steps)
    paths = vd.record_trace(trace, outdir)

    # 2. Story printout.
    story = vd.narrate_window(trace.frame, start=0, n=args.story_steps)
    flips = vd.goal_flip_report(trace.frame)
    story_path = outdir / f"story_seed{args.seed}.txt"
    story_path.write_text(story + "\n\n" + flips + "\n")

    # 3. Change frequencies.
    freqs = vd.change_frequencies(trace.frame)
    img = vd.plot_change_frequencies(freqs, outdir / f"change_frequency_seed{args.seed}.png")
    text_chart = vd.text_bar_chart(freqs)

    # 4. Mechanism-agreement table.
    mech = vd.mechanism_agreement(trace.frame, trace.noise)

    # 5. Plain-English summary memo.
    memo = _build_memo(args, check, freqs, mech, text_chart, img, paths)
    memo_path = outdir / f"SUMMARY_seed{args.seed}.md"
    memo_path.write_text(memo)

    print(memo)
    print(f"\nArtifacts written to: {outdir}")


def _build_memo(args, check, freqs, mech, text_chart, img, paths) -> str:
    lines = []
    lines.append(f"# Chunk 2 summary — wrapping the world (seed {args.seed}, {args.n_steps} steps)\n")
    lines.append("## Verify-on-arrival")
    lines.append(f"- Live world matches the noise defaults read from source: **{check['ok']}**.")
    lines.append(f"- Variables in order: {', '.join(check['var_names'])}.")
    lines.append(f"- Ground-truth agent loop: {', '.join(check['true_loop'])}.\n")

    lines.append("## How often each variable changes")
    lines.append("The goal G should barely move (~1.5%); pure noise W should move ~50%.\n")
    lines.append("```")
    lines.append(text_chart)
    lines.append("```")
    if img:
        lines.append(f"\nChart image: `{Path(img).name}`\n")
    else:
        lines.append("\n(Chart image skipped: matplotlib unavailable; text chart above.)\n")

    lines.append("## Does the world obey its designed loop?")
    lines.append("For each relationship the design claims, the disagreement rate predicted by the")
    lines.append("built-in noise sits next to the rate actually observed. Close numbers = world behaves as described.\n")
    lines.append("| Relationship | Predicted | Observed |")
    lines.append("|---|---|---|")
    for _, r in mech.iterrows():
        lines.append(f"| {r['relationship']} | {r['predicted_disagreement']:.3f} | {r['observed_disagreement']:.3f} |")
    lines.append("")

    lines.append("## Files")
    lines.append(f"- Trace: `{Path(paths['csv']).name}` (+ metadata `{Path(paths['meta']).name}`)")
    lines.append(f"- Story printout + goal-flip report: `story_seed{args.seed}.txt`")
    lines.append(f"- This memo: `SUMMARY_seed{args.seed}.md`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
