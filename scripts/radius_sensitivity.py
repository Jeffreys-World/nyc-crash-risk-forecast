#!/usr/bin/env python3
"""Re-run the backtest across the three join radii and report how the headline moves.

The published result rests on three distances that were chosen by judgement and never
tested:

    MAX_JOIN_DISTANCE_FT    150   how far a crash may be from a segment to attach to it
    INTERSECTION_RADIUS_FT  100   how close to a node a crash must be to count as
                                  intersection-related rather than mid-block
    VZV_BUFFER_FT            50   how far a DOT priority feature reaches when deciding
                                  which units are on the published list

Each is defensible. So is 100, or 250, or 25. This script answers the only question
that matters about that: does the finding survive a different, equally defensible
choice, or was it an artifact of three numbers nobody examined?

One-at-a-time from the baseline rather than a full grid. 3x3x3 is 27 rebuilds to test a
claim that is really three separate "does this one knob matter" questions, and the
interaction terms would need far more runs than they would earn. Where a knob turns out
to matter, that is the finding to chase, not a 27-cell table.

    python scripts/radius_sensitivity.py
    python scripts/radius_sensitivity.py --snapshot 2026-08-13
    python scripts/radius_sensitivity.py --quick   # intersection radius only

Writes `data/processed/radius-sensitivity.{csv,md}`. Never touches `run-summary.json`
or `top-50-ranked.csv`: those are the published headline, and an exploratory sweep must
not be able to overwrite them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DEFAULT_RADII,
    MIN_CAPTURE_RATE_LIFT_PP,
    PROCESSED_DIR,
    RAW_DIR,
    JoinRadii,
)
from src.pipeline import run  # noqa: E402

log = logging.getLogger("radius_sensitivity")

# The values swept, per knob. The baseline sits in each list so every axis contains the
# published setting and the table reads as a comparison rather than three unrelated runs.
#
# Ranges are bounded by what is physically defensible, not by what is convenient:
#   corridor join  100 ft is about one lane-width of geocoder slack; 250 ft starts
#                  letting a crash cross to a parallel street mid-block.
#   intersection   50 ft is roughly the junction box itself; 150 ft reaches a third of
#                  the way down a short Manhattan block.
#   VZV buffer     25 ft is tight enough to miss a genuine centerline-vs-VZV offset;
#                  100 ft starts pulling in the parallel street.
SWEEPS: dict[str, tuple[str, tuple[float, ...]]] = {
    "corridor join distance": ("max_join_distance_ft", (100.0, 150.0, 250.0)),
    "intersection radius": ("intersection_radius_ft", (50.0, 100.0, 150.0)),
    "VZV label buffer": ("vzv_buffer_ft", (25.0, 50.0, 100.0)),
}

QUICK_ONLY = "intersection radius"


def settings(quick: bool = False) -> list[tuple[str, str, float, JoinRadii]]:
    """Every (knob, field, value, radii) row the table shows - baseline included per knob.

    Nine rows across three axes, not seven. The baseline setting sits on all three axes
    (it is 150 *and* 100 *and* 50), and a block that jumps from 50 ft to 150 ft with the
    published 100 ft missing is a table a reader has to reassemble in their head.

    It is still only run once: `sweep` memoises on `radii.tag`. That is not just a saved
    two minutes - the pipeline is deterministic, so three separately-executed baseline
    rows would be identical by construction, and printing them as though they were three
    measurements would imply a spread that does not exist.
    """
    sweeps = {QUICK_ONLY: SWEEPS[QUICK_ONLY]} if quick else SWEEPS

    out: list[tuple[str, str, float, JoinRadii]] = []
    for knob, (field, values) in sweeps.items():
        for value in values:
            radii = JoinRadii(**{**DEFAULT_RADII.as_dict(), field: value})
            out.append((knob, field, value, radii))

    return out


def sweep(
    snapshot: Path | None,
    quick: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run every setting and collect the headline numbers into one frame."""
    plan = settings(quick=quick)
    distinct = {radii.tag for _, _, _, radii in plan}
    log.info("%d row(s) over %d distinct setting(s)", len(plan), len(distinct))

    done: dict[str, tuple[object, float]] = {}
    rows: list[dict[str, object]] = []

    for i, (knob, field, value, radii) in enumerate(plan, start=1):
        is_baseline = radii == DEFAULT_RADII
        log.info(
            "[%d/%d] %s = %g ft%s (%s)",
            i,
            len(plan),
            knob,
            value,
            "  <- published baseline" if is_baseline else "",
            radii.tag,
        )

        if radii.tag in done:
            summary, elapsed = done[radii.tag]
            log.info("    already run at these radii; reusing that result")
        else:
            t0 = time.time()
            # write_artifacts=False without exception, baseline included. The committed
            # headline is produced by `python -m src.pipeline`, and a sweep that can
            # rewrite it is a sweep that can quietly redefine what it measured against.
            summary = run(
                snapshot=snapshot,
                use_cache=use_cache,
                radii=radii,
                write_artifacts=False,
            )
            elapsed = time.time() - t0
            done[radii.tag] = (summary, elapsed)

        rows.append(
            {
                "knob": knob,
                "field": field,
                "value_ft": value,
                "is_baseline": is_baseline,
                "tag": radii.tag,
                "universe_units": summary.universe_units,
                "corridors": summary.corridors,
                "intersections": summary.intersections,
                "crashes_assigned": summary.crashes_assigned,
                "crashes_dropped": summary.crashes_dropped,
                "priority_units_n": summary.priority_units,
                "holdout_casualties": summary.holdout_casualties,
                "r1_pp": summary.r1_citywide_pp,
                "r2_pp": summary.r2_citywide_pp,
                "r3_pp": summary.r3_citywide_pp,
                "lift_pp": summary.lift_pp,
                "ci_low_pp": summary.ci_low_pp,
                "ci_high_pp": summary.ci_high_pp,
                "clears_bar": summary.clears_bar,
                "seconds": round(elapsed, 1),
            }
        )
        log.info(
            "    R1 %.1f%%  R2 %.1f%%  R3 %.1f%%  lift %+.1fpp [%+.1f, %+.1f]  %s  (%.0fs)",
            summary.r1_citywide_pp,
            summary.r2_citywide_pp,
            summary.r3_citywide_pp,
            summary.lift_pp,
            summary.ci_low_pp,
            summary.ci_high_pp,
            "CLEARS" if summary.clears_bar else "DOES NOT CLEAR",
            elapsed,
        )

    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame, snapshot_name: str) -> str:
    """A table a reader can paste into the README without reformatting it."""
    baseline = frame[frame["is_baseline"]].iloc[0]

    lines = [
        "# Sensitivity of the headline to the three join radii",
        "",
        f"Snapshot `{snapshot_name}`. One knob varied at a time; the other two hold at "
        "their published values.",
        "",
        f"Baseline (150 / 100 / 50 ft): R1 {baseline['r1_pp']:.1f}%, "
        f"R2 {baseline['r2_pp']:.1f}%, R3 {baseline['r3_pp']:.1f}%, "
        f"lift {baseline['lift_pp']:+.1f}pp.",
        "",
        "`N` is the size of DOT's list, which every ranking is made to match. `holdout` "
        "is the capture-rate denominator - it moves with the corridor join distance "
        "because a crash that attaches to no unit at all is outside the universe being "
        "scored, so the rates on each row are shares of slightly different totals.",
        "",
        "| knob | ft | N | holdout | R1 | R2 | R3 | lift | 95% CI | vs baseline | bar |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |",
    ]

    for knob in frame["knob"].unique():
        for _, row in frame[frame["knob"] == knob].iterrows():
            delta = row["lift_pp"] - baseline["lift_pp"]
            marker = " **(baseline)**" if row["is_baseline"] else ""
            lines.append(
                f"| {knob}{marker} | {row['value_ft']:.0f} | {row['priority_units_n']:,} | "
                f"{row['holdout_casualties']:,} | "
                f"{row['r1_pp']:.1f}% | {row['r2_pp']:.1f}% | {row['r3_pp']:.1f}% | "
                f"{row['lift_pp']:+.1f}pp | "
                f"[{row['ci_low_pp']:+.1f}, {row['ci_high_pp']:+.1f}] | "
                f"{delta:+.1f}pp | {'yes' if row['clears_bar'] else 'NO'} |"
            )

    worst = frame.loc[frame["lift_pp"].idxmin()]
    best = frame.loc[frame["lift_pp"].idxmax()]
    lines += [
        "",
        f"Lift ranges from {worst['lift_pp']:+.1f}pp ({worst['knob']} at "
        f"{worst['value_ft']:.0f} ft) to {best['lift_pp']:+.1f}pp ({best['knob']} at "
        f"{best['value_ft']:.0f} ft).",
        "",
        (
            f"Every setting clears the pre-registered {MIN_CAPTURE_RATE_LIFT_PP:.0f}pp bar."
            if bool(frame["clears_bar"].all())
            else "**At least one setting does not clear the pre-registered bar.** "
            "The headline is radius-dependent and the README has to say so."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", help="snapshot directory name, e.g. 2026-08-13")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"sweep only the {QUICK_ONLY}, the knob most likely to matter",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="rebuild units for every setting even if a matching cache entry exists",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    snapshot = (RAW_DIR / args.snapshot) if args.snapshot else None
    frame = sweep(snapshot, quick=args.quick, use_cache=not args.no_cache)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "radius-sensitivity.csv"
    md_path = PROCESSED_DIR / "radius-sensitivity.md"

    frame.to_csv(csv_path, index=False)
    snapshot_name = (snapshot or Path("latest")).name
    md_path.write_text(to_markdown(frame, snapshot_name), encoding="utf-8")

    log.info("wrote %s", csv_path)
    log.info("wrote %s", md_path)

    print("\n" + to_markdown(frame, snapshot_name))

    if not bool(frame["clears_bar"].all()):
        # Non-zero, deliberately. A sweep that breaks the headline is the single most
        # important thing this script can find, and it must not scroll past in a log.
        log.error("at least one radius setting does not clear the pre-registered bar")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
