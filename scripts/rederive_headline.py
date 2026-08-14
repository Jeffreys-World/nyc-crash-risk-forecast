#!/usr/bin/env python3
"""Recompute the published headline from the committed artifacts, sharing no code with it.

The headline has been verified three times and none of those checks could have caught the
bug they were meant to. Re-running `src/pipeline.py` re-runs `src/backtest.py`; pulling a
fresh snapshot and re-running still re-runs `src/backtest.py`. A wrong `capture_rate` or a
wrong `select_citywide_top_n` reproduces perfectly every time, on every machine, and stays
invisible for exactly as long as every check goes through the same code.

So this file goes around it. It reads `data/processed/scored-units.parquet` — the per-unit
inputs to the scoring layer — and rebuilds every number in `run-summary.json` from scratch:

    R1  casualties at DOT's published units, over all holdout casualties
    R2  casualties at the top-N units by trailing 36-month count
    R3  casualties at the top-N units by Empirical Bayes estimate
        the lift, and its bootstrap confidence interval
        both selection regimes, citywide and borough-stratified
        the treated / untreated split
        the Empirical Bayes blend itself, rebuilt from `spf_prediction` and the fitted k

Then it compares each one against the committed summary and exits non-zero on any
disagreement. It imports numpy, pandas, pyarrow and the standard library. It imports
nothing from `src/`, and `tests/test_rederive.py` asserts that statically, because the
independence is the entire value and a convenience import would quietly end it.

The constants below are restated rather than imported for the same reason. If one of them
drifts from `src/config.py`, the comparison fails — which is the correct outcome, not a
nuisance: it means the published summary and the values it claims to have used no longer
agree.

## What this does NOT check

Being clear about this matters more than the check itself, because a re-derivation
reported without its limits reads as more assurance than it is.

* **The spatial join.** Which unit each crash landed on is taken as given. This is where
  the harder bugs in this project have actually been — the MultiLineString that became one
  unit spanning two places, the VZV labels that stopped at segments while the crashes went
  to nodes. None of that is visible from here.
* **The feature build.** `casualties_36mo` and `holdout_casualties` are read, not rebuilt.
* **The negative-binomial fit.** The dispersion k comes from the summary. The blend that
  consumes it is rebuilt; the fit that produced it is not.
* **The pull.** The snapshot is upstream of everything here.

What it does check is the scoring: selection, ranking, tie-breaking, capture rates, the
bootstrap, and the EB arithmetic. That is the layer the headline is a direct statement
about, and it was the only layer with no independent check on it.

    python scripts/rederive_headline.py
    python scripts/rederive_headline.py --bootstrap-iterations 200   # a fast smoke run

Writes `data/processed/rederivation.md`. Exits 1 if anything disagrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

# Restated from src/config.py on purpose. See the module docstring.
BOROUGH_CUMULATIVE_SHARE = {"corridor": 0.50, "intersection": 0.15}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 20260812
MIN_CAPTURE_RATE_LIFT_PP = 5.0
MIN_PREDICTION = 1e-9
HOLDOUT_START = pd.Timestamp("2024-01-01")

# Tight because none of this arithmetic should drift. Holdout casualties are integers held
# as floats, so every sum below is exact; the only genuine float noise is in the bootstrap
# percentiles, and it lands far below this. A tolerance loose enough to absorb a real bug
# would make the whole exercise decorative.
DEFAULT_TOLERANCE_PP = 1e-6


class RederivationError(RuntimeError):
    """The committed artifacts cannot be re-derived, or do not describe the same run."""


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------


def tiebreak_key(unit_ids: np.ndarray) -> np.ndarray:
    """A content-free ordering key for units with equal scores.

    Reimplemented from the rule `src/backtest.py` documents: a blake2b digest of the id,
    eight bytes, big-endian, ascending. It has to match, or the two selections differ at
    the cut for reasons that have nothing to do with a bug — most units are tied at zero,
    so the tie-break decides thousands of the 38,909 picks.

    That is a real limit on this check and worth naming: a wrong tie-break rule would be
    reproduced here rather than caught. What is caught is a tie-break applied wrongly —
    to the wrong column, in the wrong direction, or not at all.

    Returned as uint64 because the top bit is set roughly half the time and int64 would
    overflow on those ids.
    """
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(str(uid).encode("utf-8"), digest_size=8).digest(), "big"
            )
            for uid in unit_ids
        ],
        dtype=np.uint64,
    )


def rank_order(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Row positions in ranked order: score descending, tie-break ascending.

    `np.lexsort` rather than a dataframe sort, deliberately — a different mechanism
    reaching the same order is worth more here than a tidier one that happens to be the
    mechanism under test. The last key is primary, so `-scores` gives descending.

    Negating is exact for float64 (it flips one sign bit), so this is a true reversal and
    not an approximation of one. NaN would break it silently, which is why the caller
    rejects NaN scores outright rather than letting them sort to an arbitrary end.
    """
    if np.isnan(scores).any():
        raise RederivationError(
            "score column contains NaN. NaN sorts to one end without raising, so the "
            "selection would be decided by missingness rather than by the ranking."
        )
    return np.lexsort((tiebreak, -scores))


def top_n(order: np.ndarray, n: int) -> np.ndarray:
    """A boolean mask over rows for the first N in ranked order."""
    if n <= 0:
        raise RederivationError(f"citywide N must be positive, got {n}")
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:n]] = True
    return mask


def tie_block_at_cut(scores: np.ndarray, order: np.ndarray, n: int) -> tuple[int, int]:
    """`(units sharing the cut-off score, selections the tie-break decided)`.

    Not a check on anything — a disclosure, and the more useful half is the second number.
    Every unit sharing the boundary score is equally ranked by the method; which of them
    made the list was decided by a hash of its id. So a ranking that reaches its budget
    inside a tie block has stopped ranking and started drawing lots, and the count of
    picks that came out of the hat is the honest measure of how far that goes.

    A worksheet that presents the last of those as the 38,909th most dangerous location in
    New York is reporting the tie-break as if it were the model.
    """
    if n >= len(order):
        return 0, 0
    boundary = scores[order[n - 1]]
    block = int((scores == boundary).sum())
    strictly_above = int((scores > boundary).sum())
    return block, n - strictly_above


def share(part: int, whole: int) -> str:
    """A percentage that stays informative when it is small.

    `13 of 38,909` at one decimal renders as `0.0%`, which reads as "none" where it means
    "thirteen locations were decided by a hash". The extra digits cost nothing, and
    rounding a real quantity down to nothing is the only thing in this report that could
    mislead on its own.
    """
    if whole <= 0:
        return "n/a"
    value = 100.0 * part / whole
    return f"{value:.1f}%" if value == 0 or value >= 0.1 else f"{value:.2g}%"


# --------------------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------------------


def capture_pp(casualties: np.ndarray, selected: np.ndarray) -> float:
    """Percentage of holdout casualties occurring at selected units.

    The zero denominator is refused rather than returned as 0%, matching the guard the
    pipeline documents. An undefined rate that arrives as a number is how a blank cell
    reaches a README looking like a measurement.
    """
    total = float(casualties.sum())
    if total <= 0:
        raise RederivationError(
            "holdout casualties sum to zero, so the capture rate has no denominator. "
            "This is undefined, not 0%."
        )
    return 100.0 * float(casualties[selected].sum()) / total


def stratified_mask(
    frame: pd.DataFrame, scores: np.ndarray, tiebreak: np.ndarray
) -> np.ndarray:
    """DOT's own rule: rank within borough and unit type, stop at a cumulative share.

    Corridors stop at 50% of that borough's trailing casualties, intersections at 15%.
    Units with no borough take no part — left in, they become a phantom sixth borough with
    its own stopping rule and their own quota of extra picks.

    The share is taken over `casualties_36mo`, the same column DOT's rule stops on, while
    the *ranking* may be either that column (R2) or the EB estimate (R3). Scoring is still
    against the full citywide holdout total, so the denominator is unchanged by which
    units the rule happened to reach.
    """
    casualties = frame["casualties_36mo"].to_numpy(dtype=float)
    borough = frame["borough"].to_numpy()
    unit_type = frame["unit_type"].to_numpy()
    placeable = pd.notna(frame["borough"]).to_numpy()

    selected = np.zeros(len(frame), dtype=bool)
    for b in pd.unique(borough[placeable]):
        for t in pd.unique(unit_type):
            share = BOROUGH_CUMULATIVE_SHARE.get(str(t))
            if share is None:
                continue
            rows = np.flatnonzero(placeable & (borough == b) & (unit_type == t))
            if rows.size == 0:
                continue
            total = float(casualties[rows].sum())
            if total <= 0:
                continue

            local = rows[rank_order(scores[rows], tiebreak[rows])]
            cumulative = np.cumsum(casualties[local]) / total
            reached = np.flatnonzero(cumulative >= share)
            cutoff = int(reached[0]) + 1 if reached.size else len(local)
            selected[local[:cutoff]] = True

    if not selected.any():
        raise RederivationError(
            "borough-stratified selection picked nothing, which means the casualty "
            "column reaching it is wrong"
        )
    return selected


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------


def bootstrap_ci(
    casualties: np.ndarray,
    in_a: np.ndarray,
    in_b: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    level: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """Percentile interval on the capture-rate difference (A - B), resampling units.

    The seed and the draw shape are pinned to the published run's. They have to be: a
    bootstrap on a different draw sequence produces a different, equally valid interval,
    and then the two numbers cannot be compared at all. So what is independent here is the
    arithmetic on each resample, not the resampling design — the design is a fixed input,
    like the dispersion.

    The resample is taken by gathering, `values[idx]`, where the pipeline accumulates
    multiplicities with `bincount` and takes a dot product. Different operations, and for
    integer-valued casualties held as float64 both are exact, so agreement to the last bit
    is the expected result rather than a lucky one.
    """
    total_all = float(casualties.sum())
    if total_all <= 0:
        raise RederivationError("cannot bootstrap: no holdout casualties")

    values_a = casualties * in_a
    values_b = casualties * in_b

    rng = np.random.default_rng(seed)
    n = len(casualties)
    diffs = np.full(iterations, np.nan, dtype=float)
    degenerate = 0

    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        total = casualties[idx].sum()
        if total <= 0:
            degenerate += 1
            continue
        diffs[i] = 100.0 * (values_a[idx].sum() / total - values_b[idx].sum() / total)

    usable = diffs[np.isfinite(diffs)]
    if usable.size == 0:
        raise RederivationError("every bootstrap resample had a zero denominator")

    alpha = (1.0 - level) / 2.0
    return (
        float(np.percentile(usable, 100 * alpha)),
        float(np.percentile(usable, 100 * (1 - alpha))),
        degenerate,
    )


# --------------------------------------------------------------------------------------
# The Empirical Bayes blend
# --------------------------------------------------------------------------------------


def rebuild_eb(frame: pd.DataFrame, dispersion: dict[str, float]) -> np.ndarray:
    """Rebuild `eb_estimate` from the SPF prediction and the fitted dispersion.

        w  = 1 / (1 + k * P)
        EB = w * P + (1 - w) * observed

    Per unit type, because corridors and intersections are fitted separately and k is what
    sets the blend weight. Predictions at or below 1e-9 are floored before the weight is
    formed, matching the guard the pipeline documents: at P = 0 the weight is exactly 1 and
    the blend would discard the site's observed crashes entirely.

    Units with no prediction were never fitted — degenerate exposure — and keep their
    observed count, which is what leaves them rankable and countable rather than dropped
    out of the capture-rate denominator.
    """
    predicted = frame["spf_prediction"].to_numpy(dtype=float)
    observed = frame["casualties_36mo"].to_numpy(dtype=float)
    unit_type = frame["unit_type"].to_numpy()

    out = observed.copy()
    fitted = ~np.isnan(predicted)
    for t, k in dispersion.items():
        if not np.isfinite(k) or k <= 0:
            raise RederivationError(
                f"dispersion for {t!r} is {k!r}; w = 1/(1+k*P) is undefined there"
            )
        rows = fitted & (unit_type == t)
        if not rows.any():
            continue
        p = np.maximum(predicted[rows], MIN_PREDICTION)
        w = 1.0 / (1.0 + k * p)
        out[rows] = w * p + (1.0 - w) * observed[rows]

    unnamed = fitted & ~np.isin(unit_type, list(dispersion))
    if unnamed.any():
        raise RederivationError(
            f"{int(unnamed.sum())} fitted unit(s) have a unit_type with no dispersion in "
            f"the summary; their EB estimate cannot be rebuilt"
        )
    return out


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


@dataclass
class Check:
    """One published number against its re-derivation."""

    label: str
    published: float
    rederived: float
    tolerance: float
    unit: str = "pp"

    @property
    def delta(self) -> float:
        return self.rederived - self.published

    @property
    def agrees(self) -> bool:
        return bool(abs(self.delta) <= self.tolerance)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # ranking -> (units sharing the cut-off score, selections the tie-break decided)
    ties: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add(
        self, label: str, published: float, rederived: float, tolerance: float, unit="pp"
    ) -> None:
        self.checks.append(Check(label, float(published), float(rederived), tolerance, unit))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.agrees]


def load_artifacts(processed: Path) -> tuple[pd.DataFrame, dict, dict]:
    """The per-unit frame, its embedded provenance, and the published summary.

    The provenance cross-check is the point of reading the metadata at all. The frame and
    the summary are written as one record but land as three separate files, so "these two
    describe the same run" is an assumption until something verifies it — and comparing a
    150 ft frame against a 100 ft summary would report a mismatch that looks exactly like
    a scoring bug.
    """
    units_path = processed / "scored-units.parquet"
    summary_path = processed / "run-summary.json"
    for path in (units_path, summary_path):
        if not path.exists():
            raise RederivationError(
                f"{path} not found. Run `python -m src.pipeline` to produce the "
                f"committed artifacts first."
            )

    table = pq.read_table(units_path)
    raw = table.schema.metadata or {}
    provenance = {
        k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"
    }
    frame = table.to_pandas()
    summary = json.loads(summary_path.read_text())

    if provenance.get("snapshot_date") != summary["snapshot_date"]:
        raise RederivationError(
            f"the per-unit frame was built from snapshot "
            f"{provenance.get('snapshot_date')!r} but the summary reports "
            f"{summary['snapshot_date']!r}. These are two different runs."
        )
    frame_radii = json.loads(provenance.get("join_radii", "{}"))
    if frame_radii != summary.get("join_radii"):
        raise RederivationError(
            f"the per-unit frame was built at radii {frame_radii} but the summary "
            f"reports {summary.get('join_radii')}. These are two different runs."
        )

    duplicated = int(frame["unit_id"].duplicated().sum())
    if duplicated:
        raise RederivationError(
            f"{duplicated} duplicated unit_id(s) in the per-unit frame. Casualties would "
            f"be counted more than once on both sides of every rate below."
        )
    return frame, provenance, summary


def rederive(
    frame: pd.DataFrame,
    summary: dict,
    iterations: int = BOOTSTRAP_ITERATIONS,
    tolerance_pp: float = DEFAULT_TOLERANCE_PP,
) -> Report:
    """Rebuild every published number and compare. Returns the comparison, does not print."""
    report = Report()

    unit_ids = frame["unit_id"].to_numpy()
    holdout = frame["holdout_casualties"].to_numpy(dtype=float)
    trailing = frame["casualties_36mo"].to_numpy(dtype=float)
    eb = frame["eb_estimate"].to_numpy(dtype=float)
    is_priority = frame["is_priority"].to_numpy(dtype=bool)
    tiebreak = tiebreak_key(unit_ids)

    # Shape first. A rate that matches on a frame of the wrong size is a coincidence, not
    # a confirmation, so the counts are checked before anything divides by them.
    report.add("universe units", summary["universe_units"], len(frame), 0, "units")
    report.add("holdout casualties", summary["holdout_casualties"], holdout.sum(), 0, "cas")
    report.add("priority units (N)", summary["priority_units"], is_priority.sum(), 0, "units")
    report.add("citywide N", summary["citywide_n"], int(is_priority.sum()), 0, "units")

    n = int(is_priority.sum())

    # The blend, rebuilt from the SPF prediction and the fitted k rather than trusted.
    rebuilt = rebuild_eb(frame, summary["dispersion"])
    eb_gap = float(np.nanmax(np.abs(rebuilt - eb))) if len(eb) else 0.0
    # 1e-12 absolute on estimates that run to ~19: the blend is a handful of float
    # operations, so anything above rounding here is a different formula, not drift.
    #
    # The label carries no pipe characters because it is rendered into a markdown table,
    # where `max |x - y|` silently splits one cell into three.
    report.add("EB blend, max abs(rebuilt - published)", 0.0, eb_gap, 1e-12, "cas")

    # R1 is not a ranking. It is the published list itself, taken as a selection.
    r1_pp = capture_pp(holdout, is_priority)
    report.add("R1 DOT published, citywide", summary["r1_citywide_pp"], r1_pp, tolerance_pp)

    order_r2 = rank_order(trailing, tiebreak)
    order_r3 = rank_order(eb, tiebreak)
    in_r2 = top_n(order_r2, n)
    in_r3 = top_n(order_r3, n)

    r2_pp = capture_pp(holdout, in_r2)
    r3_pp = capture_pp(holdout, in_r3)
    report.add("R2 raw count, citywide", summary["r2_citywide_pp"], r2_pp, tolerance_pp)
    report.add("R3 empirical bayes, citywide", summary["r3_citywide_pp"], r3_pp, tolerance_pp)

    lift = r3_pp - r2_pp
    report.add("lift R3 - R2", summary["lift_pp"], lift, tolerance_pp)

    low, high, degenerate = bootstrap_ci(holdout, in_r3, in_r2, iterations=iterations)
    if iterations != BOOTSTRAP_ITERATIONS:
        # A reduced run resamples a different number of times, so its interval is a
        # different estimate of the same quantity and cannot be held to 1e-6. Compared
        # loosely and labelled, rather than skipped silently or reported as a match.
        report.notes.append(
            f"bootstrap ran {iterations} iterations against the published "
            f"{BOOTSTRAP_ITERATIONS}; its interval is compared at 0.5pp, not "
            f"{tolerance_pp:g}pp"
        )
        ci_tolerance = 0.5
    else:
        ci_tolerance = tolerance_pp
    report.add("bootstrap CI low", summary["ci_low_pp"], low, ci_tolerance)
    report.add("bootstrap CI high", summary["ci_high_pp"], high, ci_tolerance)
    if degenerate:
        report.notes.append(f"{degenerate}/{iterations} resamples had a zero denominator")

    # Regime A: DOT's own stopping rule, applied to each ranking.
    strat_r2 = stratified_mask(frame, trailing, tiebreak)
    strat_r3 = stratified_mask(frame, eb, tiebreak)
    report.add(
        "R2 raw count, borough-stratified",
        summary["r2_stratified_pp"],
        capture_pp(holdout, strat_r2),
        tolerance_pp,
    )
    report.add(
        "R3 empirical bayes, borough-stratified",
        summary["r3_stratified_pp"],
        capture_pp(holdout, strat_r3),
        tolerance_pp,
    )

    # The endogeneity control: R1 split by whether the street was rebuilt, and when.
    treated = frame["treated"].fillna(False).to_numpy(dtype=bool)
    when = pd.to_datetime(frame["treatment_date"], errors="coerce")
    dated = when.notna().to_numpy()
    before = dated & (when < HOLDOUT_START).to_numpy()
    during = dated & (when >= HOLDOUT_START).to_numpy()

    for label, mask, key in (
        ("R1 at units treated before the holdout", treated & before, "treated_before_holdout_pp"),
        ("R1 at units treated during the holdout", treated & during, "treated_during_holdout_pp"),
        ("R1 at untreated units", ~treated, "untreated_pp"),
    ):
        # Each group is its own universe: the denominator is that group's casualties, not
        # the city's. A group scored against the citywide total would report a share of
        # New York rather than a capture rate, and all three would look catastrophic.
        report.add(
            label, summary[key], capture_pp(holdout[mask], is_priority[mask]), tolerance_pp
        )

    # Counted over every dated unit, not only the flagged ones — which is how the pipeline
    # counts them, and the two have to be the same question or the comparison is empty.
    report.add(
        "units treated before the holdout",
        summary["treated_before_holdout_units"],
        before.sum(),
        0,
        "units",
    )
    report.add(
        "units treated during the holdout",
        summary["treated_during_holdout_units"],
        during.sum(),
        0,
        "units",
    )

    # Disclosure, not comparison: the summary makes no claim about any of this.
    #
    # It is the most interesting thing the re-derivation found. R2's cut at N=38,909 lands
    # inside the block of units tied at zero trailing casualties, so most of what it
    # "ranked" it drew by hash — which is the README's central claim about count-based
    # ranking, arriving as a measurement rather than an argument.
    for label, scores, order in (("R2", trailing, order_r2), ("R3", eb, order_r3)):
        block, drawn = tie_block_at_cut(scores, order, n)
        report.ties[label] = (block, drawn)
        report.notes.append(
            f"{label}: {block:,} units share the cut-off score, so {drawn:,} of its "
            f"{n:,} selections ({share(drawn, n)}) were decided by the tie-break "
            f"and not by the ranking"
        )
    report.notes.append(
        f"pre-registered bar: lift {lift:+.2f}pp against a {MIN_CAPTURE_RATE_LIFT_PP:.1f}pp "
        f"threshold, CI [{low:+.2f}, {high:+.2f}] "
        f"{'excludes' if (low > 0 or high < 0) else 'includes'} zero"
    )
    return report


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def render(report: Report, summary: dict, provenance: dict, iterations: int) -> str:
    lines = [
        "# Independent re-derivation of the headline",
        "",
        f"Snapshot `{summary['snapshot_date']}`, radii "
        f"`{json.dumps(summary.get('join_radii', {}), sort_keys=True)}`, "
        f"{iterations:,} bootstrap iterations. Frame written by "
        f"`{provenance.get('produced_by', 'unknown')}`.",
        "",
        "Every number below was rebuilt from `data/processed/scored-units.parquet` by "
        "`scripts/rederive_headline.py`, which imports nothing from `src/`. It checks the "
        "scoring layer only — not the spatial join, not the feature build, not the "
        "negative-binomial fit. That limit is the important part; the script's docstring "
        "says why.",
        "",
        "| Quantity | Published | Re-derived | Δ | |",
        "|---|---:|---:|---:|---|",
    ]
    for c in report.checks:
        mark = "ok" if c.agrees else "**MISMATCH**"
        fmt = "{:,.0f}" if c.unit == "units" else "{:,.4f}"
        lines.append(
            f"| {c.label} | {fmt.format(c.published)} | {fmt.format(c.rederived)} | "
            f"{c.delta:+.2e} | {mark} |"
        )

    lines += [
        "",
        f"**{len(report.checks) - len(report.failures)} of {len(report.checks)} agree.** "
        + (
            "The headline re-derives."
            if not report.failures
            else "**The headline does not re-derive.** See the mismatched rows above."
        ),
        "",
        "## How much of each ranking was decided by the tie-break",
        "",
        "Nothing above depends on this and the summary makes no claim about it. It is "
        "reported because it is the sharpest thing the re-derivation can see: a ranking "
        "whose budget runs out inside a block of equal scores has stopped ranking, and "
        "the rest of its picks came out of a hash.",
        "",
        "| Ranking | Units sharing the cut-off score | Picks the tie-break decided |",
        "|---|---:|---:|",
    ]
    n = next((int(c.rederived) for c in report.checks if c.label == "citywide N"), 0)
    for label, (block, drawn) in report.ties.items():
        of_n = f" ({share(drawn, n)} of {n:,})" if n else ""
        lines.append(f"| {label} | {block:,} | {drawn:,}{of_n} |")

    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in report.notes]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--processed",
        type=Path,
        default=PROCESSED,
        help="directory holding the committed artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
        help="fewer runs faster but cannot match the published interval exactly",
    )
    parser.add_argument(
        "--tolerance-pp", type=float, default=DEFAULT_TOLERANCE_PP, help=argparse.SUPPRESS
    )
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args(argv)

    try:
        frame, provenance, summary = load_artifacts(args.processed)
        report = rederive(
            frame,
            summary,
            iterations=args.bootstrap_iterations,
            tolerance_pp=args.tolerance_pp,
        )
    except RederivationError as exc:
        print(f"\nRE-DERIVATION FAILED: {exc}\n", file=sys.stderr)
        return 1

    width = max(len(c.label) for c in report.checks)
    print()
    print("=" * (width + 46))
    print("INDEPENDENT RE-DERIVATION")
    print("=" * (width + 46))
    for c in report.checks:
        fmt = "{:>14,.0f}" if c.unit == "units" else "{:>14,.4f}"
        print(
            f"  {c.label:<{width}} {fmt.format(c.published)} {fmt.format(c.rederived)}  "
            f"{'ok' if c.agrees else 'MISMATCH'}"
        )
    print()
    for note in report.notes:
        print(f"  note: {note}")
    print("=" * (width + 46))

    if not args.no_write:
        out = args.processed / "rederivation.md"
        # Explicit utf-8: the table header carries a Δ, and Windows' default cp1252
        # encoder raises on it rather than degrading, so the report dies after doing all
        # the work. Every markdown artifact in this repo is utf-8; only this one said so.
        out.write_text(
            render(report, summary, provenance, args.bootstrap_iterations),
            encoding="utf-8",
        )
        print(f"  wrote {out}")

    if report.failures:
        print(f"\n  {len(report.failures)} MISMATCH(ES). The headline does not re-derive.\n")
        return 1
    print(f"\n  all {len(report.checks)} quantities re-derive.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
