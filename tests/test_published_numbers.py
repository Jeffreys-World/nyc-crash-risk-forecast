"""The README's numbers, checked against the artifacts they claim to come from.

Lint passes and 267 tests pass while the README quotes a figure that stopped being true
three commits ago. That is not hypothetical: the test count sat at "203" until 2026-08-13,
wrong for weeks, in a repo whose entire argument is that its numbers can be trusted. A
green suite next to a stale headline is worse than no suite, because it certifies the part
that was never in doubt.

So every published figure is extracted from `README.md` by an anchor that names it, and
compared against `data/processed/run-summary.json` or `radius-sensitivity.csv` — the files
the pipeline actually wrote.

Two design choices worth defending:

* **Each anchor must match exactly once.** A pattern that stops matching fails loudly
  rather than passing vacuously, which is the failure mode of every "assert the string is
  in the file" check. Rewriting the sentence around a number is then a deliberate act:
  the test names which claim it lost.
* **Comparison is on the rendered string, not the float.** The README says `48.7%`; the
  summary holds `48.74024032338446`. Comparing `f"{value:.1f}"` checks the number as a
  reader sees it, and sidesteps a tolerance argument that has no right answer.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
PROCESSED = REPO_ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def summary() -> dict:
    path = PROCESSED / "run-summary.json"
    if not path.exists():
        pytest.skip(f"{path} not committed; nothing to check the README against")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sweep() -> pd.DataFrame:
    path = PROCESSED / "radius-sensitivity.csv"
    if not path.exists():
        pytest.skip(f"{path} not committed")
    return pd.read_csv(path)


def anchored(readme: str, pattern: str) -> tuple[str, ...]:
    """The one match for `pattern`, or a failure naming the claim that went missing."""
    matches = re.findall(pattern, readme)
    assert len(matches) == 1, (
        f"expected exactly one README passage matching {pattern!r}, found "
        f"{len(matches)}. Either the sentence carrying this number was rewritten — in "
        f"which case update the anchor — or the number is now quoted in two places that "
        f"can drift apart."
    )
    match = matches[0]
    return match if isinstance(match, tuple) else (match,)


def pct(value: float) -> str:
    return f"{value:.1f}"


def count(value: float) -> str:
    return f"{round(value):,}"


# --------------------------------------------------------------------------------------
# The headline
# --------------------------------------------------------------------------------------


def test_the_three_capture_rates_match_the_run(readme, summary):
    """The table the whole repo exists to produce."""
    total = summary["holdout_casualties"]

    for label, key in (
        (r"\*\*R1\*\* DOT's published Vision Zero list", "r1_citywide_pp"),
        (r"\*\*R2\*\* raw trailing casualty count", "r2_citywide_pp"),
        (r"\*\*R3\*\* Empirical Bayes \(SPF \+ observed\)", "r3_citywide_pp"),
    ):
        captured, rate = anchored(
            readme, rf"\| {label} \| ([\d,]+) \| \*\*([\d.]+)%\*\* \|"
        )
        assert rate == pct(summary[key]), f"{key}: README says {rate}%"
        # The count is the rate's own arithmetic, so a row where they disagree is a row
        # where one of them was hand-edited.
        assert captured == count(summary[key] / 100.0 * total), (
            f"{key}: README says {captured} casualties, which is not "
            f"{summary[key]:.4f}% of {total}"
        )


def test_the_holdout_denominator_matches(readme, summary):
    (share_of,) = anchored(readme, r"\| Share of ([\d,]+) \|")
    assert share_of == count(summary["holdout_casualties"])

    (provenance,) = anchored(readme, r"\| Holdout casualties, 2024.2025 \| ([\d,]+) \|")
    assert provenance == count(summary["holdout_casualties"])


def test_the_lift_and_its_interval_match(readme, summary):
    lift, low, high = anchored(
        readme, r"\*\*R3 . R2 = \+([\d.]+)pp, 95% CI \[\+([\d.]+), \+([\d.]+)\]\.?\*\*"
    )
    assert lift == pct(summary["lift_pp"])
    assert low == pct(summary["ci_low_pp"])
    assert high == pct(summary["ci_high_pp"])
    assert summary["ci_excludes_zero"], "the README quotes an interval that includes zero"


def test_the_selection_size_matches(readme, summary):
    (n,) = anchored(readme, r"each ranking selecting ([\d,]+) locations")
    assert n == count(summary["citywide_n"])

    (row_n,) = anchored(readme, r"\| Regime B, citywide top-N \(([\d,]+) units\)")
    assert row_n == count(summary["citywide_n"])


def test_the_n_sweep_row_at_the_published_n_matches(readme, summary):
    """The one row of the N-sweep that this run also produces has to agree with it.

    The other rows come from a sweep that is not committed, so they are unverifiable here
    and are left alone rather than checked badly.
    """
    r2, r3, lift = anchored(
        readme,
        rf"\| {summary['citywide_n']:,} \| ([\d.]+)% \| ([\d.]+)% \| \*\*\+([\d.]+)pp\*\*",
    )
    assert (r2, r3, lift) == (
        pct(summary["r2_citywide_pp"]),
        pct(summary["r3_citywide_pp"]),
        pct(summary["lift_pp"]),
    )


def test_both_selection_regimes_match(readme, summary):
    r2, r3, lift = anchored(
        readme,
        r"\| Regime A, borough-stratified \([\d,]+ / [\d,]+ units\) \| ([\d.]+)% \| "
        r"([\d.]+)% \| \+([\d.]+)pp \|",
    )
    assert r2 == pct(summary["r2_stratified_pp"])
    assert r3 == pct(summary["r3_stratified_pp"])
    assert lift == pct(summary["r3_stratified_pp"] - summary["r2_stratified_pp"])

    r2, r3, lift = anchored(
        readme,
        r"\| Regime B, citywide top-N \([\d,]+ units\) \| ([\d.]+)% \| ([\d.]+)% \| "
        r"\+([\d.]+)pp \|",
    )
    assert r2 == pct(summary["r2_citywide_pp"])
    assert r3 == pct(summary["r3_citywide_pp"])
    assert lift == pct(summary["lift_pp"])


def test_the_treatment_split_matches(readme, summary):
    """The endogeneity control. Its three rates are the reason R1's 48.7% is not a verdict."""
    for row, key in (
        (r"Treated before the holdout", "treated_before_holdout_pp"),
        (r"Treated during the holdout", "treated_during_holdout_pp"),
        (r"Untreated", "untreated_pp"),
    ):
        (rate,) = anchored(
            readme, rf"\| {row} \| [\d,]+ \| [\d,]+ \| \*?\*?([\d.]+)%\*?\*? \|"
        )
        assert rate == pct(summary[key]), f"{key}: README says {rate}%"


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def test_the_snapshot_vintage_matches(readme, summary):
    (vintage,) = anchored(readme, r"\*\*Snapshot vintage:\*\* `([\d-]+)`")
    assert vintage == summary["snapshot_date"]


def test_the_provenance_table_matches(readme, summary):
    (pulled,) = anchored(readme, r"\| Crashes pulled \(2016-01-01 onward\) \| ([\d,]+) \|")
    assert pulled == count(summary["crashes_pulled"])

    assigned, share = anchored(
        readme, r"\| Crashes assigned to a unit \| ([\d,]+) \(([\d.]+)%\) \|"
    )
    assert assigned == count(summary["crashes_assigned"])
    assert share == pct(100.0 * summary["crashes_assigned"] / summary["crashes_pulled"])

    universe, corridors, intersections = anchored(
        readme,
        r"\| Units in the universe \| ([\d,]+) \(([\d,]+) corridors, "
        r"([\d,]+) intersections\) \|",
    )
    assert universe == count(summary["universe_units"])
    assert corridors == count(summary["corridors"])
    assert intersections == count(summary["intersections"])


def test_the_universe_size_in_the_headline_matches(readme, summary):
    (universe,) = anchored(readme, r"\*\*[\d,]+ of ([\d,]+) units \([\d.]+%\)\*\*")
    assert universe == count(summary["universe_units"])


def test_the_treatment_counts_match(readme, summary):
    (sip,) = anchored(readme, r"([\d,]+) units carry an SIP")
    assert sip == count(summary["treated_units"])

    (during,) = anchored(readme, r"so \*{0,2}([\d,]+)\*{0,2} units\s+were rebuilt")
    assert during == count(summary["treated_during_holdout_units"])


def test_the_corridor_dispersion_matches(readme, summary):
    """5.70 is quoted as the punchline of the discarded run, so it has to be this run's."""
    (dispersion,) = anchored(readme, r"Dispersion went from [\d.]+ to ([\d.]+)")
    assert dispersion == f"{summary['dispersion']['corridor']:.2f}"


# --------------------------------------------------------------------------------------
# The sensitivity sweep
# --------------------------------------------------------------------------------------


def test_the_radius_sweep_table_matches(readme, sweep):
    """Nine lifts across three knobs, each read back from the sweep that produced them."""
    rows = {
        "Corridor join distance": ("corridor join distance", (100.0, 150.0, 250.0)),
        "VZV label buffer": ("VZV label buffer", (25.0, 50.0, 100.0)),
        "Intersection radius": ("intersection radius", (50.0, 100.0, 150.0)),
    }
    for label, (knob, values) in rows.items():
        quoted = anchored(
            readme,
            rf"\| {label} \| [\d /*]+ft \| \+([\d.]+) / \*\*\+([\d.]+)\*\* / \+([\d.]+)pp \|",
        )
        for value, said in zip(values, quoted, strict=True):
            measured = sweep[(sweep["knob"] == knob) & (sweep["value_ft"] == value)]
            assert len(measured) == 1, f"{knob} at {value} ft is not in the sweep"
            assert said == pct(measured["lift_pp"].iloc[0]), (
                f"{knob} at {value} ft: README says +{said}pp, sweep measured "
                f"+{measured['lift_pp'].iloc[0]:.4f}pp"
            )


def test_the_quoted_lift_range_spans_the_whole_sweep(readme, sweep):
    low, high = anchored(readme, r"lift spans \+([\d.]+)pp to \+([\d.]+)pp")
    assert low == pct(sweep["lift_pp"].min())
    assert high == pct(sweep["lift_pp"].max())
    assert bool(sweep["clears_bar"].all()), "the README claims every setting clears the bar"


def test_the_vzv_buffer_moves_n_by_what_the_sweep_says(readme, sweep):
    quoted = anchored(readme, r"([\d,]+) at 25 ft, ([\d,]+) at 50 ft, ([\d,]+) at 100 ft")
    buffer_rows = sweep[sweep["knob"] == "VZV label buffer"].set_index("value_ft")
    for value, said in zip((25.0, 50.0, 100.0), quoted, strict=True):
        assert said == count(buffer_rows.loc[value, "priority_units_n"])


# --------------------------------------------------------------------------------------
# The claim about the suite itself
# --------------------------------------------------------------------------------------


def test_the_quoted_test_count_is_the_real_one(readme):
    """The number that was wrong for weeks, now checked by the thing it counts.

    Collected in a subprocess rather than measured in-process: pytest is already running,
    and asking it to collect itself from inside a test is a reentrancy problem with no
    upside. Collection does not execute anything, so this costs a few seconds and no
    fixtures.
    """
    result = subprocess.run(
        # `-o addopts=` clears the ini options for the child run. Without it this check
        # depends on pyproject's `-q`: at one level of quiet pytest still prints the
        # "N tests collected" total, at two it prints per-file counts and no total, so
        # adding a `-q` to addopts would break a test about the README.
        [sys.executable, "-m", "pytest", "--collect-only", "-o", "addopts=",
         "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    found = re.search(r"(\d+) tests? collected", result.stdout)
    assert found, (
        f"could not read a test count from pytest's collection output:\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    collected = int(found.group(1))

    quoted = set(re.findall(r"(\d+) tests", readme))
    assert quoted, "the README no longer states a test count"
    assert quoted == {str(collected)}, (
        f"the README says {sorted(quoted)} tests; pytest collects {collected}. "
        f"A test count is the cheapest possible claim to keep true."
    )
