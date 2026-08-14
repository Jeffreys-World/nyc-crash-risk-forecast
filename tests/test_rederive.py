"""The re-derivation must agree with the pipeline, and must stay independent of it.

Two different things are tested here and the second is easy to lose sight of.

The first is agreement: given a scored frame, `scripts/rederive_headline.py` and
`src/backtest.py` have to produce the same capture rates, the same lift, the same
interval, the same treated/untreated split. That is what the script asserts on the real
committed artifacts, so it had better hold on a frame whose answer is known.

The second is independence, and it is the whole reason the script exists. A re-derivation
that imports `src.backtest.capture_rate` checks nothing at all — it would reproduce that
function's bugs exactly and report a match. Nothing stops a later edit from adding that
import for convenience, except a test that fails when it appears. So the imports are
parsed and asserted, statically, rather than trusted.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    Selection,
    bootstrap_capture_difference,
    capture_rate,
    select_borough_stratified,
    select_citywide_top_n,
    split_by_treatment,
)
from src.pipeline import HOLDOUT_START, RunSummary, _write_artifacts
from src.spf import empirical_bayes

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rederive_headline.py"

DISPERSION = {"corridor": 1.3, "intersection": 0.7}


def _load_script():
    """Import the script by path, the way a reader running it from a shell would.

    Registered in `sys.modules` before it executes, not after. `@dataclass` resolves its
    annotations through `sys.modules[cls.__module__]`, so a module that is executing while
    absent from that table raises an AttributeError from inside `dataclasses` — a failure
    that says nothing about the script and everything about how it was loaded.
    """
    spec = importlib.util.spec_from_file_location("rederive_headline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rederive_module():
    return _load_script()


# --------------------------------------------------------------------------------------
# Independence
# --------------------------------------------------------------------------------------


def test_the_rederivation_imports_nothing_from_src():
    """The independence claim, asserted rather than described.

    `scripts/radius_sensitivity.py` legitimately imports `src` and inserts the repo root
    on `sys.path` to do it. This script must not, and the distinction is not stylistic:
    the sensitivity sweep re-runs the pipeline on purpose, while this one exists precisely
    because re-running the pipeline cannot catch a bug inside it.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    from_src = [name for name in imported if name == "src" or name.startswith("src.")]
    assert not from_src, (
        f"scripts/rederive_headline.py imports {from_src} from the code it is supposed to "
        f"be checking. A re-derivation sharing an implementation with its subject "
        f"reproduces that implementation's bugs and reports a match."
    )

    # A sys.path insert is how the import would come back without looking like one.
    source = SCRIPT.read_text(encoding="utf-8")
    assert "sys.path" not in source, (
        "scripts/rederive_headline.py manipulates sys.path, which is the setup step for "
        "importing src. Nothing in it should need the repo root importable."
    )


# --------------------------------------------------------------------------------------
# A small scored frame whose answers are known
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    """A miniature scored universe covering every branch the re-derivation walks.

    Built to contain, deliberately: both unit types with different dispersions, two
    boroughs plus one unit with no borough at all, units treated before and during the
    holdout alongside untreated ones, an unfitted unit with no SPF prediction, and a block
    of exactly-tied scores at the selection boundary. Every one of those is a path where
    the two implementations could disagree while both looked reasonable.
    """
    rng = np.random.default_rng(20260814)
    n = 60
    unit_type = np.where(np.arange(n) % 2 == 0, "corridor", "intersection")
    unit_id = [f"{'C' if t == 'corridor' else 'I'}{i}" for i, t in enumerate(unit_type)]

    borough = np.where(np.arange(n) % 2 == 0, "MANHATTAN", "BROOKLYN").astype(object)
    borough[7] = None  # a bridge or shoreline segment: real, and excluded from Regime A

    casualties = rng.poisson(1.5, n)
    holdout = rng.poisson(1.2, n) + 1  # every unit carries at least one, so no group is empty
    prediction = rng.uniform(0.01, 3.0, n)
    prediction[3] = np.nan  # degenerate exposure: never fitted, keeps its observed count

    frame = pd.DataFrame(
        {
            "unit_id": unit_id,
            "unit_type": unit_type,
            "borough": borough,
            "casualties_36mo": casualties.astype(int),
            "holdout_casualties": holdout.astype(int),
            "spf_prediction": prediction,
            "is_priority": np.arange(n) % 3 == 0,
            "treated": np.arange(n) % 4 != 0,
            "treatment_date": pd.to_datetime(
                [
                    None
                    if i % 4 == 0
                    else ("2022-05-01" if i % 4 == 1 else "2024-06-01")
                    for i in range(n)
                ]
            ),
        }
    )

    eb = pd.Series(np.nan, index=frame.index, dtype=float)
    for unit_type_name, k in DISPERSION.items():
        rows = (frame["unit_type"] == unit_type_name) & frame["spf_prediction"].notna()
        eb.loc[rows] = empirical_bayes(
            frame.loc[rows, "spf_prediction"], frame.loc[rows, "casualties_36mo"], k
        )
    eb = eb.fillna(frame["casualties_36mo"].astype(float))
    frame["eb_estimate"] = eb
    return frame


def _publish(scored: pd.DataFrame, tmp_path: Path) -> RunSummary:
    """Score the frame with `src/backtest.py` and write the artifacts, as `run()` does.

    A miniature of `src.pipeline.run`'s scoring half rather than a call into it, because
    `run()` starts from a snapshot on disk. Everything below this line is the code the
    re-derivation is checking; everything the re-derivation does is the other
    implementation. If they agree here they are two roads to the same number.
    """
    priority = scored[scored["is_priority"]]
    n = int(len(priority))

    r1 = Selection("R1", "citywide", priority["unit_id"].tolist())
    r2 = select_citywide_top_n(scored, "casualties_36mo", n, "R2")
    r3 = select_citywide_top_n(scored, "eb_estimate", n, "R3")
    r2s = select_borough_stratified(scored, "casualties_36mo", "casualties_36mo", "R2s")
    r3s = select_borough_stratified(scored, "eb_estimate", "casualties_36mo", "R3s")

    ci = bootstrap_capture_difference(scored, r3, r2)
    split = split_by_treatment(scored, r1, holdout_start=HOLDOUT_START)
    when = pd.to_datetime(scored["treatment_date"], errors="coerce")

    summary = RunSummary(
        snapshot_date="2026-08-13",
        crashes_pulled=0,
        crashes_assigned=0,
        crashes_dropped=0,
        universe_units=int(len(scored)),
        corridors=int((scored["unit_type"] == "corridor").sum()),
        intersections=int((scored["unit_type"] == "intersection").sum()),
        priority_units=n,
        treated_units=int(scored["treated"].sum()),
        train_window="2019-2023",
        holdout_window="2024-2025",
        holdout_casualties=int(scored["holdout_casualties"].sum()),
        dispersion=dict(DISPERSION),
        citywide_n=n,
        r1_citywide_pp=capture_rate(r1, scored).rate_pp,
        r2_citywide_pp=capture_rate(r2, scored).rate_pp,
        r3_citywide_pp=capture_rate(r3, scored).rate_pp,
        r2_stratified_pp=capture_rate(r2s, scored).rate_pp,
        r3_stratified_pp=capture_rate(r3s, scored).rate_pp,
        lift_pp=capture_rate(r3, scored).rate_pp - capture_rate(r2, scored).rate_pp,
        ci_low_pp=ci.lower_pp,
        ci_high_pp=ci.upper_pp,
        ci_excludes_zero=ci.excludes_zero,
        clears_bar=True,
        verdict="fixture",
        treated_before_holdout_pp=split["treated_before_holdout"].rate_pp,
        treated_during_holdout_pp=split["treated_during_holdout"].rate_pp,
        untreated_pp=split["untreated"].rate_pp,
        treated_before_holdout_units=int((when.notna() & (when < HOLDOUT_START)).sum()),
        treated_during_holdout_units=int((when.notna() & (when >= HOLDOUT_START)).sum()),
        join_radii={
            "max_join_distance_ft": 150.0,
            "intersection_radius_ft": 100.0,
            "vzv_buffer_ft": 50.0,
        },
    )
    _write_artifacts(summary, scored, processed_dir=tmp_path)
    return summary


@pytest.fixture
def published(scored: pd.DataFrame, tmp_path: Path) -> Path:
    _publish(scored, tmp_path)
    return tmp_path


# --------------------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------------------


def test_every_published_number_rederives(rederive_module, published):
    frame, _, summary = rederive_module.load_artifacts(published)
    report = rederive_module.rederive(frame, summary)

    assert report.checks, "the re-derivation checked nothing"
    assert not report.failures, "\n".join(
        f"{c.label}: published {c.published!r}, re-derived {c.rederived!r}"
        for c in report.failures
    )


def test_it_checks_every_rate_the_summary_publishes(rederive_module, published):
    """A green re-derivation that quietly skipped the stratified regime is worse than red.

    The failure mode this guards is a check that stops covering something without
    anything going red: drop the borough-stratified rows and every remaining row still
    agrees, so the report reads exactly as it does now while checking less.
    """
    frame, _, summary = rederive_module.load_artifacts(published)
    labels = " ".join(c.label for c in rederive_module.rederive(frame, summary).checks)

    for expected in ("R1", "R2", "R3", "borough-stratified", "CI low", "CI high", "EB blend"):
        assert expected in labels, f"the re-derivation never checks {expected!r}"


def test_the_eb_blend_is_rebuilt_not_copied(rederive_module, scored):
    """`rebuild_eb` has to reproduce `src.spf.empirical_bayes`, including its floor."""
    rebuilt = rederive_module.rebuild_eb(scored, DISPERSION)
    np.testing.assert_allclose(rebuilt, scored["eb_estimate"].to_numpy(), rtol=0, atol=1e-12)

    # The unfitted unit keeps its observed count rather than picking up a prediction.
    unfitted = scored["spf_prediction"].isna()
    assert unfitted.any(), "the fixture lost its unfitted unit"
    np.testing.assert_array_equal(
        rebuilt[unfitted.to_numpy()],
        scored.loc[unfitted, "casualties_36mo"].to_numpy(dtype=float),
    )


# --------------------------------------------------------------------------------------
# It has to be able to fail
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "r1_citywide_pp",
        "r2_citywide_pp",
        "r3_citywide_pp",
        "lift_pp",
        "r2_stratified_pp",
        "untreated_pp",
        "holdout_casualties",
    ],
)
def test_a_wrong_published_number_is_caught(rederive_module, published, key):
    """Every quantity the summary publishes has to be one the check can fail on.

    Parametrised rather than spot-checked because "is this number actually compared, or
    only printed" is the question that decides whether the guard protects anything, and it
    has to be asked of each one separately.
    """
    frame, _, summary = rederive_module.load_artifacts(published)
    summary[key] = float(summary[key]) + 1.0

    failures = rederive_module.rederive(frame, summary).failures
    assert failures, f"corrupting {key} did not fail the re-derivation"


def test_a_frame_from_a_different_run_is_refused(rederive_module, published):
    """The provenance cross-check, which is what makes the comparison meaningful at all.

    Compare a 150 ft frame against a 100 ft summary and the mismatches look exactly like
    a scoring bug. Refusing outright is the difference between "these numbers disagree"
    and "these numbers are not about the same thing."
    """
    import json

    summary_path = published / "run-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["join_radii"]["intersection_radius_ft"] = 150.0
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(rederive_module.RederivationError, match="two different runs"):
        rederive_module.load_artifacts(published)


def test_a_duplicated_unit_is_refused(rederive_module, scored, tmp_path):
    """Duplicates inflate both sides of every rate while the percentages stay plausible."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _publish(scored, tmp_path)
    original = pq.read_table(tmp_path / "scored-units.parquet")

    doubled = pd.concat([scored, scored.head(1)], ignore_index=True)
    table = pa.Table.from_pandas(
        doubled[[f.name for f in original.schema]], preserve_index=False
    ).replace_schema_metadata(original.schema.metadata)
    pq.write_table(table, tmp_path / "scored-units.parquet")

    with pytest.raises(rederive_module.RederivationError, match="duplicated"):
        rederive_module.load_artifacts(tmp_path)


def test_a_missing_artifact_says_how_to_produce_it(rederive_module, tmp_path):
    with pytest.raises(rederive_module.RederivationError, match="src.pipeline"):
        rederive_module.load_artifacts(tmp_path)


# --------------------------------------------------------------------------------------
# The tie-break disclosure
# --------------------------------------------------------------------------------------


def test_a_fully_tied_ranking_reports_every_pick_as_drawn(rederive_module):
    """When every score is equal, the ranking decided nothing and the report must say so.

    This is not hypothetical at full scale: 206,321 of New York's 220,033 units are tied
    at zero trailing casualties, so the raw-count ranking reaches its 38,909-unit budget
    inside the tie block and draws most of its list by hash.
    """
    scores = np.zeros(50)
    tiebreak = rederive_module.tiebreak_key(np.array([f"U{i}" for i in range(50)]))
    order = rederive_module.rank_order(scores, tiebreak)

    block, drawn = rederive_module.tie_block_at_cut(scores, order, 10)
    assert block == 50
    assert drawn == 10

    # Distinct scores: the ranking decided everything and nothing was drawn.
    distinct = np.arange(50, dtype=float)
    order = rederive_module.rank_order(distinct, tiebreak)
    block, drawn = rederive_module.tie_block_at_cut(distinct, order, 10)
    assert (block, drawn) == (1, 1)


def test_the_tiebreak_matches_the_pipelines(rederive_module):
    """Two implementations of the same blake2b rule, on ids that exercise the top bit."""
    from src.backtest import _tiebreak_key

    ids = pd.Series([f"C{i}" for i in range(200)] + [f"I{i}" for i in range(200)])
    theirs = _tiebreak_key(ids).to_numpy()
    ours = rederive_module.tiebreak_key(ids.to_numpy())

    assert [int(x) for x in theirs] == [int(x) for x in ours]


def test_nan_scores_are_refused_rather_than_sorted(rederive_module):
    scores = np.array([1.0, np.nan, 2.0])
    tiebreak = rederive_module.tiebreak_key(np.array(["A", "B", "C"]))
    with pytest.raises(rederive_module.RederivationError, match="NaN"):
        rederive_module.rank_order(scores, tiebreak)


def test_an_empty_holdout_is_undefined_not_zero(rederive_module):
    with pytest.raises(rederive_module.RederivationError, match="undefined, not 0%"):
        rederive_module.capture_pp(np.zeros(10), np.ones(10, dtype=bool))
