"""End-to-end: fixture city through every stage, asserting the result is stable.

This is the test that turns "reproducible" from a claim into a fact. It runs the real
pipeline — universe, assignment, features, negative-binomial SPF, Empirical Bayes,
both selection regimes, capture rate, bootstrap, verdict — over a synthetic city small
enough to reason about and large enough to fit.

What it asserts is *determinism and invariants*, not a hardcoded magic number: two runs
must agree exactly, casualties must be conserved, and no NaN may reach the verdict. A
golden-number assertion belongs here too, but only once pinned from a green run on a
pinned dependency set. Writing one from a guess would defeat the purpose.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.backtest import (
    apply_preregistered_bar,
    bootstrap_capture_difference,
    capture_rate,
    select_borough_stratified,
    select_citywide_top_n,
    split_by_treatment,
)
from src.config import CRS_GEOGRAPHIC
from src.features import add_pedestrian_casualties, build_features
from src.spatial import assign_crashes_to_units, build_universe, crashes_to_gdf
from src.spf import fit_and_blend

GRID = 12
LON0, LAT0 = -73.990, 40.740
STEP = 0.0015

TRAIN_END = pd.Timestamp("2024-01-01")
HOLDOUT_END = pd.Timestamp("2026-01-01")


def build_grid() -> gpd.GeoDataFrame:
    """A GRID x GRID lattice of streets across two boroughs."""
    lons = [LON0 + i * STEP for i in range(GRID)]
    lats = [LAT0 + j * STEP for j in range(GRID)]
    rows = []

    for j, lat in enumerate(lats):
        borough = "MANHATTAN" if j < GRID // 2 else "BROOKLYN"
        for i in range(GRID - 1):
            rows.append(
                {
                    "street": f"EW{j}",
                    "borough": borough,
                    "geometry": LineString([(lons[i], lat), (lons[i + 1], lat)]),
                }
            )
    for i, lon in enumerate(lons):
        for j in range(GRID - 1):
            borough = "MANHATTAN" if j < GRID // 2 else "BROOKLYN"
            rows.append(
                {
                    "street": f"NS{i}",
                    "borough": borough,
                    "geometry": LineString([(lon, lats[j]), (lon, lats[j + 1])]),
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_GEOGRAPHIC)


def build_crashes(seed: int = 20260812) -> pd.DataFrame:
    """Crashes concentrated on a few corridors, spread over train and holdout years.

    Concentration matters: a uniform sprinkle gives every unit the same risk and the
    ranking comparison becomes vacuous.
    """
    rng = np.random.default_rng(seed)
    lons = [LON0 + i * STEP for i in range(GRID)]
    lats = [LAT0 + j * STEP for j in range(GRID)]

    hot_rows = {2, 3, 8}
    records = []

    for j, lat in enumerate(lats):
        intensity = 14 if j in hot_rows else 3
        for i in range(GRID - 1):
            for _ in range(rng.poisson(intensity)):
                offset = rng.uniform(0.0002, STEP - 0.0002)
                year = int(rng.integers(2019, 2026))
                month = int(rng.integers(1, 13))
                day = int(rng.integers(1, 28))
                hour = int(rng.integers(0, 24))
                records.append(
                    {
                        "crash_date": f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00.000",
                        "latitude": lat + rng.normal(0, 0.00002),
                        "longitude": lons[i] + offset,
                        "number_of_pedestrians_killed": int(rng.random() < 0.02),
                        "number_of_pedestrians_injured": int(rng.integers(0, 3)),
                        "contributing_factor_vehicle_1": rng.choice(
                            ["Unspecified", "Driver Inattention/Distraction", "Unsafe Speed"]
                        ),
                        "contributing_factor_vehicle_2": None,
                    }
                )

    return pd.DataFrame(records)


def run_pipeline(seed: int = 20260812) -> dict:
    """Every stage, in order. Returns the numbers a README would quote."""
    universe = build_universe(build_grid())

    crashes = add_pedestrian_casualties(build_crashes(seed))
    points, report = crashes_to_gdf(crashes)
    assigned, report = assign_crashes_to_units(points, universe, report)
    report.validate()

    assigned["crash_date"] = pd.to_datetime(assigned["crash_date"])
    train = assigned[assigned["crash_date"] < TRAIN_END]
    holdout_crashes = assigned[
        (assigned["crash_date"] >= TRAIN_END) & (assigned["crash_date"] < HOLDOUT_END)
    ]

    features, _ = build_features(train, universe, TRAIN_END, months=(36,))

    scored, spf_results = fit_and_blend(
        features,
        target="casualties_36mo",
        predictors=["night_share"],
        observed_col="casualties_36mo",
    )

    holdout_totals = holdout_crashes.groupby("unit_id")["pedestrian_casualties"].sum()
    scored["holdout_casualties"] = scored["unit_id"].map(holdout_totals).fillna(0).astype(int)
    scored["treated"] = scored.index % 3 == 0  # deterministic stand-in for the SIP join

    naive = select_citywide_top_n(scored, "casualties_36mo", 40, "R2 raw count")
    eb = select_citywide_top_n(scored, "eb_estimate", 40, "R3 empirical bayes")

    naive_rate = capture_rate(naive, scored)
    eb_rate = capture_rate(eb, scored)
    ci = bootstrap_capture_difference(scored, eb, naive, iterations=500)
    verdict = apply_preregistered_bar(eb_rate, naive_rate, ci)

    stratified = select_borough_stratified(
        scored, "eb_estimate", "casualties_36mo", "R3 borough-stratified"
    )

    return {
        "report": report,
        "universe_size": len(universe),
        "spf": spf_results,
        "scored": scored,
        "naive_rate": naive_rate,
        "eb_rate": eb_rate,
        "ci": ci,
        "verdict": verdict,
        "stratified": stratified,
        "treatment_split": split_by_treatment(scored, eb),
    }


@pytest.fixture(scope="module")
def pipeline() -> dict:
    return run_pipeline()


class TestPipelineRuns:
    def test_completes_every_stage(self, pipeline):
        assert pipeline["universe_size"] > 0
        assert pipeline["spf"]

    def test_crash_accounting_balances(self, pipeline):
        pipeline["report"].validate()

    def test_spf_converged_with_positive_dispersion(self, pipeline):
        for result in pipeline["spf"].values():
            assert result.converged
            assert result.dispersion > 0


class TestNoSilentFailureReachesTheResult:
    def test_capture_rates_are_defined(self, pipeline):
        assert pipeline["eb_rate"].defined
        assert pipeline["naive_rate"].defined

    def test_no_nan_in_the_scored_table(self, pipeline):
        assert pipeline["scored"]["eb_estimate"].notna().all()
        assert pipeline["scored"]["holdout_casualties"].notna().all()

    def test_no_infinity_in_the_scored_table(self, pipeline):
        numeric = pipeline["scored"].select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()

    def test_verdict_lift_is_finite(self, pipeline):
        assert np.isfinite(pipeline["verdict"].lift_pp)


class TestInvariants:
    def test_capture_rates_are_proportions(self, pipeline):
        for rate in (pipeline["eb_rate"], pipeline["naive_rate"]):
            assert 0.0 <= rate.rate <= 1.0

    def test_denominators_match_total_holdout_casualties(self, pipeline):
        total = pipeline["scored"]["holdout_casualties"].sum()
        assert pipeline["eb_rate"].total == total
        assert pipeline["naive_rate"].total == total

    def test_both_rankings_score_against_the_same_denominator(self, pipeline):
        """Otherwise the comparison is between two different questions."""
        assert pipeline["eb_rate"].total == pipeline["naive_rate"].total

    def test_treatment_split_partitions_the_holdout(self, pipeline):
        split = pipeline["treatment_split"]
        total = pipeline["scored"]["holdout_casualties"].sum()
        assert split["treated"].total + split["untreated"].total == total

    def test_borough_stratified_selects_in_both_boroughs(self, pipeline):
        keys = pipeline["stratified"].per_borough
        assert any("MANHATTAN" in k for k in keys)
        assert any("BROOKLYN" in k for k in keys)

    def test_stratified_selects_fewer_units_than_the_universe(self, pipeline):
        assert 0 < pipeline["stratified"].n < pipeline["universe_size"]

    def test_ranking_beats_selecting_at_random(self, pipeline):
        """A sanity floor. If a risk ranking cannot beat its own selection share, the
        pipeline is wired wrong somewhere upstream."""
        share_of_units = pipeline["eb_rate"].n_selected / pipeline["universe_size"]
        assert pipeline["eb_rate"].rate > share_of_units


class TestReproducibility:
    def test_two_runs_agree_exactly(self):
        """The reproduction path the README promises a reader."""
        first = run_pipeline()
        second = run_pipeline()
        assert first["eb_rate"].captured == second["eb_rate"].captured
        assert first["eb_rate"].total == second["eb_rate"].total
        assert first["verdict"].lift_pp == second["verdict"].lift_pp

    def test_bootstrap_bounds_are_reproducible(self):
        first = run_pipeline()
        second = run_pipeline()
        assert first["ci"].lower_pp == second["ci"].lower_pp
        assert first["ci"].upper_pp == second["ci"].upper_pp

    def test_selection_is_order_independent(self, pipeline):
        scored = pipeline["scored"]
        shuffled = scored.sample(frac=1, random_state=3).reset_index(drop=True)
        a = select_citywide_top_n(scored, "eb_estimate", 40, "R3")
        b = select_citywide_top_n(shuffled, "eb_estimate", 40, "R3")
        assert a.unit_ids == b.unit_ids


class TestVerdictIsHonest:
    def test_verdict_states_which_condition_decided_it(self, pipeline):
        assert pipeline["verdict"].reason

    def test_verdict_respects_the_preregistered_threshold(self, pipeline):
        """The bar is not adjustable after the fact."""
        from src.config import MIN_CAPTURE_RATE_LIFT_PP

        verdict = pipeline["verdict"]
        assert verdict.threshold_pp == MIN_CAPTURE_RATE_LIFT_PP
        if verdict.clears_bar:
            assert verdict.lift_pp >= MIN_CAPTURE_RATE_LIFT_PP
            assert verdict.ci.excludes_zero
