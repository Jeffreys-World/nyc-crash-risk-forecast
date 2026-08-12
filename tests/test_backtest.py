"""Selection regimes, capture rate, and CRITICAL gap 3: the zero denominator.

A borough with no holdout casualties gives 0/0. Left alone that is a NaN, and a NaN
reaches the README looking exactly like a measurement. `CaptureRate.rate` returns None
when undefined, never NaN, so the undefined case has to be handled explicitly by whoever
reads it.

Also covered: selection-rule fidelity. Regime A must reproduce DOT's published rule
exactly (corridors to 50% of borough casualties, intersections to 15%), because the
whole comparison rests on the model being judged under DOT's own terms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BacktestError,
    Selection,
    apply_preregistered_bar,
    bootstrap_capture_difference,
    capture_rate,
    select_borough_stratified,
    select_citywide_top_n,
    split_by_treatment,
)


@pytest.fixture
def holdout() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [f"C{i}" for i in range(10)],
            "unit_type": "corridor",
            "borough": ["MANHATTAN"] * 5 + ["BROOKLYN"] * 5,
            "holdout_casualties": [10, 8, 6, 4, 2, 9, 7, 5, 3, 1],
            "score": [10, 8, 6, 4, 2, 9, 7, 5, 3, 1],
            "treated": [True, False] * 5,
        }
    )


class TestCitywideSelection:
    def test_selects_exactly_n(self, holdout):
        assert select_citywide_top_n(holdout, "score", 3, "R3").n == 3

    def test_selects_the_highest_scores(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 3, "R3")
        assert set(selection.unit_ids) == {"C0", "C5", "C1"}

    def test_rejects_a_non_positive_n(self, holdout):
        with pytest.raises(BacktestError, match="must be positive"):
            select_citywide_top_n(holdout, "score", 0, "R3")

    def test_caps_at_the_universe_and_says_so(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 99, "R3")
        assert selection.n == 10
        assert selection.notes

    def test_missing_score_column_raises(self, holdout):
        with pytest.raises(BacktestError, match="not present"):
            select_citywide_top_n(holdout, "nope", 3, "R3")


class TestTieBreaking:
    """Ties at the boundary decide the headline number. They cannot be arbitrary."""

    @pytest.fixture
    def all_tied(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unit_id": ["C3", "C1", "C2", "C0"],
                "unit_type": "corridor",
                "borough": "MANHATTAN",
                "score": [5, 5, 5, 5],
                "holdout_casualties": [1, 1, 1, 1],
            }
        )

    def test_ties_are_deterministic_across_calls(self, all_tied):
        first = select_citywide_top_n(all_tied, "score", 2, "R3").unit_ids
        second = select_citywide_top_n(all_tied, "score", 2, "R3").unit_ids
        assert first == second

    def test_result_is_stable_across_row_ordering(self, all_tied):
        first = select_citywide_top_n(all_tied, "score", 2, "R3").unit_ids
        shuffled = all_tied.sample(frac=1, random_state=7).reset_index(drop=True)
        assert select_citywide_top_n(shuffled, "score", 2, "R3").unit_ids == first

    def test_tiebreak_does_not_favour_one_unit_type(self):
        """Regression: the tie-break was silently rigging the naive baseline.

        Found on the 2026-08-12 run. Unit ids are `C…` for corridors and `I…` for
        intersections, and ties sorted on unit_id ascending, so `"C" < "I"` put every
        corridor ahead of every intersection. Most units have a trailing count of zero,
        so the baseline spent its entire quota on corridors, which hold 14% of
        casualties against the intersections' 86%. The measured lift was inflated by
        alphabetical order.

        A tie-break must be arbitrary but uncorrelated with anything predicting the
        outcome. With 500 tied units of each type and 500 picks, a correlated key gives
        500/0; an uncorrelated one lands near 250/250.
        """
        n = 500
        tied = pd.DataFrame(
            {
                "unit_id": [f"C{i}" for i in range(n)] + [f"I{i}" for i in range(n)],
                "unit_type": ["corridor"] * n + ["intersection"] * n,
                "borough": "MANHATTAN",
                "score": 0,
                "holdout_casualties": 0,
            }
        )
        picked = select_citywide_top_n(tied, "score", n, "R2")
        chosen = tied[tied["unit_id"].isin(picked.unit_ids)]
        corridors = int((chosen["unit_type"] == "corridor").sum())

        assert 0.35 * n < corridors < 0.65 * n, (
            f"tie-break favoured one unit type: {corridors}/{n} corridors"
        )

    def test_tiebreak_is_stable_across_processes(self):
        """Python's built-in hash() is salted per process; this key must not be."""
        from src.backtest import _tiebreak_key

        assert _tiebreak_key(pd.Series(["C0"])).iloc[0] == 10475129511078190953


class TestBoroughStratifiedSelection:
    def test_reproduces_dots_fifty_percent_corridor_rule(self):
        """Selection-rule fidelity, against a hand-checkable case.

        One borough, casualties 10/8/6/4/2 summing to 30. Half is 15. Taking the top
        two reaches 18, which is the first cumulative total at or above 50%.
        """
        units = pd.DataFrame(
            {
                "unit_id": ["A", "B", "C", "D", "E"],
                "unit_type": "corridor",
                "borough": "MANHATTAN",
                "score": [10, 8, 6, 4, 2],
                "casualties": [10, 8, 6, 4, 2],
            }
        )
        selection = select_borough_stratified(units, "score", "casualties", "R3")
        assert selection.unit_ids == ["A", "B"]

    def test_applies_the_fifteen_percent_intersection_rule(self):
        units = pd.DataFrame(
            {
                "unit_id": ["A", "B", "C", "D", "E"],
                "unit_type": "intersection",
                "borough": "MANHATTAN",
                "score": [10, 8, 6, 4, 2],
                "casualties": [10, 8, 6, 4, 2],
            }
        )
        selection = select_borough_stratified(units, "score", "casualties", "R3")
        assert selection.unit_ids == ["A"]  # 10/30 = 33%, clears 15% at the first unit

    def test_selects_within_every_borough(self, holdout):
        selection = select_borough_stratified(
            holdout, "score", "holdout_casualties", "R3"
        )
        assert any("MANHATTAN" in k for k in selection.per_borough)
        assert any("BROOKLYN" in k for k in selection.per_borough)

    def test_borough_with_zero_casualties_selects_nothing_and_is_noted(self):
        """The zero-denominator case at selection time."""
        units = pd.DataFrame(
            {
                "unit_id": ["A", "B", "C"],
                "unit_type": "corridor",
                "borough": ["MANHATTAN", "MANHATTAN", "STATEN ISLAND"],
                "score": [5, 3, 1],
                "casualties": [5, 3, 0],
            }
        )
        selection = select_borough_stratified(units, "score", "casualties", "R3")
        assert "C" not in selection.unit_ids
        assert any("STATEN ISLAND" in n for n in selection.notes)

    def test_borough_smaller_than_the_rule_requires_selects_all(self):
        units = pd.DataFrame(
            {
                "unit_id": ["A"],
                "unit_type": "corridor",
                "borough": "STATEN ISLAND",
                "score": [3],
                "casualties": [3],
            }
        )
        selection = select_borough_stratified(units, "score", "casualties", "R3")
        assert selection.unit_ids == ["A"]

    def test_null_borough_does_not_become_a_phantom_borough(self):
        """Regression: ISSUE-002 — a null borough got its own 50% stopping rule.

        Found by /qa on 2026-08-12.
        Report: .gstack/qa-reports/qa-report-nyc-crash-risk-forecast-2026-08-12.md

        Unit A had the highest score and no borough. It was grouped under a null key,
        given its own borough budget, and selected — handing the model a pick DOT
        never had and inflating its capture rate.
        """
        units = pd.DataFrame(
            {
                "unit_id": ["A", "B", "C"],
                "unit_type": "corridor",
                "borough": [None, "MANHATTAN", "MANHATTAN"],
                "score": [9, 5, 3],
                "casualties": [9, 5, 3],
            }
        )
        selection = select_borough_stratified(units, "score", "casualties", "R3")

        assert "A" not in selection.unit_ids
        assert not any("nan" in k.lower() for k in selection.per_borough)
        assert any("no borough" in n for n in selection.notes)

    def test_all_null_boroughs_raises_rather_than_selecting_nothing(self):
        units = pd.DataFrame(
            {
                "unit_id": ["A", "B"],
                "unit_type": "corridor",
                "borough": [None, None],
                "score": [9, 5],
                "casualties": [9, 5],
            }
        )
        with pytest.raises(BacktestError, match="every unit lacks a borough"):
            select_borough_stratified(units, "score", "casualties", "R3")

    def test_requires_a_borough_column(self):
        units = pd.DataFrame(
            {"unit_id": ["A"], "unit_type": "corridor", "score": [1], "casualties": [1]}
        )
        with pytest.raises(BacktestError, match="borough"):
            select_borough_stratified(units, "score", "casualties", "R3")

    def test_raises_when_nothing_could_be_selected(self):
        units = pd.DataFrame(
            {
                "unit_id": ["A"],
                "unit_type": "corridor",
                "borough": "MANHATTAN",
                "score": [1],
                "casualties": [0],
            }
        )
        with pytest.raises(BacktestError, match="picked nothing"):
            select_borough_stratified(units, "score", "casualties", "R3")


class TestCaptureRate:
    def test_computes_the_share_of_holdout_casualties(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 2, "R3")  # C0=10, C5=9
        result = capture_rate(selection, holdout)
        assert result.total == 55
        assert result.captured == 19
        assert result.rate_pp == pytest.approx(100 * 19 / 55)

    def test_zero_denominator_is_undefined_not_zero(self, holdout):
        """CRITICAL gap 3, stated directly."""
        empty = holdout.copy()
        empty["holdout_casualties"] = 0
        result = capture_rate(select_citywide_top_n(empty, "score", 2, "R3"), empty)
        assert not result.defined
        assert result.rate is None

    def test_undefined_rate_is_never_nan(self, holdout):
        """None forces a decision. NaN silently formats into a README."""
        empty = holdout.copy()
        empty["holdout_casualties"] = 0
        result = capture_rate(select_citywide_top_n(empty, "score", 2, "R3"), empty)
        assert result.rate is not None or result.rate is None
        assert not (isinstance(result.rate, float) and np.isnan(result.rate))

    def test_undefined_summary_says_undefined(self, holdout):
        empty = holdout.copy()
        empty["holdout_casualties"] = 0
        result = capture_rate(select_citywide_top_n(empty, "score", 2, "R3"), empty)
        assert "UNDEFINED" in result.summary()

    def test_duplicate_unit_id_raises_rather_than_double_counting(self):
        """Regression: ISSUE-004 — a repeated unit_id inflated both sides of the rate.

        Found by /qa on 2026-08-12.
        Report: .gstack/qa-reports/qa-report-nyc-crash-risk-forecast-2026-08-12.md

        Unit A appearing twice made the denominator 20 instead of 15 and the numerator
        10 instead of 5. The result was a clean-looking 50% that was wrong twice over.
        """
        duplicated = pd.DataFrame(
            {"unit_id": ["A", "A", "B"], "holdout_casualties": [5, 5, 10]}
        )
        selection = Selection(name="R3", regime="citywide", unit_ids=["A"])
        with pytest.raises(BacktestError, match="duplicated unit_id"):
            capture_rate(selection, duplicated)

    def test_missing_casualty_column_raises(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 2, "R3")
        with pytest.raises(BacktestError, match="holdout_casualties"):
            capture_rate(selection, holdout.drop(columns=["holdout_casualties"]))

    def test_selecting_everything_captures_everything(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 10, "R3")
        assert capture_rate(selection, holdout).rate == pytest.approx(1.0)


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self, holdout):
        a = select_citywide_top_n(holdout, "score", 5, "R3")
        b = Selection(name="R2", regime="citywide", unit_ids=["C4", "C9"])
        ci = bootstrap_capture_difference(holdout, a, b, iterations=500)
        assert ci.lower_pp <= ci.point_estimate_pp <= ci.upper_pp

    def test_identical_selections_give_a_zero_difference(self, holdout):
        a = select_citywide_top_n(holdout, "score", 5, "R3")
        ci = bootstrap_capture_difference(holdout, a, a, iterations=500)
        assert ci.point_estimate_pp == pytest.approx(0.0)
        assert not ci.excludes_zero

    def test_a_large_real_difference_excludes_zero(self, holdout):
        strong = select_citywide_top_n(holdout, "score", 5, "R3")
        weak = Selection(name="R2", regime="citywide", unit_ids=["C9"])
        ci = bootstrap_capture_difference(holdout, strong, weak, iterations=2000)
        assert ci.excludes_zero

    def test_is_reproducible_for_a_fixed_seed(self, holdout):
        a = select_citywide_top_n(holdout, "score", 5, "R3")
        b = Selection(name="R2", regime="citywide", unit_ids=["C4"])
        first = bootstrap_capture_difference(holdout, a, b, iterations=500, seed=42)
        second = bootstrap_capture_difference(holdout, a, b, iterations=500, seed=42)
        assert first.lower_pp == second.lower_pp
        assert first.upper_pp == second.upper_pp

    def test_refuses_to_bootstrap_an_empty_holdout(self, holdout):
        empty = holdout.copy()
        empty["holdout_casualties"] = 0
        a = select_citywide_top_n(empty, "score", 2, "R3")
        with pytest.raises(BacktestError, match="zero casualties"):
            bootstrap_capture_difference(empty, a, a, iterations=100)

    def test_single_casualty_unit_is_refused_not_reported(self):
        """Too sparse to support an interval is a result, not a number to quote.

        With casualties in one unit, roughly exp(-1) of resamples draw none of it and
        have no denominator at all. An interval built on the survivors would look
        precise and mean nothing.
        """
        sparse = pd.DataFrame(
            {
                "unit_id": [f"U{i}" for i in range(20)],
                "holdout_casualties": [1] + [0] * 19,
                "score": [1] + [0] * 19,
            }
        )
        a = Selection(name="R3", regime="citywide", unit_ids=["U0"])
        b = Selection(name="R2", regime="citywide", unit_ids=["U1"])
        with pytest.raises(BacktestError, match="concentrated in too few units"):
            bootstrap_capture_difference(sparse, a, b, iterations=1000)

    def test_a_few_degenerate_resamples_are_noted_not_fatal(self):
        """Three casualty-bearing units out of five: ~1% degenerate, under the limit."""
        borderline = pd.DataFrame(
            {
                "unit_id": ["A", "B", "C", "D", "E"],
                "holdout_casualties": [3, 2, 1, 0, 0],
                "score": [3, 2, 1, 0, 0],
            }
        )
        a = Selection(name="R3", regime="citywide", unit_ids=["A"])
        b = Selection(name="R2", regime="citywide", unit_ids=["C"])
        ci = bootstrap_capture_difference(borderline, a, b, iterations=2000)

        assert np.isfinite(ci.lower_pp) and np.isfinite(ci.upper_pp)
        assert ci.lower_pp <= ci.point_estimate_pp <= ci.upper_pp

    def test_dense_holdout_has_no_degenerate_resamples(self):
        dense = pd.DataFrame(
            {
                "unit_id": [f"U{i}" for i in range(10)],
                "holdout_casualties": [5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
                "score": [5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
            }
        )
        a = Selection(name="R3", regime="citywide", unit_ids=["U0", "U1"])
        b = Selection(name="R2", regime="citywide", unit_ids=["U8", "U9"])
        ci = bootstrap_capture_difference(dense, a, b, iterations=2000)
        assert ci.note == ""

    def test_spread_out_casualties_produce_a_real_interval(self):
        spread = pd.DataFrame(
            {
                "unit_id": [f"U{i}" for i in range(10)],
                "holdout_casualties": [5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
                "score": [5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
            }
        )
        a = Selection(name="R3", regime="citywide", unit_ids=["U0", "U1"])
        b = Selection(name="R2", regime="citywide", unit_ids=["U8", "U9"])
        ci = bootstrap_capture_difference(spread, a, b, iterations=2000)
        assert ci.upper_pp - ci.lower_pp > 0

    def test_threshold_is_reachable_by_construction(self):
        """Guards that can never fire are decoration. This one has to be able to."""
        from src.backtest import MAX_DEGENERATE_RESAMPLE_FRACTION

        # A single casualty-bearing unit degenerates at about exp(-1) = 37%.
        assert MAX_DEGENERATE_RESAMPLE_FRACTION < np.exp(-1)


class TestPreregisteredBar:
    """Both conditions, together. Relaxing either after the fact voids the exercise."""

    def _rates(self, eb_pp: float, naive_pp: float):
        eb = capture_rate(
            Selection("R3", "citywide", ["A"]),
            pd.DataFrame({"unit_id": ["A", "B"], "holdout_casualties": [eb_pp, 100 - eb_pp]}),
        )
        naive = capture_rate(
            Selection("R2", "citywide", ["A"]),
            pd.DataFrame({"unit_id": ["A", "B"], "holdout_casualties": [naive_pp, 100 - naive_pp]}),
        )
        return eb, naive

    def _ci(self, excludes_zero: bool):
        from src.backtest import BootstrapCI

        return BootstrapCI(
            point_estimate_pp=6.0,
            lower_pp=2.0 if excludes_zero else -2.0,
            upper_pp=10.0,
            level=0.95,
            iterations=1000,
            excludes_zero=excludes_zero,
        )

    def test_clears_when_lift_and_ci_both_hold(self):
        eb, naive = self._rates(40, 30)
        verdict = apply_preregistered_bar(eb, naive, self._ci(True))
        assert verdict.clears_bar

    def test_fails_when_the_ci_includes_zero(self):
        eb, naive = self._rates(40, 30)
        verdict = apply_preregistered_bar(eb, naive, self._ci(False))
        assert not verdict.clears_bar
        assert "CI includes zero" in verdict.reason

    def test_fails_when_the_lift_is_real_but_too_small(self):
        eb, naive = self._rates(32, 30)
        verdict = apply_preregistered_bar(eb, naive, self._ci(True))
        assert not verdict.clears_bar
        assert "below the" in verdict.reason

    def test_fails_on_both_counts(self):
        eb, naive = self._rates(31, 30)
        verdict = apply_preregistered_bar(eb, naive, self._ci(False))
        assert not verdict.clears_bar

    def test_undefined_rate_cannot_clear_the_bar(self):
        undefined = capture_rate(
            Selection("R3", "citywide", ["A"]),
            pd.DataFrame({"unit_id": ["A"], "holdout_casualties": [0]}),
        )
        _, naive = self._rates(40, 30)
        verdict = apply_preregistered_bar(undefined, naive, self._ci(True))
        assert not verdict.clears_bar
        assert "undefined" in verdict.reason


class TestTreatmentSplit:
    def test_reports_treated_and_untreated_separately(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 5, "R1")
        split = split_by_treatment(holdout, selection)
        assert set(split) == {"treated", "untreated"}

    def test_denominators_partition_the_holdout(self, holdout):
        """The split must account for every casualty, not overlap or lose any."""
        selection = select_citywide_top_n(holdout, "score", 5, "R1")
        split = split_by_treatment(holdout, selection)
        assert split["treated"].total + split["untreated"].total == 55

    def test_missing_treated_column_raises(self, holdout):
        selection = select_citywide_top_n(holdout, "score", 5, "R1")
        with pytest.raises(BacktestError, match="confounded"):
            split_by_treatment(holdout.drop(columns=["treated"]), selection)

    def test_absent_group_is_undefined_not_zero(self, holdout):
        none_treated = holdout.copy()
        none_treated["treated"] = False
        selection = select_citywide_top_n(none_treated, "score", 5, "R1")
        split = split_by_treatment(none_treated, selection)
        assert not split["treated"].defined
