"""Features, the label, and CRITICAL gap 2: the log(0) exposure offset.

A zero-length segment makes `log(length)` negative infinity. statsmodels accepts it,
then either diverges after many iterations or returns parameters that look like numbers
and mean nothing. The guard has to sit in feature construction, before the fit sees it.

Also covered: zero must mean zero, not NaN. A unit with no crashes in a trailing window
is *quiet*, and quiet is information. NaN would drop it from the fit entirely and change
which streets the SPF was even trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureError,
    add_pedestrian_casualties,
    build_features,
    exposure_term,
    factor_mix,
    temporal_concentration,
    trailing_casualties,
)


@pytest.fixture
def units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["C1", "C2", "I1"],
            "unit_type": ["corridor", "corridor", "intersection"],
            "length_ft": [500.0, 0.0, np.nan],
            "leg_count": [np.nan, np.nan, 4],
        }
    )


@pytest.fixture
def assigned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["C1", "C1", "I1"],
            "crash_date": pd.to_datetime(
                ["2023-01-15 20:00", "2023-06-10 03:00", "2023-07-04 14:00"]
            ),
            "pedestrian_casualties": [2, 1, 3],
            "contributing_factor_vehicle_1": [
                "Driver Inattention/Distraction",
                "Unspecified",
                "Unsafe Speed",
            ],
            "contributing_factor_vehicle_2": [None, None, None],
        }
    )


class TestLabel:
    def test_casualties_are_killed_plus_injured(self):
        crashes = pd.DataFrame(
            {
                "crash_date": ["2023-01-01"],
                "number_of_pedestrians_killed": [1],
                "number_of_pedestrians_injured": [2],
            }
        )
        assert add_pedestrian_casualties(crashes)["pedestrian_casualties"].iloc[0] == 3

    def test_missing_counts_become_zero_not_nan(self):
        crashes = pd.DataFrame(
            {
                "crash_date": ["2023-01-01"],
                "number_of_pedestrians_killed": [None],
                "number_of_pedestrians_injured": ["2"],
            }
        )
        assert add_pedestrian_casualties(crashes)["pedestrian_casualties"].iloc[0] == 2

    def test_raises_when_a_label_column_is_absent(self):
        """Better to stop than to model a label that silently became all zeros."""
        crashes = pd.DataFrame({"crash_date": ["2023-01-01"]})
        with pytest.raises(FeatureError, match="missing"):
            add_pedestrian_casualties(crashes)

    def test_undated_crashes_are_excluded(self):
        crashes = pd.DataFrame(
            {
                "crash_date": ["2023-01-01", "not-a-date"],
                "number_of_pedestrians_killed": [0, 0],
                "number_of_pedestrians_injured": [1, 5],
            }
        )
        assert len(add_pedestrian_casualties(crashes)) == 1


class TestExposureGuard:
    def test_zero_length_never_produces_negative_infinity(self, units):
        """CRITICAL gap 2, stated directly."""
        out, _ = exposure_term(units)
        assert not np.isinf(out["log_exposure"]).any()

    def test_zero_length_is_marked_invalid(self, units):
        out, _ = exposure_term(units)
        assert not out.loc[out["unit_id"] == "C2", "exposure_valid"].iloc[0]

    def test_invalid_units_are_kept_in_the_table(self, units):
        """They stay in the universe so the capture-rate denominator is the real total."""
        out, _ = exposure_term(units)
        assert len(out) == 3

    def test_intersections_use_leg_count(self, units):
        out, _ = exposure_term(units)
        assert out.loc[out["unit_id"] == "I1", "exposure"].iloc[0] == 4

    def test_valid_corridor_gets_log_of_its_length(self, units):
        out, _ = exposure_term(units)
        assert out.loc[out["unit_id"] == "C1", "log_exposure"].iloc[0] == pytest.approx(
            np.log(500.0)
        )

    def test_exclusions_are_reported(self, units):
        _, report = exposure_term(units)
        assert report.excluded_degenerate_exposure == 1
        assert report.units_in == 3

    def test_missing_exposure_is_recorded_as_a_fallback(self):
        """A unit modelled on an invented exposure cannot be defended."""
        units = pd.DataFrame(
            {"unit_id": ["C1"], "unit_type": ["corridor"], "length_ft": [np.nan]}
        )
        out, report = exposure_term(units)
        assert report.exposure_fallbacks == 1
        assert not out["exposure_valid"].iloc[0]
        assert report.notes


class TestTrailingCounts:
    def test_quiet_unit_returns_zero_not_nan(self, assigned, units):
        """Quiet is information. NaN would drop the row from the fit."""
        out, _ = trailing_casualties(assigned, units, pd.Timestamp("2024-01-01"), (12,))
        c2 = out.loc[out["unit_id"] == "C2", "casualties_12mo"].iloc[0]
        assert c2 == 0
        assert not pd.isna(c2)

    def test_counts_are_integers(self, assigned, units):
        out, _ = trailing_casualties(assigned, units, pd.Timestamp("2024-01-01"), (12,))
        assert out["casualties_12mo"].dtype.kind == "i"

    def test_sums_casualties_within_the_window(self, assigned, units):
        out, _ = trailing_casualties(assigned, units, pd.Timestamp("2024-01-01"), (12,))
        assert out.loc[out["unit_id"] == "C1", "casualties_12mo"].iloc[0] == 3

    def test_window_excludes_crashes_after_as_of(self, assigned, units):
        """The whole point of `as_of`: no holdout crash may reach a training feature."""
        out, _ = trailing_casualties(assigned, units, pd.Timestamp("2023-03-01"), (12,))
        assert out.loc[out["unit_id"] == "C1", "casualties_12mo"].iloc[0] == 2

    def test_underfilled_window_is_flagged(self, assigned, units):
        """An under-filled window mimics a genuinely quiet street."""
        _, report = trailing_casualties(assigned, units, pd.Timestamp("2024-01-01"), (120,))
        assert report.window_truncated
        assert report.notes

    def test_fully_covered_window_is_not_flagged(self, assigned, units):
        _, report = trailing_casualties(assigned, units, pd.Timestamp("2023-12-01"), (6,))
        assert not report.window_truncated

    def test_empty_assignment_yields_all_zeros(self, units):
        empty = pd.DataFrame(columns=["unit_id", "crash_date", "pedestrian_casualties"])
        out, _ = trailing_casualties(empty, units, pd.Timestamp("2024-01-01"), (12,))
        assert (out["casualties_12mo"] == 0).all()


class TestMixFeatures:
    def test_factor_shares_are_proportions(self, assigned, units):
        out = factor_mix(assigned, units, pd.Timestamp("2024-01-01"))
        cols = [c for c in out.columns if c.startswith("factor_")]
        assert cols
        assert ((out[cols] >= 0) & (out[cols] <= 1)).all().all()

    def test_unspecified_is_tracked_not_discarded(self, assigned, units):
        """The most common factor in this dataset. Sparse, but real."""
        out = factor_mix(assigned, units, pd.Timestamp("2024-01-01"))
        assert "factor_unspecified" in out.columns
        assert out.loc[out["unit_id"] == "C1", "factor_unspecified"].iloc[0] > 0

    def test_denominator_is_mentions_not_crashes(self):
        """ISSUE-005: pins the documented semantics so they cannot drift.

        Found by /qa on 2026-08-12. One crash naming two factors gives each 0.5, not
        1.0, because the denominator counts factor mentions.
        """
        units = pd.DataFrame({"unit_id": ["A"], "unit_type": ["corridor"]})
        one_crash_two_factors = pd.DataFrame(
            {
                "unit_id": ["A"],
                "crash_date": pd.to_datetime(["2023-06-01"]),
                "contributing_factor_vehicle_1": ["Unsafe Speed"],
                "contributing_factor_vehicle_2": ["Unspecified"],
            }
        )
        out = factor_mix(one_crash_two_factors, units, pd.Timestamp("2024-01-01"))
        assert out["factor_unsafe_speed"].iloc[0] == pytest.approx(0.5)
        assert out["factor_unspecified"].iloc[0] == pytest.approx(0.5)

    def test_units_without_crashes_get_zero_share(self, assigned, units):
        out = factor_mix(assigned, units, pd.Timestamp("2024-01-01"))
        assert out.loc[out["unit_id"] == "C2", "factor_unspecified"].iloc[0] == 0.0

    def test_night_share_uses_the_evening_and_overnight_window(self, assigned, units):
        """C1's two crashes are at 20:00 and 03:00, both night."""
        out = temporal_concentration(assigned, units, pd.Timestamp("2024-01-01"))
        assert out.loc[out["unit_id"] == "C1", "night_share"].iloc[0] == pytest.approx(1.0)

    def test_daytime_crash_is_not_night(self, assigned, units):
        out = temporal_concentration(assigned, units, pd.Timestamp("2024-01-01"))
        assert out.loc[out["unit_id"] == "I1", "night_share"].iloc[0] == pytest.approx(0.0)


class TestBuildFeatures:
    def test_produces_one_row_per_unit(self, assigned, units):
        out, _ = build_features(assigned, units, pd.Timestamp("2024-01-01"))
        assert len(out) == len(units)

    def test_carries_the_as_of_date(self, assigned, units):
        out, _ = build_features(assigned, units, pd.Timestamp("2024-01-01"))
        assert (out["as_of"] == pd.Timestamp("2024-01-01")).all()

    def test_no_infinities_survive_assembly(self, assigned, units):
        out, _ = build_features(assigned, units, pd.Timestamp("2024-01-01"))
        numeric = out.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()
