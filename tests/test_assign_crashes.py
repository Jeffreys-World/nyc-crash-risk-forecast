"""Crash-to-unit assignment: every record assigned or accounted for, none lost.

Four exclusion buckets are kept separate on purpose. A crash with no coordinates is a
reporting gap, a crash at (0, 0) is a known geocoder artifact, a crash in Los Angeles is
a data error, and a crash 400 ft from any street is a geometry limitation. Collapsing
them into one "bad rows" counter would hide which one is growing.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.config import CRS_PROJECTED, MAX_JOIN_DISTANCE_FT
from src.spatial import (
    AssignmentReport,
    SpatialJoinError,
    _nearest_within,
    assign_crashes_to_units,
    crashes_to_gdf,
)


class TestExclusionBuckets:
    def test_counts_missing_coordinates(self, crashes):
        _, report = crashes_to_gdf(crashes)
        assert report.missing_coords == 1

    def test_counts_null_island_separately(self, crashes):
        """(0, 0) is in the Gulf of Guinea. It is a geocoder failure, not a location."""
        _, report = crashes_to_gdf(crashes)
        assert report.null_island == 1

    def test_counts_out_of_bounds(self, crashes):
        _, report = crashes_to_gdf(crashes)
        assert report.outside_nyc == 1  # the Los Angeles row

    def test_keeps_the_valid_crashes(self, crashes):
        points, report = crashes_to_gdf(crashes)
        assert report.total_input == 7
        assert len(points) == 4  # 3 on-grid + 1 far but still inside NYC

    def test_nothing_is_dropped_before_being_counted(self, crashes):
        points, report = crashes_to_gdf(crashes)
        assert len(points) + report.missing_coords + report.null_island + report.outside_nyc == 7


class TestAssignment:
    def test_far_crash_lands_in_the_beyond_distance_bucket(self, crashes, universe):
        points, report = crashes_to_gdf(crashes)
        _, report = assign_crashes_to_units(points, universe, report)
        assert report.beyond_max_distance == 1

    def test_the_books_balance(self, crashes, universe):
        """assigned + dropped == input, or validate() raises."""
        points, report = crashes_to_gdf(crashes)
        _, report = assign_crashes_to_units(points, universe, report)
        assert report.assigned + report.dropped == report.total_input
        report.validate()

    def test_every_grid_crash_is_assigned(self, crashes, universe):
        points, report = crashes_to_gdf(crashes)
        assigned, report = assign_crashes_to_units(points, universe, report)
        assert report.assigned == 3
        assert len(assigned) == 3

    def test_each_crash_is_assigned_exactly_once(self, crashes, universe):
        """Double assignment would inflate both units and the capture rate."""
        points, report = crashes_to_gdf(crashes)
        assigned, _ = assign_crashes_to_units(points, universe, report)
        assert not assigned.index.duplicated().any()
        assert len(assigned) == len(assigned.drop_duplicates())

    def test_crashes_at_intersections_go_to_nodes(self, universe):
        """A crash on a corner is intersection-related, not mid-block."""
        from tests.conftest import LATS, LONS, make_crash

        on_corner = pd.DataFrame([make_crash(LONS[1], LATS[1], "2022-01-01T00:00:00.000")])
        points, report = crashes_to_gdf(on_corner)
        assigned, report = assign_crashes_to_units(points, universe, report)
        assert report.assigned_intersection == 1
        assert assigned["unit_type"].iloc[0] == "intersection"

    def test_validate_raises_when_records_vanish(self):
        report = AssignmentReport(total_input=10, assigned_corridor=3, missing_coords=1)
        with pytest.raises(SpatialJoinError, match="lost silently"):
            report.validate()

    def test_raises_on_an_empty_universe(self, crashes):
        points, report = crashes_to_gdf(crashes)
        empty = gpd.GeoDataFrame(
            {"unit_id": [], "unit_type": []},
            geometry=gpd.GeoSeries([], crs=CRS_PROJECTED),
            crs=CRS_PROJECTED,
        )
        with pytest.raises(SpatialJoinError, match="universe is empty"):
            assign_crashes_to_units(points, empty, report)


class TestDeterministicTieBreak:
    """A crash exactly equidistant from two segments must not be counted twice."""

    @pytest.fixture
    def equidistant(self):
        segments = gpd.GeoDataFrame(
            {"unit_id": ["C_beta", "C_alpha"], "unit_type": ["corridor", "corridor"]},
            geometry=[
                LineString([(990_000, 200_000), (990_500, 200_000)]),
                LineString([(990_000, 200_200), (990_500, 200_200)]),
            ],
            crs=CRS_PROJECTED,
        )
        crash = gpd.GeoDataFrame(
            {"crash_id": [1]}, geometry=[Point(990_250, 200_100)], crs=CRS_PROJECTED
        )
        return crash, segments

    def test_returns_exactly_one_match(self, equidistant):
        crash, segments = equidistant
        result = _nearest_within(crash, segments, MAX_JOIN_DISTANCE_FT)
        assert len(result) == 1

    def test_tie_breaks_on_unit_id_ascending(self, equidistant):
        """Arbitrary, but identical on every machine and every re-run."""
        crash, segments = equidistant
        result = _nearest_within(crash, segments, MAX_JOIN_DISTANCE_FT)
        assert result["unit_id"].iloc[0] == "C_alpha"

    def test_is_stable_across_input_ordering(self, equidistant):
        crash, segments = equidistant
        first = _nearest_within(crash, segments, MAX_JOIN_DISTANCE_FT)["unit_id"].iloc[0]
        reordered = segments.iloc[::-1].reset_index(drop=True)
        second = _nearest_within(crash, reordered, MAX_JOIN_DISTANCE_FT)["unit_id"].iloc[0]
        assert first == second


class TestCRSEnforcementInJoins:
    def test_nearest_within_rejects_unprojected_input(self, crashes, universe):
        """The guard has to hold at the join, not just at frame construction."""
        from src.config import CRS_GEOGRAPHIC
        from src.spatial import CRSError

        geographic = gpd.GeoDataFrame(
            {"crash_id": [1]}, geometry=[Point(-73.98, 40.75)], crs=CRS_GEOGRAPHIC
        )
        with pytest.raises(CRSError):
            _nearest_within(geographic, universe, MAX_JOIN_DISTANCE_FT)


class TestReportSummary:
    def test_summary_reports_every_bucket(self, crashes, universe):
        points, report = crashes_to_gdf(crashes)
        _, report = assign_crashes_to_units(points, universe, report)
        text = report.summary()
        for fragment in ("assigned", "missing coords", "(0,0)", "outside NYC", "beyond"):
            assert fragment in text

    def test_percentage_does_not_divide_by_zero(self):
        assert "0.0%" in AssignmentReport(total_input=0).summary()
