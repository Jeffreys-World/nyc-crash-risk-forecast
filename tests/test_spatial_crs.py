"""CRITICAL gap 1: distances measured in the wrong coordinate system.

In EPSG:4326 a `max_distance=150` means 150 *degrees*. That covers the planet, so every
crash matches whichever segment the spatial index happens to return first, the pipeline
runs to completion, and the headline number is garbage. Nothing raises. These tests are
the only thing standing between that bug and a published result.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.config import CRS_GEOGRAPHIC, CRS_PROJECTED
from src.spatial import (
    CRSError,
    SpatialJoinError,
    assert_projected,
    build_node_universe,
    build_segment_universe,
    build_universe,
    to_projected,
)


class TestAssertProjected:
    def test_rejects_missing_crs(self):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)])
        with pytest.raises(CRSError, match="no CRS"):
            assert_projected(gdf, "test frame")

    def test_rejects_geographic_crs(self):
        """The actual bug: WGS84 degrees where feet are intended."""
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(-73.98, 40.75)], crs=CRS_GEOGRAPHIC)
        with pytest.raises(CRSError, match="projected CRS"):
            assert_projected(gdf, "test frame")

    def test_accepts_projected_crs(self, projected_points):
        assert_projected(projected_points, "test frame")  # must not raise

    def test_error_names_the_frame(self):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(-73.98, 40.75)], crs=CRS_GEOGRAPHIC)
        with pytest.raises(CRSError, match="crashes"):
            assert_projected(gdf, "crashes")


class TestToProjected:
    def test_refuses_to_guess_source_crs(self):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)])
        with pytest.raises(CRSError, match="cannot reproject from unknown"):
            to_projected(gdf, "test frame")

    def test_reprojects_and_lands_in_nyc(self):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(-73.98, 40.75)], crs=CRS_GEOGRAPHIC)
        out = to_projected(gdf)
        assert out.crs.equals(CRS_PROJECTED)
        # State Plane Long Island puts Manhattan near x~990k, y~210k in feet.
        assert 900_000 < out.geometry.iloc[0].x < 1_090_000
        assert 110_000 < out.geometry.iloc[0].y < 280_000

    def test_is_a_noop_when_already_projected(self, projected_points):
        assert to_projected(projected_points).crs.equals(CRS_PROJECTED)


class TestSegmentUniverse:
    def test_lengths_are_in_feet_not_degrees(self, centerline):
        segments = build_segment_universe(centerline)
        # ~0.001 degrees at this latitude is a few hundred feet. In degrees these
        # lengths would be ~0.001, which is the failure this asserts against.
        assert segments["length_ft"].min() > 100
        assert segments["length_ft"].max() < 1000

    def test_flags_degenerate_segments_without_dropping_them(self, centerline):
        """Zero-length segments stay in the universe so the denominator is honest."""
        from shapely.geometry import LineString

        degenerate = centerline.copy()
        degenerate.loc[len(degenerate)] = {
            "street": "zero",
            "borough": "MANHATTAN",
            "geometry": LineString([(-73.980, 40.750), (-73.980, 40.750)]),
        }
        segments = build_segment_universe(degenerate)
        assert len(segments) == len(centerline) + 1
        assert segments["degenerate_length"].sum() == 1


class TestNodeUniverse:
    def test_builds_nine_nodes_from_a_three_by_three_grid(self, centerline):
        segments = build_segment_universe(centerline)
        nodes = build_node_universe(segments)
        assert len(nodes) == 9

    def test_leg_count_matches_grid_topology(self, centerline):
        """Corners meet 2 streets, edges 3, the centre 4."""
        nodes = build_node_universe(build_segment_universe(centerline))
        assert sorted(nodes["leg_count"].tolist()) == [2, 2, 2, 2, 3, 3, 3, 3, 4]

    def test_nodes_inherit_borough_from_their_streets(self, centerline):
        """Without this, borough-stratified selection drops every intersection."""
        nodes = build_node_universe(build_segment_universe(centerline))
        assert "borough" in nodes.columns
        assert nodes["borough"].notna().all()
        assert set(nodes["borough"]) == {"MANHATTAN"}

    def test_raises_when_there_are_no_endpoints(self):
        empty = gpd.GeoDataFrame(
            {"a": []}, geometry=gpd.GeoSeries([], crs=CRS_PROJECTED), crs=CRS_PROJECTED
        )
        with pytest.raises(SpatialJoinError, match="no segment endpoints"):
            build_node_universe(empty)


class TestBuildUniverse:
    def test_contains_both_unit_types(self, universe):
        assert set(universe["unit_type"]) == {"corridor", "intersection"}
        assert (universe["unit_type"] == "corridor").sum() == 12
        assert (universe["unit_type"] == "intersection").sum() == 9

    def test_unit_ids_are_unique(self, universe):
        """A duplicate unit_id would double-count casualties in the capture rate."""
        assert not universe["unit_id"].duplicated().any()

    def test_output_is_projected(self, universe):
        assert_projected(universe, "universe")

    def test_every_unit_has_a_borough(self, universe):
        assert universe["borough"].notna().all()
