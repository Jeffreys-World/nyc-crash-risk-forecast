"""The three join radii are parameters, and the units cache knows it.

Two failure modes are covered here, and the second is the dangerous one.

  1. A radius passed in is ignored, and the sweep silently measures the default three
     times. Loud enough to notice, since every row of the table would be identical.

  2. A radius passed in is honoured, but the units cache is keyed only on file
     contents. The first setting builds and caches; every later setting hits that
     entry and returns the *first* setting's units. The table then shows a headline
     that does not move at all, which reads as "the result is robust to the radii"
     when what actually happened is that the radii were never varied.

The second is the same class of bug the code fingerprint was added to fix, arriving
from the opposite direction: there, the files changed and the key did not; here, the
arguments change and the files do not.
"""

from __future__ import annotations

from dataclasses import replace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.config import (
    CRS_GEOGRAPHIC,
    CRS_PROJECTED,
    DEFAULT_RADII,
    INTERSECTION_RADIUS_FT,
    MAX_JOIN_DISTANCE_FT,
    VZV_BUFFER_FT,
    JoinRadii,
)
from src.spatial import AssignmentReport, assign_crashes_to_units, join_vzv_labels


class TestDefaultsMatchConfig:
    """The dataclass must not drift from the constants the README quotes."""

    def test_default_radii_are_the_config_constants(self):
        assert DEFAULT_RADII.max_join_distance_ft == MAX_JOIN_DISTANCE_FT
        assert DEFAULT_RADII.intersection_radius_ft == INTERSECTION_RADIUS_FT
        assert DEFAULT_RADII.vzv_buffer_ft == VZV_BUFFER_FT

    def test_published_baseline_is_150_100_50(self):
        """The numbers underneath the published headline, asserted rather than assumed."""
        assert DEFAULT_RADII.as_dict() == {
            "max_join_distance_ft": 150.0,
            "intersection_radius_ft": 100.0,
            "vzv_buffer_ft": 50.0,
        }

    def test_join_vzv_labels_defaults_to_the_config_buffer(self, universe, vzv_corridors,
                                                           vzv_intersections):
        """A literal default here would silently outlive a change to config."""
        from_default, _ = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        explicit, _ = join_vzv_labels(
            universe, vzv_corridors, vzv_intersections, buffer_ft=VZV_BUFFER_FT
        )
        assert from_default["is_priority"].tolist() == explicit["is_priority"].tolist()


class TestCacheTagSeparatesSettings:
    """`tag` is the cache-key fragment. Distinct radii must produce distinct tags."""

    def test_baseline_tag_is_readable(self):
        assert DEFAULT_RADII.tag == "j150-i100-v50"

    @pytest.mark.parametrize(
        "field",
        ["max_join_distance_ft", "intersection_radius_ft", "vzv_buffer_ft"],
    )
    def test_changing_any_one_radius_changes_the_tag(self, field):
        changed = JoinRadii(**{**DEFAULT_RADII.as_dict(), field: 999.0})
        assert changed.tag != DEFAULT_RADII.tag

    def test_every_swept_setting_has_a_distinct_tag(self):
        """The property the sweep depends on, checked over the values it actually uses."""
        tags = set()
        for field, values in (
            ("max_join_distance_ft", (100.0, 150.0, 250.0)),
            ("intersection_radius_ft", (50.0, 100.0, 150.0)),
            ("vzv_buffer_ft", (25.0, 50.0, 100.0)),
        ):
            for value in values:
                tags.add(JoinRadii(**{**DEFAULT_RADII.as_dict(), field: value}).tag)
        # Three axes of three, sharing one baseline: seven distinct settings.
        assert len(tags) == 7

    def test_fractional_radii_do_not_collide(self):
        """`f"{x:g}"` must not round 100.4 and 100.6 onto the same key."""
        a = JoinRadii(**{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 100.4})
        b = JoinRadii(**{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 100.6})
        assert a.tag != b.tag

    def test_radii_are_hashable_and_comparable(self):
        """`radii == DEFAULT_RADII` gates whether data/processed/ may be written."""
        assert JoinRadii() == DEFAULT_RADII
        assert len({JoinRadii(), DEFAULT_RADII}) == 1


class TestRadiusActuallyMovesTheAssignment:
    """A crash between a node and a segment changes hands when the radius moves."""

    @pytest.fixture
    def one_node_city(self) -> gpd.GeoDataFrame:
        """Two segments meeting at a node, so `build_universe` yields both unit types."""
        from src.spatial import build_universe

        lon, lat = -73.980, 40.750
        rows = [
            {
                "street": "WEST LEG",
                "borough": "MANHATTAN",
                "geometry": LineString([(lon - 0.004, lat), (lon, lat)]),
            },
            {
                "street": "EAST LEG",
                "borough": "MANHATTAN",
                "geometry": LineString([(lon, lat), (lon + 0.004, lat)]),
            },
            {
                "street": "NORTH LEG",
                "borough": "MANHATTAN",
                "geometry": LineString([(lon, lat), (lon, lat + 0.004)]),
            },
        ]
        return build_universe(
            gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_GEOGRAPHIC)
        )

    def _crash_120ft_up_the_block(self, one_node_city) -> gpd.GeoDataFrame:
        """A point on the north leg, 120 ft from the junction.

        Placed in projected feet directly rather than in degrees, so the distance is
        exact and the test is not asserting on a reprojection approximation.
        """
        node = one_node_city[one_node_city["unit_type"] == "intersection"].iloc[0]
        x, y = node.geometry.x, node.geometry.y
        return gpd.GeoDataFrame(
            {"crash_id": ["c1"]},
            geometry=[Point(x, y + 120.0)],
            crs=CRS_PROJECTED,
        )

    def test_tight_radius_sends_it_to_the_segment(self, one_node_city):
        crash = self._crash_120ft_up_the_block(one_node_city)
        report = AssignmentReport(total_input=1)
        radii = JoinRadii(**{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 100.0})

        assigned, report = assign_crashes_to_units(
            crash, one_node_city, report, radii=radii
        )
        assert report.assigned_intersection == 0
        assert report.assigned_corridor == 1
        assert assigned["unit_type"].iloc[0] == "corridor"

    def test_wide_radius_claims_it_for_the_node(self, one_node_city):
        crash = self._crash_120ft_up_the_block(one_node_city)
        report = AssignmentReport(total_input=1)
        radii = JoinRadii(**{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 150.0})

        assigned, report = assign_crashes_to_units(
            crash, one_node_city, report, radii=radii
        )
        assert report.assigned_intersection == 1
        assert report.assigned_corridor == 0
        assert assigned["unit_type"].iloc[0] == "intersection"

    def test_max_join_distance_decides_assigned_versus_dropped(self, universe, crashes):
        """The far crash is beyond 150 ft but well inside a generous radius."""
        from src.spatial import crashes_to_gdf

        points, base = crashes_to_gdf(crashes)

        # `replace` rather than a fresh report: the exclusion counters set by
        # `crashes_to_gdf` are part of the balance `validate()` checks, and
        # `assign_crashes_to_units` mutates the report it is handed.
        tight = assign_crashes_to_units(
            points, universe, replace(base), radii=DEFAULT_RADII
        )[1]
        wide = assign_crashes_to_units(
            points,
            universe,
            replace(base),
            radii=JoinRadii(
                **{**DEFAULT_RADII.as_dict(), "max_join_distance_ft": 200_000.0}
            ),
        )[1]

        assert tight.beyond_max_distance == 1
        assert wide.beyond_max_distance == 0
        assert wide.assigned > tight.assigned

    def test_omitting_radii_matches_passing_the_defaults(self, universe, crashes):
        """The default path must not diverge from the explicit one."""
        from src.spatial import crashes_to_gdf

        points, base = crashes_to_gdf(crashes)
        implicit, r_implicit = assign_crashes_to_units(points, universe, replace(base))
        explicit, r_explicit = assign_crashes_to_units(
            points, universe, replace(base), radii=DEFAULT_RADII
        )
        assert r_implicit.summary() == r_explicit.summary()
        pd.testing.assert_series_equal(
            implicit["unit_id"].reset_index(drop=True),
            explicit["unit_id"].reset_index(drop=True),
        )


class TestTheReportDescribesTheRunThatProducedIt:
    def test_summary_quotes_the_radius_actually_used(self, universe, crashes):
        from src.spatial import crashes_to_gdf

        points, base = crashes_to_gdf(crashes)
        _, report = assign_crashes_to_units(
            points,
            universe,
            replace(base),
            radii=JoinRadii(**{**DEFAULT_RADII.as_dict(), "max_join_distance_ft": 250.0}),
        )
        assert "beyond 250ft" in report.summary()
        assert "beyond 150ft" not in report.summary()


class TestTheSweepPlanSurvivesRetuning:
    """`src/config.py` invites retuning the intersection radius. The plan must follow it.

    With the swept values written out literally, changing a published constant leaves no
    row flagged baseline. The report builder then dies selecting it - after every rebuild
    has run and after the CSV has already been written over.
    """

    def test_every_axis_contains_the_published_value(self):
        from scripts.radius_sensitivity import SWEEPS, settings

        published = DEFAULT_RADII.as_dict()
        by_knob: dict[str, set[float]] = {}
        for knob, field, value, _ in settings():
            by_knob.setdefault((knob, field), set()).add(value)

        assert len(by_knob) == len(SWEEPS)
        for (knob, field), values in by_knob.items():
            assert published[field] in values, f"{knob} has no baseline row"

    def test_exactly_one_row_per_axis_is_the_baseline(self):
        from scripts.radius_sensitivity import settings

        for knob in {k for k, _, _, _ in settings()}:
            flagged = [r for k, _, _, r in settings() if k == knob and r == DEFAULT_RADII]
            assert len(flagged) == 1, knob

    def test_a_retuned_config_still_yields_a_baseline_row(self, monkeypatch):
        """The regression: 120 ft is in no hardcoded list, and must still appear."""
        from scripts import radius_sensitivity

        retuned = JoinRadii(
            **{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 120.0}
        )
        monkeypatch.setattr(radius_sensitivity, "DEFAULT_RADII", retuned)

        rows = radius_sensitivity.settings()
        assert any(r == retuned for _, _, _, r in rows)
        for knob in {k for k, _, _, _ in rows}:
            assert any(r == retuned for k, _, _, r in rows if k == knob), knob

    def test_quick_sweeps_one_axis(self):
        from scripts.radius_sensitivity import QUICK_ONLY, settings

        assert {k for k, _, _, _ in settings(quick=True)} == {QUICK_ONLY}


class TestReportRefusesToMisrepresentItself:
    def test_an_undefined_capture_rate_is_named_not_formatted(self):
        """`rate_pp` is `float | None` by contract; None must not become a traceback."""
        from scripts.radius_sensitivity import _pp

        assert _pp(None) == "UNDEFINED"
        assert _pp(float("nan")) == "UNDEFINED"
        assert _pp(82.535) == "82.5%"

    def test_a_frame_with_no_baseline_fails_loudly(self):
        from scripts.radius_sensitivity import to_markdown

        frame = pd.DataFrame(
            [
                {
                    "knob": "intersection radius", "value_ft": 50.0, "is_baseline": False,
                    "priority_units_n": 1, "holdout_casualties": 1, "r1_pp": 1.0,
                    "r2_pp": 1.0, "r3_pp": 1.0, "lift_pp": 1.0, "ci_low_pp": 0.1,
                    "ci_high_pp": 2.0, "clears_bar": True,
                }
            ]
        )
        with pytest.raises(SystemExit, match="no row matches the published radii"):
            to_markdown(frame, "2026-08-13")

    def test_a_partial_sweep_says_so_in_the_report(self):
        """A one-axis table must not read as though all three were tested."""
        from scripts.radius_sensitivity import to_markdown

        frame = pd.DataFrame(
            [
                {
                    "knob": "intersection radius", "value_ft": v, "is_baseline": base,
                    "priority_units_n": 38909, "holdout_casualties": 18059, "r1_pp": 48.7,
                    "r2_pp": 64.1, "r3_pp": 82.5, "lift_pp": 18.4, "ci_low_pp": 17.5,
                    "ci_high_pp": 19.3, "clears_bar": True,
                }
                for v, base in ((50.0, False), (100.0, True), (150.0, False))
            ]
        )
        assert "Partial sweep" in to_markdown(frame, "2026-08-13")

    def test_the_baseline_line_follows_config(self, monkeypatch):
        """`Baseline (150 / 100 / 50 ft)` must be derived, not typed."""
        from scripts import radius_sensitivity

        monkeypatch.setattr(
            radius_sensitivity,
            "DEFAULT_RADII",
            JoinRadii(**{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 120.0}),
        )
        frame = pd.DataFrame(
            [
                {
                    "knob": "intersection radius", "value_ft": 120.0, "is_baseline": True,
                    "priority_units_n": 1, "holdout_casualties": 1, "r1_pp": 1.0,
                    "r2_pp": 1.0, "r3_pp": 1.0, "lift_pp": 1.0, "ci_low_pp": 0.1,
                    "ci_high_pp": 2.0, "clears_bar": True,
                }
            ]
        )
        assert "Baseline (150 / 120 / 50 ft)" in radius_sensitivity.to_markdown(
            frame, "2026-08-13"
        )


class TestExploratoryRunsCannotOverwriteThePublishedResult:
    """`data/processed/` is the headline the README quotes.

    A run at 250 ft is a question, not an answer. If it can overwrite run-summary.json
    the repo starts disagreeing with its own README, and the disagreement is invisible
    because both files still look like the output of a real run - which they are, just
    not of the run being described.

    `run()` is stubbed rather than executed: the guard being tested is in `main`'s
    argument handling, and the full pipeline takes minutes on real data.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        from src import pipeline as pipeline_module

        calls: list[dict] = []

        class FakeSummary:
            holdout_window = "2024-2025"
            holdout_casualties = 1
            citywide_n = 1
            r1_citywide_pp = r2_citywide_pp = r3_citywide_pp = 1.0
            lift_pp = ci_low_pp = ci_high_pp = 1.0
            clears_bar = True
            verdict = "stub"

        def fake_run(**kwargs):
            calls.append(kwargs)
            return FakeSummary()

        monkeypatch.setattr(pipeline_module, "run", fake_run)
        return calls

    def test_default_radii_write_the_artifacts(self, captured):
        from src.pipeline import main

        assert main([]) == 0
        assert captured[0]["write_artifacts"] is True

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--max-join-distance-ft", "250"),
            ("--intersection-radius-ft", "150"),
            ("--vzv-buffer-ft", "100"),
        ],
    )
    def test_any_non_default_radius_blocks_the_write(self, captured, flag, value):
        from src.pipeline import main

        assert main([flag, value]) == 0
        assert captured[0]["write_artifacts"] is False

    def test_passing_the_defaults_explicitly_still_writes(self, captured):
        """`--intersection-radius-ft 100` is the baseline, not a non-default run."""
        from src.pipeline import main

        assert main(["--intersection-radius-ft", "100"]) == 0
        assert captured[0]["write_artifacts"] is True

    def test_no_write_blocks_the_write_at_default_radii(self, captured):
        from src.pipeline import main

        assert main(["--no-write"]) == 0
        assert captured[0]["write_artifacts"] is False

    def test_the_radii_reach_run(self, captured):
        from src.pipeline import main

        main(["--vzv-buffer-ft", "25"])
        assert captured[0]["radii"].vzv_buffer_ft == 25.0
        assert captured[0]["radii"].max_join_distance_ft == MAX_JOIN_DISTANCE_FT


class TestUnitsCacheIsKeyedOnTheRadii:
    """The stale-cache hole, from the arguments side.

    Every file the code fingerprint hashes is byte-identical across these two calls.
    Only the arguments differ, so if the radii are not in the key the second call
    returns the first call's units and the sweep reports a fiction.
    """

    @pytest.fixture
    def snapshot(self, tmp_path, monkeypatch, centerline, crashes, vzv_corridors,
                 vzv_intersections, sip_layer):
        """A snapshot directory the pipeline's own loaders can read."""
        from src import pipeline as pipeline_module

        raw = tmp_path / "raw" / "2026-01-01"
        raw.mkdir(parents=True)
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(pipeline_module, "CACHE_DIR", cache)

        def write_geo(gdf: gpd.GeoDataFrame, name: str) -> None:
            frame = pd.DataFrame(gdf.drop(columns=["geometry"]))
            frame["geometry_wkt"] = gdf.geometry.to_wkt()
            frame.to_parquet(raw / f"{name}.parquet", index=False)

        line = centerline.copy()
        line["boroughcode"] = "1"
        line["full_street_name"] = line["street"]
        line["rw_type"] = "1"
        write_geo(line.drop(columns=["borough"]), "centerline")
        write_geo(vzv_corridors, "vzv_corridors")
        write_geo(vzv_intersections, "vzv_intersections")
        write_geo(sip_layer, "sip_corridors")
        write_geo(sip_layer, "sip_intersections")
        crashes.to_parquet(raw / "crashes.parquet", index=False)
        (raw / "manifest.json").write_text("{}")

        return raw, cache

    def test_two_radii_write_two_cache_entries(self, snapshot):
        from src.pipeline import build_scored_units

        raw, cache = snapshot
        wide = JoinRadii(**{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 400.0})

        build_scored_units(raw, use_cache=True, radii=DEFAULT_RADII)
        build_scored_units(raw, use_cache=True, radii=wide)

        parquets = sorted(p.name for p in cache.glob("units-*.parquet"))
        assert len(parquets) == 2, parquets
        assert any(DEFAULT_RADII.tag in name for name in parquets)
        assert any(wide.tag in name for name in parquets)

    def test_a_wider_intersection_radius_changes_the_units(self, snapshot):
        """Not just a different filename - a different answer.

        A cache hit would return a byte-identical frame, so this is the assertion that
        actually distinguishes "keyed correctly" from "keyed on something irrelevant".
        """
        from src.pipeline import build_scored_units

        raw, _ = snapshot
        tight, _ = build_scored_units(raw, use_cache=True, radii=DEFAULT_RADII)
        wide, _ = build_scored_units(
            raw,
            use_cache=True,
            radii=JoinRadii(
                **{**DEFAULT_RADII.as_dict(), "intersection_radius_ft": 400.0}
            ),
        )

        def intersection_share(units: pd.DataFrame) -> float:
            at_nodes = units[units["unit_type"] == "intersection"]
            return float(at_nodes["casualties_36mo"].sum())

        assert intersection_share(wide) > intersection_share(tight)

    def test_the_second_call_at_the_same_radii_is_a_cache_hit(self, snapshot):
        """The cache still has to work; keying it correctly must not disable it."""
        from src.pipeline import build_scored_units

        raw, cache = snapshot
        build_scored_units(raw, use_cache=True, radii=DEFAULT_RADII)
        written = {p: p.stat().st_mtime_ns for p in cache.glob("units-*.parquet")}

        build_scored_units(raw, use_cache=True, radii=DEFAULT_RADII)
        assert {p: p.stat().st_mtime_ns for p in cache.glob("units-*.parquet")} == written
