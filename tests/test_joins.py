"""VZV priority labels and SIP treatment.

Two failure modes, both of which would quietly change what the project is measuring:

* A VZV feature that matches zero units. Dropping it silently shrinks the published
  priority list, which is the exact artifact this project is scored against.
* A SIP record with no completion date. Treatment that cannot be placed in time cannot
  be controlled for, and the treated/untreated split is the endogeneity control the
  whole comparison depends on.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.config import CRS_GEOGRAPHIC
from src.spatial import SpatialJoinError, join_sip_treatment, join_vzv_labels


class TestVZVLabels:
    def test_labels_units_on_a_priority_corridor(self, universe, vzv_corridors, vzv_intersections):
        labelled, _ = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        assert labelled["is_priority"].sum() > 0

    def test_orphan_corridor_is_flagged_not_dropped(
        self, universe, vzv_corridors, vzv_intersections
    ):
        """The second fixture corridor is nowhere near the grid, by design."""
        _, report = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        assert report.vzv_corridors_in == 2
        assert report.vzv_corridors_matched == 1
        assert any(k.startswith("corridor:") for k in report.unmatched_keys)

    def test_universe_size_is_unchanged(self, universe, vzv_corridors, vzv_intersections):
        """Labelling must not filter. A shorter universe changes the denominator."""
        labelled, _ = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        assert len(labelled) == len(universe)

    def test_corridor_labels_reach_the_nodes_along_the_corridor(
        self, universe, vzv_corridors, vzv_intersections
    ):
        """Deliberate reversal of an earlier assertion, which encoded a real bug.

        This test previously asserted corridor labels must NOT touch intersections.
        That was wrong. A VZV priority corridor is a stretch of street including its
        junctions, and crashes within 100ft of a junction are assigned to the node. With
        86% of pedestrian casualties occurring at intersections, leaving those nodes
        unlabelled meant DOT's list could not capture the casualties it was chosen for.
        """
        labelled, _ = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        nodes_on_corridor = labelled[
            (labelled["unit_type"] == "intersection") & labelled["is_priority"]
        ]
        assert not nodes_on_corridor.empty

    def test_intersection_layer_never_labels_corridors(
        self, universe, vzv_corridors, vzv_intersections
    ):
        """The reverse direction stays closed: a priority point is not a priority street."""
        import geopandas as gpd

        empty = gpd.GeoDataFrame(
            {"name": []}, geometry=gpd.GeoSeries([], crs=CRS_GEOGRAPHIC), crs=CRS_GEOGRAPHIC
        )
        labelled, _ = join_vzv_labels(universe, empty, vzv_intersections)
        assert labelled.loc[labelled["unit_type"] == "corridor", "is_priority"].sum() == 0

    def test_vzv_source_accumulates_instead_of_overwriting(
        self, universe, vzv_corridors, vzv_intersections
    ):
        """Regression: ISSUE-009 — the intersection layer erased corridor provenance.

        Found by /qa on 2026-08-12. A node can be both a VZV priority intersection and
        a point on a VZV priority corridor. A plain assignment let the second layer
        overwrite the first, so the record read "intersection" for a unit on both lists.
        """
        labelled, _ = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        sources = set(labelled.loc[labelled["is_priority"], "vzv_source"].dropna())

        # The fixture's VZV point sits on the fixture's VZV corridor, so at least one
        # unit must carry both.
        assert any("+" in s for s in sources), f"no combined provenance in {sources}"

    def test_empty_vzv_layer_labels_nothing_and_does_not_raise(self, universe, vzv_intersections):
        empty = gpd.GeoDataFrame(
            {"name": []}, geometry=gpd.GeoSeries([], crs=CRS_GEOGRAPHIC), crs=CRS_GEOGRAPHIC
        )
        labelled, report = join_vzv_labels(universe, empty, vzv_intersections)
        assert report.vzv_corridors_in == 0
        assert len(labelled) == len(universe)

    def test_name_join_would_have_over_labelled(self, universe, vzv_corridors, vzv_intersections):
        """The reason the join is spatial.

        The fixture's priority corridor covers one block of one E-W street. A join on
        street name would label both blocks of that street. Spatially, only the
        overlapping segment is labelled.
        """
        labelled, _ = join_vzv_labels(universe, vzv_corridors, vzv_intersections)
        priority_corridors = labelled[
            (labelled["unit_type"] == "corridor") & labelled["is_priority"]
        ]
        assert len(priority_corridors) < 12


class TestSIPTreatment:
    def test_tags_treated_units(self, universe, sip_layer):
        treated, _ = join_sip_treatment(universe, [sip_layer])
        assert treated["treated"].sum() > 0

    def test_undated_project_is_excluded_and_counted(self, universe, sip_layer):
        _, report = join_sip_treatment(universe, [sip_layer])
        assert report.sip_records_in == 2
        assert report.sip_missing_date == 1

    def test_treatment_date_is_recorded(self, universe, sip_layer):
        treated, _ = join_sip_treatment(universe, [sip_layer])
        dates = treated.loc[treated["treated"], "treatment_date"]
        assert dates.notna().all()
        assert (dates == pd.Timestamp("2021-06-30")).all()

    def test_earliest_treatment_wins(self, universe):
        """A unit rebuilt twice was first affected by the first project."""
        geom = LineString([(-73.980, 40.750), (-73.979, 40.750)])
        early = gpd.GeoDataFrame(
            [{"completion_date": "2019-01-01", "geometry": geom}],
            geometry="geometry",
            crs=CRS_GEOGRAPHIC,
        )
        late = gpd.GeoDataFrame(
            [{"completion_date": "2023-01-01", "geometry": geom}],
            geometry="geometry",
            crs=CRS_GEOGRAPHIC,
        )
        treated, _ = join_sip_treatment(universe, [late, early])
        dates = treated.loc[treated["treated"], "treatment_date"]
        assert dates.min() == pd.Timestamp("2019-01-01")

    def test_raises_when_no_date_column_exists(self, universe):
        """Silently treating everything as untreated would void the control."""
        no_date = gpd.GeoDataFrame(
            [{"project": "x", "geometry": LineString([(-73.980, 40.750), (-73.979, 40.750)])}],
            geometry="geometry",
            crs=CRS_GEOGRAPHIC,
        )
        with pytest.raises(SpatialJoinError, match="no completion-date column"):
            join_sip_treatment(universe, [no_date])

    def test_untreated_units_stay_false_not_null(self, universe, sip_layer):
        """`treated` is consumed as a boolean mask; a null would silently misclassify."""
        treated, _ = join_sip_treatment(universe, [sip_layer])
        assert treated["treated"].notna().all()
        assert treated["treated"].dtype == bool
