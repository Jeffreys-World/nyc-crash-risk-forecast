"""Build the unit universe and attach crashes, VZV labels, and SIP treatment to it.

This is the highest-risk stage in the project. Two of the three silent failure modes
live here:

  1. CRS mismatch. Buffering or measuring distance in WGS84 degrees while intending
     feet is wrong by roughly 364,000x and raises nothing. Every function that touches
     distance calls `assert_projected` first.
  2. Records lost without a trace. A crash with no coordinates, a crash at (0, 0), a
     crash outside NYC, and a crash too far from any street are four different stories.
     Each is counted into an `AssignmentReport` rather than quietly filtered away.

The unit of analysis is a *segment*, not a street name. A VZV corridor is Broadway from
W 135th to W 153rd, not all of Broadway, so a name join would assign every Broadway
crash to that corridor and inflate priority-corridor counts in the model's favor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.config import (
    CRS_GEOGRAPHIC,
    CRS_PROJECTED,
    INTERSECTION_RADIUS_FT,
    MAX_JOIN_DISTANCE_FT,
    MIN_SEGMENT_LENGTH_FT,
    NYC_BOUNDS_FT,
    ROAD_NUMERIC_COLUMNS,
    ROADWAY_TYPE_STREET,
)

log = logging.getLogger(__name__)

UnitType = Literal["corridor", "intersection"]


class CRSError(ValueError):
    """A geometric operation was attempted in the wrong coordinate system."""


class SpatialJoinError(RuntimeError):
    """A join produced a result that cannot be trusted downstream."""


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def assert_projected(gdf: gpd.GeoDataFrame, what: str = "frame") -> None:
    """Refuse to measure distance in degrees.

    Called before every buffer, length, and nearest-neighbour operation in this module.
    The check is cheap and the failure it prevents is invisible: in EPSG:4326 a
    `max_distance=150` means 150 *degrees*, which covers the planet, so every crash
    would match the first segment the index happens to return.
    """
    if gdf.crs is None:
        raise CRSError(
            f"{what} has no CRS. Set one explicitly - guessing is how the "
            f"degrees-for-feet bug gets in."
        )
    if not gdf.crs.equals(CRS_PROJECTED):
        raise CRSError(
            f"{what} is in {gdf.crs.to_string()}, expected {CRS_PROJECTED}. "
            f"Distances here are in feet and only valid in a projected CRS."
        )


def to_projected(gdf: gpd.GeoDataFrame, what: str = "frame") -> gpd.GeoDataFrame:
    """Reproject into the working CRS, requiring the source CRS to be declared."""
    if gdf.crs is None:
        raise CRSError(f"{what} has no CRS; cannot reproject from unknown.")
    if gdf.crs.equals(CRS_PROJECTED):
        return gdf
    log.info("reprojecting %s: %s -> %s", what, gdf.crs.to_string(), CRS_PROJECTED)
    return gdf.to_crs(CRS_PROJECTED)


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------


@dataclass
class AssignmentReport:
    """Every crash is either assigned or accounted for by exactly one counter here."""

    total_input: int = 0
    missing_coords: int = 0
    null_island: int = 0
    outside_nyc: int = 0
    beyond_max_distance: int = 0
    assigned_intersection: int = 0
    assigned_corridor: int = 0

    @property
    def assigned(self) -> int:
        return self.assigned_intersection + self.assigned_corridor

    @property
    def dropped(self) -> int:
        return (
            self.missing_coords
            + self.null_island
            + self.outside_nyc
            + self.beyond_max_distance
        )

    def validate(self) -> None:
        """The books must balance. An imbalance means a record vanished."""
        if self.assigned + self.dropped != self.total_input:
            raise SpatialJoinError(
                f"crash accounting does not balance: {self.assigned} assigned + "
                f"{self.dropped} dropped != {self.total_input} input. Records were "
                f"lost silently."
            )

    def summary(self) -> str:
        pct = (100.0 * self.assigned / self.total_input) if self.total_input else 0.0
        return (
            f"crashes: {self.total_input} in, {self.assigned} assigned ({pct:.1f}%) "
            f"[{self.assigned_intersection} intersection, {self.assigned_corridor} corridor], "
            f"{self.dropped} dropped "
            f"[missing coords {self.missing_coords}, (0,0) {self.null_island}, "
            f"outside NYC {self.outside_nyc}, beyond {MAX_JOIN_DISTANCE_FT:.0f}ft "
            f"{self.beyond_max_distance}]"
        )


@dataclass
class LabelJoinReport:
    """VZV and SIP joins. Unmatched features are flagged, never dropped."""

    vzv_corridors_in: int = 0
    vzv_corridors_matched: int = 0
    vzv_intersections_in: int = 0
    vzv_intersections_matched: int = 0
    sip_records_in: int = 0
    sip_missing_date: int = 0
    unmatched_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"VZV corridors {self.vzv_corridors_matched}/{self.vzv_corridors_in} matched, "
            f"VZV intersections {self.vzv_intersections_matched}/{self.vzv_intersections_in} "
            f"matched, SIP {self.sip_records_in} in "
            f"({self.sip_missing_date} missing a date, excluded)"
        )


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------


def build_segment_universe(centerline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centerline segments as corridor units, with length as the exposure term.

    Zero-length and sub-foot segments are kept in the universe but flagged. They are
    excluded at fit time rather than here, because dropping them silently would change
    the denominator of the capture rate without anyone noticing.
    """
    segments = to_projected(centerline, "centerline").copy()
    assert_projected(segments, "centerline")

    segments = segments[segments.geometry.notna() & ~segments.geometry.is_empty].copy()

    # Explode multi-part geometries into one row per LineString. A MultiLineString
    # left intact becomes a single unit whose length is the sum of disconnected
    # pieces, so its exposure term is wrong and the unit spans two places at once -
    # crashes from both would pool into one risk score. DCP LION ships
    # MultiLineStrings, and it is one of the live centerline candidates.
    multipart = int((segments.geometry.geom_type == "MultiLineString").sum())
    if multipart:
        log.info("exploding %d multi-part centerline geometries into single parts", multipart)
        segments = segments.explode(index_parts=False, ignore_index=True)

    segments["length_ft"] = segments.geometry.length
    segments["unit_type"] = "corridor"
    segments["unit_id"] = "C" + segments.index.astype(str)

    # Socrata serves every field as a string. Coerced here so the SPF sees numbers
    # rather than silently dropping the column at fit time.
    for column in ROAD_NUMERIC_COLUMNS:
        if column in segments.columns:
            segments[column] = pd.to_numeric(segments[column], errors="coerce")

    # rw_type 1 is a surface street; every higher code is a highway, ramp, or bridge.
    # This is the limited-access-highway signal the founding borough-gap finding turned
    # on, so the model gets to see it explicitly instead of rediscovering it.
    if "rw_type" in segments.columns:
        segments["is_highway"] = (
            segments["rw_type"].astype(str).str.strip() != ROADWAY_TYPE_STREET
        ).astype(int)

    degenerate = int((segments["length_ft"] < MIN_SEGMENT_LENGTH_FT).sum())
    if degenerate:
        log.warning(
            "%d segments are shorter than %.1f ft. Flagged as degenerate; they are "
            "excluded at fit time (log(0) offset) and counted in the exclusion report.",
            degenerate,
            MIN_SEGMENT_LENGTH_FT,
        )
    segments["degenerate_length"] = segments["length_ft"] < MIN_SEGMENT_LENGTH_FT

    return segments.reset_index(drop=True)


def build_node_universe(segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Intersection nodes, derived from centerline endpoints shared by 2+ segments.

    Leg count stands in for the exposure term at intersections, since the HSM's normal
    exposure (entering vehicle volume) is not available in this slice. It is a weak
    proxy and the README says so.
    """
    assert_projected(segments, "segments")

    has_borough = "borough" in segments.columns
    attribute_cols = [c for c in (*ROAD_NUMERIC_COLUMNS, "is_highway") if c in segments.columns]

    records: list[dict[str, object]] = []
    for row in segments.itertuples():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            coords = [c for part in geom.geoms for c in (part.coords[0], part.coords[-1])]
        elif geom.geom_type == "LineString":
            coords = [geom.coords[0], geom.coords[-1]]
        else:
            continue
        # Round to the foot so float noise does not split one intersection into two.
        for x, y in coords:
            record: dict[str, object] = {"xy": (round(x, 0), round(y, 0))}
            if has_borough:
                record["borough"] = getattr(row, "borough", None)
            for column in attribute_cols:
                record[column] = getattr(row, column, None)
            records.append(record)

    if not records:
        raise SpatialJoinError("no segment endpoints found; cannot build nodes")

    endpoint_frame = pd.DataFrame.from_records(records)
    counts = endpoint_frame["xy"].value_counts()
    junctions = counts[counts >= 2]

    data: dict[str, object] = {
        "unit_id": [f"I{i}" for i in range(len(junctions))],
        "unit_type": "intersection",
        "leg_count": junctions.to_numpy(),
    }

    if has_borough:
        # A node inherits the borough of the streets meeting there. Without this,
        # every intersection would be borough-less and borough-stratified selection
        # would silently drop the entire intersection universe.
        modal = (
            endpoint_frame.dropna(subset=["borough"])
            .groupby("xy")["borough"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        )
        data["borough"] = [modal.get(xy) for xy in junctions.index]

    # An intersection has no geometry of its own, so its road characteristics are those
    # of the streets meeting there. The aggregations are chosen to describe the worst
    # approach rather than the average one: a junction where one leg is a 50mph highway
    # ramp is a highway junction, and averaging that away would hide exactly the site
    # type the founding finding is about.
    aggregations = {
        "posted_speed": "max",
        "streetwidth": "max",
        "number_travel_lanes": "sum",
        "is_highway": "max",
    }
    for column in attribute_cols:
        grouped = endpoint_frame.groupby("xy")[column].agg(aggregations[column])
        data[column] = [grouped.get(xy) for xy in junctions.index]

    nodes = gpd.GeoDataFrame(
        data, geometry=[Point(xy) for xy in junctions.index], crs=CRS_PROJECTED
    )
    log.info("built %d intersection nodes from %d segments", len(nodes), len(segments))
    return nodes


def build_universe(centerline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The full candidate universe: every corridor segment and every intersection.

    Both unit types are needed because DOT's priority list ranks them as separate
    universes with separate stopping rules (corridors to 50% of borough casualties,
    intersections to 15%).
    """
    segments = build_segment_universe(centerline)
    nodes = build_node_universe(segments)

    keep = ["unit_id", "unit_type", "geometry"]
    # `full_street_name` and `rw_type` are carried, not dropped, because both are load
    # bearing for this project's argument rather than decoration:
    #
    #   full_street_name  is the entire reason the unit of analysis is a named
    #                     centerline segment instead of an anonymous ~100m grid cell.
    #                     A ranked list a DOT budget-holder can act on says "Broadway
    #                     from W 135th to W 153rd", not "cell 31847".
    #   rw_type           distinguishes limited-access highway from surface street.
    #                     The founding finding of this project is that the rows with no
    #                     borough are highway rows and are deadlier, so a model that
    #                     cannot tell the two apart cannot check whether it repeats the
    #                     same blind spot.
    carried = (
        "length_ft",
        "degenerate_length",
        "borough",
        "full_street_name",
        "rw_type",
        *ROAD_NUMERIC_COLUMNS,
        "is_highway",
    )
    seg_cols = keep + [c for c in carried if c in segments]
    # borough has to survive onto nodes too, or borough-stratified selection silently
    # drops the entire intersection universe. The road characteristics ride along for
    # the same reason they do on segments: they are the SPF's predictors.
    node_cols = keep + [
        c for c in ("leg_count", "borough", *ROAD_NUMERIC_COLUMNS, "is_highway") if c in nodes
    ]

    universe = pd.concat(
        [segments[seg_cols], nodes[node_cols]], ignore_index=True, sort=False
    )
    universe = gpd.GeoDataFrame(universe, geometry="geometry", crs=CRS_PROJECTED)

    if universe["unit_id"].duplicated().any():
        raise SpatialJoinError("unit_id is not unique across the universe")

    assert_projected(universe, "universe")
    log.info(
        "universe: %d units (%d corridors, %d intersections)",
        len(universe),
        int((universe["unit_type"] == "corridor").sum()),
        int((universe["unit_type"] == "intersection").sum()),
    )
    return universe


# --------------------------------------------------------------------------------------
# Crashes
# --------------------------------------------------------------------------------------


def crashes_to_gdf(
    crashes: pd.DataFrame,
    report: AssignmentReport | None = None,
) -> tuple[gpd.GeoDataFrame, AssignmentReport]:
    """Turn raw crash rows into projected points, counting every exclusion.

    Three distinct exclusions, deliberately not collapsed into one "bad geometry"
    bucket: a crash with no coordinates is a reporting gap, a crash at (0, 0) is a
    known geocoder artifact, and a crash outside NYC is a data error. They have
    different causes and different implications for coverage.
    """
    report = report or AssignmentReport()
    report.total_input = len(crashes)

    frame = crashes.copy()
    frame["latitude"] = pd.to_numeric(frame.get("latitude"), errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.get("longitude"), errors="coerce")

    missing = frame["latitude"].isna() | frame["longitude"].isna()
    report.missing_coords = int(missing.sum())
    frame = frame[~missing]

    at_null_island = (frame["latitude"] == 0.0) & (frame["longitude"] == 0.0)
    report.null_island = int(at_null_island.sum())
    frame = frame[~at_null_island]

    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs=CRS_GEOGRAPHIC,
    )
    points = to_projected(points, "crashes")
    assert_projected(points, "crashes")

    minx, miny, maxx, maxy = NYC_BOUNDS_FT
    x, y = points.geometry.x, points.geometry.y
    in_bounds = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    report.outside_nyc = int((~in_bounds).sum())
    points = points[in_bounds].copy()

    log.info(
        "geocoded %d/%d crashes (missing %d, (0,0) %d, outside NYC %d)",
        len(points),
        report.total_input,
        report.missing_coords,
        report.null_island,
        report.outside_nyc,
    )
    return points, report


def _nearest_within(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    max_distance: float,
) -> pd.DataFrame:
    """`sjoin_nearest` with a deterministic tie-break.

    A crash exactly equidistant from two segments (dead centre of an intersection, a
    perfectly square block) yields two rows from `sjoin_nearest`. Left as-is that crash
    would be double-counted, which inflates both units. Ties break on `unit_id`
    ascending: arbitrary, but identical on every machine and every re-run.
    """
    assert_projected(left, "left frame")
    assert_projected(right, "right frame")

    joined = gpd.sjoin_nearest(
        left,
        right[["unit_id", "unit_type", "geometry"]],
        how="inner",
        max_distance=max_distance,
        distance_col="join_distance_ft",
    )
    joined = joined.reset_index(names="_left_idx")
    joined = joined.sort_values(["_left_idx", "join_distance_ft", "unit_id"])
    return joined.drop_duplicates(subset="_left_idx", keep="first")


def assign_crashes_to_units(
    crashes: gpd.GeoDataFrame,
    universe: gpd.GeoDataFrame,
    report: AssignmentReport,
) -> tuple[pd.DataFrame, AssignmentReport]:
    """Attach each crash to one unit: intersection first, then corridor.

    Two stages rather than one nearest-neighbour call against the whole universe. A
    crash in the middle of an intersection is physically nearest to the node, but a
    crash 60 ft up the block is nearest to the segment while still being
    intersection-related in every conventional classification. Stage one claims
    everything within `INTERSECTION_RADIUS_FT` of a node; stage two takes the rest.

    Uses `sjoin_nearest`, which is R-tree indexed. The naive `.apply()` over rows is
    O(n*m) across roughly 800k crashes and 120k segments and does not finish.
    """
    assert_projected(crashes, "crashes")
    assert_projected(universe, "universe")

    # The two-stage handoff identifies crashes by index label, so a non-unique index
    # silently collapses distinct crashes into one. Concatenating two snapshot parquet
    # files without ignore_index=True is enough to produce that, and the only thing
    # that would catch it is the accounting check at the end. Reset here so the
    # precondition is enforced rather than assumed.
    crashes = crashes.reset_index(drop=True)

    nodes = universe[universe["unit_type"] == "intersection"]
    segments = universe[universe["unit_type"] == "corridor"]

    if nodes.empty and segments.empty:
        raise SpatialJoinError("universe is empty; nothing to assign crashes to")

    parts: list[pd.DataFrame] = []
    remaining = crashes

    if not nodes.empty:
        at_nodes = _nearest_within(remaining, nodes, INTERSECTION_RADIUS_FT)
        report.assigned_intersection = len(at_nodes)
        parts.append(at_nodes)
        remaining = remaining.loc[~remaining.index.isin(at_nodes["_left_idx"])]

    if not segments.empty and not remaining.empty:
        at_segments = _nearest_within(remaining, segments, MAX_JOIN_DISTANCE_FT)
        report.assigned_corridor = len(at_segments)
        parts.append(at_segments)
        remaining = remaining.loc[~remaining.index.isin(at_segments["_left_idx"])]

    report.beyond_max_distance = len(remaining)
    report.validate()
    log.info(report.summary())

    if not parts:
        return pd.DataFrame(columns=["unit_id", "unit_type"]), report

    assigned = pd.concat(parts, ignore_index=True)
    return assigned.drop(columns=["_left_idx", "index_right"], errors="ignore"), report


# --------------------------------------------------------------------------------------
# Labels and treatment
# --------------------------------------------------------------------------------------


def join_vzv_labels(
    universe: gpd.GeoDataFrame,
    vzv_corridors: gpd.GeoDataFrame,
    vzv_intersections: gpd.GeoDataFrame,
    report: LabelJoinReport | None = None,
    buffer_ft: float = 50.0,
) -> tuple[gpd.GeoDataFrame, LabelJoinReport]:
    """Flag which units are on DOT's published priority list.

    Spatial, not by name: a VZV corridor is a segment of a street, so a name join would
    label the entire street and hand the priority list credit for crashes it never
    claimed.

    A VZV feature that matches zero units is *flagged, not dropped*. Zero matches means
    the geometry or the buffer is wrong, and silently returning a shorter priority list
    would quietly shrink the artifact this whole project is scored against.
    """
    report = report or LabelJoinReport()
    assert_projected(universe, "universe")

    out = universe.copy()
    out["is_priority"] = False
    out["vzv_source"] = pd.NA

    # A VZV priority corridor is a stretch of street *including its junctions*, so it
    # labels both segments and the nodes along it. Restricting corridor labels to
    # segments was wrong and badly so: crashes within 100ft of a junction are assigned
    # to the node, 86% of pedestrian casualties happen at intersections, and the nodes
    # along a priority corridor were left unlabelled. DOT's list was structurally
    # prevented from capturing the casualties it was selected for, which showed up on
    # the 2026-08-12 run as an implausible 11.9% capture rate.
    for label, vzv, unit_types in (
        ("corridor", vzv_corridors, ("corridor", "intersection")),
        ("intersection", vzv_intersections, ("intersection",)),
    ):
        if vzv is None or vzv.empty:
            log.warning("VZV %s layer is empty; no priority labels applied", label)
            continue

        vzv_p = to_projected(vzv, f"vzv_{label}")
        assert_projected(vzv_p, f"vzv_{label}")
        vzv_p = vzv_p.copy()
        vzv_p["_vzv_idx"] = range(len(vzv_p))
        vzv_p["geometry"] = vzv_p.geometry.buffer(buffer_ft)

        targets = out[out["unit_type"].isin(unit_types)]
        hits = gpd.sjoin(
            targets, vzv_p[["_vzv_idx", "geometry"]], how="inner", predicate="intersects"
        )

        matched_units = set(hits["unit_id"])
        out.loc[out["unit_id"].isin(matched_units), "is_priority"] = True
        out.loc[out["unit_id"].isin(matched_units), "vzv_source"] = label

        matched_features = hits["_vzv_idx"].nunique()
        unmatched = set(vzv_p["_vzv_idx"]) - set(hits["_vzv_idx"])
        if unmatched:
            log.warning(
                "%d VZV %s feature(s) matched zero units at a %.0f ft buffer. Flagged, "
                "not dropped - check geometry or widen the buffer.",
                len(unmatched),
                label,
                buffer_ft,
            )
            report.unmatched_keys.extend(f"{label}:{i}" for i in sorted(unmatched))

        if label == "corridor":
            report.vzv_corridors_in = len(vzv_p)
            report.vzv_corridors_matched = matched_features
        else:
            report.vzv_intersections_in = len(vzv_p)
            report.vzv_intersections_matched = matched_features

    log.info(
        "priority units: %d of %d", int(out["is_priority"].sum()), len(out)
    )
    return out, report


def join_sip_treatment(
    universe: gpd.GeoDataFrame,
    sip_layers: list[gpd.GeoDataFrame],
    # `end_date` is what the live SIP datasets (if4c-w48d, shr7-eqdc) actually use.
    # The other names were guesses made before the schema was inspected; they are kept
    # so the fixture tests and any older extract still resolve.
    date_column_candidates: tuple[str, ...] = (
        "end_date",
        "completion_date",
        "comp_date",
        "date_complete",
    ),
    report: LabelJoinReport | None = None,
    buffer_ft: float = 50.0,
) -> tuple[gpd.GeoDataFrame, LabelJoinReport]:
    """Tag which units received a Street Improvement Project, and when.

    This is the endogeneity control, not a nicety. VZV priority locations were selected
    *in order to* receive these projects, so their later casualty counts reflect the
    intervention. Without the split, a result showing DOT's locations underperforming is
    unreadable: it could mean the ranking was wrong, or it could mean the treatment
    worked.

    A SIP record with no completion date cannot place the treatment in time relative to
    the holdout window, so it is excluded and counted rather than assumed.
    """
    report = report or LabelJoinReport()
    assert_projected(universe, "universe")

    out = universe.copy()
    out["treated"] = False
    out["treatment_date"] = pd.NaT

    for i, sip in enumerate(sip_layers):
        if sip is None or sip.empty:
            continue

        report.sip_records_in += len(sip)
        sip_p = to_projected(sip, f"sip_{i}").copy()
        assert_projected(sip_p, f"sip_{i}")

        date_col = next((c for c in date_column_candidates if c in sip_p.columns), None)
        if date_col is None:
            raise SpatialJoinError(
                f"SIP layer {i} has no completion-date column (tried "
                f"{date_column_candidates}). Treatment timing cannot be established."
            )

        sip_p["_treatment_date"] = pd.to_datetime(sip_p[date_col], errors="coerce")
        undated = sip_p["_treatment_date"].isna()
        report.sip_missing_date += int(undated.sum())
        if undated.any():
            log.warning(
                "SIP layer %d: %d record(s) have no usable completion date; excluded",
                i,
                int(undated.sum()),
            )
        sip_p = sip_p[~undated]
        if sip_p.empty:
            continue

        sip_p["geometry"] = sip_p.geometry.buffer(buffer_ft)
        hits = gpd.sjoin(
            out, sip_p[["_treatment_date", "geometry"]], how="inner", predicate="intersects"
        )
        if hits.empty:
            continue

        # Earliest treatment wins: a unit rebuilt twice was first affected by the first
        # project, and that is the date the holdout comparison has to respect.
        earliest = hits.groupby("unit_id")["_treatment_date"].min()
        idx = out["unit_id"].map(earliest)
        newly = idx.notna()
        out.loc[newly, "treated"] = True
        out.loc[newly, "treatment_date"] = np.minimum(
            out.loc[newly, "treatment_date"].fillna(pd.Timestamp.max),
            idx[newly],
        )

    log.info(
        "treated units: %d of %d. %s",
        int(out["treated"].sum()),
        len(out),
        report.summary(),
    )
    return out, report
