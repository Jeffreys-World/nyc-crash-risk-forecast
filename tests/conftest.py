"""A miniature synthetic city, built so every edge case has a known right answer.

Real NYC data cannot test the edge cases that matter here: you cannot ask the live feed
for a crash exactly equidistant from two segments, or a borough with zero holdout
casualties. The fixture is a 3x3 grid of streets at real Manhattan coordinates, so
reprojection into EPSG:2263 lands inside NYC bounds and the CRS guards see genuine
values rather than toy numbers.

Grid geometry (WGS84):
    longitudes -73.980, -73.979, -73.978   (~277 ft apart at this latitude)
    latitudes   40.750,  40.751,  40.752   (~365 ft apart)

That yields 12 segments and 9 intersection nodes.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.config import CRS_GEOGRAPHIC, CRS_PROJECTED

LONS = [-73.980, -73.979, -73.978]
LATS = [40.750, 40.751, 40.752]

# A second cluster far from the grid, used to test the max-join-distance bucket.
FAR_LON, FAR_LAT = -73.900, 40.700


@pytest.fixture
def centerline() -> gpd.GeoDataFrame:
    """12 street segments forming a 3x3 grid, in WGS84 with a declared CRS."""
    rows = []
    for lat in LATS:  # east-west streets
        for i in range(len(LONS) - 1):
            rows.append(
                {
                    "street": f"E-W {lat}",
                    "borough": "MANHATTAN",
                    "geometry": LineString([(LONS[i], lat), (LONS[i + 1], lat)]),
                }
            )
    for lon in LONS:  # north-south avenues
        for j in range(len(LATS) - 1):
            rows.append(
                {
                    "street": f"N-S {lon}",
                    "borough": "MANHATTAN",
                    "geometry": LineString([(lon, LATS[j]), (lon, LATS[j + 1])]),
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_GEOGRAPHIC)


@pytest.fixture
def two_borough_centerline(centerline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The same grid with the northern row relabelled, for stratified-selection tests."""
    out = centerline.copy()
    out.loc[out["street"].str.contains("40.752"), "borough"] = "BROOKLYN"
    return out


def make_crash(
    lon: float | None,
    lat: float | None,
    date: str,
    killed: int = 0,
    injured: int = 1,
    factor: str = "Unspecified",
) -> dict:
    return {
        "crash_date": date,
        "latitude": lat,
        "longitude": lon,
        "number_of_pedestrians_killed": killed,
        "number_of_pedestrians_injured": injured,
        "contributing_factor_vehicle_1": factor,
        "contributing_factor_vehicle_2": None,
    }


@pytest.fixture
def crashes() -> pd.DataFrame:
    """A crash set that exercises every exclusion bucket exactly once.

    Composition, by design:
      3 clean crashes on the grid
      1 with null coordinates
      1 at (0, 0)
      1 outside NYC bounds
      1 beyond the max join distance from any segment
    """
    rows = [
        make_crash(-73.9795, 40.7500, "2022-03-01T08:00:00.000", injured=2),
        make_crash(-73.9790, 40.7510, "2022-06-15T19:30:00.000", killed=1, injured=0),
        make_crash(-73.9785, 40.7520, "2023-01-20T23:15:00.000", injured=1),
        make_crash(None, None, "2022-04-01T12:00:00.000"),
        make_crash(0.0, 0.0, "2022-05-01T12:00:00.000"),
        make_crash(-118.24, 34.05, "2022-07-01T12:00:00.000"),  # Los Angeles
        make_crash(FAR_LON, FAR_LAT, "2022-08-01T12:00:00.000"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def vzv_corridors() -> gpd.GeoDataFrame:
    """One corridor overlapping the southern E-W street, one matching nothing."""
    return gpd.GeoDataFrame(
        [
            {
                "name": "on-grid",
                "geometry": LineString([(LONS[0], LATS[0]), (LONS[1], LATS[0])]),
            },
            {
                "name": "orphan",
                "geometry": LineString([(FAR_LON, FAR_LAT), (FAR_LON + 0.001, FAR_LAT)]),
            },
        ],
        geometry="geometry",
        crs=CRS_GEOGRAPHIC,
    )


@pytest.fixture
def vzv_intersections() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [{"name": "grid-corner", "geometry": Point(LONS[0], LATS[0])}],
        geometry="geometry",
        crs=CRS_GEOGRAPHIC,
    )


@pytest.fixture
def sip_layer() -> gpd.GeoDataFrame:
    """One dated project on the grid, one with no date at all."""
    return gpd.GeoDataFrame(
        [
            {
                "completion_date": "2021-06-30",
                "geometry": LineString([(LONS[0], LATS[0]), (LONS[1], LATS[0])]),
            },
            {
                "completion_date": None,
                "geometry": LineString([(LONS[1], LATS[1]), (LONS[2], LATS[1])]),
            },
        ],
        geometry="geometry",
        crs=CRS_GEOGRAPHIC,
    )


@pytest.fixture
def universe(centerline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    from src.spatial import build_universe

    return build_universe(centerline)


@pytest.fixture
def modelable_units() -> pd.DataFrame:
    """A unit table large and varied enough for a negative binomial to converge.

    Counts are drawn from an actual NB process rather than hand-written, because a
    hand-written table tends to be underdispersed and the fit then legitimately
    complains that there is no overdispersion to model.
    """
    rng = np.random.default_rng(20260812)
    n = 400
    length = rng.uniform(200, 2000, size=n)
    night = rng.uniform(0, 1, size=n)

    mu = np.exp(-8.0 + 0.15 * night) * length
    counts = rng.negative_binomial(n=2.0, p=2.0 / (2.0 + mu))

    return pd.DataFrame(
        {
            "unit_id": [f"C{i}" for i in range(n)],
            "unit_type": "corridor",
            "borough": rng.choice(["MANHATTAN", "BROOKLYN"], size=n),
            "length_ft": length,
            "night_share": night,
            "casualties_36mo": counts,
            "holdout_casualties": rng.poisson(np.maximum(mu, 0.01)),
        }
    )


@pytest.fixture
def projected_points() -> gpd.GeoDataFrame:
    """Two points already in EPSG:2263, for CRS-guard tests."""
    return gpd.GeoDataFrame(
        {"unit_id": ["A", "B"]},
        geometry=[Point(990_000, 200_000), Point(990_100, 200_000)],
        crs=CRS_PROJECTED,
    )
