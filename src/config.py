"""Single source of truth for dataset IDs, CRS, windows, and the pre-registered bar.

Every constant a reviewer would want to check lives here rather than inline in a
pipeline stage, so "what did this run actually use" is answerable by reading one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # gitignored: dated snapshots, re-pullable
CACHE_DIR = DATA_DIR / "cache"      # gitignored: expensive spatial joins
PROCESSED_DIR = DATA_DIR / "processed"  # committed: small aggregated intermediate

# --------------------------------------------------------------------------------------
# Socrata sources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SocrataSource:
    """One NYC Open Data resource.

    `where` is a SoQL filter applied server-side. Filtering at the API keeps the crash
    pull to the years we model instead of the full ~2.2M-row history.
    """

    key: str
    dataset_id: str
    description: str
    where: str | None = None

    @property
    def url(self) -> str:
        return f"https://data.cityofnewyork.us/resource/{self.dataset_id}.json"


# The crash label window. Trailing features need history before the training window
# opens, so the pull starts earlier than TRAIN_YEARS.
PULL_START = "2016-01-01T00:00:00.000"

SOURCES: dict[str, SocrataSource] = {
    "crashes": SocrataSource(
        key="crashes",
        dataset_id="h9gi-nx95",
        description="Motor Vehicle Collisions - Crashes",
        where=f"crash_date >= '{PULL_START}'",
    ),
    "vzv_corridors": SocrataSource(
        key="vzv_corridors",
        dataset_id="kdda-2wcy",
        description="Vision Zero priority corridors",
    ),
    "vzv_intersections": SocrataSource(
        key="vzv_intersections",
        dataset_id="2nj7-jxah",
        description="Vision Zero priority intersections",
    ),
    "sip_corridors": SocrataSource(
        key="sip_corridors",
        dataset_id="wqhs-q6wd",
        description="Street Improvement Projects - corridors (treatment)",
    ),
    "sip_intersections": SocrataSource(
        key="sip_intersections",
        dataset_id="79sh-heg3",
        description="Street Improvement Projects - intersections (treatment)",
    ),
}

# Centerline source is NOT yet pinned. Candidates are 3mf9-qshr, inkn-q76z, and DCP
# LION (a Bytes download, not Socrata). Whichever is selected gets added to SOURCES
# and recorded in the README with its vintage. Until then the pipeline raises rather
# than guessing, so a placeholder can never silently become the published universe.
CENTERLINE_CANDIDATES = ("3mf9-qshr", "inkn-q76z", "LION (DCP Bytes)")
CENTERLINE_SOURCE: SocrataSource | None = None

# --------------------------------------------------------------------------------------
# Coordinate reference systems
# --------------------------------------------------------------------------------------

# Socrata delivers lat/long in WGS84.
CRS_GEOGRAPHIC = "EPSG:4326"

# NAD83 / New York Long Island (ftUS). Units are FEET. Every distance, buffer, and
# length in this project is computed here and nowhere else. Measuring in EPSG:4326
# degrees while intending feet is wrong by roughly 364,000x and raises nothing, which
# is why `src.spatial.assert_projected` gates every geometric operation.
CRS_PROJECTED = "EPSG:2263"
PROJECTED_UNIT = "us-ft"

# A crash farther than this from any centerline segment is not assigned. NYC crash
# geocoding lands on the street or the nearest intersection; 150 ft absorbs that
# jitter without letting a crash jump to a parallel street mid-block.
MAX_JOIN_DISTANCE_FT = 150.0

# Coordinates NYC's feed emits for "no geocode." Real (0, 0) is in the Gulf of Guinea.
NULL_ISLAND = (0.0, 0.0)

# Generous bounding box around the five boroughs, in CRS_PROJECTED feet.
NYC_BOUNDS_FT = (900_000.0, 110_000.0, 1_090_000.0, 280_000.0)

# --------------------------------------------------------------------------------------
# Modeling windows
# --------------------------------------------------------------------------------------

TRAIN_YEARS = (2019, 2023)    # inclusive, SPF is fit here
HOLDOUT_YEARS = (2024, 2025)  # inclusive, every ranking is scored here

# Trailing feature window, in months, ending at the training window's close.
TRAILING_MONTHS = (12, 24, 36)

# --------------------------------------------------------------------------------------
# DOT's published selection rule
# --------------------------------------------------------------------------------------

# DOT ranks within borough and stops at a cumulative share of that borough's KSI.
# Regime A reproduces this rule exactly so the model is judged on DOT's own terms.
BOROUGH_CUMULATIVE_SHARE = {
    "corridor": 0.50,
    "intersection": 0.15,
}

BOROUGHS = ("MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND")

# --------------------------------------------------------------------------------------
# The pre-registered bar (README: "The bar, set in advance")
# --------------------------------------------------------------------------------------

# Empirical Bayes beats raw trailing count only if BOTH hold on the holdout:
#   1. capture-rate difference >= MIN_CAPTURE_RATE_LIFT_PP percentage points, and
#   2. the bootstrap CI on that difference excludes zero.
# Changing either number after seeing a result invalidates the pre-registration. If a
# run needs different values, say so explicitly in the README next to the result.
MIN_CAPTURE_RATE_LIFT_PP = 5.0
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 20260812

# --------------------------------------------------------------------------------------
# Empirical Bayes guards
# --------------------------------------------------------------------------------------

# w = 1 / (1 + k * P) is only meaningful for k > 0. A non-positive dispersion means the
# NB fit collapsed toward Poisson or failed; blending on it would silently produce a
# ranking that is just the SPF prediction.
MIN_DISPERSION = 1e-8

# A zero-length segment makes log(length) negative infinity. Segments at or below this
# are excluded from fitting and counted in the exclusion report.
MIN_SEGMENT_LENGTH_FT = 1.0
