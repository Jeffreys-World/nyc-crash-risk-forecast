"""Single source of truth for dataset IDs, CRS, windows, and the pre-registered bar.

Every constant a reviewer would want to check lives here rather than inline in a
pipeline stage, so "what did this run actually use" is answerable by reading one file.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # gitignored: dated snapshots, re-pullable
CACHE_DIR = DATA_DIR / "cache"      # gitignored: expensive spatial joins
PROCESSED_DIR = DATA_DIR / "processed"  # committed: small aggregated intermediate

# --------------------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------------------

# Loaded from the repo root rather than the caller's working directory, so the token
# resolves the same whether the pull is run from the repo root, from scripts/, or by
# pytest. `override=False` means a real environment variable always wins over the file,
# which is what CI and any future deploy will set.
load_dotenv(REPO_ROOT / ".env", override=False)

# Socrata app token. Raises the NYC Open Data rate limit; anonymous requests are
# throttled hard and this project walks roughly 800k crash rows.
#
# This is a rate-limit identifier, not an authentication credential: it grants no write
# access and no access to anything not already public. Socrata's own guidance permits
# embedding app tokens in client-side code. It is kept in a gitignored `.env` anyway,
# because a token in git history is a token you cannot quietly rotate.
def _clean_token(raw: str | None) -> str | None:
    """Normalise a token to a usable value or None.

    Whitespace is stripped and a blank value becomes None. This is not cosmetic: a
    whitespace-only token makes `requests` raise InvalidHeader before the request is
    even sent, so the pull dies on a traceback about header validity rather than
    saying the token is missing. Pasting a token into `.env` and picking up a stray
    space is an ordinary slip.
    """
    if raw is None:
        return None
    return raw.strip() or None


SOCRATA_APP_TOKEN: str | None = _clean_token(os.environ.get("SOCRATA_APP_TOKEN"))

# --------------------------------------------------------------------------------------
# Socrata sources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SocrataSource:
    """One NYC Open Data resource.

    `where` is a SoQL filter applied server-side. Filtering at the API keeps the crash
    pull to the years we model instead of the full 2.27M-row history.

    `select` narrows the columns server-side. The crash dataset has ~30 columns and
    1.5M matching rows; pulling only the eight the model uses cuts the payload by
    roughly three quarters and is the difference between a pull that finishes and one
    that times out.

    `is_geo` marks resources whose rows carry a `the_geom` GeoJSON object. Those are
    converted to WKT at snapshot time so the parquet stays a plain table.
    """

    key: str
    dataset_id: str
    description: str
    where: str | None = None
    select: str | None = None
    is_geo: bool = False

    @property
    def url(self) -> str:
        return f"https://data.cityofnewyork.us/resource/{self.dataset_id}.json"


# The crash label window. Trailing features need history before the training window
# opens, so the pull starts earlier than TRAIN_YEARS.
PULL_START = "2016-01-01T00:00:00.000"

CRASH_COLUMNS = (
    "crash_date",
    "latitude",
    "longitude",
    "borough",
    "number_of_pedestrians_killed",
    "number_of_pedestrians_injured",
    "contributing_factor_vehicle_1",
    "contributing_factor_vehicle_2",
)

# The four VZV/SIP resources below were originally pinned to their `map` IDs
# (kdda-2wcy, 2nj7-jxah, wqhs-q6wd, 79sh-heg3). Those exist and report row counts, but
# they are visualization canvases: `displayType: visualization_canvas_map`, zero
# API-accessible columns, and every row returns `{}`. Each has a real `dataset` twin,
# pinned here. Verifying that a dataset *exists* is not the same as verifying its data
# can be read.
SOURCES: dict[str, SocrataSource] = {
    "crashes": SocrataSource(
        key="crashes",
        dataset_id="h9gi-nx95",
        description="Motor Vehicle Collisions - Crashes",
        where=f"crash_date >= '{PULL_START}'",
        select=", ".join(CRASH_COLUMNS),
    ),
    "vzv_corridors": SocrataSource(
        key="vzv_corridors",
        dataset_id="36nr-7fbp",
        description="VZV Priority Corridors (199 segments, geometry only)",
        is_geo=True,
    ),
    "vzv_intersections": SocrataSource(
        key="vzv_intersections",
        dataset_id="tmt9-43em",
        description="VZV Priority Intersections (304 points)",
        is_geo=True,
    ),
    "sip_corridors": SocrataSource(
        key="sip_corridors",
        dataset_id="if4c-w48d",
        description="VZV Street Improvement Projects - corridors (treatment)",
        is_geo=True,
    ),
    "sip_intersections": SocrataSource(
        key="sip_intersections",
        dataset_id="shr7-eqdc",
        description="VZV Street Improvement Projects - intersections (treatment)",
        is_geo=True,
    ),
}

# Pinned 2026-08-12 after schema inspection. `inkn-q76z` is the NYC DCP/DoITT street
# Centerline: 122,244 segments, MultiLineString geometry, and crucially it carries
# `boroughcode` and a precomputed `segmentlength` in feet, so borough and the exposure
# term both come free rather than needing a separate spatial join.
#
# `3mf9-qshr` is the same Centerline data with zero API-accessible columns, the same
# trap as the VZV map views. DCP LION was not needed once inkn-q76z proved sufficient,
# and avoiding it keeps the pipeline to a single API rather than a manual Bytes
# download that no reader could reproduce from a clone.
CENTERLINE_CANDIDATES = ("3mf9-qshr", "inkn-q76z", "LION (DCP Bytes)")

CENTERLINE_SOURCE: SocrataSource | None = SocrataSource(
    key="centerline",
    dataset_id="inkn-q76z",
    description="NYC street Centerline (122,244 segments)",
    is_geo=True,
)

# rw_type distinguishes road class. Type 1 is a normal street; the higher codes cover
# highways, ramps, and bridges. This is the limited-access-highway signal the original
# borough-gap finding turned on, so it is carried through as a feature rather than
# discarded.
ROADWAY_TYPE_STREET = "1"

# Road characteristics used as SPF predictors.
#
# These describe the SITE, not its crash history, and that distinction is the whole
# point. An HSM Safety Performance Function predicts expected crashes from geometry and
# exposure; the observed crash record enters separately, through the Empirical Bayes
# blend. Feeding crash-derived features (night share, contributing-factor mix) into the
# SPF double-counts the very history EB exists to weigh, and it produced a degenerate
# fit on the 2026-08-12 run: intercept -29.5, night_share +21.9, and 74% of the city
# predicted at exp(-30). Those features were really encoding "did this unit ever have a
# crash", because they are 0.0 both for a unit with no crashes and for a unit whose
# crashes were all in daylight.
SPF_PREDICTORS = (
    "posted_speed",
    "number_travel_lanes",
    "streetwidth",
    "is_highway",
    "road_attrs_imputed",
)

# Raw numeric columns lifted from the centerline. Socrata serves everything as strings.
ROAD_NUMERIC_COLUMNS = ("posted_speed", "number_travel_lanes", "streetwidth")

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

# A crash within this radius of an intersection node is treated as intersection-related
# and assigned to the node rather than to the adjoining segment. This mirrors how crash
# records are conventionally classified, and it matters because DOT's priority list
# ranks corridors and intersections as separate universes with separate stopping rules.
#
# Of the three distances here, this is the one the 2026-08-13 sweep found the headline
# actually turns on: 50 / 100 / 150 ft give a lift of +16.1 / +18.4 / +19.9pp, roughly
# 2pp per 50 ft, with nothing else moving - same N, same crashes assigned, same
# capture-rate denominator. Only which unit each crash lands on changes. Anything that
# revisits this value should re-run `scripts/radius_sensitivity.py` and say what moved.
INTERSECTION_RADIUS_FT = 100.0

# How far a VZV priority feature reaches when deciding which units are on DOT's list.
# The VZV geometry and the centerline are two independent renderings of the same street,
# so they do not lie exactly on top of each other; the buffer absorbs that offset.
#
# It sets `is_priority`, which is both R1's footprint *and* N, the size of every
# ranking's selection - so widening it moves all three capture rates at once. The sweep
# showed that is exactly what happens and no more: 25 / 50 / 100 ft put 35,461 / 38,909 /
# 43,111 units on the list, and the lift at each tracks the published N-sweep. The
# labelling is not separately sensitive; N is.
VZV_BUFFER_FT = 50.0

# The same idea for Street Improvement Projects. Kept separate because it only decides
# the treated/untreated split, never the headline.
SIP_BUFFER_FT = 50.0


@dataclass(frozen=True)
class JoinRadii:
    """The three distances that decide what lands where.

    Bundled rather than read from module scope at each call site so a sensitivity run
    can vary them, and — the part that matters — so the *effective* values can reach the
    units cache key. Keyed on file contents alone, an override applied in memory leaves
    the fingerprint unchanged and the run silently reuses units built at a different
    radius, which is the stale-cache correctness bug this project already fixed once.
    """

    max_join_distance_ft: float = MAX_JOIN_DISTANCE_FT
    intersection_radius_ft: float = INTERSECTION_RADIUS_FT
    vzv_buffer_ft: float = VZV_BUFFER_FT

    @property
    def tag(self) -> str:
        """Compact, readable cache-key fragment: `j150-i100-v50`.

        Readable rather than hashed on purpose. A sweep leaves one parquet per setting
        in `data/cache/`, and being able to tell them apart by looking is worth more
        than six characters of path.
        """
        def _n(value: float) -> str:
            return f"{value:g}"

        return (
            f"j{_n(self.max_join_distance_ft)}"
            f"-i{_n(self.intersection_radius_ft)}"
            f"-v{_n(self.vzv_buffer_ft)}"
        )

    def as_dict(self) -> dict[str, float]:
        """All fields, derived rather than hand-listed.

        This is the round trip in both directions - the sensitivity sweep builds each
        setting with `JoinRadii(**{**DEFAULT_RADII.as_dict(), field: value})`, and
        `RunSummary.join_radii` is what a run records about itself. Spelled out by hand,
        a fourth radius added here would silently drop out of both: the sweep would hold
        it at its default while claiming to vary everything, and the summary would omit
        it from the provenance it exists to carry.
        """
        return asdict(self)


DEFAULT_RADII = JoinRadii()

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

# Centerline encodes borough as a numeric code.
BOROUGH_CODES = {
    "1": "MANHATTAN",
    "2": "BRONX",
    "3": "BROOKLYN",
    "4": "QUEENS",
    "5": "STATEN ISLAND",
}

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
