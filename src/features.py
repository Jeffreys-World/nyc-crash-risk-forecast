"""Unit-level features for the Safety Performance Function.

The label is **pedestrian casualties** = killed + injured. It is not KSI. The public
crash dataset carries no injury-severity field at all, so DOT's "killed and severely
injured" target is not reproducible from public data. Calling this KSI anywhere would
be an overclaim, so the column is named for what it actually counts.

The silent failure guarded here is the `log(0)` exposure offset. A zero-length segment
makes `log(length)` negative infinity, and statsmodels does not reject it at call time:
the fit either diverges after many iterations or returns parameters that look plausible
and are meaningless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import MIN_SEGMENT_LENGTH_FT, ROAD_NUMERIC_COLUMNS, TRAILING_MONTHS

log = logging.getLogger(__name__)

PED_KILLED = "number_of_pedestrians_killed"
PED_INJURED = "number_of_pedestrians_injured"
CRASH_DATE = "crash_date"

FACTOR_COLUMNS = (
    "contributing_factor_vehicle_1",
    "contributing_factor_vehicle_2",
)

# "Unspecified" is the single most common contributing factor in this dataset. It is
# kept as a real, informative sparse category rather than dropped as noise: a corridor
# whose crashes are overwhelmingly unexplained is a different kind of place than one
# with a consistent, named failure.
TRACKED_FACTORS = (
    "Driver Inattention/Distraction",
    "Failure to Yield Right-of-Way",
    "Unsafe Speed",
    "Traffic Control Disregarded",
    "Unspecified",
)


class FeatureError(ValueError):
    """A feature could not be computed in a way that is safe to model on."""


@dataclass
class FeatureReport:
    """What the feature build had to exclude or fall back on."""

    units_in: int = 0
    excluded_degenerate_exposure: int = 0
    exposure_fallbacks: int = 0
    window_truncated: bool = False
    earliest_crash: pd.Timestamp | None = None
    requested_window_start: pd.Timestamp | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"units {self.units_in}",
            f"excluded for degenerate exposure {self.excluded_degenerate_exposure}",
            f"exposure fallbacks {self.exposure_fallbacks}",
        ]
        if self.window_truncated:
            parts.append(
                f"trailing window truncated at data start {self.earliest_crash!s} "
                f"(requested {self.requested_window_start!s})"
            )
        return "; ".join(parts)


# --------------------------------------------------------------------------------------
# Label
# --------------------------------------------------------------------------------------


def add_pedestrian_casualties(crashes: pd.DataFrame) -> pd.DataFrame:
    """killed + injured pedestrians, per crash. Never called KSI."""
    out = crashes.copy()
    for col in (PED_KILLED, PED_INJURED):
        if col not in out.columns:
            raise FeatureError(f"crash frame is missing {col}; cannot build the label")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out["pedestrian_casualties"] = out[PED_KILLED] + out[PED_INJURED]
    out[CRASH_DATE] = pd.to_datetime(out[CRASH_DATE], errors="coerce")

    undated = out[CRASH_DATE].isna()
    if undated.any():
        log.warning("%d crash(es) have no parseable date; excluded", int(undated.sum()))
        out = out[~undated]

    return out


# --------------------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------------------


def exposure_term(
    units: pd.DataFrame,
    report: FeatureReport | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Attach the log-exposure offset, refusing to hand `log(0)` to the fit.

    Corridors use segment length in feet. Intersections use leg count, which is a weak
    proxy for the HSM's usual exposure (entering vehicle volume) and is documented as
    such in the README. Traffic volume is the P2 upgrade in TODOS.md.

    Units whose exposure is non-positive are marked `exposure_valid=False` rather than
    dropped here, so the capture-rate denominator downstream still sees the full
    universe and the exclusion is visible in the report.
    """
    report = report or FeatureReport()
    out = units.copy()
    report.units_in = len(out)

    exposure = pd.Series(np.nan, index=out.index, dtype=float)

    is_corridor = out["unit_type"] == "corridor"
    if "length_ft" in out.columns:
        exposure[is_corridor] = pd.to_numeric(
            out.loc[is_corridor, "length_ft"], errors="coerce"
        )

    is_intersection = out["unit_type"] == "intersection"
    if "leg_count" in out.columns:
        exposure[is_intersection] = pd.to_numeric(
            out.loc[is_intersection, "leg_count"], errors="coerce"
        )

    missing = exposure.isna()
    if missing.any():
        # Falling back is recorded, never assumed. A unit modelled on a made-up
        # exposure is a unit whose prediction cannot be defended.
        report.exposure_fallbacks = int(missing.sum())
        report.notes.append(
            f"{int(missing.sum())} unit(s) had no exposure value and were marked invalid"
        )
        log.warning("%d unit(s) have no exposure value", int(missing.sum()))

    out["exposure"] = exposure
    out["exposure_valid"] = exposure.notna() & (exposure >= MIN_SEGMENT_LENGTH_FT)

    invalid = int((~out["exposure_valid"]).sum())
    report.excluded_degenerate_exposure = invalid
    if invalid:
        log.warning(
            "%d unit(s) have exposure below %.1f. Excluded from fitting to avoid a "
            "log(0) offset; they remain in the universe for scoring.",
            invalid,
            MIN_SEGMENT_LENGTH_FT,
        )

    # Only computed where valid. NaN here is intentional and checked before fitting.
    out["log_exposure"] = np.where(
        out["exposure_valid"], np.log(out["exposure"].where(out["exposure_valid"])), np.nan
    )

    if np.isinf(out["log_exposure"]).any():
        raise FeatureError(
            "log_exposure contains an infinite value. This is the log(0) failure the "
            "exposure guard exists to prevent."
        )

    return out, report


# --------------------------------------------------------------------------------------
# Trailing counts
# --------------------------------------------------------------------------------------


def impute_road_attributes(
    units: pd.DataFrame,
    columns: tuple[str, ...] = ROAD_NUMERIC_COLUMNS,
    report: FeatureReport | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Median-impute missing road characteristics, and flag that it happened.

    17.6% of centerline segments carry no posted speed, and 7% no lane count or width.
    Letting those rows fall out of the fit would repeat, inside this model, the exact
    failure the project was built to expose: NYC's borough bar chart drops 32% of
    crashes because one field is blank, and the dropped rows are the deadlier ones.

    So the value is imputed rather than dropped, and `road_attrs_imputed` carries the
    fact into the model as its own predictor. If imputed sites behave differently, the
    coefficient says so out loud instead of the difference vanishing with the rows.
    Median rather than mean because these are bounded, lumpy quantities (speeds cluster
    at 25, lanes at 2) where a mean invents values that no street has.
    """
    report = report or FeatureReport()
    out = units.copy()

    present = [c for c in columns if c in out.columns]
    if not present:
        out["road_attrs_imputed"] = 0
        return out, report

    missing_any = out[present].isna().any(axis=1)
    out["road_attrs_imputed"] = missing_any.astype(int)

    for column in present:
        values = pd.to_numeric(out[column], errors="coerce")
        n_missing = int(values.isna().sum())
        if not n_missing:
            out[column] = values
            continue

        # Within unit_type: a corridor's typical width is not an intersection's.
        filled = values.fillna(values.groupby(out["unit_type"]).transform("median"))
        filled = filled.fillna(values.median())
        out[column] = filled

        report.notes.append(f"{column}: imputed {n_missing} missing value(s) with median")
        log.info("%s: imputed %d missing value(s)", column, n_missing)

    still_missing = int(out[present].isna().any(axis=1).sum())
    if still_missing:
        raise FeatureError(
            f"{still_missing} unit(s) still have a missing road attribute after "
            f"imputation. The median itself was undefined, which means the column is "
            f"empty rather than sparse."
        )

    return out, report


def trailing_casualties(
    assigned: pd.DataFrame,
    units: pd.DataFrame,
    as_of: pd.Timestamp,
    months: tuple[int, ...] = TRAILING_MONTHS,
    report: FeatureReport | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Casualty and crash counts in trailing windows ending at `as_of`.

    A unit with no crashes in a window gets 0, not NaN. That distinction decides whether
    the SPF sees a quiet street as quiet or as unknown, and NaN would silently drop the
    row from the fit.

    A window reaching back past the first crash record is truncated to the data start
    and flagged. Left unflagged, an under-filled window looks like a genuinely quiet
    location, which is exactly the regression-to-the-mean artifact this project exists
    to correct.
    """
    report = report or FeatureReport()
    out = units.copy()

    if assigned.empty:
        for m in months:
            out[f"casualties_{m}mo"] = 0
            out[f"crashes_{m}mo"] = 0
        return out, report

    frame = assigned.copy()
    frame[CRASH_DATE] = pd.to_datetime(frame[CRASH_DATE], errors="coerce")
    frame = frame[frame[CRASH_DATE].notna()]

    earliest = frame[CRASH_DATE].min()
    report.earliest_crash = earliest

    for m in months:
        start = as_of - pd.DateOffset(months=m)
        if start < earliest:
            report.window_truncated = True
            report.requested_window_start = start
            report.notes.append(
                f"{m}mo window starts {start.date()} but data starts "
                f"{earliest.date()}; window is under-filled"
            )
            log.warning(
                "%dmo trailing window starts %s, before data start %s - truncated",
                m,
                start.date(),
                earliest.date(),
            )

        window = frame[(frame[CRASH_DATE] > start) & (frame[CRASH_DATE] <= as_of)]
        grouped = window.groupby("unit_id").agg(
            casualties=("pedestrian_casualties", "sum"),
            crashes=("pedestrian_casualties", "size"),
        )

        # fillna(0) is the "quiet, not unknown" decision made explicit.
        out[f"casualties_{m}mo"] = (
            out["unit_id"].map(grouped["casualties"]).fillna(0).astype(int)
        )
        out[f"crashes_{m}mo"] = (
            out["unit_id"].map(grouped["crashes"]).fillna(0).astype(int)
        )

    return out, report


# --------------------------------------------------------------------------------------
# Mix features
# --------------------------------------------------------------------------------------


def factor_mix(
    assigned: pd.DataFrame,
    units: pd.DataFrame,
    as_of: pd.Timestamp,
    months: int = 36,
) -> pd.DataFrame:
    """Share of a unit's factor *mentions* attributed to each tracked factor.

    The denominator is non-null factor mentions, not crashes. A crash naming two
    contributing factors contributes two mentions, so a single crash with "Unsafe
    Speed" and "Unspecified" yields 0.5 for each rather than 1.0. That is the intended
    behaviour for a mix feature, but it is not the same as a share of crashes, and
    reading it as one would overstate how dominant any single factor is.

    Shares, not counts, so the feature describes the character of a location rather
    than restating its volume. Volume is already carried by the trailing counts, and
    letting it in twice would give it two votes in the fit.
    """
    out = units.copy()
    factor_cols = [c for c in FACTOR_COLUMNS if c in assigned.columns]

    if assigned.empty or not factor_cols:
        for factor in TRACKED_FACTORS:
            out[f"factor_{_slug(factor)}"] = 0.0
        return out

    frame = assigned.copy()
    frame[CRASH_DATE] = pd.to_datetime(frame[CRASH_DATE], errors="coerce")
    start = as_of - pd.DateOffset(months=months)
    frame = frame[(frame[CRASH_DATE] > start) & (frame[CRASH_DATE] <= as_of)]

    long = frame.melt(
        id_vars=["unit_id"], value_vars=factor_cols, value_name="factor"
    ).dropna(subset=["factor"])

    totals = long.groupby("unit_id").size()
    for factor in TRACKED_FACTORS:
        hits = long[long["factor"] == factor].groupby("unit_id").size()
        share = (hits / totals).replace([np.inf, -np.inf], np.nan)
        out[f"factor_{_slug(factor)}"] = out["unit_id"].map(share).fillna(0.0)

    return out


def temporal_concentration(
    assigned: pd.DataFrame,
    units: pd.DataFrame,
    as_of: pd.Timestamp,
    months: int = 36,
) -> pd.DataFrame:
    """Night and weekend shares of a unit's crashes.

    Two locations with identical counts behave differently if one's crashes cluster
    after dark. Night is 18:00-05:59, which brackets the evening peak and the
    low-visibility overnight window.
    """
    out = units.copy()

    if assigned.empty or CRASH_DATE not in assigned.columns:
        out["night_share"] = 0.0
        out["weekend_share"] = 0.0
        return out

    frame = assigned.copy()
    frame[CRASH_DATE] = pd.to_datetime(frame[CRASH_DATE], errors="coerce")
    start = as_of - pd.DateOffset(months=months)
    frame = frame[(frame[CRASH_DATE] > start) & (frame[CRASH_DATE] <= as_of)]

    if frame.empty:
        out["night_share"] = 0.0
        out["weekend_share"] = 0.0
        return out

    hour = frame[CRASH_DATE].dt.hour
    frame["is_night"] = (hour >= 18) | (hour < 6)
    frame["is_weekend"] = frame[CRASH_DATE].dt.dayofweek >= 5

    grouped = frame.groupby("unit_id").agg(
        night_share=("is_night", "mean"), weekend_share=("is_weekend", "mean")
    )
    out["night_share"] = out["unit_id"].map(grouped["night_share"]).fillna(0.0)
    out["weekend_share"] = out["unit_id"].map(grouped["weekend_share"]).fillna(0.0)
    return out


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def build_features(
    assigned: pd.DataFrame,
    units: pd.DataFrame,
    as_of: pd.Timestamp,
    months: tuple[int, ...] = TRAILING_MONTHS,
) -> tuple[pd.DataFrame, FeatureReport]:
    """The full unit-level feature table as of a point in time.

    `as_of` exists so the training features can be built without any knowledge of the
    holdout years. A random split, or a feature computed over the whole history, would
    leak future crashes into training and make the forecast claim meaningless.
    """
    report = FeatureReport()

    features, report = exposure_term(units, report)
    features, report = impute_road_attributes(features, report=report)
    features, report = trailing_casualties(assigned, features, as_of, months, report)
    features = factor_mix(assigned, features, as_of)
    features = temporal_concentration(assigned, features, as_of)

    features["as_of"] = as_of
    log.info("features built as of %s: %s", as_of.date(), report.summary())
    return features, report


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
