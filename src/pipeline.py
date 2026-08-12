"""End-to-end run: snapshots -> universe -> features -> SPF -> EB -> backtest.

    python -m src.pipeline

Reads only dated snapshots, never the API. Every expensive stage caches to parquet so
a re-run is fast and a reader can inspect any intermediate.

The output is the number the README quotes, and it is printed whether or not it is
flattering. The bar it has to clear was fixed before this file could produce anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import from_wkt

from src.backtest import (
    Selection,
    apply_preregistered_bar,
    bootstrap_capture_difference,
    capture_rate,
    select_borough_stratified,
    select_citywide_top_n,
    split_by_treatment,
)
from src.config import (
    BOROUGH_CODES,
    CACHE_DIR,
    CRS_GEOGRAPHIC,
    HOLDOUT_YEARS,
    PROCESSED_DIR,
    RAW_DIR,
    SPF_PREDICTORS,
    TRAIN_YEARS,
)
from src.features import add_pedestrian_casualties, build_features
from src.spatial import (
    assign_crashes_to_units,
    build_universe,
    crashes_to_gdf,
    join_sip_treatment,
    join_vzv_labels,
)
from src.spf import fit_and_blend

log = logging.getLogger("pipeline")

TRAIN_END = pd.Timestamp(f"{TRAIN_YEARS[1] + 1}-01-01")
HOLDOUT_START = TRAIN_END
HOLDOUT_END = pd.Timestamp(f"{HOLDOUT_YEARS[1] + 1}-01-01")

# Site characteristics, not crash history. See SPF_PREDICTORS in src/config.py for why
# the crash-derived features that were here on the 2026-08-12 run are gone.
PREDICTORS = list(SPF_PREDICTORS)


@dataclass
class RunSummary:
    """Everything the README needs, in one serialisable object."""

    snapshot_date: str
    crashes_pulled: int
    crashes_assigned: int
    crashes_dropped: int
    universe_units: int
    corridors: int
    intersections: int
    priority_units: int
    treated_units: int
    train_window: str
    holdout_window: str
    holdout_casualties: int
    dispersion: dict[str, float]
    citywide_n: int
    r1_citywide_pp: float | None
    r2_citywide_pp: float | None
    r3_citywide_pp: float | None
    r2_stratified_pp: float | None
    r3_stratified_pp: float | None
    lift_pp: float
    ci_low_pp: float
    ci_high_pp: float
    ci_excludes_zero: bool
    clears_bar: bool
    verdict: str
    treated_pp: float | None
    untreated_pp: float | None


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def latest_snapshot(root: Path = RAW_DIR) -> Path:
    """Most recent dated snapshot directory holding a manifest."""
    candidates = sorted(p for p in root.glob("*") if (p / "manifest.json").exists())
    if not candidates:
        raise FileNotFoundError(
            f"no snapshots under {root}. Run scripts/pull_snapshots.py first."
        )
    return candidates[-1]


def load_geo(snapshot: Path, key: str) -> gpd.GeoDataFrame:
    """Read a snapshot parquet whose geometry was stored as WKT."""
    frame = pd.read_parquet(snapshot / f"{key}.parquet")
    if "geometry_wkt" not in frame.columns:
        raise ValueError(f"{key}: snapshot has no geometry_wkt column")
    frame = frame[frame["geometry_wkt"].notna()].copy()
    return gpd.GeoDataFrame(
        frame.drop(columns=["geometry_wkt"]),
        geometry=from_wkt(frame["geometry_wkt"]),
        crs=CRS_GEOGRAPHIC,
    )


def prepare_centerline(snapshot: Path) -> gpd.GeoDataFrame:
    """Centerline with borough resolved from its numeric code."""
    gdf = load_geo(snapshot, "centerline")
    gdf["borough"] = gdf["boroughcode"].astype(str).str.strip().map(BOROUGH_CODES)

    unmapped = int(gdf["borough"].isna().sum())
    if unmapped:
        # Not fatal: these stay in the universe and in the citywide regime, and
        # select_borough_stratified excludes and counts them.
        log.warning("%d centerline segment(s) have an unrecognised borough code", unmapped)

    return gdf


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


def build_scored_units(snapshot: Path, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Universe + assigned crashes + labels + treatment, cached to parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"units-{snapshot.name}.parquet"
    stats_path = CACHE_DIR / f"units-{snapshot.name}.json"

    if use_cache and cache.exists() and stats_path.exists():
        log.info("using cached units: %s", cache)
        return pd.read_parquet(cache), json.loads(stats_path.read_text())

    t0 = time.time()
    log.info("building universe from centerline")
    universe = build_universe(prepare_centerline(snapshot))
    log.info("universe: %d units in %.1fs", len(universe), time.time() - t0)

    log.info("labelling VZV priority units")
    universe, label_report = join_vzv_labels(
        universe, load_geo(snapshot, "vzv_corridors"), load_geo(snapshot, "vzv_intersections")
    )

    log.info("joining SIP treatment")
    universe, label_report = join_sip_treatment(
        universe,
        [load_geo(snapshot, "sip_corridors"), load_geo(snapshot, "sip_intersections")],
        report=label_report,
    )

    log.info("loading crashes")
    crashes = add_pedestrian_casualties(pd.read_parquet(snapshot / "crashes.parquet"))
    points, report = crashes_to_gdf(crashes)

    log.info("assigning %d crashes to %d units", len(points), len(universe))
    t0 = time.time()
    assigned, report = assign_crashes_to_units(points, universe, report)
    log.info("assignment done in %.1fs", time.time() - t0)

    assigned["crash_date"] = pd.to_datetime(assigned["crash_date"])
    train = assigned[assigned["crash_date"] < TRAIN_END]
    holdout = assigned[
        (assigned["crash_date"] >= HOLDOUT_START) & (assigned["crash_date"] < HOLDOUT_END)
    ]

    log.info("building features as of %s", TRAIN_END.date())
    features, _ = build_features(train, universe, TRAIN_END, months=(12, 24, 36))

    totals = holdout.groupby("unit_id")["pedestrian_casualties"].sum()
    features["holdout_casualties"] = (
        features["unit_id"].map(totals).fillna(0).astype(int)
    )

    units = pd.DataFrame(features.drop(columns=["geometry"], errors="ignore"))
    stats = {
        "crashes_pulled": report.total_input,
        "crashes_assigned": report.assigned,
        "crashes_dropped": report.dropped,
        "assignment_summary": report.summary(),
        "label_summary": label_report.summary(),
        "train_crashes": int(len(train)),
        "holdout_crashes": int(len(holdout)),
    }

    units.to_parquet(cache, index=False)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return units, stats


def run(snapshot: Path | None = None, use_cache: bool = True) -> RunSummary:
    snapshot = snapshot or latest_snapshot()
    log.info("snapshot: %s", snapshot)

    units, stats = build_scored_units(snapshot, use_cache=use_cache)

    log.info("fitting SPF and blending")
    scored, spf_results = fit_and_blend(
        units,
        target="casualties_36mo",
        predictors=PREDICTORS,
        observed_col="casualties_36mo",
    )

    priority = scored[scored["is_priority"]]
    citywide_n = int(len(priority))
    log.info("DOT published list: %d units", citywide_n)

    r1 = Selection(
        name="R1 DOT published",
        regime="citywide",
        unit_ids=priority["unit_id"].tolist(),
    )
    r2 = select_citywide_top_n(scored, "casualties_36mo", citywide_n, "R2 raw count")
    r3 = select_citywide_top_n(scored, "eb_estimate", citywide_n, "R3 empirical bayes")

    r1_rate = capture_rate(r1, scored)
    r2_rate = capture_rate(r2, scored)
    r3_rate = capture_rate(r3, scored)
    for r in (r1_rate, r2_rate, r3_rate):
        log.info(r.summary())

    ci = bootstrap_capture_difference(scored, r3, r2)
    verdict = apply_preregistered_bar(r3_rate, r2_rate, ci)
    log.info(ci.summary())
    log.info(verdict.summary())

    r2s = select_borough_stratified(
        scored, "casualties_36mo", "casualties_36mo", "R2 stratified"
    )
    r3s = select_borough_stratified(
        scored, "eb_estimate", "casualties_36mo", "R3 stratified"
    )
    r2s_rate = capture_rate(r2s, scored)
    r3s_rate = capture_rate(r3s, scored)
    log.info(r2s_rate.summary())
    log.info(r3s_rate.summary())

    split = split_by_treatment(scored, r1)

    summary = RunSummary(
        snapshot_date=snapshot.name,
        crashes_pulled=stats["crashes_pulled"],
        crashes_assigned=stats["crashes_assigned"],
        crashes_dropped=stats["crashes_dropped"],
        universe_units=int(len(scored)),
        corridors=int((scored["unit_type"] == "corridor").sum()),
        intersections=int((scored["unit_type"] == "intersection").sum()),
        priority_units=citywide_n,
        treated_units=int(scored["treated"].sum()),
        train_window=f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
        holdout_window=f"{HOLDOUT_YEARS[0]}-{HOLDOUT_YEARS[1]}",
        holdout_casualties=int(scored["holdout_casualties"].sum()),
        dispersion={k: v.dispersion for k, v in spf_results.items()},
        citywide_n=citywide_n,
        r1_citywide_pp=r1_rate.rate_pp,
        r2_citywide_pp=r2_rate.rate_pp,
        r3_citywide_pp=r3_rate.rate_pp,
        r2_stratified_pp=r2s_rate.rate_pp,
        r3_stratified_pp=r3s_rate.rate_pp,
        lift_pp=verdict.lift_pp,
        ci_low_pp=ci.lower_pp,
        ci_high_pp=ci.upper_pp,
        ci_excludes_zero=ci.excludes_zero,
        clears_bar=verdict.clears_bar,
        verdict=verdict.reason,
        treated_pp=split["treated"].rate_pp,
        untreated_pp=split["untreated"].rate_pp,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "run-summary.json"
    out.write_text(json.dumps(asdict(summary), indent=2, default=str) + "\n")
    log.info("wrote %s", out)

    # Selected defensively: an older cached units parquet may predate a carried column,
    # and losing the ranked CSV to a KeyError after a multi-minute fit is a poor trade.
    wanted = [
        "unit_id",
        "unit_type",
        "borough",
        "full_street_name",
        "rw_type",
        "eb_estimate",
        "casualties_36mo",
        "holdout_casualties",
        "is_priority",
        "treated",
    ]
    available = [c for c in wanted if c in scored.columns]
    missing = [c for c in wanted if c not in scored.columns]
    if missing:
        log.warning(
            "ranked output is missing %s - rebuild with --no-cache to pick them up",
            ", ".join(missing),
        )
    ranked = scored.nlargest(50, "eb_estimate")[available]
    ranked.to_csv(PROCESSED_DIR / "top-50-ranked.csv", index=False)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="rebuild cached stages")
    parser.add_argument("--snapshot", help="snapshot directory name, e.g. 2026-08-12")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    snapshot = (RAW_DIR / args.snapshot) if args.snapshot else None
    summary = run(snapshot=snapshot, use_cache=not args.no_cache)

    print("\n" + "=" * 72)
    print("HEADLINE")
    print("=" * 72)
    print(f"  holdout {summary.holdout_window}: {summary.holdout_casualties} pedestrian casualties")
    print(f"  each ranking selects {summary.citywide_n} locations (size of DOT's list)")
    print()
    print(f"  R1 DOT published      {summary.r1_citywide_pp:.1f}%")
    print(f"  R2 raw trailing count {summary.r2_citywide_pp:.1f}%")
    print(f"  R3 empirical bayes    {summary.r3_citywide_pp:.1f}%")
    print()
    print(f"  R3 - R2 = {summary.lift_pp:+.1f}pp "
          f"[95% CI {summary.ci_low_pp:+.1f}, {summary.ci_high_pp:+.1f}]")
    print(f"  {'CLEARS' if summary.clears_bar else 'DOES NOT CLEAR'} the pre-registered bar")
    print(f"  {summary.verdict}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
