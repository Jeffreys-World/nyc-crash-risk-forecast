"""End-to-end run: snapshots -> universe -> features -> SPF -> EB -> backtest.

    python -m src.pipeline

Reads only dated snapshots, never the API. Every expensive stage caches to parquet so
a re-run is fast and a reader can inspect any intermediate.

The output is the number the README quotes, and it is printed whether or not it is
flattering. The bar it has to clear was fixed before this file could produce anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    DEFAULT_RADII,
    HOLDOUT_YEARS,
    PROCESSED_DIR,
    RAW_DIR,
    SPF_PREDICTORS,
    TRAIN_YEARS,
    JoinRadii,
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

TRAIN_START = pd.Timestamp(f"{TRAIN_YEARS[0]}-01-01")
TRAIN_END = pd.Timestamp(f"{TRAIN_YEARS[1] + 1}-01-01")
HOLDOUT_START = TRAIN_END
HOLDOUT_END = pd.Timestamp(f"{HOLDOUT_YEARS[1] + 1}-01-01")

# Site characteristics, not crash history. See SPF_PREDICTORS in src/config.py for why
# the crash-derived features that were here on the 2026-08-12 run are gone.
PREDICTORS = list(SPF_PREDICTORS)

# The per-unit frame `scripts/rederive_headline.py` reads.
#
# Ten columns of the units frame's thirty-five: exactly what the scoring layer touches,
# and nothing it does not. That is what makes the file small enough to commit — under
# 4 MB for 220,033 units against 8.3 MB for the full cached frame — while still being the
# *complete* input to every number in run-summary.json. A re-derivation that had to be
# handed a subset of the inputs would be checking a subset of the claim.
#
# `spf_prediction` is here even though no selection ranks on it, because it is what lets
# the re-derivation rebuild `eb_estimate` from the HSM formula rather than trusting the
# column. Without it the blend — the one piece of arithmetic the whole method turns on —
# would be the one thing an independent check could not check.
SCORED_UNITS_COLUMNS = (
    "unit_id",
    "unit_type",
    "borough",
    "casualties_36mo",
    "holdout_casualties",
    "spf_prediction",
    "eb_estimate",
    "is_priority",
    "treated",
    "treatment_date",
)


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
    treated_before_holdout_pp: float | None
    treated_during_holdout_pp: float | None
    untreated_pp: float | None
    treated_before_holdout_units: int
    treated_during_holdout_units: int
    # The three join distances this run actually used. Provenance, not configuration:
    # a summary that does not say which radii produced it cannot be compared against
    # another summary, which is precisely what a sensitivity sweep needs to do.
    join_radii: dict[str, float]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


# Modules whose behaviour determines the contents of the cached units frame. config is
# included because the join radius, CRS, and windows all live there.
_CACHE_KEY_MODULES = ("config.py", "spatial.py", "features.py")


def _code_fingerprint() -> str:
    """Short hash of the source files that determine the cached units frame."""
    digest = hashlib.blake2b(digest_size=6)
    for name in _CACHE_KEY_MODULES:
        path = Path(__file__).with_name(name)
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()


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


def build_scored_units(
    snapshot: Path,
    use_cache: bool = True,
    radii: JoinRadii | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Universe + assigned crashes + labels + treatment, cached to parquet."""
    radii = radii or DEFAULT_RADII
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # The cache key includes a fingerprint of the code that produces the cached frame,
    # not just the snapshot date. Keyed on the date alone, editing a feature or a spatial
    # join and re-running silently reuses units built by the previous version of the
    # code, and the run reports numbers that no longer correspond to the source. That
    # happened during this project's own build: a stale units file survived a change to
    # the label join and produced a summary that disagreed with the log beside it.
    #
    # For a repo whose claim is "clone this and reproduce the number", a cache that can
    # serve a stale intermediate is a correctness bug, not a performance detail.
    #
    # The radii tag is in the key for the same reason and closes the same hole from the
    # other side: they are now arguments, so two runs can differ in what they built
    # while every file the fingerprint hashes is byte-identical. Without the tag a
    # sensitivity sweep would rebuild once and then serve that first result for every
    # subsequent setting, reporting a flat, entirely fictional insensitivity.
    fingerprint = _code_fingerprint()
    stem = f"units-{snapshot.name}-{radii.tag}-{fingerprint}"
    cache = CACHE_DIR / f"{stem}.parquet"
    stats_path = CACHE_DIR / f"{stem}.json"

    if use_cache and cache.exists() and stats_path.exists():
        log.info("using cached units: %s", cache)
        return pd.read_parquet(cache), json.loads(stats_path.read_text())

    stale = sorted(CACHE_DIR.glob(f"units-{snapshot.name}-{radii.tag}-*.parquet"))
    if stale:
        log.info(
            "code changed since %d cached build(s) at these radii; rebuilding",
            len(stale),
        )

    t0 = time.time()
    log.info("building universe from centerline")
    universe = build_universe(prepare_centerline(snapshot))
    log.info("universe: %d units in %.1fs", len(universe), time.time() - t0)

    log.info("labelling VZV priority units at a %.0f ft buffer", radii.vzv_buffer_ft)
    universe, label_report = join_vzv_labels(
        universe,
        load_geo(snapshot, "vzv_corridors"),
        load_geo(snapshot, "vzv_intersections"),
        buffer_ft=radii.vzv_buffer_ft,
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

    log.info(
        "assigning %d crashes to %d units (intersection %.0f ft, corridor %.0f ft)",
        len(points),
        len(universe),
        radii.intersection_radius_ft,
        radii.max_join_distance_ft,
    )
    t0 = time.time()
    assigned, report = assign_crashes_to_units(points, universe, report, radii=radii)
    log.info("assignment done in %.1fs", time.time() - t0)

    assigned["crash_date"] = pd.to_datetime(assigned["crash_date"])
    # Bounded at both ends. `crash_date < TRAIN_END` alone silently included every crash
    # back to the 2016 pull start, so the run reported a 2019-2023 training window while
    # actually training on 2016-2023. It made little difference to the trailing counts,
    # which only look back 36 months, but a provenance field that does not describe the
    # run is the kind of quiet inaccuracy this project exists to avoid.
    train = assigned[
        (assigned["crash_date"] >= TRAIN_START) & (assigned["crash_date"] < TRAIN_END)
    ]
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
        "join_radii": radii.as_dict(),
    }

    units.to_parquet(cache, index=False)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return units, stats


def scored_units_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """The committed per-unit frame: the scoring layer's inputs, and only those."""
    missing = [c for c in SCORED_UNITS_COLUMNS if c not in scored.columns]
    if missing:
        # Unlike the top-50 CSV below, this one is not written defensively. A frame
        # missing a column is a frame the re-derivation cannot check, and shipping a
        # partial one would let the guard pass while covering less than it claims.
        raise KeyError(
            f"scored frame is missing {missing}; the re-derivation artifact needs every "
            f"column in SCORED_UNITS_COLUMNS. Rebuild with --no-cache."
        )
    return scored[list(SCORED_UNITS_COLUMNS)].copy()


def _write_artifacts(
    summary: RunSummary, scored: pd.DataFrame, processed_dir: Path = PROCESSED_DIR
) -> list[Path]:
    """Write the three committed artifacts, staging each before it replaces the last.

    Staged because these three files are read as one record. `run-summary.json` was
    previously written first and the ranked CSV second, so a failure in between left a
    complete-looking summary on disk next to a stale CSV describing a different run —
    and now a stale per-unit frame too, which is worse: the re-derivation would compare
    one run's units against another run's headline and report a mismatch that is really
    a half-finished write. Everything is built, then everything is moved.

    `os.replace` is atomic per file, not across the set, so a crash mid-move can still
    leave a mixed directory. Closing that properly needs the snapshot and radii the
    re-derivation already cross-checks between the frame's metadata and the summary,
    which is why that check exists rather than being assumed away here.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []

    summary_tmp = processed_dir / "run-summary.json.tmp"
    summary_tmp.write_text(json.dumps(asdict(summary), indent=2, default=str) + "\n")
    staged.append((summary_tmp, processed_dir / "run-summary.json"))

    # Provenance travels *inside* the parquet, not beside it. A per-unit frame that does
    # not say which snapshot and which radii produced it cannot be told apart from one
    # built at 150 ft in a directory whose summary says 100, and that is precisely the
    # stale-intermediate failure this project has already had once.
    frame = scored_units_frame(scored)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"snapshot_date": summary.snapshot_date.encode(),
            b"join_radii": json.dumps(summary.join_radii).encode(),
            b"dispersion": json.dumps(summary.dispersion).encode(),
            b"universe_units": str(summary.universe_units).encode(),
            b"citywide_n": str(summary.citywide_n).encode(),
            b"produced_by": b"src.pipeline.run",
        }
    )
    units_tmp = processed_dir / "scored-units.parquet.tmp"
    pq.write_table(table, units_tmp, compression="zstd")
    staged.append((units_tmp, processed_dir / "scored-units.parquet"))

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
    ranked_tmp = processed_dir / "top-50-ranked.csv.tmp"
    scored.nlargest(50, "eb_estimate")[available].to_csv(ranked_tmp, index=False)
    staged.append((ranked_tmp, processed_dir / "top-50-ranked.csv"))

    written: list[Path] = []
    for tmp, final in staged:
        os.replace(tmp, final)
        log.info("wrote %s", final)
        written.append(final)
    return written


def run(
    snapshot: Path | None = None,
    use_cache: bool = True,
    radii: JoinRadii | None = None,
    write_artifacts: bool = True,
) -> RunSummary:
    """The full run. Returns the summary; optionally writes the committed artifacts.

    `write_artifacts=False` is what the sensitivity sweep uses. `data/processed/` holds
    the published headline, and a sweep at 250 ft silently overwriting it with a result
    the README does not describe would be worse than not running the sweep at all.
    """
    radii = radii or DEFAULT_RADII
    snapshot = snapshot or latest_snapshot()
    log.info("snapshot: %s, radii: %s", snapshot, radii.tag)

    units, stats = build_scored_units(snapshot, use_cache=use_cache, radii=radii)

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

    split = split_by_treatment(scored, r1, holdout_start=HOLDOUT_START)
    for label, rate in split.items():
        log.info("%s: %s", label, rate.summary())

    treatment_when = pd.to_datetime(scored["treatment_date"], errors="coerce")

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
        treated_before_holdout_pp=split["treated_before_holdout"].rate_pp,
        treated_during_holdout_pp=split["treated_during_holdout"].rate_pp,
        untreated_pp=split["untreated"].rate_pp,
        treated_before_holdout_units=int(
            (treatment_when.notna() & (treatment_when < HOLDOUT_START)).sum()
        ),
        treated_during_holdout_units=int(
            (treatment_when.notna() & (treatment_when >= HOLDOUT_START)).sum()
        ),
        join_radii=radii.as_dict(),
    )

    if not write_artifacts:
        log.info("write_artifacts=False: leaving %s untouched", PROCESSED_DIR)
        return summary

    _write_artifacts(summary, scored)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="rebuild cached stages")
    parser.add_argument("--snapshot", help="snapshot directory name, e.g. 2026-08-12")
    # Exposed so a single non-default setting can be reproduced from a shell without
    # importing anything. `scripts/radius_sensitivity.py` sweeps all three at once.
    parser.add_argument(
        "--max-join-distance-ft",
        type=float,
        default=DEFAULT_RADII.max_join_distance_ft,
        help="crash-to-segment join radius (default %(default)s)",
    )
    parser.add_argument(
        "--intersection-radius-ft",
        type=float,
        default=DEFAULT_RADII.intersection_radius_ft,
        help="crash-to-node claim radius (default %(default)s)",
    )
    parser.add_argument(
        "--vzv-buffer-ft",
        type=float,
        default=DEFAULT_RADII.vzv_buffer_ft,
        help="VZV priority-label buffer (default %(default)s)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="do not write data/processed/; print the headline only",
    )
    args = parser.parse_args(argv)

    radii = JoinRadii(
        max_join_distance_ft=args.max_join_distance_ft,
        intersection_radius_ft=args.intersection_radius_ft,
        vzv_buffer_ft=args.vzv_buffer_ft,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    snapshot = (RAW_DIR / args.snapshot) if args.snapshot else None
    # Non-default radii never touch data/processed/ even without --no-write. The
    # committed artifacts are the published headline; overwriting them from an
    # exploratory run is how a repo starts disagreeing with its own README.
    write = not args.no_write and radii == DEFAULT_RADII
    if not write and not args.no_write:
        log.info("non-default radii %s: not writing data/processed/", radii.tag)
    summary = run(
        snapshot=snapshot,
        use_cache=not args.no_cache,
        radii=radii,
        write_artifacts=write,
    )

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
