#!/usr/bin/env python3
"""Pull dated Socrata snapshots to parquet.

Every input to this project is a live, mutating API. NYPD amends past crash records
retroactively, so even a date-filtered query returns different data week to week. The
pipeline therefore never touches the API: this script writes dated snapshots, and every
downstream stage reads only snapshots.

    python scripts/pull_snapshots.py                  # today's snapshot, skip existing
    python scripts/pull_snapshots.py --force          # re-pull even if present
    python scripts/pull_snapshots.py --only crashes   # one source
    python scripts/pull_snapshots.py --app-token TOK  # higher rate limit

Writes:
    data/raw/<YYYY-MM-DD>/<key>.parquet
    data/raw/<YYYY-MM-DD>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import shape

# Allow `python scripts/pull_snapshots.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CENTERLINE_SOURCE,
    RAW_DIR,
    SOCRATA_APP_TOKEN,
    SOURCES,
    SocrataSource,
)
from src.socrata import SocrataError, fetch_socrata  # noqa: E402

log = logging.getLogger("pull_snapshots")

MANIFEST_NAME = "manifest.json"


def snapshot_dir(when: date | None = None) -> Path:
    return RAW_DIR / (when or date.today()).isoformat()


def geojson_to_wkt(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Convert Socrata's `the_geom` GeoJSON objects into a WKT column.

    Parquet has no native geometry type, and a column of nested dicts round-trips
    badly. WKT keeps the snapshot a plain, portable table that any reader can open
    without geospatial libraries, and `shapely.from_wkt` restores it exactly.

    A feature whose geometry fails to parse is kept with a null geometry rather than
    dropped, so the count in the manifest still matches what the API reported and the
    loss shows up downstream as an explicit unmatched feature.
    """
    if "the_geom" not in frame.columns:
        raise SocrataError(
            f"{key}: expected a `the_geom` column on a geo source but found "
            f"{list(frame.columns)[:8]}"
        )

    def _convert(value: object) -> str | None:
        if not isinstance(value, dict):
            return None
        try:
            return shape(value).wkt
        except Exception:  # malformed geometry from the API
            return None

    frame = frame.copy()
    frame["geometry_wkt"] = frame["the_geom"].map(_convert)
    frame = frame.drop(columns=["the_geom"])

    bad = int(frame["geometry_wkt"].isna().sum())
    if bad:
        log.warning("%s: %d row(s) had unparseable geometry; kept with null WKT", key, bad)

    return frame


def pull_one(
    source: SocrataSource,
    out_dir: Path,
    session: requests.Session,
    app_token: str | None,
    force: bool,
) -> dict[str, object]:
    """Fetch one source to parquet. Returns its manifest entry."""
    out_path = out_dir / f"{source.key}.parquet"

    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        log.info("%s: already present (%d rows), skipping", source.key, len(existing))
        return {
            "key": source.key,
            "dataset_id": source.dataset_id,
            "description": source.description,
            "where": source.where,
            "rows": int(len(existing)),
            "columns": int(existing.shape[1]),
            "status": "skipped_existing",
        }

    log.info("%s: pulling %s", source.key, source.dataset_id)
    rows = fetch_socrata(source, session=session, app_token=app_token)
    frame = pd.DataFrame.from_records(rows)

    if source.is_geo:
        frame = geojson_to_wkt(frame, source.key)

    # Write to a temp path first so an interrupted run cannot leave a truncated
    # parquet that a later `--force`-less run would happily treat as complete.
    tmp_path = out_path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    log.info("%s: wrote %d rows x %d cols", source.key, len(frame), frame.shape[1])
    return {
        "key": source.key,
        "dataset_id": source.dataset_id,
        "description": source.description,
        "where": source.where,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "status": "pulled",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-pull sources already on disk")
    parser.add_argument("--only", action="append", help="pull only this source key (repeatable)")
    parser.add_argument(
        "--app-token",
        default=SOCRATA_APP_TOKEN,
        help="Socrata app token; defaults to SOCRATA_APP_TOKEN from the environment or .env",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Never log the token itself, only whether one was found. A token echoed into CI
    # output is a token in a log aggregator.
    log.info(
        "Socrata app token: %s",
        "present" if args.app_token else "MISSING (anonymous requests are throttled hard)",
    )

    sources = dict(SOURCES)
    if CENTERLINE_SOURCE is not None:
        sources["centerline"] = CENTERLINE_SOURCE
    else:
        log.warning(
            "centerline source is not pinned in src/config.py - skipping it. The "
            "pipeline cannot build its unit of analysis until one is selected."
        )

    if args.only:
        unknown = set(args.only) - set(sources)
        if unknown:
            parser.error(f"unknown source key(s): {', '.join(sorted(unknown))}")
        sources = {k: v for k, v in sources.items() if k in args.only}

    out_dir = snapshot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("snapshot directory: %s", out_dir)

    session = requests.Session()
    entries: list[dict[str, object]] = []
    failures: list[str] = []

    for source in sources.values():
        try:
            entries.append(pull_one(source, out_dir, session, args.app_token, args.force))
        except SocrataError as exc:
            # Keep going so one dead dataset does not cost the whole pull, but record
            # the failure and exit non-zero. A partial snapshot must never look clean.
            log.error("%s: FAILED - %s", source.key, exc)
            failures.append(source.key)
            entries.append(
                {
                    "key": source.key,
                    "dataset_id": source.dataset_id,
                    "description": source.description,
                    "where": source.where,
                    "rows": 0,
                    "columns": 0,
                    "status": f"failed: {exc}",
                }
            )

    manifest = {
        "pulled_at": datetime.now(UTC).isoformat(),
        "snapshot_date": out_dir.name,
        "centerline_pinned": CENTERLINE_SOURCE is not None,
        "sources": entries,
        "total_rows": sum(int(e["rows"]) for e in entries),
        "failures": failures,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    log.info("manifest: %s", out_dir / MANIFEST_NAME)
    log.info("total rows: %d", manifest["total_rows"])

    if failures:
        log.error("pull incomplete - failed sources: %s", ", ".join(failures))
        return 1

    log.info("pull complete. Record the snapshot date and row counts in the README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
