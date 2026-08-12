"""Socrata paginated fetch with a completeness assertion.

The failure this module exists to prevent: Socrata pagination can return fewer rows
than exist without raising anything. A short page, a dropped connection mid-walk, or a
server-side timeout all look identical to "you reached the end." The pipeline downstream
would then model a partial dataset and report a confident, wrong number.

The guard is to ask the API how many rows it holds *before* walking, then assert the
walk collected exactly that many.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.config import SocrataSource

log = logging.getLogger(__name__)

PAGE_SIZE = 50_000
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0
TIMEOUT_SECONDS = 120


class SocrataError(RuntimeError):
    """Raised when a pull cannot be trusted. Never swallowed by callers."""


class IncompletePullError(SocrataError):
    """Pagination returned a different row count than the API reported."""


def _request_json(
    url: str,
    params: dict[str, Any],
    session: requests.Session,
    app_token: str | None = None,
) -> list[dict[str, Any]]:
    """GET with bounded retries on transient failures.

    Retries 429 and 5xx, which are load-shedding rather than "your query is wrong."
    A 4xx other than 429 means the request itself is malformed, so retrying it would
    just be a slower failure.
    """
    headers = {"X-App-Token": app_token} if app_token else {}
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url, params=params, headers=headers, timeout=TIMEOUT_SECONDS
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise SocrataError(
                    f"transient HTTP {response.status_code} from {url}"
                )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, SocrataError, ValueError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                break
            sleep_for = BACKOFF_BASE_SECONDS * (2**attempt)
            log.warning(
                "request failed (attempt %d/%d): %s - retrying in %.1fs",
                attempt + 1,
                MAX_RETRIES,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    raise SocrataError(f"exhausted {MAX_RETRIES} attempts for {url}") from last_error


def fetch_row_count(
    source: SocrataSource,
    session: requests.Session | None = None,
    app_token: str | None = None,
) -> int:
    """Ask the API how many rows match, before fetching any of them."""
    session = session or requests.Session()
    params: dict[str, Any] = {"$select": "count(*) as n"}
    if source.where:
        params["$where"] = source.where

    payload = _request_json(source.url, params, session, app_token)
    if not payload:
        raise SocrataError(f"{source.key}: count query returned no rows")

    try:
        return int(payload[0]["n"])
    except (KeyError, ValueError, TypeError) as exc:
        raise SocrataError(
            f"{source.key}: could not parse count response {payload[0]!r}"
        ) from exc


def fetch_socrata(
    source: SocrataSource,
    session: requests.Session | None = None,
    app_token: str | None = None,
    page_size: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Walk every page of a Socrata resource, asserting the walk was complete.

    Ordering by `:id` is required for correctness, not tidiness. Socrata does not
    guarantee a stable order across paged requests without an explicit sort, so an
    unordered walk can return the same row twice and miss another entirely while
    still producing the expected total.

    Raises:
        IncompletePullError: the walk collected a different number of rows than the
            API reported. The data is not usable; the caller must not proceed.
    """
    session = session or requests.Session()
    expected = fetch_row_count(source, session, app_token)
    log.info("%s: API reports %d rows", source.key, expected)

    if expected == 0:
        raise SocrataError(
            f"{source.key}: API reports 0 rows. Either the filter is wrong or the "
            f"dataset moved. Refusing to write an empty snapshot."
        )

    rows: list[dict[str, Any]] = []
    offset = 0

    while offset < expected:
        params: dict[str, Any] = {
            "$limit": page_size,
            "$offset": offset,
            "$order": ":id",
        }
        if source.where:
            params["$where"] = source.where
        if source.select:
            params["$select"] = source.select

        page = _request_json(source.url, params, session, app_token)
        if not page:
            # An empty page before reaching `expected` is the silent-truncation case.
            break

        rows.extend(page)
        offset += len(page)
        log.info("%s: %d/%d rows", source.key, len(rows), expected)

    if len(rows) != expected:
        raise IncompletePullError(
            f"{source.key}: expected {expected} rows, collected {len(rows)}. "
            f"Pagination was truncated - this snapshot would silently model partial "
            f"data. Re-run the pull."
        )

    return rows
