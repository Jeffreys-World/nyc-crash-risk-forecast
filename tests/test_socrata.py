"""Socrata pagination, and the truncation that does not raise.

The failure: a paged walk stops early and returns fewer rows than exist. A short page, a
dropped connection, or a server-side timeout all look identical to "you reached the
end." The pipeline then models partial data and reports a confident wrong number.

The guard is to ask the API for `count(*)` first and assert the walk collected exactly
that many. These tests exercise that assertion from both sides.
"""

from __future__ import annotations

import pytest
import requests

from src.config import SocrataSource
from src.socrata import (
    IncompletePullError,
    SocrataError,
    fetch_row_count,
    fetch_socrata,
)

SOURCE = SocrataSource(key="test", dataset_id="abcd-1234", description="test source")


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves a fixed row list, honouring $limit/$offset and $select=count(*)."""

    def __init__(self, rows, *, reported_count=None, fail_times=0, status=500, short_page_at=None):
        self.rows = rows
        self.reported_count = len(rows) if reported_count is None else reported_count
        self.fail_times = fail_times
        self.status = status
        self.short_page_at = short_page_at
        self.calls = 0
        self.page_calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        params = params or {}

        if self.fail_times > 0:
            self.fail_times -= 1
            return FakeResponse([], status_code=self.status)

        if str(params.get("$select", "")).startswith("count"):
            return FakeResponse([{"n": str(self.reported_count)}])

        self.page_calls += 1
        offset = int(params.get("$offset", 0))
        limit = int(params.get("$limit", 50_000))

        if self.short_page_at is not None and self.page_calls >= self.short_page_at:
            return FakeResponse([])  # silent truncation

        return FakeResponse(self.rows[offset : offset + limit])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.socrata.time.sleep", lambda _: None)


def rows(n: int) -> list[dict]:
    return [{"id": i} for i in range(n)]


class TestRowCount:
    def test_reads_the_count(self):
        assert fetch_row_count(SOURCE, FakeSession(rows(42))) == 42

    def test_applies_the_where_clause(self):
        source = SocrataSource("t", "abcd-1234", "t", where="crash_date >= '2019-01-01'")
        session = FakeSession(rows(5))
        fetch_row_count(source, session)
        assert session.calls == 1

    def test_unparseable_count_raises(self):
        class BadCount(FakeSession):
            def get(self, url, params=None, headers=None, timeout=None):
                return FakeResponse([{"wrong_key": "5"}])

        with pytest.raises(SocrataError, match="could not parse count"):
            fetch_row_count(SOURCE, BadCount([]))


class TestPagination:
    def test_walks_a_single_page(self):
        assert len(fetch_socrata(SOURCE, FakeSession(rows(10)), page_size=100)) == 10

    def test_walks_multiple_pages(self):
        session = FakeSession(rows(250))
        assert len(fetch_socrata(SOURCE, session, page_size=100)) == 250
        assert session.page_calls == 3

    def test_preserves_every_row_exactly_once(self):
        result = fetch_socrata(SOURCE, FakeSession(rows(250)), page_size=100)
        assert [r["id"] for r in result] == list(range(250))

    def test_requests_a_stable_sort(self):
        """Without an explicit $order, a paged walk can repeat one row and drop another."""
        captured = {}

        class Recording(FakeSession):
            def get(self, url, params=None, headers=None, timeout=None):
                if not str((params or {}).get("$select", "")).startswith("count"):
                    captured.update(params or {})
                return super().get(url, params, headers, timeout)

        fetch_socrata(SOURCE, Recording(rows(10)), page_size=5)
        assert captured.get("$order") == ":id"


class TestTruncationGuard:
    def test_short_walk_raises(self):
        """The core guard: API says 250, walk collects 100."""
        session = FakeSession(rows(250), short_page_at=2)
        with pytest.raises(IncompletePullError, match="Pagination was truncated"):
            fetch_socrata(SOURCE, session, page_size=100)

    def test_error_reports_both_counts(self):
        session = FakeSession(rows(250), short_page_at=2)
        with pytest.raises(IncompletePullError) as exc:
            fetch_socrata(SOURCE, session, page_size=100)
        assert "250" in str(exc.value) and "100" in str(exc.value)

    def test_over_reported_count_also_raises(self):
        """Trusting the walk when the API claims more rows would hide a real gap."""
        session = FakeSession(rows(10), reported_count=99)
        with pytest.raises(IncompletePullError):
            fetch_socrata(SOURCE, session, page_size=100)


class TestEmptyAndFailure:
    def test_zero_rows_refuses_to_write_a_snapshot(self):
        """An empty snapshot silently becomes a pipeline full of zeros."""
        with pytest.raises(SocrataError, match="Refusing to write an empty snapshot"):
            fetch_socrata(SOURCE, FakeSession([], reported_count=0))

    def test_retries_transient_server_errors(self):
        session = FakeSession(rows(10), fail_times=2, status=503)
        assert len(fetch_socrata(SOURCE, session, page_size=100)) == 10

    def test_retries_rate_limiting(self):
        session = FakeSession(rows(10), fail_times=1, status=429)
        assert len(fetch_socrata(SOURCE, session, page_size=100)) == 10

    def test_gives_up_after_the_retry_budget(self):
        session = FakeSession(rows(10), fail_times=99, status=503)
        with pytest.raises(SocrataError, match="exhausted"):
            fetch_socrata(SOURCE, session, page_size=100)

    def test_client_errors_are_not_retried(self):
        """A 400 means the query is wrong; retrying is just a slower failure."""
        session = FakeSession(rows(10), fail_times=99, status=400)
        with pytest.raises(SocrataError):
            fetch_socrata(SOURCE, session, page_size=100)
