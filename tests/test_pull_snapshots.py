"""Snapshot idempotency and manifest provenance.

Re-running the pull must not silently re-download or, worse, half-overwrite an existing
snapshot. The manifest is the only record of which vintage of a mutating dataset
produced a published number, so it has to be written even when the pull partly fails.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import pull_snapshots
from src.config import SocrataSource
from src.socrata import SocrataError

SOURCE = SocrataSource(key="crashes", dataset_id="abcd-1234", description="test")


@pytest.fixture
def fake_rows():
    return [{"id": i, "value": f"row{i}"} for i in range(5)]


@pytest.fixture
def patched(monkeypatch, fake_rows):
    """Replace the network call with a counter so re-runs are observable."""
    calls = {"n": 0}

    def fake_fetch(source, session=None, app_token=None, page_size=None):
        calls["n"] += 1
        return fake_rows

    monkeypatch.setattr(pull_snapshots, "fetch_socrata", fake_fetch)
    return calls


class TestPullOne:
    def test_writes_parquet(self, tmp_path, patched):
        entry = pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        assert (tmp_path / "crashes.parquet").exists()
        assert entry["rows"] == 5
        assert entry["status"] == "pulled"

    def test_rerun_skips_the_network(self, tmp_path, patched):
        """Idempotency: the second call must not hit the API."""
        pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        entry = pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        assert patched["n"] == 1
        assert entry["status"] == "skipped_existing"

    def test_force_repulls(self, tmp_path, patched):
        pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=True)
        assert patched["n"] == 2

    def test_rerun_preserves_row_count(self, tmp_path, patched):
        first = pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        second = pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        assert first["rows"] == second["rows"]

    def test_no_temp_file_is_left_behind(self, tmp_path, patched):
        """A truncated .tmp would be read as complete by a later run."""
        pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        assert not list(tmp_path.glob("*.tmp"))

    def test_written_parquet_round_trips(self, tmp_path, patched, fake_rows):
        pull_snapshots.pull_one(SOURCE, tmp_path, None, None, force=False)
        frame = pd.read_parquet(tmp_path / "crashes.parquet")
        assert len(frame) == len(fake_rows)
        assert list(frame["value"]) == [r["value"] for r in fake_rows]


class TestFailureHandling:
    def test_a_failed_source_is_recorded_and_exits_non_zero(self, tmp_path, monkeypatch):
        """A partial snapshot must never look clean."""
        monkeypatch.setattr(pull_snapshots, "RAW_DIR", tmp_path)
        monkeypatch.setattr(pull_snapshots, "snapshot_dir", lambda when=None: tmp_path / "d")
        monkeypatch.setattr(pull_snapshots, "SOURCES", {"crashes": SOURCE})
        # Pin state is a config decision, not this test's subject. Held at None so the
        # test asserts failure handling rather than which sources happen to be pinned.
        monkeypatch.setattr(pull_snapshots, "CENTERLINE_SOURCE", None)

        def boom(*args, **kwargs):
            raise SocrataError("pagination truncated")

        monkeypatch.setattr(pull_snapshots, "fetch_socrata", boom)

        assert pull_snapshots.main([]) == 1

        manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
        assert manifest["failures"] == ["crashes"]
        assert "pagination truncated" in manifest["sources"][0]["status"]

    def test_manifest_records_provenance(self, tmp_path, monkeypatch, patched):
        monkeypatch.setattr(pull_snapshots, "snapshot_dir", lambda when=None: tmp_path / "d")
        monkeypatch.setattr(pull_snapshots, "SOURCES", {"crashes": SOURCE})
        monkeypatch.setattr(pull_snapshots, "CENTERLINE_SOURCE", None)

        assert pull_snapshots.main([]) == 0

        manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
        assert manifest["total_rows"] == 5
        assert manifest["snapshot_date"] == "d"
        assert manifest["pulled_at"]

    def test_manifest_records_the_centerline_pin(self, tmp_path, monkeypatch, patched):
        """Provenance: a reader must be able to tell whether the universe was built."""
        monkeypatch.setattr(pull_snapshots, "snapshot_dir", lambda when=None: tmp_path / "d")
        monkeypatch.setattr(pull_snapshots, "SOURCES", {})
        monkeypatch.setattr(pull_snapshots, "CENTERLINE_SOURCE", SOURCE)

        assert pull_snapshots.main([]) == 0
        manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
        assert manifest["centerline_pinned"] is True

    def test_unknown_source_key_is_rejected(self, tmp_path, monkeypatch, patched):
        monkeypatch.setattr(pull_snapshots, "snapshot_dir", lambda when=None: tmp_path / "d")
        with pytest.raises(SystemExit):
            pull_snapshots.main(["--only", "not_a_source"])
