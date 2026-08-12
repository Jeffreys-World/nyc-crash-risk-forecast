"""Guards against committing a credential.

A secret in git history cannot be quietly removed. Rotating it is the only real fix,
and that means noticing in the first place. These tests fail loudly at the point the
mistake is made rather than after the push.

They deliberately do not require a `.env` to exist: a fresh clone has none, and the
suite has to pass there.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from src.config import REPO_ROOT, _clean_token

ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Socrata app tokens are 25 characters of mixed-case alphanumerics.
TOKEN_SHAPE = re.compile(r"\b[A-Za-z0-9]{25}\b")


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.split()


class TestEnvIsNotTracked:
    def test_dotenv_is_not_in_git(self):
        """The whole point. If this fails, a credential is in the repo."""
        assert ".env" not in _tracked_files()

    def test_gitignore_actually_ignores_dotenv(self):
        """Tests the rule, not just its presence in the file."""
        result = subprocess.run(
            ["git", "check-ignore", "-q", ".env"], cwd=REPO_ROOT, check=False
        )
        assert result.returncode == 0, ".env is not covered by .gitignore"

    def test_example_file_is_tracked(self):
        """The negation pattern must not accidentally exclude the template."""
        assert ".env.example" in _tracked_files()


class TestExampleHoldsNoRealValue:
    def test_example_exists(self):
        assert ENV_EXAMPLE.exists(), ".env.example is the setup contract; it must exist"

    def test_example_contains_no_token_shaped_string(self):
        """Catches the classic slip: filling in the template and committing it."""
        matches = TOKEN_SHAPE.findall(ENV_EXAMPLE.read_text())
        assert not matches, f".env.example contains a token-shaped value: {matches}"

    def test_example_names_the_variable_the_code_reads(self):
        assert "SOCRATA_APP_TOKEN" in ENV_EXAMPLE.read_text()


class TestNoTokenInTrackedSource:
    def test_no_token_shaped_literal_in_tracked_text_files(self):
        """Sweeps every tracked source file, not just the ones we remembered."""
        suffixes = {".py", ".toml", ".md", ".yml", ".yaml", ".json", ".cfg", ".txt"}
        offenders: list[str] = []

        for rel in _tracked_files():
            path = REPO_ROOT / rel
            if path.suffix not in suffixes or not path.exists():
                continue
            if rel == "tests/test_secrets.py":  # this file describes the shape
                continue
            for line in path.read_text(errors="ignore").splitlines():
                if "SOCRATA_APP_TOKEN" in line and TOKEN_SHAPE.search(line):
                    offenders.append(f"{rel}: {line.strip()[:60]}")

        assert not offenders, f"possible committed token: {offenders}"


class TestTokenNormalisation:
    """`_clean_token` is tested directly rather than by reloading the module.

    An earlier version of this file reloaded `src.config` under monkeypatch to inspect
    the constant. The restoring reload ran while monkeypatch was still active, so the
    module kept the patched state and every later test saw a None token. Testing the
    pure function has no such ordering hazard.
    """

    def test_missing_becomes_none(self):
        assert _clean_token(None) is None

    def test_empty_becomes_none(self):
        assert _clean_token("") is None

    def test_whitespace_only_becomes_none(self):
        """requests raises InvalidHeader on a whitespace header value, before sending."""
        assert _clean_token("   ") is None

    def test_surrounding_whitespace_is_stripped(self):
        assert _clean_token("  abc123  ") == "abc123"

    def test_real_value_passes_through(self):
        # Deliberately not token-shaped. A committed fixture that looks like a real
        # credential invites someone to wonder whether it is one.
        assert _clean_token("EXAMPLE-NOT-A-TOKEN") == "EXAMPLE-NOT-A-TOKEN"


class TestConfigLoading:
    @pytest.mark.skipif(not ENV_FILE.exists(), reason="no local .env; fresh clone")
    def test_local_env_supplies_a_token(self):
        from src.config import SOCRATA_APP_TOKEN

        assert SOCRATA_APP_TOKEN, ".env exists but no token was loaded from it"
