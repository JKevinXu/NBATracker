"""Tests for nba_tracker.exporter — rendering and file export."""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from nba_tracker.exporter import export_stats, _fmt
from nba_tracker.models import (
    Player,
    TeamLeader,
    TeamLeaders,
    TeamSummary,
)


# ── _fmt helper ───────────────────────────────────────────────────────


def test_fmt_percentage() -> None:
    assert _fmt(0.482, "FG_PCT") == "48.2%"
    assert _fmt(0.370, "FG3_PCT") == "37.0%"
    assert _fmt(0.900, "FT_PCT") == "90.0%"


def test_fmt_stat() -> None:
    assert _fmt(25.3, "PTS") == "25.3"
    assert _fmt(7.0, "REB") == "7.0"


def test_fmt_name() -> None:
    assert _fmt("LeBron James", "PLAYER_NAME") == "LeBron James"


# ── export_stats to CSV ──────────────────────────────────────────────


def test_export_csv() -> None:
    df = pd.DataFrame(
        [{"PLAYER_NAME": "Alice", "PTS": 28.0, "REB": 5.0}]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.csv")
        result = export_stats(df, "TST", "2024-25", "csv", path)
        assert os.path.exists(result)
        content = open(result).read()
        assert "Alice" in content
        assert "28.0" in content


# ── export_stats to JSON ─────────────────────────────────────────────


def test_export_json() -> None:
    df = pd.DataFrame(
        [{"PLAYER_NAME": "Bob", "PTS": 15.0, "AST": 3.0}]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        result = export_stats(df, "TST", "2024-25", "json", path)
        assert os.path.exists(result)
        data = json.loads(open(result).read())
        assert data[0]["PLAYER_NAME"] == "Bob"


# ── export_stats default filename ────────────────────────────────────


def test_export_default_filename() -> None:
    df = pd.DataFrame([{"PLAYER_NAME": "X", "PTS": 1.0}])
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            result = export_stats(df, "LAL", "2024-25", "csv", None)
            assert "LAL_2024_25_stats.csv" in result
            assert os.path.exists(result)
        finally:
            os.chdir(orig_cwd)
