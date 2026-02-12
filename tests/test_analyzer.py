"""Tests for nba_tracker.analyzer — leaders, summary, comparison."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from nba_tracker.analyzer import (
    compare_players,
    compute_team_leaders,
    compute_team_summary,
)


# ── helpers ───────────────────────────────────────────────────────────


def _sample_df() -> pd.DataFrame:
    """Return a small DataFrame mimicking LeagueDashPlayerStats output."""
    return pd.DataFrame(
        [
            {
                "PLAYER_ID": 1,
                "PLAYER_NAME": "Alice",
                "TEAM_ID": 100,
                "TEAM_ABBREVIATION": "TST",
                "GP": 50,
                "MIN": 35.0,
                "PTS": 28.0,
                "REB": 5.0,
                "AST": 8.0,
                "STL": 1.5,
                "BLK": 0.5,
                "TOV": 3.0,
                "FG_PCT": 0.480,
                "FG3_PCT": 0.370,
                "FT_PCT": 0.900,
            },
            {
                "PLAYER_ID": 2,
                "PLAYER_NAME": "Bob",
                "TEAM_ID": 100,
                "TEAM_ABBREVIATION": "TST",
                "GP": 60,
                "MIN": 30.0,
                "PTS": 15.0,
                "REB": 10.0,
                "AST": 3.0,
                "STL": 0.8,
                "BLK": 2.0,
                "TOV": 1.5,
                "FG_PCT": 0.520,
                "FG3_PCT": 0.330,
                "FT_PCT": 0.750,
            },
        ]
    )


# ── compute_team_leaders ──────────────────────────────────────────────


def test_compute_team_leaders_basic() -> None:
    df = _sample_df()
    leaders = compute_team_leaders(df, "TST", "2024-25")
    assert leaders.team_abbreviation == "TST"
    assert leaders.season == "2024-25"

    leader_map = {ld.category: ld.player_name for ld in leaders.leaders}
    assert leader_map["Points"] == "Alice"
    assert leader_map["Rebounds"] == "Bob"
    assert leader_map["Blocks"] == "Bob"
    assert leader_map["Assists"] == "Alice"


# ── compute_team_summary ──────────────────────────────────────────────


@patch("nba_tracker.fetcher.resolve_team_id", return_value=100)
@patch("nba_tracker.fetcher.resolve_team_name", return_value="Test Team")
def test_compute_team_summary(mock_name, mock_id) -> None:
    df = _sample_df()
    summary = compute_team_summary(df, "TST", "2024-25")
    assert summary.num_players == 2
    assert summary.total_points == pytest.approx(43.0)
    assert summary.total_rebounds == pytest.approx(15.0)
    assert summary.avg_fg_pct == pytest.approx(0.50)


# ── compare_players ───────────────────────────────────────────────────


def test_compare_players() -> None:
    df = _sample_df()
    row_a = df.iloc[0]
    row_b = df.iloc[1]
    result = compare_players([row_a, row_b], "2024-25")
    assert len(result.players) == 2
    assert result.players[0].player_name == "Alice"
    assert result.players[1].player_name == "Bob"
    # Alice scores more points
    assert result.category_leaders["Points"] == "Alice"
    # Bob has more rebounds
    assert result.category_leaders["Rebounds"] == "Bob"
