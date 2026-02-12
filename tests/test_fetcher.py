"""Tests for nba_tracker.fetcher — resolution helpers and caching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nba_tracker.fetcher import (
    resolve_player_id,
    resolve_team_id,
    resolve_team_name,
)


# ── resolve_team_id ───────────────────────────────────────────────────

_FAKE_TEAMS = [
    {"id": 1610612747, "abbreviation": "LAL", "full_name": "Los Angeles Lakers"},
    {"id": 1610612738, "abbreviation": "BOS", "full_name": "Boston Celtics"},
]


@patch("nba_tracker.fetcher.static_teams.get_teams", return_value=_FAKE_TEAMS)
def test_resolve_team_id_valid(mock_get: MagicMock) -> None:
    assert resolve_team_id("LAL") == 1610612747
    assert resolve_team_id("lal") == 1610612747


@patch("nba_tracker.fetcher.static_teams.get_teams", return_value=_FAKE_TEAMS)
def test_resolve_team_id_invalid(mock_get: MagicMock) -> None:
    with pytest.raises(ValueError, match="Unknown team"):
        resolve_team_id("XYZ")


# ── resolve_team_name ─────────────────────────────────────────────────


@patch("nba_tracker.fetcher.static_teams.get_teams", return_value=_FAKE_TEAMS)
def test_resolve_team_name(mock_get: MagicMock) -> None:
    assert resolve_team_name("BOS") == "Boston Celtics"


@patch("nba_tracker.fetcher.static_teams.get_teams", return_value=_FAKE_TEAMS)
def test_resolve_team_name_invalid(mock_get: MagicMock) -> None:
    with pytest.raises(ValueError):
        resolve_team_name("XYZ")


# ── resolve_player_id ─────────────────────────────────────────────────

_FAKE_PLAYERS = [
    {"id": 2544, "full_name": "LeBron James"},
    {"id": 201566, "full_name": "Russell Westbrook"},
]


@patch("nba_tracker.fetcher.static_players.get_players", return_value=_FAKE_PLAYERS)
def test_resolve_player_id_exact(mock_get: MagicMock) -> None:
    assert resolve_player_id("LeBron James") == 2544


@patch("nba_tracker.fetcher.static_players.get_players", return_value=_FAKE_PLAYERS)
def test_resolve_player_id_partial(mock_get: MagicMock) -> None:
    assert resolve_player_id("lebron") == 2544


@patch("nba_tracker.fetcher.static_players.get_players", return_value=_FAKE_PLAYERS)
def test_resolve_player_id_not_found(mock_get: MagicMock) -> None:
    assert resolve_player_id("Unknown Player") is None


# ── get_team_roster (cache hit path) ──────────────────────────────────


@patch("nba_tracker.fetcher.cache")
def test_get_team_roster_cache_hit(mock_cache: MagicMock) -> None:
    from nba_tracker.fetcher import get_team_roster

    mock_cache.get.return_value = [
        {
            "player_id": 1,
            "player_name": "Test Player",
            "team_id": 100,
            "team_abbreviation": "TST",
            "jersey_number": "0",
            "position": "G",
            "height": "6-3",
            "weight": "200",
            "season": "2024-25",
        }
    ]
    players = get_team_roster("TST", "2024-25")
    assert len(players) == 1
    assert players[0].player_name == "Test Player"


# ── get_team_player_stats (cache hit path) ────────────────────────────


@patch("nba_tracker.fetcher.cache")
def test_get_team_player_stats_cache_hit(mock_cache: MagicMock) -> None:
    from nba_tracker.fetcher import get_team_player_stats

    mock_cache.get.return_value = [
        {"PLAYER_NAME": "Test", "PTS": 25.0, "REB": 7.0, "AST": 5.0}
    ]
    df = get_team_player_stats("TST", "2024-25")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["PTS"] == 25.0
