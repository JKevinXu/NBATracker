"""Data fetching layer — pulls data from nba_api with local caching."""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
from nba_api.stats.endpoints import (
    CommonTeamRoster,
    LeagueDashPlayerStats,
    PlayerCareerStats,
)
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams

from nba_tracker import cache
from nba_tracker.models import Player

# ── helpers ───────────────────────────────────────────────────────────

# Delay between API calls to respect NBA.com rate limits (seconds).
_API_DELAY = 0.6


def _delay() -> None:
    time.sleep(_API_DELAY)


def resolve_team_id(team_abbr: str) -> int:
    """Return the NBA team ID for a 3-letter abbreviation (e.g. 'LAL')."""
    teams = static_teams.get_teams()
    for t in teams:
        if t["abbreviation"].upper() == team_abbr.upper():
            return int(t["id"])
    raise ValueError(
        f"Unknown team abbreviation '{team_abbr}'. "
        f"Valid values: {', '.join(t['abbreviation'] for t in teams)}"
    )


def resolve_team_name(team_abbr: str) -> str:
    """Return the full team name for a 3-letter abbreviation."""
    teams = static_teams.get_teams()
    for t in teams:
        if t["abbreviation"].upper() == team_abbr.upper():
            return str(t["full_name"])
    raise ValueError(f"Unknown team abbreviation '{team_abbr}'.")


def resolve_player_id(player_name: str) -> Optional[int]:
    """Return a player ID by full name (case-insensitive, partial match)."""
    all_players = static_players.get_players()
    name_lower = player_name.lower()
    # Try exact match first
    for p in all_players:
        if p["full_name"].lower() == name_lower:
            return int(p["id"])
    # Partial match fallback
    for p in all_players:
        if name_lower in p["full_name"].lower():
            return int(p["id"])
    return None


# ── roster ────────────────────────────────────────────────────────────


def get_team_roster(team_abbr: str, season: str) -> List[Player]:
    """Fetch the roster for *team_abbr* in *season*, with caching."""
    cache_key = f"roster:{team_abbr}:{season}"
    cached = cache.get(cache_key)
    if cached is not None:
        return [Player(**p) for p in cached]

    team_id = resolve_team_id(team_abbr)
    _delay()
    roster_data = CommonTeamRoster(
        team_id=team_id, season=season
    ).get_data_frames()[0]

    players: List[Player] = []
    for _, row in roster_data.iterrows():
        players.append(
            Player(
                player_id=int(row.get("PLAYER_ID", 0)),
                player_name=str(row.get("PLAYER", "")),
                team_id=team_id,
                team_abbreviation=team_abbr,
                jersey_number=str(row.get("NUM", "")),
                position=str(row.get("POSITION", "")),
                height=str(row.get("HEIGHT", "")),
                weight=str(row.get("WEIGHT", "")),
                season=season,
            )
        )

    cache.put(cache_key, [p.model_dump() for p in players])
    return players


# ── team player stats ─────────────────────────────────────────────────


def get_team_player_stats(team_abbr: str, season: str) -> pd.DataFrame:
    """Return a DataFrame of per-player season stats for *team_abbr*.

    Columns include: PLAYER_NAME, GP, MIN, PTS, REB, AST, STL, BLK, TOV,
    FG_PCT, FG3_PCT, FT_PCT, and more from LeagueDashPlayerStats.
    """
    cache_key = f"team_stats:{team_abbr}:{season}"
    cached = cache.get(cache_key)
    if cached is not None:
        return pd.DataFrame(cached)

    team_id = resolve_team_id(team_abbr)
    _delay()
    df = LeagueDashPlayerStats(
        team_id_nullable=team_id,
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    # Keep useful columns and rename for clarity
    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    cache.put(cache_key, df.to_dict(orient="records"))
    return df


# ── single player stats (for comparison) ──────────────────────────────


def get_player_stats_by_name(
    player_name: str, season: str
) -> Optional[pd.Series]:
    """Return a single-row Series of season stats for *player_name*, or None."""
    cache_key = f"player:{player_name.lower()}:{season}"
    cached = cache.get(cache_key)
    if cached is not None:
        return pd.Series(cached)

    player_id = resolve_player_id(player_name)
    if player_id is None:
        return None

    _delay()
    # Use league-wide player stats and filter
    df = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    match = df[df["PLAYER_ID"] == player_id]
    if match.empty:
        return None

    row = match.iloc[0]
    cache.put(cache_key, row.to_dict())
    return row
