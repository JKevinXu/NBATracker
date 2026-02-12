"""Statistical analysis helpers for NBA player and team data."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from nba_tracker.models import (
    ComparisonResult,
    PlayerStats,
    TeamLeader,
    TeamLeaders,
    TeamSummary,
)

# Stat categories used for leaders / comparisons
STAT_CATEGORIES: List[str] = [
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "MIN",
]

CATEGORY_LABELS: Dict[str, str] = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
    "FG_PCT": "FG%",
    "FG3_PCT": "3P%",
    "FT_PCT": "FT%",
    "MIN": "Minutes",
    "GP": "Games Played",
}


# ── team leaders ──────────────────────────────────────────────────────


def compute_team_leaders(
    df: pd.DataFrame, team_abbr: str, season: str
) -> TeamLeaders:
    """Identify the statistical leader in each major category for a team.

    Parameters
    ----------
    df : DataFrame
        Per-player stats (from ``get_team_player_stats``).
    team_abbr : str
        Team abbreviation (e.g. ``"LAL"``).
    season : str
        Season string (e.g. ``"2024-25"``).

    Returns
    -------
    TeamLeaders
        A model with one ``TeamLeader`` per category.
    """
    leaders: List[TeamLeader] = []
    for cat in STAT_CATEGORIES:
        if cat not in df.columns:
            continue
        idx = df[cat].idxmax()
        leaders.append(
            TeamLeader(
                category=CATEGORY_LABELS.get(cat, cat),
                player_name=str(df.loc[idx, "PLAYER_NAME"]),
                value=float(df.loc[idx, cat]),
            )
        )
    return TeamLeaders(
        team_abbreviation=team_abbr,
        season=season,
        leaders=leaders,
    )


# ── team summary / aggregations ───────────────────────────────────────


def compute_team_summary(
    df: pd.DataFrame, team_abbr: str, season: str
) -> TeamSummary:
    """Compute aggregate statistics across all players on a team.

    Parameters
    ----------
    df : DataFrame
        Per-player stats.
    team_abbr : str
        Team abbreviation.
    season : str
        Season string.

    Returns
    -------
    TeamSummary
    """
    from nba_tracker.fetcher import resolve_team_id, resolve_team_name

    team_id = resolve_team_id(team_abbr)
    team_name = resolve_team_name(team_abbr)

    def _safe_sum(col: str) -> float:
        return float(df[col].sum()) if col in df.columns else 0.0

    def _safe_mean(col: str) -> float:
        return float(df[col].mean()) if col in df.columns else 0.0

    return TeamSummary(
        team_id=team_id,
        team_name=team_name,
        team_abbreviation=team_abbr,
        season=season,
        num_players=len(df),
        total_points=_safe_sum("PTS"),
        total_rebounds=_safe_sum("REB"),
        total_assists=_safe_sum("AST"),
        total_steals=_safe_sum("STL"),
        total_blocks=_safe_sum("BLK"),
        avg_fg_pct=_safe_mean("FG_PCT"),
        avg_fg3_pct=_safe_mean("FG3_PCT"),
        avg_ft_pct=_safe_mean("FT_PCT"),
    )


# ── player comparison ─────────────────────────────────────────────────


def compare_players(
    player_rows: List[pd.Series], season: str
) -> ComparisonResult:
    """Build a side-by-side comparison of multiple players.

    Parameters
    ----------
    player_rows : list[pd.Series]
        One Series per player (from ``get_player_stats_by_name``).
    season : str
        Season string.

    Returns
    -------
    ComparisonResult
    """
    stats_list: List[PlayerStats] = []
    for row in player_rows:
        stats_list.append(
            PlayerStats(
                player_id=int(row.get("PLAYER_ID", 0)),
                player_name=str(row.get("PLAYER_NAME", "")),
                team_id=int(row.get("TEAM_ID", 0)),
                team_abbreviation=str(row.get("TEAM_ABBREVIATION", "")),
                GP=int(row.get("GP", 0)),
                MIN=float(row.get("MIN", 0)),
                PTS=float(row.get("PTS", 0)),
                REB=float(row.get("REB", 0)),
                AST=float(row.get("AST", 0)),
                STL=float(row.get("STL", 0)),
                BLK=float(row.get("BLK", 0)),
                TOV=float(row.get("TOV", 0)),
                FG_PCT=float(row.get("FG_PCT", 0)),
                FG3_PCT=float(row.get("FG3_PCT", 0)),
                FT_PCT=float(row.get("FT_PCT", 0)),
            )
        )

    # Determine who leads each category
    category_leaders: Dict[str, str] = {}
    compare_cats = ["PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "MIN"]
    for cat in compare_cats:
        best_val = -1.0
        best_name = ""
        attr = cat.lower()
        # Map aliases
        attr_map = {
            "pts": "points",
            "reb": "rebounds",
            "ast": "assists",
            "stl": "steals",
            "blk": "blocks",
            "tov": "turnovers",
            "fg_pct": "fg_pct",
            "fg3_pct": "fg3_pct",
            "ft_pct": "ft_pct",
            "min": "minutes",
        }
        field = attr_map.get(attr, attr)
        for ps in stats_list:
            val = getattr(ps, field, 0.0)
            if val > best_val:
                best_val = val
                best_name = ps.player_name
        category_leaders[CATEGORY_LABELS.get(cat, cat)] = best_name

    return ComparisonResult(
        season=season,
        players=stats_list,
        category_leaders=category_leaders,
    )
