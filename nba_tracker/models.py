"""Pydantic data models for NBA player and team statistics."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Player(BaseModel):
    """Basic player information from a team roster."""

    player_id: int
    player_name: str
    team_id: int
    team_abbreviation: str
    jersey_number: str = ""
    position: str = ""
    height: str = ""
    weight: str = ""
    season: str = ""


class PlayerStats(BaseModel):
    """Season-average statistics for a single player."""

    player_id: int
    player_name: str
    team_id: int
    team_abbreviation: str
    games_played: int = Field(0, alias="GP")
    minutes: float = Field(0.0, alias="MIN")
    points: float = Field(0.0, alias="PTS")
    rebounds: float = Field(0.0, alias="REB")
    assists: float = Field(0.0, alias="AST")
    steals: float = Field(0.0, alias="STL")
    blocks: float = Field(0.0, alias="BLK")
    turnovers: float = Field(0.0, alias="TOV")
    fg_pct: float = Field(0.0, alias="FG_PCT")
    fg3_pct: float = Field(0.0, alias="FG3_PCT")
    ft_pct: float = Field(0.0, alias="FT_PCT")

    model_config = {"populate_by_name": True}


class TeamSummary(BaseModel):
    """Aggregated statistics for an entire team."""

    team_id: int
    team_name: str
    team_abbreviation: str
    season: str
    num_players: int = 0
    total_points: float = 0.0
    total_rebounds: float = 0.0
    total_assists: float = 0.0
    total_steals: float = 0.0
    total_blocks: float = 0.0
    avg_fg_pct: float = 0.0
    avg_fg3_pct: float = 0.0
    avg_ft_pct: float = 0.0


class TeamLeader(BaseModel):
    """A single statistical category leader for a team."""

    category: str
    player_name: str
    value: float


class TeamLeaders(BaseModel):
    """Collection of statistical leaders for a team."""

    team_abbreviation: str
    season: str
    leaders: List[TeamLeader] = []


class ComparisonResult(BaseModel):
    """Side-by-side comparison of multiple players."""

    season: str
    players: List[PlayerStats] = []
    category_leaders: dict[str, str] = {}
