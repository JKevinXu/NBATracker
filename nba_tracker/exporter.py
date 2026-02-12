"""Output formatting — rich terminal tables, CSV, and JSON export."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from nba_tracker.models import (
    ComparisonResult,
    Player,
    TeamLeaders,
    TeamSummary,
)

console = Console()

# ── roster table ──────────────────────────────────────────────────────


def render_roster_table(
    players: List[Player], team_abbr: str, season: str
) -> None:
    """Print a rich table of the team roster."""
    table = Table(
        title=f"{team_abbr} Roster — {season}",
        show_lines=True,
    )
    table.add_column("#", justify="center", style="bold")
    table.add_column("Player", style="cyan")
    table.add_column("Pos", justify="center")
    table.add_column("Height", justify="center")
    table.add_column("Weight", justify="center")

    for p in players:
        table.add_row(
            p.jersey_number,
            p.player_name,
            p.position,
            p.height,
            p.weight,
        )

    console.print(table)


# ── per-player stats table ────────────────────────────────────────────

_STAT_COLS = [
    "PLAYER_NAME",
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

_COL_LABELS = {
    "PLAYER_NAME": "Player",
    "GP": "GP",
    "MIN": "MIN",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
    "FG_PCT": "FG%",
    "FG3_PCT": "3P%",
    "FT_PCT": "FT%",
}


def _fmt(val: object, col: str) -> str:
    """Format a cell value for display."""
    if col in ("FG_PCT", "FG3_PCT", "FT_PCT"):
        try:
            return f"{float(val) * 100:.1f}%"
        except (ValueError, TypeError):
            return str(val)
    if col in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"):
        try:
            return f"{float(val):.1f}"
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def render_stats_table(
    df: pd.DataFrame, team_abbr: str, season: str
) -> None:
    """Print a rich table of per-player season stats."""
    table = Table(
        title=f"{team_abbr} Player Stats — {season} (Per Game)",
        show_lines=True,
    )

    cols = [c for c in _STAT_COLS if c in df.columns]
    for c in cols:
        justify = "left" if c == "PLAYER_NAME" else "right"
        style = "cyan" if c == "PLAYER_NAME" else None
        table.add_column(_COL_LABELS.get(c, c), justify=justify, style=style)

    for _, row in df.iterrows():
        table.add_row(*[_fmt(row[c], c) for c in cols])

    console.print(table)


# ── leaders table ─────────────────────────────────────────────────────


def render_leaders_table(leaders: TeamLeaders) -> None:
    """Print a rich table of statistical leaders."""
    table = Table(
        title=f"{leaders.team_abbreviation} Leaders — {leaders.season}",
        show_lines=True,
    )
    table.add_column("Category", style="bold")
    table.add_column("Player", style="cyan")
    table.add_column("Value", justify="right")

    for ld in leaders.leaders:
        # Format percentages
        if "%" in ld.category:
            val_str = f"{ld.value * 100:.1f}%"
        else:
            val_str = f"{ld.value:.1f}"
        table.add_row(ld.category, ld.player_name, val_str)

    console.print(table)


# ── summary table ─────────────────────────────────────────────────────


def render_summary_table(summary: TeamSummary) -> None:
    """Print an aggregate summary for a team."""
    table = Table(
        title=f"{summary.team_name} ({summary.team_abbreviation}) — {summary.season} Summary",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Players on roster", str(summary.num_players))
    table.add_row("Total PTS (per game avg sum)", f"{summary.total_points:.1f}")
    table.add_row("Total REB", f"{summary.total_rebounds:.1f}")
    table.add_row("Total AST", f"{summary.total_assists:.1f}")
    table.add_row("Total STL", f"{summary.total_steals:.1f}")
    table.add_row("Total BLK", f"{summary.total_blocks:.1f}")
    table.add_row("Avg FG%", f"{summary.avg_fg_pct * 100:.1f}%")
    table.add_row("Avg 3P%", f"{summary.avg_fg3_pct * 100:.1f}%")
    table.add_row("Avg FT%", f"{summary.avg_ft_pct * 100:.1f}%")

    console.print(table)


# ── comparison table ──────────────────────────────────────────────────


def render_comparison_table(result: ComparisonResult) -> None:
    """Print a side-by-side player comparison table."""
    table = Table(
        title=f"Player Comparison — {result.season}",
        show_lines=True,
    )
    table.add_column("Stat", style="bold")
    for ps in result.players:
        table.add_column(ps.player_name, justify="right", style="cyan")
    table.add_column("Leader", justify="center", style="green")

    rows = [
        ("Points", "points"),
        ("Rebounds", "rebounds"),
        ("Assists", "assists"),
        ("Steals", "steals"),
        ("Blocks", "blocks"),
        ("Turnovers", "turnovers"),
        ("FG%", "fg_pct"),
        ("3P%", "fg3_pct"),
        ("FT%", "ft_pct"),
        ("Minutes", "minutes"),
        ("Games", "games_played"),
    ]

    for label, attr in rows:
        vals = []
        for ps in result.players:
            v = getattr(ps, attr, 0)
            if "%" in label:
                vals.append(f"{v * 100:.1f}%")
            elif attr == "games_played":
                vals.append(str(int(v)))
            else:
                vals.append(f"{v:.1f}")
        leader = result.category_leaders.get(label, "")
        table.add_row(label, *vals, leader)

    console.print(table)


# ── file export ───────────────────────────────────────────────────────


def export_stats(
    df: pd.DataFrame,
    team_abbr: str,
    season: str,
    fmt: str,
    output_path: Optional[str] = None,
) -> str:
    """Export a stats DataFrame to CSV or JSON and return the file path."""
    if output_path is None:
        safe_season = season.replace("-", "_")
        ext = "csv" if fmt == "csv" else "json"
        output_path = f"{team_abbr}_{safe_season}_stats.{ext}"

    path = Path(output_path)

    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", indent=2)

    return str(path)
