#!/usr/bin/env python3
"""Golden State Warriors — comprehensive team analysis script.

Run:
    python scripts/warriors_analysis.py [--season 2024-25]
"""

from __future__ import annotations

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from nba_tracker.fetcher import get_team_roster, get_team_player_stats
from nba_tracker.analyzer import (
    STAT_CATEGORIES,
    CATEGORY_LABELS,
    compute_team_leaders,
    compute_team_summary,
)
from nba_tracker.exporter import _fmt

TEAM = "GSW"
console = Console()


# ── helpers ───────────────────────────────────────────────────────────


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _bar(value: float, max_value: float, width: int = 25) -> str:
    """Return a simple text bar chart segment."""
    if max_value == 0:
        return ""
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


# ── sections ──────────────────────────────────────────────────────────


def print_header(season: str) -> None:
    title = Text()
    title.append("Golden State Warriors", style="bold yellow")
    title.append(f"  —  {season} Season Analysis", style="dim")
    console.print(Panel(title, border_style="yellow", padding=(1, 2)))
    console.print()


def print_roster(season: str) -> None:
    console.rule("[bold yellow]Roster[/bold yellow]")
    players = get_team_roster(TEAM, season)
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("#", justify="center", style="bold yellow")
    table.add_column("Player", style="cyan")
    table.add_column("Pos", justify="center")
    table.add_column("Ht", justify="center")
    table.add_column("Wt", justify="center")

    for p in sorted(players, key=lambda x: x.position or "Z"):
        table.add_row(p.jersey_number, p.player_name, p.position, p.height, p.weight)

    console.print(table)
    console.print(f"  [dim]Total players: {len(players)}[/dim]\n")


def print_stats(df: pd.DataFrame, season: str) -> None:
    console.rule("[bold yellow]Per-Player Stats (Per Game)[/bold yellow]")

    show_cols = [
        "PLAYER_NAME", "GP", "MIN", "PTS", "REB", "AST",
        "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT",
    ]
    col_labels = {
        "PLAYER_NAME": "Player", "GP": "GP", "MIN": "MIN", "PTS": "PTS",
        "REB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK",
        "TOV": "TOV", "FG_PCT": "FG%", "FG3_PCT": "3P%", "FT_PCT": "FT%",
    }
    cols = [c for c in show_cols if c in df.columns]

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    for c in cols:
        justify = "left" if c == "PLAYER_NAME" else "right"
        style = "cyan" if c == "PLAYER_NAME" else None
        table.add_column(col_labels.get(c, c), justify=justify, style=style)

    sorted_df = df.sort_values("PTS", ascending=False) if "PTS" in df.columns else df
    for _, row in sorted_df.iterrows():
        table.add_row(*[_fmt(row[c], c) for c in cols])

    console.print(table)
    console.print()


def print_leaders(df: pd.DataFrame, season: str) -> None:
    console.rule("[bold yellow]Statistical Leaders[/bold yellow]")
    leaders = compute_team_leaders(df, TEAM, season)

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Category", style="bold")
    table.add_column("Leader", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for ld in leaders.leaders:
        val = _pct(ld.value) if "%" in ld.category else f"{ld.value:.1f}"
        table.add_row(ld.category, ld.player_name, val)

    console.print(table)
    console.print()


def print_summary(df: pd.DataFrame, season: str) -> None:
    console.rule("[bold yellow]Team Aggregate Summary[/bold yellow]")
    summary = compute_team_summary(df, TEAM, season)

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Roster size", str(summary.num_players))
    table.add_row("Total PTS (avg sum)", f"{summary.total_points:.1f}")
    table.add_row("Total REB", f"{summary.total_rebounds:.1f}")
    table.add_row("Total AST", f"{summary.total_assists:.1f}")
    table.add_row("Total STL", f"{summary.total_steals:.1f}")
    table.add_row("Total BLK", f"{summary.total_blocks:.1f}")
    table.add_row("Avg FG%", _pct(summary.avg_fg_pct))
    table.add_row("Avg 3P%", _pct(summary.avg_fg3_pct))
    table.add_row("Avg FT%", _pct(summary.avg_ft_pct))

    console.print(table)
    console.print()


def print_scoring_distribution(df: pd.DataFrame) -> None:
    """Visual bar chart of scoring distribution across the roster."""
    console.rule("[bold yellow]Scoring Distribution[/bold yellow]")
    if "PTS" not in df.columns:
        console.print("[dim]No PTS data available.[/dim]")
        return

    sorted_df = df.sort_values("PTS", ascending=False).head(10)
    max_pts = sorted_df["PTS"].max()

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("Player", style="cyan", width=22)
    table.add_column("PPG", justify="right", style="green", width=6)
    table.add_column("Distribution", width=30)

    for _, row in sorted_df.iterrows():
        pts = float(row["PTS"])
        bar = _bar(pts, max_pts)
        table.add_row(str(row["PLAYER_NAME"]), f"{pts:.1f}", f"[yellow]{bar}[/yellow]")

    console.print(table)
    console.print(f"  [dim]Showing top {len(sorted_df)} scorers[/dim]\n")


def print_offensive_vs_defensive(df: pd.DataFrame) -> None:
    """Break down top offensive and defensive contributors."""
    console.rule("[bold yellow]Offensive vs Defensive Breakdown[/bold yellow]")

    off_cols = {"PTS": "Points", "AST": "Assists", "FG_PCT": "FG%", "FG3_PCT": "3P%"}
    def_cols = {"REB": "Rebounds", "STL": "Steals", "BLK": "Blocks"}

    # Offensive top contributors
    console.print("\n  [bold]Top Offensive Contributors[/bold]")
    off_table = Table(box=box.SIMPLE, show_lines=False, padding=(0, 1))
    off_table.add_column("Player", style="cyan", width=22)
    off_table.add_column("PTS", justify="right")
    off_table.add_column("AST", justify="right")
    off_table.add_column("FG%", justify="right")
    off_table.add_column("3P%", justify="right")

    # Rank by combined offensive score: PTS + AST (simple proxy)
    off_df = df.copy()
    if "PTS" in off_df.columns and "AST" in off_df.columns:
        off_df["_off_score"] = off_df["PTS"] + off_df["AST"]
        off_df = off_df.sort_values("_off_score", ascending=False).head(5)
        for _, row in off_df.iterrows():
            off_table.add_row(
                str(row["PLAYER_NAME"]),
                f"{float(row.get('PTS', 0)):.1f}",
                f"{float(row.get('AST', 0)):.1f}",
                _pct(float(row.get("FG_PCT", 0))),
                _pct(float(row.get("FG3_PCT", 0))),
            )
    console.print(off_table)

    # Defensive top contributors
    console.print("\n  [bold]Top Defensive Contributors[/bold]")
    def_table = Table(box=box.SIMPLE, show_lines=False, padding=(0, 1))
    def_table.add_column("Player", style="cyan", width=22)
    def_table.add_column("REB", justify="right")
    def_table.add_column("STL", justify="right")
    def_table.add_column("BLK", justify="right")

    def_df = df.copy()
    if all(c in def_df.columns for c in ["REB", "STL", "BLK"]):
        def_df["_def_score"] = def_df["REB"] + def_df["STL"] * 2 + def_df["BLK"] * 2
        def_df = def_df.sort_values("_def_score", ascending=False).head(5)
        for _, row in def_df.iterrows():
            def_table.add_row(
                str(row["PLAYER_NAME"]),
                f"{float(row.get('REB', 0)):.1f}",
                f"{float(row.get('STL', 0)):.1f}",
                f"{float(row.get('BLK', 0)):.1f}",
            )
    console.print(def_table)
    console.print()


def print_efficiency(df: pd.DataFrame) -> None:
    """Highlight players with best shooting efficiency (min 10 GP)."""
    console.rule("[bold yellow]Shooting Efficiency (min 10 GP)[/bold yellow]")

    eff_df = df.copy()
    if "GP" in eff_df.columns:
        eff_df = eff_df[eff_df["GP"] >= 10]

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("Player", style="cyan", width=22)
    table.add_column("GP", justify="right")
    table.add_column("FG%", justify="right")
    table.add_column("3P%", justify="right")
    table.add_column("FT%", justify="right")
    table.add_column("PTS", justify="right")

    if "FG_PCT" in eff_df.columns:
        eff_df = eff_df.sort_values("FG_PCT", ascending=False).head(8)

    for _, row in eff_df.iterrows():
        table.add_row(
            str(row["PLAYER_NAME"]),
            str(int(row.get("GP", 0))),
            _pct(float(row.get("FG_PCT", 0))),
            _pct(float(row.get("FG3_PCT", 0))),
            _pct(float(row.get("FT_PCT", 0))),
            f"{float(row.get('PTS', 0)):.1f}",
        )

    console.print(table)
    console.print()


# ── main ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
def main(season: str) -> None:
    """Run a full Golden State Warriors analysis for the given season."""
    print_header(season)

    console.print("[dim]Fetching roster…[/dim]")
    print_roster(season)

    console.print("[dim]Fetching player stats…[/dim]")
    df = get_team_player_stats(TEAM, season)

    print_stats(df, season)
    print_leaders(df, season)
    print_summary(df, season)
    print_scoring_distribution(df)
    print_offensive_vs_defensive(df)
    print_efficiency(df)

    console.rule("[bold yellow]Analysis Complete[/bold yellow]")
    console.print(
        f"\n  [dim]Data cached locally. Run [bold]python -m nba_tracker.cli cache clear[/bold] to refresh.[/dim]\n"
    )


if __name__ == "__main__":
    main()
