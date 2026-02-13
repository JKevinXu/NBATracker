#!/usr/bin/env python3
"""Golden State Warriors — Recent Games Report.

Fetches the last N team game logs and top individual performances,
then prints a rich terminal report.

Run:
    python scripts/warriors_recent_games.py [--season 2024-25] [--games 10]
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from nba_api.stats.endpoints import TeamGameLog, PlayerGameLog
from nba_api.stats.static import teams as static_teams

TEAM_ABBR = "GSW"
TEAM_ID = 1610612744  # Golden State Warriors
API_DELAY = 0.6

console = Console()


# ── helpers ───────────────────────────────────────────────────────────


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _plus_minus(val: float) -> str:
    if val > 0:
        return f"[green]+{val:.0f}[/green]"
    elif val < 0:
        return f"[red]{val:.0f}[/red]"
    return "0"


def _wl_style(wl: str) -> str:
    return "[bold green]W[/bold green]" if wl == "W" else "[bold red]L[/bold red]"


def _streak(results: list[str]) -> str:
    """Compute current streak from most-recent-first results."""
    if not results:
        return "-"
    current = results[0]
    count = 0
    for r in results:
        if r == current:
            count += 1
        else:
            break
    style = "green" if current == "W" else "red"
    return f"[{style}]{current}{count}[/{style}]"


# ── data fetching ─────────────────────────────────────────────────────


def fetch_team_game_log(season: str) -> pd.DataFrame:
    """Return the full season game log for the Warriors."""
    time.sleep(API_DELAY)
    gl = TeamGameLog(
        team_id=TEAM_ID,
        season=season,
        season_type_all_star="Regular Season",
    )
    return gl.get_data_frames()[0]


def fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    """Return the full season game log for a single player."""
    time.sleep(API_DELAY)
    gl = PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    return gl.get_data_frames()[0]


# ── report sections ──────────────────────────────────────────────────


def print_header(season: str, n_games: int) -> None:
    title = Text()
    title.append("Golden State Warriors", style="bold yellow")
    title.append(f"  —  Last {n_games} Games Report ({season})", style="dim")
    console.print(Panel(title, border_style="yellow", padding=(1, 2)))
    console.print()


def print_record_snapshot(df: pd.DataFrame, recent: pd.DataFrame) -> None:
    """Overall and recent W-L record snapshot."""
    console.rule("[bold yellow]Record Snapshot[/bold yellow]")

    total_w = int(df.iloc[0]["W"]) if len(df) > 0 else 0
    total_l = int(df.iloc[0]["L"]) if len(df) > 0 else 0
    total_pct = f"{df.iloc[0]['W_PCT']:.3f}" if len(df) > 0 else "-"

    recent_wl = recent["WL"].tolist()
    recent_w = recent_wl.count("W")
    recent_l = recent_wl.count("L")
    streak_str = _streak(recent_wl)

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Season record", f"{total_w}-{total_l}  ({total_pct})")
    table.add_row(f"Last {len(recent)} games", f"{recent_w}-{recent_l}")
    table.add_row("Current streak", streak_str)
    table.add_row("Games played", str(len(df)))

    console.print(table)
    console.print()


def print_recent_games_table(recent: pd.DataFrame) -> None:
    """Game-by-game results for the recent window."""
    console.rule("[bold yellow]Game-by-Game Results[/bold yellow]")

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Date", style="dim")
    table.add_column("Matchup", style="cyan")
    table.add_column("W/L", justify="center")
    table.add_column("PTS", justify="right")
    table.add_column("OPP", justify="right")
    table.add_column("+/-", justify="right")
    table.add_column("FG%", justify="right")
    table.add_column("3P%", justify="right")
    table.add_column("REB", justify="right")
    table.add_column("AST", justify="right")
    table.add_column("TOV", justify="right")

    for _, row in recent.iterrows():
        pts = int(row["PTS"])
        # Estimate opponent points from context (we'll compute margin)
        # For team game log, we only have our own PTS. Use W/L + context.
        matchup = str(row["MATCHUP"])
        wl = str(row["WL"])
        fg_pct = _pct(float(row["FG_PCT"]))
        fg3_pct = _pct(float(row["FG3_PCT"]))
        reb = str(int(row["REB"]))
        ast = str(int(row["AST"]))
        tov = str(int(row["TOV"]))

        table.add_row(
            str(row["GAME_DATE"]),
            matchup,
            _wl_style(wl),
            str(pts),
            "",  # opponent pts not in this endpoint
            "",
            fg_pct,
            fg3_pct,
            reb,
            ast,
            tov,
        )

    console.print(table)
    console.print()


def print_recent_averages(recent: pd.DataFrame) -> None:
    """Averages across the recent game window."""
    console.rule("[bold yellow]Recent Averages[/bold yellow]")

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Stat", style="bold")
    table.add_column("Average", justify="right", style="green")

    n = len(recent)
    stats = {
        "Points": recent["PTS"].mean(),
        "Rebounds": recent["REB"].mean(),
        "Assists": recent["AST"].mean(),
        "Steals": recent["STL"].mean(),
        "Blocks": recent["BLK"].mean(),
        "Turnovers": recent["TOV"].mean(),
        "FG%": recent["FG_PCT"].mean(),
        "3P%": recent["FG3_PCT"].mean(),
        "FT%": recent["FT_PCT"].mean(),
        "3PM per game": recent["FG3M"].mean(),
        "Off Rebounds": recent["OREB"].mean(),
        "Def Rebounds": recent["DREB"].mean(),
    }

    for label, val in stats.items():
        if "%" in label:
            table.add_row(label, _pct(val))
        else:
            table.add_row(label, f"{val:.1f}")

    console.print(table)
    console.print()


def print_trends(recent: pd.DataFrame) -> None:
    """Show scoring and shooting trend across the recent games (oldest→newest)."""
    console.rule("[bold yellow]Scoring Trend (oldest → newest)[/bold yellow]")

    # Reverse so oldest is first (game log comes newest-first)
    trend = recent.iloc[::-1].reset_index(drop=True)

    max_pts = trend["PTS"].max()
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, padding=(0, 1))
    table.add_column("Game", justify="center", style="dim", width=5)
    table.add_column("vs", style="cyan", width=14)
    table.add_column("PTS", justify="right", width=5)
    table.add_column("W/L", justify="center", width=4)
    table.add_column("", width=30)

    for i, (_, row) in enumerate(trend.iterrows(), 1):
        pts = int(row["PTS"])
        bar_len = int(round(pts / max_pts * 25)) if max_pts > 0 else 0
        bar = "█" * bar_len + "░" * (25 - bar_len)
        # Extract opponent from matchup string
        matchup = str(row["MATCHUP"])
        opp = matchup.split(" ")[-1] if " " in matchup else matchup
        wl = str(row["WL"])
        color = "green" if wl == "W" else "red"
        table.add_row(
            str(i),
            opp,
            str(pts),
            _wl_style(wl),
            f"[{color}]{bar}[/{color}]",
        )

    console.print(table)
    console.print()


def print_highs_and_lows(recent: pd.DataFrame) -> None:
    """Best and worst games in the recent window."""
    console.rule("[bold yellow]Highs & Lows[/bold yellow]")

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("", style="bold", width=22)
    table.add_column("Value", justify="right", style="green", width=8)
    table.add_column("Game", style="cyan")

    def _game_label(row: pd.Series) -> str:
        return f"{row['GAME_DATE']}  {row['MATCHUP']}"

    # Highs
    best_pts_idx = recent["PTS"].idxmax()
    best_pts = recent.loc[best_pts_idx]
    table.add_row("Most points", str(int(best_pts["PTS"])), _game_label(best_pts))

    best_reb_idx = recent["REB"].idxmax()
    best_reb = recent.loc[best_reb_idx]
    table.add_row("Most rebounds", str(int(best_reb["REB"])), _game_label(best_reb))

    best_ast_idx = recent["AST"].idxmax()
    best_ast = recent.loc[best_ast_idx]
    table.add_row("Most assists", str(int(best_ast["AST"])), _game_label(best_ast))

    best_fg_idx = recent["FG_PCT"].idxmax()
    best_fg = recent.loc[best_fg_idx]
    table.add_row("Best FG%", _pct(best_fg["FG_PCT"]), _game_label(best_fg))

    best_3p_idx = recent["FG3_PCT"].idxmax()
    best_3p = recent.loc[best_3p_idx]
    table.add_row("Best 3P%", _pct(best_3p["FG3_PCT"]), _game_label(best_3p))

    # Lows
    table.add_row("", "", "")
    worst_pts_idx = recent["PTS"].idxmin()
    worst_pts = recent.loc[worst_pts_idx]
    table.add_row("Fewest points", str(int(worst_pts["PTS"])), _game_label(worst_pts))

    worst_fg_idx = recent["FG_PCT"].idxmin()
    worst_fg = recent.loc[worst_fg_idx]
    table.add_row("Worst FG%", _pct(worst_fg["FG_PCT"]), _game_label(worst_fg))

    most_tov_idx = recent["TOV"].idxmax()
    most_tov = recent.loc[most_tov_idx]
    table.add_row("Most turnovers", str(int(most_tov["TOV"])), _game_label(most_tov))

    console.print(table)
    console.print()


def print_top_individual_performances(recent: pd.DataFrame, season: str) -> None:
    """Fetch player game logs for key Warriors and highlight best recent games."""
    console.rule("[bold yellow]Top Individual Performances[/bold yellow]")

    # Key players to check (id, name)
    key_players = [
        (201939, "Stephen Curry"),
        (203110, "Draymond Green"),
        (1629621, "Andrew Wiggins"),
        (1630228, "Jonathan Kuminga"),
        (1631218, "Brandin Podziemski"),
        (1627790, "Buddy Hield"),
    ]

    # Get game IDs from recent team games
    recent_game_ids = set(recent["Game_ID"].tolist())

    all_performances: list[dict] = []

    console.print("  [dim]Fetching individual game logs…[/dim]")
    for player_id, player_name in key_players:
        try:
            pdf = fetch_player_game_log(player_id, season)
            # Filter to only games in our recent window
            player_recent = pdf[pdf["Game_ID"].isin(recent_game_ids)]
            for _, row in player_recent.iterrows():
                all_performances.append(
                    {
                        "player": player_name,
                        "date": str(row.get("GAME_DATE", "")),
                        "matchup": str(row.get("MATCHUP", "")),
                        "pts": int(row.get("PTS", 0)),
                        "reb": int(row.get("REB", 0)),
                        "ast": int(row.get("AST", 0)),
                        "stl": int(row.get("STL", 0)),
                        "blk": int(row.get("BLK", 0)),
                        "fg_pct": float(row.get("FG_PCT", 0)),
                        "fg3_pct": float(row.get("FG3_PCT", 0)),
                        "min": float(row.get("MIN", 0)),
                    }
                )
        except Exception:
            pass  # skip if player data unavailable

    if not all_performances:
        console.print("  [dim]No individual data available.[/dim]\n")
        return

    # Sort by points and show top 10 performances
    all_performances.sort(key=lambda x: x["pts"], reverse=True)
    top = all_performances[:10]

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Player", style="cyan", width=22)
    table.add_column("Date", style="dim")
    table.add_column("vs", style="dim")
    table.add_column("PTS", justify="right", style="bold green")
    table.add_column("REB", justify="right")
    table.add_column("AST", justify="right")
    table.add_column("STL", justify="right")
    table.add_column("BLK", justify="right")
    table.add_column("FG%", justify="right")
    table.add_column("MIN", justify="right")

    for p in top:
        opp = p["matchup"].split(" ")[-1] if " " in p["matchup"] else p["matchup"]
        table.add_row(
            p["player"],
            p["date"],
            opp,
            str(p["pts"]),
            str(p["reb"]),
            str(p["ast"]),
            str(p["stl"]),
            str(p["blk"]),
            _pct(p["fg_pct"]),
            f"{p['min']:.0f}",
        )

    console.print(table)
    console.print(f"  [dim]Top {len(top)} individual scoring performances[/dim]\n")


def print_home_vs_away(recent: pd.DataFrame) -> None:
    """Home vs away split within the recent window."""
    console.rule("[bold yellow]Home vs Away Split[/bold yellow]")

    home = recent[recent["MATCHUP"].str.contains("vs.")]
    away = recent[~recent["MATCHUP"].str.contains("vs.")]

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("", style="bold", width=10)
    table.add_column("Games", justify="right")
    table.add_column("W-L", justify="center")
    table.add_column("Avg PTS", justify="right")
    table.add_column("Avg REB", justify="right")
    table.add_column("Avg AST", justify="right")
    table.add_column("Avg FG%", justify="right")
    table.add_column("Avg 3P%", justify="right")

    for label, subset in [("Home", home), ("Away", away)]:
        if subset.empty:
            table.add_row(label, "0", "-", "-", "-", "-", "-", "-")
            continue
        n = len(subset)
        w = (subset["WL"] == "W").sum()
        l = n - w
        table.add_row(
            label,
            str(n),
            f"{w}-{l}",
            f"{subset['PTS'].mean():.1f}",
            f"{subset['REB'].mean():.1f}",
            f"{subset['AST'].mean():.1f}",
            _pct(subset["FG_PCT"].mean()),
            _pct(subset["FG3_PCT"].mean()),
        )

    console.print(table)
    console.print()


# ── main ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
@click.option("--games", default=10, type=int, help="Number of recent games to analyze.")
def main(season: str, games: int) -> None:
    """Generate a recent-games report for the Golden State Warriors."""
    print_header(season, games)

    console.print("[dim]Fetching team game log…[/dim]")
    df = fetch_team_game_log(season)

    if df.empty:
        console.print("[red]No game data found.[/red]")
        return

    recent = df.head(games).copy()

    print_record_snapshot(df, recent)
    print_recent_games_table(recent)
    print_recent_averages(recent)
    print_trends(recent)
    print_highs_and_lows(recent)
    print_home_vs_away(recent)
    print_top_individual_performances(recent, season)

    console.rule("[bold yellow]Report Complete[/bold yellow]")
    console.print()


if __name__ == "__main__":
    main()
