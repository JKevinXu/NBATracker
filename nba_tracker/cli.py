"""CLI entry point for the NBA Tracker tool."""

from __future__ import annotations

import click

from nba_tracker import __version__


@click.group()
@click.version_option(version=__version__, prog_name="nba-tracker")
def main() -> None:
    """NBA Player Statistics Analyzer — per-team player stats from the command line."""


# ── team command group ────────────────────────────────────────────────
@main.group()
def team() -> None:
    """Commands for team-level data (roster, stats, leaders, summary)."""


@team.command()
@click.argument("team_abbr")
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
def roster(team_abbr: str, season: str) -> None:
    """List all players on TEAM_ABBR for a given season."""
    from nba_tracker.fetcher import get_team_roster
    from nba_tracker.exporter import render_roster_table

    players = get_team_roster(team_abbr.upper(), season)
    render_roster_table(players, team_abbr.upper(), season)


@team.command()
@click.argument("team_abbr")
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
@click.option("--sort", "sort_by", default="PTS", help="Column to sort by.")
@click.option("--min-games", default=0, type=int, help="Minimum games played filter.")
def stats(team_abbr: str, season: str, sort_by: str, min_games: int) -> None:
    """Show per-player season stats for TEAM_ABBR."""
    from nba_tracker.fetcher import get_team_player_stats
    from nba_tracker.exporter import render_stats_table

    df = get_team_player_stats(team_abbr.upper(), season)
    if min_games > 0:
        df = df[df["GP"] >= min_games]
    sort_col = sort_by.upper()
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    render_stats_table(df, team_abbr.upper(), season)


@team.command()
@click.argument("team_abbr")
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
def leaders(team_abbr: str, season: str) -> None:
    """Show statistical leaders for TEAM_ABBR."""
    from nba_tracker.fetcher import get_team_player_stats
    from nba_tracker.analyzer import compute_team_leaders
    from nba_tracker.exporter import render_leaders_table

    df = get_team_player_stats(team_abbr.upper(), season)
    team_leaders = compute_team_leaders(df, team_abbr.upper(), season)
    render_leaders_table(team_leaders)


@team.command()
@click.argument("team_abbr")
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
def summary(team_abbr: str, season: str) -> None:
    """Show aggregate team statistics for TEAM_ABBR."""
    from nba_tracker.fetcher import get_team_player_stats
    from nba_tracker.analyzer import compute_team_summary
    from nba_tracker.exporter import render_summary_table

    df = get_team_player_stats(team_abbr.upper(), season)
    team_summary = compute_team_summary(df, team_abbr.upper(), season)
    render_summary_table(team_summary)


# ── compare command ───────────────────────────────────────────────────
@main.command()
@click.argument("players", nargs=-1, required=True)
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
def compare(players: tuple[str, ...], season: str) -> None:
    """Compare two or more PLAYERS side-by-side (use full names in quotes)."""
    from nba_tracker.fetcher import get_player_stats_by_name
    from nba_tracker.analyzer import compare_players
    from nba_tracker.exporter import render_comparison_table

    player_stats = []
    for name in players:
        ps = get_player_stats_by_name(name, season)
        if ps is not None:
            player_stats.append(ps)
        else:
            click.echo(f"Warning: could not find stats for '{name}'", err=True)
    if len(player_stats) < 2:
        raise click.ClickException("Need at least 2 valid players to compare.")
    result = compare_players(player_stats, season)
    render_comparison_table(result)


# ── export command ────────────────────────────────────────────────────
@main.command()
@click.argument("team_abbr")
@click.option("--season", default="2024-25", help="NBA season (e.g. 2024-25).")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json"], case_sensitive=False),
    default="csv",
    help="Export format.",
)
@click.option("--output", "output_path", default=None, help="Output file path.")
def export(team_abbr: str, season: str, fmt: str, output_path: str | None) -> None:
    """Export team player stats to CSV or JSON."""
    from nba_tracker.fetcher import get_team_player_stats
    from nba_tracker.exporter import export_stats

    df = get_team_player_stats(team_abbr.upper(), season)
    path = export_stats(df, team_abbr.upper(), season, fmt, output_path)
    click.echo(f"Exported to {path}")


# ── cache command group ───────────────────────────────────────────────
@main.group()
def cache() -> None:
    """Manage the local data cache."""


@cache.command()
def clear() -> None:
    """Clear all cached data."""
    from nba_tracker.cache import clear_cache

    clear_cache()
    click.echo("Cache cleared.")


if __name__ == "__main__":
    main()
