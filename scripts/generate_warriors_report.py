#!/usr/bin/env python3
"""Generate a Markdown report of Warriors recent games + season analysis."""

from __future__ import annotations

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from nba_api.stats.endpoints import TeamGameLog, PlayerGameLog
from nba_api.stats.static import teams as static_teams

from nba_tracker.fetcher import get_team_roster, get_team_player_stats
from nba_tracker.analyzer import compute_team_leaders, compute_team_summary

import click

TEAM_ABBR = "GSW"
TEAM_ID = 1610612744
DEFAULT_SEASON = "2025-26"
N_RECENT = 10
API_DELAY = 0.6


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def fetch_team_game_log(season: str) -> pd.DataFrame:
    time.sleep(API_DELAY)
    return TeamGameLog(
        team_id=TEAM_ID, season=season, season_type_all_star="Regular Season"
    ).get_data_frames()[0]


def fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    time.sleep(API_DELAY)
    return PlayerGameLog(
        player_id=player_id, season=season, season_type_all_star="Regular Season"
    ).get_data_frames()[0]


def streak(results: list[str]) -> str:
    if not results:
        return "-"
    cur = results[0]
    count = sum(1 for r in results if r == cur)
    # recount properly
    count = 0
    for r in results:
        if r == cur:
            count += 1
        else:
            break
    return f"{cur}{count}"


@click.command()
@click.option("--season", default=DEFAULT_SEASON, help="NBA season (e.g. 2025-26).")
@click.option("--output", "output_path", default=None, help="Output .md file path.")
def main(season: str, output_path: str | None) -> None:
    if output_path is None:
        safe = season.replace("-", "_")
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", f"warriors_report_{safe}.md"
        )

    SEASON = season
    print(f"Generating Warriors report for {SEASON}...")

    # -- team game log --
    gl = fetch_team_game_log(SEASON)
    recent = gl.head(N_RECENT).copy()

    total_w = int(gl.iloc[0]["W"])
    total_l = int(gl.iloc[0]["L"])
    total_pct = gl.iloc[0]["W_PCT"]
    recent_wl = recent["WL"].tolist()
    recent_w = recent_wl.count("W")
    recent_l = recent_wl.count("L")
    streak_str = streak(recent_wl)

    # -- season player stats --
    print("Fetching player stats...")
    df_stats = get_team_player_stats(TEAM_ABBR, SEASON)
    leaders = compute_team_leaders(df_stats, TEAM_ABBR, SEASON)
    summary = compute_team_summary(df_stats, TEAM_ABBR, SEASON)

    # -- roster --
    print("Fetching roster...")
    roster = get_team_roster(TEAM_ABBR, SEASON)

    # -- top individual performances --
    print("Fetching individual game logs...")
    key_players = [
        (201939, "Stephen Curry"),
        (203110, "Draymond Green"),
        (1629621, "Andrew Wiggins"),
        (1630228, "Jonathan Kuminga"),
        (1631218, "Brandin Podziemski"),
        (1627790, "Buddy Hield"),
    ]
    recent_game_ids = set(recent["Game_ID"].tolist())
    all_perfs: list[dict] = []
    for pid, pname in key_players:
        try:
            pdf = fetch_player_game_log(pid, SEASON)
            pr = pdf[pdf["Game_ID"].isin(recent_game_ids)]
            for _, row in pr.iterrows():
                opp = str(row.get("MATCHUP", ""))
                opp = opp.split(" ")[-1] if " " in opp else opp
                all_perfs.append({
                    "player": pname,
                    "date": str(row.get("GAME_DATE", "")),
                    "opp": opp,
                    "pts": int(row.get("PTS", 0)),
                    "reb": int(row.get("REB", 0)),
                    "ast": int(row.get("AST", 0)),
                    "stl": int(row.get("STL", 0)),
                    "blk": int(row.get("BLK", 0)),
                    "fg_pct": float(row.get("FG_PCT", 0)),
                    "min": float(row.get("MIN", 0)),
                })
        except Exception:
            pass

    all_perfs.sort(key=lambda x: x["pts"], reverse=True)

    # -- home / away split --
    home = recent[recent["MATCHUP"].str.contains("vs.")]
    away = recent[~recent["MATCHUP"].str.contains("vs.")]

    # -- scoring trend (oldest first) --
    trend = recent.iloc[::-1].reset_index(drop=True)

    # ====================== BUILD MARKDOWN ======================
    lines: list[str] = []
    w = lines.append

    w(f"# Golden State Warriors — {SEASON} Season Report")
    w("")
    w(f"> Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    w("")

    # --- Record ---
    w("## Season Record")
    w("")
    w(f"| Metric | Value |")
    w(f"|---|---|")
    w(f"| Overall Record | **{total_w}-{total_l}** ({total_pct:.3f}) |")
    w(f"| Last {N_RECENT} Games | **{recent_w}-{recent_l}** |")
    w(f"| Current Streak | **{streak_str}** |")
    w(f"| Games Played | {len(gl)} |")
    w("")

    # --- Recent Games ---
    w(f"## Last {N_RECENT} Games")
    w("")
    w("| Date | Matchup | W/L | PTS | FG% | 3P% | REB | AST | TOV |")
    w("|---|---|:---:|---:|---:|---:|---:|---:|---:|")
    for _, row in recent.iterrows():
        wl_icon = "**W**" if row["WL"] == "W" else "L"
        w(
            f"| {row['GAME_DATE']} | {row['MATCHUP']} | {wl_icon} "
            f"| {int(row['PTS'])} | {_pct(row['FG_PCT'])} | {_pct(row['FG3_PCT'])} "
            f"| {int(row['REB'])} | {int(row['AST'])} | {int(row['TOV'])} |"
        )
    w("")

    # --- Recent Averages ---
    w("## Recent Averages")
    w("")
    avg = {
        "Points": recent["PTS"].mean(),
        "Rebounds": recent["REB"].mean(),
        "Assists": recent["AST"].mean(),
        "Steals": recent["STL"].mean(),
        "Blocks": recent["BLK"].mean(),
        "Turnovers": recent["TOV"].mean(),
    }
    w("| Stat | Average |")
    w("|---|---:|")
    for k, v in avg.items():
        w(f"| {k} | {v:.1f} |")
    w(f"| FG% | {_pct(recent['FG_PCT'].mean())} |")
    w(f"| 3P% | {_pct(recent['FG3_PCT'].mean())} |")
    w(f"| FT% | {_pct(recent['FT_PCT'].mean())} |")
    w(f"| 3PM / game | {recent['FG3M'].mean():.1f} |")
    w("")

    # --- Scoring Trend ---
    w("## Scoring Trend (oldest to newest)")
    w("")
    max_pts = trend["PTS"].max()
    for i, (_, row) in enumerate(trend.iterrows(), 1):
        pts = int(row["PTS"])
        opp = str(row["MATCHUP"]).split(" ")[-1]
        wl = row["WL"]
        bar_len = int(round(pts / max_pts * 20)) if max_pts > 0 else 0
        bar = "█" * bar_len
        icon = "🟢" if wl == "W" else "🔴"
        w(f"| {icon} vs {opp} | `{bar}` **{pts}** |")
    w("")

    # --- Highs & Lows ---
    w("## Highs & Lows")
    w("")

    def _gl(r: pd.Series) -> str:
        return f"{r['GAME_DATE']} — {r['MATCHUP']}"

    best_pts = recent.loc[recent["PTS"].idxmax()]
    best_reb = recent.loc[recent["REB"].idxmax()]
    best_ast = recent.loc[recent["AST"].idxmax()]
    best_fg = recent.loc[recent["FG_PCT"].idxmax()]
    best_3p = recent.loc[recent["FG3_PCT"].idxmax()]
    worst_pts = recent.loc[recent["PTS"].idxmin()]
    worst_fg = recent.loc[recent["FG_PCT"].idxmin()]
    most_tov = recent.loc[recent["TOV"].idxmax()]

    w("| Category | Value | Game |")
    w("|---|---:|---|")
    w(f"| Most Points | **{int(best_pts['PTS'])}** | {_gl(best_pts)} |")
    w(f"| Most Rebounds | **{int(best_reb['REB'])}** | {_gl(best_reb)} |")
    w(f"| Most Assists | **{int(best_ast['AST'])}** | {_gl(best_ast)} |")
    w(f"| Best FG% | **{_pct(best_fg['FG_PCT'])}** | {_gl(best_fg)} |")
    w(f"| Best 3P% | **{_pct(best_3p['FG3_PCT'])}** | {_gl(best_3p)} |")
    w(f"| Fewest Points | {int(worst_pts['PTS'])} | {_gl(worst_pts)} |")
    w(f"| Worst FG% | {_pct(worst_fg['FG_PCT'])} | {_gl(worst_fg)} |")
    w(f"| Most Turnovers | {int(most_tov['TOV'])} | {_gl(most_tov)} |")
    w("")

    # --- Home vs Away ---
    w("## Home vs Away Split")
    w("")
    w("| | Games | W-L | Avg PTS | Avg REB | Avg AST | Avg FG% | Avg 3P% |")
    w("|---|---:|:---:|---:|---:|---:|---:|---:|")
    for label, subset in [("Home", home), ("Away", away)]:
        if subset.empty:
            w(f"| {label} | 0 | - | - | - | - | - | - |")
        else:
            n = len(subset)
            sw = (subset["WL"] == "W").sum()
            sl = n - sw
            w(
                f"| **{label}** | {n} | {sw}-{sl} "
                f"| {subset['PTS'].mean():.1f} | {subset['REB'].mean():.1f} "
                f"| {subset['AST'].mean():.1f} | {_pct(subset['FG_PCT'].mean())} "
                f"| {_pct(subset['FG3_PCT'].mean())} |"
            )
    w("")

    # --- Top Individual Performances ---
    w("## Top Individual Performances")
    w("")
    w("| # | Player | Date | vs | PTS | REB | AST | STL | BLK | FG% | MIN |")
    w("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, p in enumerate(all_perfs[:15], 1):
        w(
            f"| {i} | **{p['player']}** | {p['date']} | {p['opp']} "
            f"| {p['pts']} | {p['reb']} | {p['ast']} | {p['stl']} | {p['blk']} "
            f"| {_pct(p['fg_pct'])} | {p['min']:.0f} |"
        )
    w("")

    # --- Season Leaders ---
    w("## Season Statistical Leaders")
    w("")
    w("| Category | Player | Value |")
    w("|---|---|---:|")
    for ld in leaders.leaders:
        val = _pct(ld.value) if "%" in ld.category else f"{ld.value:.1f}"
        w(f"| {ld.category} | **{ld.player_name}** | {val} |")
    w("")

    # --- Season Summary ---
    w("## Team Season Summary")
    w("")
    w("| Metric | Value |")
    w("|---|---:|")
    w(f"| Roster Size | {summary.num_players} |")
    w(f"| Total PTS (per-game avg sum) | {summary.total_points:.1f} |")
    w(f"| Total REB | {summary.total_rebounds:.1f} |")
    w(f"| Total AST | {summary.total_assists:.1f} |")
    w(f"| Total STL | {summary.total_steals:.1f} |")
    w(f"| Total BLK | {summary.total_blocks:.1f} |")
    w(f"| Avg FG% | {_pct(summary.avg_fg_pct)} |")
    w(f"| Avg 3P% | {_pct(summary.avg_fg3_pct)} |")
    w(f"| Avg FT% | {_pct(summary.avg_ft_pct)} |")
    w("")

    # --- Full Roster Stats ---
    w("## Full Roster Stats (Per Game)")
    w("")
    cols = ["PLAYER_NAME", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT"]
    avail = [c for c in cols if c in df_stats.columns]
    sorted_stats = df_stats.sort_values("PTS", ascending=False) if "PTS" in df_stats.columns else df_stats

    header_map = {
        "PLAYER_NAME": "Player", "GP": "GP", "MIN": "MIN", "PTS": "PTS",
        "REB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK",
        "TOV": "TOV", "FG_PCT": "FG%", "FG3_PCT": "3P%", "FT_PCT": "FT%",
    }
    header = "| " + " | ".join(header_map.get(c, c) for c in avail) + " |"
    sep = "|" + "|".join("---:" if c != "PLAYER_NAME" else "---" for c in avail) + "|"
    w(header)
    w(sep)

    for _, row in sorted_stats.iterrows():
        cells = []
        for c in avail:
            v = row[c]
            if c in ("FG_PCT", "FG3_PCT", "FT_PCT"):
                cells.append(_pct(float(v)))
            elif c in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"):
                cells.append(f"{float(v):.1f}")
            elif c == "GP":
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        w("| " + " | ".join(cells) + " |")
    w("")

    # --- Write file ---
    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
