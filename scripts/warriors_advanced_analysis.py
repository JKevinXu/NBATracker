#!/usr/bin/env python3
"""Golden State Warriors — Advanced Analytics Report.

Generates a comprehensive Markdown report with:
  1. Advanced player metrics (TS%, USG%, OFF/DEF rating, NET rating, PIE)
  2. Player efficiency tiers
  3. Clutch performance breakdown
  4. Monthly trends
  5. Home vs Away & W/L splits
  6. Rest-day impact
  7. Rolling scoring averages (5-game window)
  8. Player consistency / variance
  9. Offensive vs Defensive impact scatter concept
 10. Key takeaways

Run:
    python scripts/warriors_advanced_analysis.py [--season 2025-26]
"""

from __future__ import annotations

import sys
import os
import time
import statistics
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashPlayerClutch,
    TeamDashboardByGeneralSplits,
    TeamGameLog,
    PlayerGameLog,
)

TEAM_ABBR = "GSW"
TEAM_ID = 1610612744
API_DELAY = 0.6


def _delay() -> None:
    time.sleep(API_DELAY)


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _plus_minus(val: float) -> str:
    if val > 0:
        return f"+{val:.1f}"
    return f"{val:.1f}"


def _tier(net_rating: float) -> str:
    if net_rating >= 8:
        return "Elite"
    elif net_rating >= 3:
        return "Positive"
    elif net_rating >= -3:
        return "Neutral"
    elif net_rating >= -8:
        return "Negative"
    return "Struggling"


# ── data fetching ─────────────────────────────────────────────────────


def fetch_advanced_player_stats(season: str) -> pd.DataFrame:
    _delay()
    return LeagueDashPlayerStats(
        team_id_nullable=TEAM_ID,
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]


def fetch_base_player_stats(season: str) -> pd.DataFrame:
    _delay()
    return LeagueDashPlayerStats(
        team_id_nullable=TEAM_ID,
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_clutch_stats(season: str) -> pd.DataFrame:
    _delay()
    return LeagueDashPlayerClutch(
        team_id_nullable=TEAM_ID,
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_team_splits(season: str) -> list[pd.DataFrame]:
    _delay()
    return TeamDashboardByGeneralSplits(
        team_id=TEAM_ID,
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()


def fetch_team_game_log(season: str) -> pd.DataFrame:
    _delay()
    return TeamGameLog(
        team_id=TEAM_ID,
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]


def fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    _delay()
    return PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]


# ── report builder ────────────────────────────────────────────────────


def build_report(season: str) -> str:
    lines: list[str] = []
    w = lines.append

    print(f"Generating advanced analytics for {TEAM_ABBR} — {season}...")

    # ─── Fetch all data ──────────────────────────────────────────
    print("  Fetching advanced player stats...")
    adv = fetch_advanced_player_stats(season)
    print("  Fetching base player stats...")
    base = fetch_base_player_stats(season)
    print("  Fetching clutch stats...")
    clutch = fetch_clutch_stats(season)
    print("  Fetching team splits...")
    splits = fetch_team_splits(season)
    print("  Fetching team game log...")
    game_log = fetch_team_game_log(season)

    # Filter to players with meaningful minutes
    adv_main = adv[adv["GP"] >= 10].copy()
    base_main = base[base["GP"] >= 10].copy()
    clutch_main = clutch[clutch["GP"] >= 5].copy()

    # Splits indices: 0=overall, 1=home/away, 2=wins/losses, 3=monthly, 5=rest days
    overall = splits[0]
    home_away = splits[1]
    wins_losses = splits[2]
    monthly = splits[3]
    rest_days = splits[5]

    total_w = int(game_log.iloc[0]["W"])
    total_l = int(game_log.iloc[0]["L"])

    # ─── HEADER ──────────────────────────────────────────────────
    w(f"# Golden State Warriors — {season} Advanced Analytics Report")
    w("")
    w(f"> Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    w(f"> Season record: **{total_w}-{total_l}** ({total_w / max(total_w + total_l, 1):.3f}) through {len(game_log)} games")
    w("")

    # ─── 1. Advanced Player Metrics ──────────────────────────────
    w("## 1. Advanced Player Metrics")
    w("")
    w("Key advanced statistics for players with 10+ games played.")
    w("")
    w("| Player | GP | MIN | TS% | USG% | OFF RTG | DEF RTG | NET RTG | AST/TO | PIE | Tier |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    adv_sorted = adv_main.sort_values("NET_RATING", ascending=False)
    for _, row in adv_sorted.iterrows():
        net = float(row["NET_RATING"])
        w(
            f"| {row['PLAYER_NAME']} | {int(row['GP'])} | {float(row['MIN']):.1f} "
            f"| {_pct(row['TS_PCT'])} | {_pct(row['USG_PCT'])} "
            f"| {float(row['OFF_RATING']):.1f} | {float(row['DEF_RATING']):.1f} "
            f"| {_plus_minus(net)} | {float(row['AST_TO']):.2f} "
            f"| {_pct(row['PIE'])} | {_tier(net)} |"
        )
    w("")

    # ─── Metric Glossary ─────────────────────────────────────────
    w("<details><summary>Glossary</summary>")
    w("")
    w("| Metric | Definition |")
    w("|---|---|")
    w("| **TS%** | True Shooting % — accounts for FG, 3P, and FT efficiency |")
    w("| **USG%** | Usage Rate — % of team plays used by a player while on court |")
    w("| **OFF RTG** | Offensive Rating — points produced per 100 possessions |")
    w("| **DEF RTG** | Defensive Rating — points allowed per 100 possessions |")
    w("| **NET RTG** | Net Rating — OFF RTG minus DEF RTG (higher = better) |")
    w("| **AST/TO** | Assist-to-Turnover Ratio |")
    w("| **PIE** | Player Impact Estimate — overall contribution metric |")
    w("")
    w("</details>")
    w("")

    # ─── 2. Efficiency Tiers ─────────────────────────────────────
    w("## 2. Player Efficiency Tiers")
    w("")
    tiers = {"Elite": [], "Positive": [], "Neutral": [], "Negative": [], "Struggling": []}
    for _, row in adv_sorted.iterrows():
        t = _tier(float(row["NET_RATING"]))
        tiers[t].append(f"{row['PLAYER_NAME']} ({_plus_minus(row['NET_RATING'])})")

    for tier_name in ["Elite", "Positive", "Neutral", "Negative", "Struggling"]:
        players = tiers[tier_name]
        if players:
            w(f"**{tier_name}** (NET RTG {'≥ +8' if tier_name == 'Elite' else '≥ +3' if tier_name == 'Positive' else '± 3' if tier_name == 'Neutral' else '≥ -8' if tier_name == 'Negative' else '< -8'})")
            for p in players:
                w(f"- {p}")
            w("")

    # ─── 3. Shooting Efficiency Deep Dive ────────────────────────
    w("## 3. Shooting Efficiency Deep Dive")
    w("")
    w("| Player | FG% | eFG% | TS% | 3P% | FT% | PTS |")
    w("|---|---:|---:|---:|---:|---:|---:|")

    base_sorted = base_main.sort_values("PTS", ascending=False)
    for _, row in base_sorted.iterrows():
        pid = int(row["PLAYER_ID"])
        adv_row = adv_main[adv_main["PLAYER_ID"] == pid]
        efg = float(adv_row.iloc[0]["EFG_PCT"]) if len(adv_row) > 0 else 0
        ts = float(adv_row.iloc[0]["TS_PCT"]) if len(adv_row) > 0 else 0
        w(
            f"| {row['PLAYER_NAME']} "
            f"| {_pct(row['FG_PCT'])} | {_pct(efg)} | {_pct(ts)} "
            f"| {_pct(row['FG3_PCT'])} | {_pct(row['FT_PCT'])} "
            f"| {float(row['PTS']):.1f} |"
        )
    w("")

    # ─── 4. Clutch Performance ───────────────────────────────────
    w("## 4. Clutch Performance")
    w("")
    w("Stats in clutch situations (last 5 minutes, score within 5 points).")
    w("")
    if not clutch_main.empty:
        w("| Player | Clutch GP | W | L | Clutch MIN | Clutch PTS | FG% | FT% | AST | TO |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        clutch_sorted = clutch_main.sort_values("PTS", ascending=False)
        for _, row in clutch_sorted.iterrows():
            w(
                f"| {row['PLAYER_NAME']} | {int(row['GP'])} "
                f"| {int(row['W'])} | {int(row['L'])} "
                f"| {float(row['MIN']):.1f} | {float(row['PTS']):.1f} "
                f"| {_pct(row['FG_PCT'])} | {_pct(row['FT_PCT'])} "
                f"| {float(row['AST']):.1f} | {float(row['TOV']):.1f} |"
            )
        w("")

        # Clutch W-L
        total_clutch_gp = clutch_main["GP"].max() if len(clutch_main) > 0 else 0
        best_clutch = clutch_sorted.iloc[0] if len(clutch_sorted) > 0 else None
        if best_clutch is not None:
            w(f"> **Top clutch performer:** {best_clutch['PLAYER_NAME']} with {float(best_clutch['PTS']):.1f} PPG in clutch minutes")
            w("")
    else:
        w("_No clutch data available._")
        w("")

    # ─── 5. Monthly Trends ───────────────────────────────────────
    w("## 5. Monthly Trends")
    w("")
    w("| Month | GP | W-L | Win% | PPG | FG% | 3P% | RPG | APG | TOV | +/- |")
    w("|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in monthly.iterrows():
        gp = int(row["GP"])
        wins = int(row["W"])
        losses = int(row["L"])
        w(
            f"| {row['GROUP_VALUE']} | {gp} | {wins}-{losses} "
            f"| {_pct(row['W_PCT'])} | {float(row['PTS']):.1f} "
            f"| {_pct(row['FG_PCT'])} | {_pct(row['FG3_PCT'])} "
            f"| {float(row['REB']):.1f} | {float(row['AST']):.1f} "
            f"| {float(row['TOV']):.1f} | {_plus_minus(row['PLUS_MINUS'])} |"
        )
    w("")

    # Best / worst month
    best_month = monthly.loc[monthly["W_PCT"].idxmax()]
    worst_month = monthly.loc[monthly["W_PCT"].idxmin()]
    w(f"> **Best month:** {best_month['GROUP_VALUE']} ({int(best_month['W'])}-{int(best_month['L'])}, {float(best_month['PTS']):.1f} PPG)")
    w(f"> **Worst month:** {worst_month['GROUP_VALUE']} ({int(worst_month['W'])}-{int(worst_month['L'])}, {float(worst_month['PTS']):.1f} PPG)")
    w("")

    # ─── 6. Home vs Away ─────────────────────────────────────────
    w("## 6. Home vs Away Splits")
    w("")
    w("| Location | GP | W-L | Win% | PPG | FG% | 3P% | RPG | APG | TOV | +/- |")
    w("|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in home_away.iterrows():
        w(
            f"| **{row['GROUP_VALUE']}** | {int(row['GP'])} "
            f"| {int(row['W'])}-{int(row['L'])} | {_pct(row['W_PCT'])} "
            f"| {float(row['PTS']):.1f} | {_pct(row['FG_PCT'])} "
            f"| {_pct(row['FG3_PCT'])} | {float(row['REB']):.1f} "
            f"| {float(row['AST']):.1f} | {float(row['TOV']):.1f} "
            f"| {_plus_minus(row['PLUS_MINUS'])} |"
        )
    w("")

    # ─── 7. Wins vs Losses Profile ───────────────────────────────
    w("## 7. Wins vs Losses Profile")
    w("")
    w("What changes when the Warriors win vs lose?")
    w("")
    w("| Metric | In Wins | In Losses | Delta |")
    w("|---|---:|---:|---:|")

    if len(wins_losses) >= 2:
        wr = wins_losses[wins_losses["GROUP_VALUE"] == "Wins"].iloc[0]
        lr = wins_losses[wins_losses["GROUP_VALUE"] == "Losses"].iloc[0]
        metrics = [
            ("PPG", "PTS"), ("FG%", "FG_PCT"), ("3P%", "FG3_PCT"),
            ("FT%", "FT_PCT"), ("RPG", "REB"), ("APG", "AST"),
            ("TOV", "TOV"), ("STL", "STL"), ("BLK", "BLK"),
        ]
        for label, col in metrics:
            wv = float(wr[col])
            lv = float(lr[col])
            delta = wv - lv
            if "%" in label:
                w(f"| {label} | {_pct(wv)} | {_pct(lv)} | {_plus_minus(delta * 100)}% |")
            else:
                w(f"| {label} | {wv:.1f} | {lv:.1f} | {_plus_minus(delta)} |")
    w("")

    # ─── 8. Rest Day Impact ──────────────────────────────────────
    w("## 8. Rest Day Impact")
    w("")
    w("| Days Rest | GP | W-L | Win% | PPG | FG% | +/- |")
    w("|---|---:|:---:|---:|---:|---:|---:|")

    for _, row in rest_days.iterrows():
        w(
            f"| {row['GROUP_VALUE']} | {int(row['GP'])} "
            f"| {int(row['W'])}-{int(row['L'])} | {_pct(row['W_PCT'])} "
            f"| {float(row['PTS']):.1f} | {_pct(row['FG_PCT'])} "
            f"| {_plus_minus(row['PLUS_MINUS'])} |"
        )
    w("")

    # ─── 9. Rolling 5-Game Team Averages ─────────────────────────
    w("## 9. Rolling 5-Game Team Averages")
    w("")
    w("Five-game rolling averages across the season (newest first).")
    w("")

    gl_reversed = game_log.iloc[::-1].reset_index(drop=True)
    pts_rolling = gl_reversed["PTS"].rolling(window=5, min_periods=5).mean()
    fg_rolling = gl_reversed["FG_PCT"].rolling(window=5, min_periods=5).mean()
    tov_rolling = gl_reversed["TOV"].rolling(window=5, min_periods=5).mean()

    w("| Window | Games | Avg PTS | Avg FG% | Avg TOV |")
    w("|---|---|---:|---:|---:|")

    # Show windows at intervals
    n = len(gl_reversed)
    window_points = list(range(4, n, 5))  # every 5 games
    if window_points and window_points[-1] != n - 1:
        window_points.append(n - 1)

    for idx in window_points:
        if pd.isna(pts_rolling.iloc[idx]):
            continue
        game_start = idx - 3  # approx
        game_end = idx + 1
        w(
            f"| Games {game_start}-{game_end} "
            f"| {gl_reversed.iloc[max(0,idx-4):idx+1]['GAME_DATE'].iloc[0]} → {gl_reversed.iloc[idx]['GAME_DATE']} "
            f"| {pts_rolling.iloc[idx]:.1f} "
            f"| {_pct(fg_rolling.iloc[idx])} "
            f"| {tov_rolling.iloc[idx]:.1f} |"
        )
    w("")

    # ─── 10. Player Consistency (Scoring Variance) ───────────────
    w("## 10. Player Scoring Consistency")
    w("")
    w("Standard deviation of points scored across games — lower = more consistent.")
    w("")

    print("  Fetching individual game logs for consistency analysis...")
    key_players = [
        (201939, "Stephen Curry"),
        (203110, "Draymond Green"),
        (1630228, "Jonathan Kuminga"),
        (1631218, "Brandin Podziemski"),
        (1627790, "Buddy Hield"),
        (1629621, "Andrew Wiggins"),
    ]
    # Also check IDs from base_main for Butler, Moody, etc.
    for _, row in base_main.iterrows():
        pid = int(row["PLAYER_ID"])
        pname = str(row["PLAYER_NAME"])
        if pid not in [p[0] for p in key_players] and int(row["GP"]) >= 20:
            key_players.append((pid, pname))

    consistency_data: list[dict] = []
    for pid, pname in key_players:
        try:
            pgl = fetch_player_game_log(pid, season)
            if len(pgl) >= 10:
                pts_list = pgl["PTS"].tolist()
                avg = statistics.mean(pts_list)
                std = statistics.stdev(pts_list)
                low = min(pts_list)
                high = max(pts_list)
                consistency_data.append({
                    "player": pname,
                    "gp": len(pgl),
                    "avg": avg,
                    "std": std,
                    "cv": std / avg if avg > 0 else 0,
                    "low": low,
                    "high": high,
                    "range": high - low,
                })
        except Exception:
            pass

    if consistency_data:
        consistency_data.sort(key=lambda x: x["cv"])

        w("| Player | GP | Avg PTS | Std Dev | CV | Low | High | Range |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in consistency_data:
            w(
                f"| {c['player']} | {c['gp']} | {c['avg']:.1f} "
                f"| {c['std']:.1f} | {c['cv']:.2f} "
                f"| {c['low']} | {c['high']} | {c['range']} |"
            )
        w("")

        most_consistent = consistency_data[0]
        most_volatile = consistency_data[-1]
        w(f"> **Most consistent:** {most_consistent['player']} (CV = {most_consistent['cv']:.2f})")
        w(f"> **Most volatile:** {most_volatile['player']} (CV = {most_volatile['cv']:.2f}, range {most_volatile['low']}-{most_volatile['high']})")
        w("")

    # ─── 11. Offensive vs Defensive Impact ───────────────────────
    w("## 11. Offensive vs Defensive Impact Matrix")
    w("")
    w("Players plotted by OFF RTG vs DEF RTG (lower DEF = better defense).")
    w("")
    w("| Player | OFF RTG | DEF RTG | NET RTG | Role |")
    w("|---|---:|---:|---:|---|")

    for _, row in adv_sorted.iterrows():
        off = float(row["OFF_RATING"])
        def_ = float(row["DEF_RATING"])
        net = float(row["NET_RATING"])

        if off >= 112 and def_ <= 110:
            role = "Two-way star"
        elif off >= 112:
            role = "Offensive engine"
        elif def_ <= 110:
            role = "Defensive anchor"
        elif off >= 108 and def_ <= 114:
            role = "Solid contributor"
        else:
            role = "Developmental"

        w(f"| {row['PLAYER_NAME']} | {off:.1f} | {def_:.1f} | {_plus_minus(net)} | {role} |")
    w("")

    # ─── 12. Key Takeaways ───────────────────────────────────────
    w("## 12. Key Takeaways")
    w("")

    # Auto-generate insights from the data
    insights: list[str] = []

    # 1. Overall record
    win_pct = total_w / max(total_w + total_l, 1)
    if win_pct >= 0.6:
        insights.append(f"The Warriors are having a strong season at {total_w}-{total_l} ({_pct(win_pct)}).")
    elif win_pct >= 0.5:
        insights.append(f"The Warriors are a borderline playoff team at {total_w}-{total_l} ({_pct(win_pct)}).")
    else:
        insights.append(f"The Warriors are below .500 at {total_w}-{total_l} ({_pct(win_pct)}), a concerning trajectory.")

    # 2. Home vs away
    if len(home_away) >= 2:
        home_row = home_away[home_away["GROUP_VALUE"] == "Home"].iloc[0]
        away_row = home_away[home_away["GROUP_VALUE"] == "Road"].iloc[0]
        home_pct = float(home_row["W_PCT"])
        away_pct = float(away_row["W_PCT"])
        if home_pct - away_pct > 0.15:
            insights.append(f"Dominant at home ({_pct(home_pct)}) but struggling on the road ({_pct(away_pct)}) — a {_pct(home_pct - away_pct)} gap.")
        elif away_pct - home_pct > 0.15:
            insights.append(f"Surprisingly better on the road ({_pct(away_pct)}) than at home ({_pct(home_pct)}).")

    # 3. Best player
    if len(adv_sorted) > 0:
        best = adv_sorted.iloc[0]
        insights.append(
            f"**{best['PLAYER_NAME']}** has the highest net rating ({_plus_minus(best['NET_RATING'])}) — "
            f"the team is significantly better when they're on the court."
        )

    # 4. Turnover concern
    avg_tov = float(overall.iloc[0]["TOV"]) if len(overall) > 0 else 0
    if avg_tov >= 15:
        insights.append(f"Turnover problems persist at {avg_tov:.1f} per game — this ranks among the league's worst.")

    # 5. Three-point shooting
    fg3_pct = float(overall.iloc[0]["FG3_PCT"]) if len(overall) > 0 else 0
    if fg3_pct >= 0.37:
        insights.append(f"Elite three-point shooting at {_pct(fg3_pct)} keeps the offense competitive.")
    elif fg3_pct <= 0.34:
        insights.append(f"Three-point shooting has regressed to {_pct(fg3_pct)} — below league average.")

    # 6. Clutch
    if len(clutch_main) > 0:
        clutch_top = clutch_main.sort_values("PTS", ascending=False).iloc[0]
        total_clutch_w = clutch_main["W"].max()
        total_clutch_l = clutch_main["L"].max()
        if total_clutch_l > total_clutch_w:
            insights.append(
                f"Clutch record is concerning ({total_clutch_w}-{total_clutch_l}) — "
                f"the team needs to close out tight games better."
            )

    # 7. Best month
    insights.append(
        f"**{best_month['GROUP_VALUE']}** was the best month "
        f"({int(best_month['W'])}-{int(best_month['L'])}, {float(best_month['PTS']):.1f} PPG, "
        f"{_plus_minus(best_month['PLUS_MINUS'])} net). "
        f"**{worst_month['GROUP_VALUE']}** was the toughest "
        f"({int(worst_month['W'])}-{int(worst_month['L'])})."
    )

    for i, insight in enumerate(insights, 1):
        w(f"{i}. {insight}")
    w("")

    w("---")
    w(f"*Report generated by NBATracker advanced analytics engine.*")
    w("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--season", default="2025-26", help="NBA season (e.g. 2025-26).")
@click.option("--output", "output_path", default=None, help="Output .md file path.")
def main(season: str, output_path: str | None) -> None:
    """Generate an advanced analytics report for the Golden State Warriors."""
    if output_path is None:
        safe = season.replace("-", "_")
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", f"warriors_advanced_{safe}.md"
        )

    report = build_report(season)

    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nAdvanced report written to {out_path}")


if __name__ == "__main__":
    main()
