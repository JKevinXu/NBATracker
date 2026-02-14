#!/usr/bin/env python3
"""Golden State Warriors — External Player Gap-Fill Analysis.

Identifies the Warriors' statistical weaknesses and scores every qualifying
non-Warriors player in the league on how well they would fill those gaps.

Scoring dimensions (weighted):
  1. Defensive impact   (25%) — DEF RTG, STL, BLK
  2. Efficient scoring  (20%) — PTS × TS% combo
  3. Rebounding         (15%) — REB, DREB
  4. Playmaking + ball  (15%) — AST/TO, low TOV rate
  5. Two-way versatility(15%) — NET RTG, PIE
  6. Clutch capability  (10%) — clutch PPG, clutch FG%

Run:
    python scripts/warriors_gap_analysis.py [--season 2025-26]
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    LeagueDashPlayerClutch,
    LeagueDashPlayerStats,
)

TEAM_ID = 1610612744  # Warriors
API_DELAY = 0.6
MIN_GP = 30
MIN_MPG = 20.0


def _delay() -> None:
    time.sleep(API_DELAY)


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _plus_minus(val: float) -> str:
    return f"+{val:.1f}" if val > 0 else f"{val:.1f}"


# ── data fetching ─────────────────────────────────────────────────────


def fetch_league_advanced(season: str) -> pd.DataFrame:
    print("  Fetching league-wide advanced stats...")
    _delay()
    return LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]


def fetch_league_base(season: str) -> pd.DataFrame:
    print("  Fetching league-wide base stats...")
    _delay()
    return LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_league_clutch(season: str) -> pd.DataFrame:
    print("  Fetching league-wide clutch stats...")
    _delay()
    return LeagueDashPlayerClutch(
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_warriors_advanced(season: str) -> pd.DataFrame:
    print("  Fetching Warriors advanced stats...")
    _delay()
    return LeagueDashPlayerStats(
        team_id_nullable=TEAM_ID,
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]


def fetch_warriors_base(season: str) -> pd.DataFrame:
    print("  Fetching Warriors base stats...")
    _delay()
    return LeagueDashPlayerStats(
        team_id_nullable=TEAM_ID,
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


# ── helpers ───────────────────────────────────────────────────────────


def norm(series: pd.Series, higher_better: bool = True) -> pd.Series:
    """Normalize a series to 0-100."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    n = (series - mn) / (mx - mn) * 100
    return n if higher_better else 100 - n


# ── report builder ────────────────────────────────────────────────────


def build_report(season: str) -> str:
    lines: list[str] = []
    w = lines.append

    print(f"Building gap-fill analysis for Warriors — {season}...")

    # ── Fetch all data ───────────────────────────────────────────
    league_adv = fetch_league_advanced(season)
    league_base = fetch_league_base(season)
    league_clutch = fetch_league_clutch(season)
    gsw_adv = fetch_warriors_advanced(season)
    gsw_base = fetch_warriors_base(season)

    # ── Warriors weakness profile ────────────────────────────────
    gsw_base_main = gsw_base[gsw_base["GP"] >= 10]
    gsw_adv_main = gsw_adv[gsw_adv["GP"] >= 10]

    gsw_avg_def_rtg = float(gsw_adv_main["DEF_RATING"].mean())
    gsw_avg_reb = float(gsw_base_main["REB"].mean())
    gsw_avg_dreb = float(gsw_base_main["DREB"].mean())
    gsw_avg_tov = float(gsw_base_main["TOV"].mean())
    gsw_third_scorer = float(gsw_base_main.sort_values("PTS", ascending=False).iloc[2]["PTS"])
    gsw_avg_net = float(gsw_adv_main["NET_RATING"].mean())

    print(f"\n  Warriors weakness profile:")
    print(f"    Avg DEF RTG: {gsw_avg_def_rtg:.1f}")
    print(f"    Avg REB: {gsw_avg_reb:.1f} (DREB: {gsw_avg_dreb:.1f})")
    print(f"    3rd scorer: {gsw_third_scorer:.1f} PPG")
    print(f"    Avg TOV: {gsw_avg_tov:.1f}")
    print(f"    Avg NET RTG: {gsw_avg_net:.1f}")
    print()

    # ── Filter non-Warriors qualifying players ───────────────────
    league_adv = league_adv[
        (league_adv["TEAM_ID"] != TEAM_ID)
        & (league_adv["GP"] >= MIN_GP)
        & (league_adv["MIN"] >= MIN_MPG)
    ].copy()

    league_base = league_base[
        (league_base["TEAM_ID"] != TEAM_ID)
        & (league_base["GP"] >= MIN_GP)
        & (league_base["MIN"] >= MIN_MPG)
    ].copy()

    # Merge advanced + base on PLAYER_ID
    merged = league_base.merge(
        league_adv[
            [
                "PLAYER_ID",
                "NET_RATING",
                "OFF_RATING",
                "DEF_RATING",
                "TS_PCT",
                "USG_PCT",
                "PIE",
                "AST_TO",
                "EFG_PCT",
            ]
        ],
        on="PLAYER_ID",
        how="inner",
        suffixes=("", "_adv"),
    )

    # Merge clutch data (left join — not all players have clutch data)
    clutch_cols = league_clutch[["PLAYER_ID", "GP", "PTS", "FG_PCT"]].rename(
        columns={"GP": "CLUTCH_GP", "PTS": "CLUTCH_PTS", "FG_PCT": "CLUTCH_FG_PCT"}
    )
    merged = merged.merge(clutch_cols, on="PLAYER_ID", how="left")
    merged["CLUTCH_PTS"] = merged["CLUTCH_PTS"].fillna(0)
    merged["CLUTCH_FG_PCT"] = merged["CLUTCH_FG_PCT"].fillna(0)
    merged["CLUTCH_GP"] = merged["CLUTCH_GP"].fillna(0)

    print(f"  Qualifying players: {len(merged)} (non-Warriors, {MIN_GP}+ GP, {MIN_MPG}+ MIN)")

    # ── Compute dimension scores ─────────────────────────────────

    # 1. Defensive impact (25%): low DEF_RATING + high STL + high BLK
    merged["d_def_rtg"] = norm(merged["DEF_RATING"], higher_better=False)
    merged["d_stl"] = norm(merged["STL"])
    merged["d_blk"] = norm(merged["BLK"])
    merged["dim_defense"] = 0.50 * merged["d_def_rtg"] + 0.25 * merged["d_stl"] + 0.25 * merged["d_blk"]

    # 2. Efficient scoring (20%): high PTS weighted by TS%
    merged["scoring_value"] = merged["PTS"] * merged["TS_PCT"]
    merged["dim_scoring"] = norm(merged["scoring_value"])

    # 3. Rebounding (15%): total REB with bonus for DREB
    merged["reb_value"] = 0.4 * merged["REB"] + 0.6 * merged["DREB"]
    merged["dim_rebounding"] = norm(merged["reb_value"])

    # 4. Playmaking + ball security (15%): high AST/TO, low TOV
    merged["dim_ast_to"] = norm(merged["AST_TO"])
    merged["dim_low_tov"] = norm(merged["TOV"], higher_better=False)
    merged["dim_playmaking"] = 0.60 * merged["dim_ast_to"] + 0.40 * merged["dim_low_tov"]

    # 5. Two-way versatility (15%): NET RTG + PIE
    merged["dim_net"] = norm(merged["NET_RATING"])
    merged["dim_pie"] = norm(merged["PIE"])
    merged["dim_versatility"] = 0.50 * merged["dim_net"] + 0.50 * merged["dim_pie"]

    # 6. Clutch capability (10%): clutch PTS + clutch FG%
    merged["dim_clutch_pts"] = norm(merged["CLUTCH_PTS"])
    merged["dim_clutch_fg"] = norm(merged["CLUTCH_FG_PCT"])
    merged["dim_clutch"] = 0.60 * merged["dim_clutch_pts"] + 0.40 * merged["dim_clutch_fg"]

    # ── Composite gap-fill score ─────────────────────────────────
    merged["gap_fill_score"] = (
        0.25 * merged["dim_defense"]
        + 0.20 * merged["dim_scoring"]
        + 0.15 * merged["dim_rebounding"]
        + 0.15 * merged["dim_playmaking"]
        + 0.15 * merged["dim_versatility"]
        + 0.10 * merged["dim_clutch"]
    )

    merged = merged.sort_values("gap_fill_score", ascending=False).reset_index(drop=True)

    # ── Determine strongest dimension per player ─────────────────
    dim_cols = {
        "dim_defense": "Defense",
        "dim_scoring": "Scoring",
        "dim_rebounding": "Rebounding",
        "dim_playmaking": "Playmaking",
        "dim_versatility": "Two-Way",
        "dim_clutch": "Clutch",
    }

    def best_dim(row: pd.Series) -> str:
        best = max(dim_cols.keys(), key=lambda c: row[c])
        return dim_cols[best]

    merged["best_fit"] = merged.apply(best_dim, axis=1)

    # ═════════════════════════════════════════════════════════════
    # BUILD MARKDOWN REPORT
    # ═════════════════════════════════════════════════════════════

    w(f"# Warriors Gap-Fill Analysis — {season}")
    w("")
    w(f"> Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    w(f"> Players evaluated: {len(merged)} qualifying non-Warriors ({MIN_GP}+ GP, {MIN_MPG}+ MPG)")
    w("")

    # ── 1. Warriors Weakness Profile ─────────────────────────────
    w("## 1. Warriors Weakness Profile")
    w("")
    w("Statistical gaps that the ideal external addition would address:")
    w("")
    w("| Weakness | Evidence | Weight |")
    w("|---|---|---:|")
    w(f"| **Defense** | Opp 114.0 PPG; avg rotation DEF RTG {gsw_avg_def_rtg:.1f} | 25% |")
    w(f"| **Scoring depth** | 3rd scorer at {gsw_third_scorer:.1f} PPG; need efficient volume | 20% |")
    w(f"| **Rebounding / interior** | Avg {gsw_avg_reb:.1f} RPG ({gsw_avg_dreb:.1f} DREB); aging big rotation | 15% |")
    w(f"| **Playmaking + ball security** | Team avg {gsw_avg_tov:.1f} TOV; need low-turnover creators | 15% |")
    w(f"| **Two-way versatility** | Avg NET RTG {gsw_avg_net:.1f}; need positive-impact players | 15% |")
    w(f"| **Clutch capability** | 8-11 clutch record; need closer alongside Curry | 10% |")
    w("")

    # ── 2. Top 20 External Players ───────────────────────────────
    w("## 2. Top 20 Players Who Best Fill Warriors Gaps")
    w("")
    w("| Rank | Player | Team | Score | PTS | REB | AST | DEF RTG | TS% | NET | Best Fit |")
    w("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    top20 = merged.head(20)
    for i, row in top20.iterrows():
        w(
            f"| {i + 1} | **{row['PLAYER_NAME']}** | {row['TEAM_ABBREVIATION']} "
            f"| {row['gap_fill_score']:.1f} "
            f"| {row['PTS']:.1f} | {row['REB']:.1f} | {row['AST']:.1f} "
            f"| {row['DEF_RATING']:.1f} | {_pct(row['TS_PCT'])} "
            f"| {_plus_minus(row['NET_RATING'])} | {row['best_fit']} |"
        )
    w("")

    # ── 3. Deep Dive on Top 5 ────────────────────────────────────
    w("## 3. Deep Dive — Top 5 Fits")
    w("")

    for i in range(min(5, len(merged))):
        p = merged.iloc[i]
        w(f"### #{i + 1} — {p['PLAYER_NAME']} ({p['TEAM_ABBREVIATION']})")
        w("")
        w(f"**Gap-Fill Score: {p['gap_fill_score']:.1f} / 100**")
        w("")
        w(f"| Stat | Value |")
        w(f"|---|---|")
        w(f"| Points | {p['PTS']:.1f} PPG |")
        w(f"| Rebounds | {p['REB']:.1f} RPG ({p['DREB']:.1f} DREB) |")
        w(f"| Assists | {p['AST']:.1f} APG |")
        w(f"| Steals / Blocks | {p['STL']:.1f} / {p['BLK']:.1f} |")
        w(f"| True Shooting | {_pct(p['TS_PCT'])} |")
        w(f"| Usage Rate | {_pct(p['USG_PCT'])} |")
        w(f"| OFF / DEF RTG | {p['OFF_RATING']:.1f} / {p['DEF_RATING']:.1f} |")
        w(f"| Net Rating | **{_plus_minus(p['NET_RATING'])}** |")
        w(f"| PIE | {_pct(p['PIE'])} |")
        w(f"| AST/TO | {p['AST_TO']:.2f} |")
        w(f"| Clutch PPG | {p['CLUTCH_PTS']:.1f} |")
        w("")

        # Dimension breakdown
        w("**Dimension Scores:**")
        w("")
        w("| Dimension | Score | Weight |")
        w("|---|---:|---:|")
        for col, label in dim_cols.items():
            weight = {"dim_defense": 25, "dim_scoring": 20, "dim_rebounding": 15, "dim_playmaking": 15, "dim_versatility": 15, "dim_clutch": 10}[col]
            w(f"| {label} | {p[col]:.1f} | {weight}% |")
        w("")

        # Fit reasoning
        reasons: list[str] = []
        if p["DEF_RATING"] < 110:
            reasons.append(f"Elite defender (DEF RTG {p['DEF_RATING']:.1f}) — directly addresses Warriors' biggest weakness")
        if p["PTS"] >= 18 and p["TS_PCT"] >= 0.58:
            reasons.append(f"Efficient scorer ({p['PTS']:.1f} PPG on {_pct(p['TS_PCT'])} TS) — could be the reliable 3rd option")
        if p["DREB"] >= 6:
            reasons.append(f"Strong rebounder ({p['DREB']:.1f} DREB) — addresses interior presence gap")
        if p["AST_TO"] >= 2.5:
            reasons.append(f"Excellent ball security ({p['AST_TO']:.2f} AST/TO) — would reduce team turnovers")
        if p["NET_RATING"] >= 5:
            reasons.append(f"High-impact player ({_plus_minus(p['NET_RATING'])} NET) — makes his team significantly better")
        if p["CLUTCH_PTS"] >= 3:
            reasons.append(f"Clutch performer ({p['CLUTCH_PTS']:.1f} clutch PPG) — would help close tight games")
        if p["BLK"] >= 1.5:
            reasons.append(f"Rim protector ({p['BLK']:.1f} BPG) — upgrades Warriors' interior defense")
        if p["STL"] >= 1.5:
            reasons.append(f"Active hands ({p['STL']:.1f} SPG) — creates transition opportunities")

        if reasons:
            w("**Why this player fills Warriors gaps:**")
            for r in reasons:
                w(f"- {r}")
            w("")

    # ── 4. Dimension Leaders ─────────────────────────────────────
    w("## 4. Best Available by Dimension")
    w("")
    w("Top 3 scorers in each gap dimension (from the full qualifying pool):")
    w("")

    for col, label in dim_cols.items():
        top3 = merged.nlargest(3, col)
        w(f"### {label}")
        w("")
        w("| Rank | Player | Team | Dim Score | Key Stat |")
        w("|---:|---|---|---:|---|")
        for j, (_, row) in enumerate(top3.iterrows()):
            if col == "dim_defense":
                key = f"DEF RTG {row['DEF_RATING']:.1f}, {row['STL']:.1f} STL, {row['BLK']:.1f} BLK"
            elif col == "dim_scoring":
                key = f"{row['PTS']:.1f} PPG on {_pct(row['TS_PCT'])} TS"
            elif col == "dim_rebounding":
                key = f"{row['REB']:.1f} RPG ({row['DREB']:.1f} DREB)"
            elif col == "dim_playmaking":
                key = f"{row['AST_TO']:.2f} AST/TO, {row['TOV']:.1f} TOV"
            elif col == "dim_versatility":
                key = f"{_plus_minus(row['NET_RATING'])} NET, {_pct(row['PIE'])} PIE"
            else:
                key = f"{row['CLUTCH_PTS']:.1f} clutch PPG, {_pct(row['CLUTCH_FG_PCT'])} FG"
            w(f"| {j + 1} | {row['PLAYER_NAME']} | {row['TEAM_ABBREVIATION']} | {row[col]:.1f} | {key} |")
        w("")

    # ── 5. Position-based Recommendations ────────────────────────
    w("## 5. Summary")
    w("")

    top1 = merged.iloc[0]
    top2 = merged.iloc[1]
    top3_p = merged.iloc[2]

    w(f"The external player the Warriors need the most based on pure statistics is "
      f"**{top1['PLAYER_NAME']}** ({top1['TEAM_ABBREVIATION']}) with a Gap-Fill Score of "
      f"**{top1['gap_fill_score']:.1f}**.")
    w("")
    w(f"| Rank | Player | Team | Gap-Fill Score | Primary Fit |")
    w(f"|---:|---|---|---:|---|")
    w(f"| 1 | **{top1['PLAYER_NAME']}** | {top1['TEAM_ABBREVIATION']} | **{top1['gap_fill_score']:.1f}** | {top1['best_fit']} |")
    w(f"| 2 | **{top2['PLAYER_NAME']}** | {top2['TEAM_ABBREVIATION']} | **{top2['gap_fill_score']:.1f}** | {top2['best_fit']} |")
    w(f"| 3 | **{top3_p['PLAYER_NAME']}** | {top3_p['TEAM_ABBREVIATION']} | **{top3_p['gap_fill_score']:.1f}** | {top3_p['best_fit']} |")
    w("")

    w("---")
    w(f"*Gap-fill analysis generated by NBATracker — {len(merged)} players evaluated across 6 dimensions.*")
    w("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--season", default="2025-26", help="NBA season (e.g. 2025-26).")
@click.option("--output", "output_path", default=None, help="Output .md file path.")
def main(season: str, output_path: str | None) -> None:
    """Generate a gap-fill analysis report for the Warriors."""
    if output_path is None:
        safe = season.replace("-", "_")
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", f"warriors_gap_analysis_{safe}.md"
        )

    report = build_report(season)

    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nGap-fill report written to {out_path}")


if __name__ == "__main__":
    main()
