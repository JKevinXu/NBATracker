#!/usr/bin/env python3
"""Golden State Warriors — Trade Deadline Plan.

Generates a Markdown report with:
  1. Warriors asset classification (untouchable / core / tradeable / expendable)
  2. Identified sellers across the league
  3. Priority trade targets ranked by realistic fit
  4. Concrete trade scenarios with value-based reasoning
  5. Impact projections for each trade

Run:
    python scripts/warriors_trade_plan.py [--season 2025-26]
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
    LeagueStandings,
    TeamGameLog,
)

TEAM_ID = 1610612744
TEAM_ABBR = "GSW"
API_DELAY = 0.6


def _delay():
    time.sleep(API_DELAY)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _pm(v: float) -> str:
    return f"+{v:.1f}" if v > 0 else f"{v:.1f}"


def norm(s: pd.Series, higher_better: bool = True) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    n = (s - mn) / (mx - mn) * 100
    return n if higher_better else 100 - n


# ── data fetching ─────────────────────────────────────────────────


def fetch_standings(season: str) -> pd.DataFrame:
    print("  Fetching standings..."); _delay()
    return LeagueStandings(season=season, league_id="00").get_data_frames()[0]


def fetch_gsw_adv(season: str) -> pd.DataFrame:
    print("  Fetching Warriors advanced stats..."); _delay()
    return LeagueDashPlayerStats(
        team_id_nullable=TEAM_ID, season=season,
        per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]


def fetch_gsw_base(season: str) -> pd.DataFrame:
    print("  Fetching Warriors base stats..."); _delay()
    return LeagueDashPlayerStats(
        team_id_nullable=TEAM_ID, season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_league_adv(season: str) -> pd.DataFrame:
    print("  Fetching league advanced stats..."); _delay()
    return LeagueDashPlayerStats(
        season=season, per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]


def fetch_league_base(season: str) -> pd.DataFrame:
    print("  Fetching league base stats..."); _delay()
    return LeagueDashPlayerStats(
        season=season, per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_league_clutch(season: str) -> pd.DataFrame:
    print("  Fetching league clutch stats..."); _delay()
    return LeagueDashPlayerClutch(
        season=season, per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_team_game_log(season: str) -> pd.DataFrame:
    print("  Fetching Warriors game log..."); _delay()
    return TeamGameLog(
        team_id=TEAM_ID, season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]


# ── asset valuation ───────────────────────────────────────────────


def compute_trade_value(row: pd.Series) -> float:
    """Compute a 0-100 trade value based on production + impact."""
    ppg = float(row.get("PTS", 0))
    rpg = float(row.get("REB", 0))
    apg = float(row.get("AST", 0))
    net = float(row.get("NET_RATING", 0))
    pie = float(row.get("PIE", 0))
    ts = float(row.get("TS_PCT", 0))
    usg = float(row.get("USG_PCT", 0))
    gp = int(row.get("GP", 0))
    mpg = float(row.get("MIN", 0))

    # Scoring value (0-30)
    scoring = min(30, ppg * ts * 2)
    # Impact (0-25)
    impact = min(25, max(0, (net + 15) / 30 * 25))
    # PIE (0-15)
    pie_score = min(15, pie * 100)
    # Volume (0-15) — minutes × games
    volume = min(15, (mpg / 36) * (gp / 55) * 15)
    # Versatility bonus (0-15) — rebounds + assists
    versatility = min(15, (rpg + apg) / 15 * 15)

    return scoring + impact + pie_score + volume + versatility


def classify_warrior(name: str, trade_val: float, ppg: float, usg: float,
                     net: float, pie: float, gp: int) -> str:
    """Classify Warriors players into trade tiers."""
    # Untouchable
    if name in ("Stephen Curry",):
        return "Untouchable"
    # Core — high-value players central to the team
    if name in ("Jimmy Butler III", "Draymond Green"):
        return "Core"
    if pie >= 0.14 and net >= 5:
        return "Core"
    # Available — solid players that could be moved in the right deal
    if trade_val >= 25 or (ppg >= 10 and gp >= 30):
        return "Available"
    # Expendable — lower-value or negative-impact players
    return "Expendable"


# ── report builder ────────────────────────────────────────────────


def build_report(season: str) -> str:
    lines: list[str] = []
    w = lines.append

    print(f"Building trade plan for {TEAM_ABBR} — {season}...")

    # Fetch data
    standings = fetch_standings(season)
    gsw_adv = fetch_gsw_adv(season)
    gsw_base = fetch_gsw_base(season)
    league_adv = fetch_league_adv(season)
    league_base = fetch_league_base(season)
    league_clutch = fetch_league_clutch(season)
    game_log = fetch_team_game_log(season)

    # ── Warriors current state ───────────────────────────────────
    gsw_stand = standings[standings["TeamCity"] == "Golden State"].iloc[0]
    curr_w = int(gsw_stand["WINS"])
    curr_l = int(gsw_stand["LOSSES"])
    seed = int(gsw_stand["PlayoffRank"])
    diff_ppg = float(gsw_stand["DiffPointsPG"])

    # ── Team records map ─────────────────────────────────────────
    team_map: dict[int, dict] = {}
    for _, r in standings.iterrows():
        tid = int(r["TeamID"])
        team_map[tid] = {
            "name": f"{r['TeamCity']} {r['TeamName']}",
            "abbr": str(r["TeamID"]),
            "wins": int(r["WINS"]),
            "losses": int(r["LOSSES"]),
            "pct": float(r["WinPCT"]),
            "rank": int(r["PlayoffRank"]),
            "conf": str(r["Conference"]),
        }

    def team_sit(tid):
        pct = team_map.get(tid, {}).get("pct", 0.5)
        if pct <= 0.30: return "Tanking"
        if pct <= 0.40: return "Lottery"
        if pct <= 0.48: return "Fringe"
        if pct <= 0.55: return "Bubble"
        return "Contender"

    # ── Warriors player valuation ────────────────────────────────
    gsw_merged = gsw_base[gsw_base["GP"] >= 5].merge(
        gsw_adv[gsw_adv["GP"] >= 5][
            ["PLAYER_ID", "NET_RATING", "OFF_RATING", "DEF_RATING",
             "TS_PCT", "USG_PCT", "PIE", "AST_TO", "EFG_PCT"]
        ],
        on="PLAYER_ID", how="inner", suffixes=("", "_adv"),
    )

    gsw_merged["trade_value"] = gsw_merged.apply(compute_trade_value, axis=1)
    gsw_merged["tier"] = gsw_merged.apply(
        lambda r: classify_warrior(
            str(r["PLAYER_NAME"]), float(r["trade_value"]),
            float(r["PTS"]), float(r["USG_PCT"]),
            float(r["NET_RATING"]), float(r["PIE"]), int(r["GP"]),
        ),
        axis=1,
    )
    gsw_merged = gsw_merged.sort_values("trade_value", ascending=False).reset_index(drop=True)

    # ── League targets ───────────────────────────────────────────
    # Filter to non-GSW, 30+ GP, 20+ MIN, from selling teams
    seller_tids = {tid for tid, info in team_map.items() if info["pct"] <= 0.45 and tid != TEAM_ID}

    targets_base = league_base[
        (league_base["TEAM_ID"].isin(seller_tids))
        & (league_base["GP"] >= 25)
        & (league_base["MIN"] >= 18)
    ].copy()
    targets_adv = league_adv[
        (league_adv["TEAM_ID"].isin(seller_tids))
        & (league_adv["GP"] >= 25)
        & (league_adv["MIN"] >= 18)
    ].copy()

    targets = targets_base.merge(
        targets_adv[["PLAYER_ID", "NET_RATING", "OFF_RATING", "DEF_RATING",
                      "TS_PCT", "USG_PCT", "PIE", "AST_TO", "EFG_PCT"]],
        on="PLAYER_ID", how="inner", suffixes=("", "_adv"),
    )

    clutch_cols = league_clutch[["PLAYER_ID", "PTS", "FG_PCT"]].rename(
        columns={"PTS": "CLUTCH_PTS", "FG_PCT": "CLUTCH_FG_PCT"}
    )
    targets = targets.merge(clutch_cols, on="PLAYER_ID", how="left")
    targets["CLUTCH_PTS"] = targets["CLUTCH_PTS"].fillna(0)
    targets["CLUTCH_FG_PCT"] = targets["CLUTCH_FG_PCT"].fillna(0)

    # Compute gap-fill score for targets
    if len(targets) > 0:
        targets["dim_defense"] = (
            0.50 * norm(targets["DEF_RATING"], False)
            + 0.25 * norm(targets["STL"])
            + 0.25 * norm(targets["BLK"])
        )
        targets["dim_scoring"] = norm(targets["PTS"] * targets["TS_PCT"])
        targets["dim_rebounding"] = norm(0.4 * targets["REB"] + 0.6 * targets["DREB"])
        targets["dim_playmaking"] = 0.60 * norm(targets["AST_TO"]) + 0.40 * norm(targets["TOV"], False)
        targets["dim_versatility"] = 0.50 * norm(targets["NET_RATING"]) + 0.50 * norm(targets["PIE"])
        targets["dim_clutch"] = 0.60 * norm(targets["CLUTCH_PTS"]) + 0.40 * norm(targets["CLUTCH_FG_PCT"])

        targets["gap_fill"] = (
            0.25 * targets["dim_defense"] + 0.20 * targets["dim_scoring"]
            + 0.15 * targets["dim_rebounding"] + 0.15 * targets["dim_playmaking"]
            + 0.15 * targets["dim_versatility"] + 0.10 * targets["dim_clutch"]
        )
        targets["trade_value"] = targets.apply(compute_trade_value, axis=1)

    targets["team_name"] = targets["TEAM_ID"].apply(lambda t: team_map.get(t, {}).get("name", "?"))
    targets["situation"] = targets["TEAM_ID"].apply(team_sit)
    targets = targets.sort_values("gap_fill", ascending=False).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # BUILD MARKDOWN REPORT
    # ═══════════════════════════════════════════════════════════════

    w(f"# Golden State Warriors — {season} Trade Deadline Plan")
    w("")
    w(f"> Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    w(f"> Current record: **{curr_w}-{curr_l}** | West #{seed} | Diff: {_pm(diff_ppg)} PPG")
    w(f"> 27 games remaining — team is in **win-now mode** around Curry's window")
    w("")

    # ── 1. Trade Mandate ─────────────────────────────────────────
    w("## 1. Trade Mandate")
    w("")
    w("The Warriors are 29-26, sitting as the **8th seed** in the West with 27 games left.")
    w("Stephen Curry is 37 years old. The championship window is closing.")
    w("")
    w("**Objective:** Acquire a player who upgrades the roster for a **playoff push** without")
    w("gutting the young core. Priority needs:")
    w("")
    w("1. **Scoring depth** — The drop from Butler (20.0 PPG) to the 3rd scorer (12.1 PPG) is too steep")
    w("2. **Interior defense / rebounding** — Big rotation is aging (Horford) or negative-impact (TJD at -6.2 NET)")
    w("3. **Closing ability** — 8-11 in clutch games; need someone who can get buckets in crunch time")
    w("")

    # ── 2. Warriors Asset Classification ─────────────────────────
    w("## 2. Warriors Asset Classification")
    w("")
    w("| Tier | Player | GP | PPG | NET RTG | PIE | Trade Value | Role |")
    w("|---|---|---:|---:|---:|---:|---:|---|")

    tier_order = {"Untouchable": 0, "Core": 1, "Available": 2, "Expendable": 3}
    gsw_sorted = gsw_merged.sort_values(
        by=["tier", "trade_value"],
        key=lambda s: s.map(tier_order) if s.name == "tier" else -s,
    )

    for _, r in gsw_sorted.iterrows():
        tier = r["tier"]
        icon = {"Untouchable": "🔒", "Core": "⭐", "Available": "📦", "Expendable": "↔️"}.get(tier, "")
        w(
            f"| **{tier}** {icon} | {r['PLAYER_NAME']} "
            f"| {int(r['GP'])} | {float(r['PTS']):.1f} "
            f"| {_pm(r['NET_RATING'])} | {_pct(r['PIE'])} "
            f"| {r['trade_value']:.1f} | {_pct(r['USG_PCT'])} USG, {float(r['MIN']):.0f} MIN |"
        )
    w("")

    # Summarize tradeable assets
    available = gsw_sorted[gsw_sorted["tier"] == "Available"]
    expendable = gsw_sorted[gsw_sorted["tier"] == "Expendable"]
    w("**Tradeable assets:**")
    for _, r in available.iterrows():
        w(f"- **{r['PLAYER_NAME']}** — {float(r['PTS']):.1f} PPG, value {r['trade_value']:.1f}")
    for _, r in expendable.iterrows():
        w(f"- {r['PLAYER_NAME']} — {float(r['PTS']):.1f} PPG, value {r['trade_value']:.1f}")
    w("")

    # ── 3. Sellers Across the League ─────────────────────────────
    w("## 3. Identified Sellers")
    w("")
    w("Teams likely to be selling at the deadline based on record and playoff positioning:")
    w("")
    w("| Team | Record | Win% | Situation | Notable Assets |")
    w("|---|---|---:|---|---|")

    # Collect seller info with their top available player
    sellers = []
    for tid, info in sorted(team_map.items(), key=lambda x: x[1]["pct"]):
        if info["pct"] <= 0.45 and tid != TEAM_ID:
            team_targets = targets[targets["TEAM_ID"] == tid].head(2)
            notable = ", ".join(
                f"{r['PLAYER_NAME']} ({float(r['PTS']):.1f} PPG)"
                for _, r in team_targets.iterrows()
            )
            sellers.append((tid, info, notable))
            w(
                f"| {info['name']} | {info['wins']}-{info['losses']} "
                f"| {_pct(info['pct'])} | {team_sit(tid)} | {notable} |"
            )
    w("")

    # ── 4. Priority Trade Targets ────────────────────────────────
    w("## 4. Priority Trade Targets")
    w("")
    w("Top targets from selling teams, ranked by gap-fill score:")
    w("")
    w("| Rank | Player | Team | Situation | PTS | REB | AST | TS% | NET | Gap-Fill | Value |")
    w("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    top_targets = targets.head(15)
    for i, (_, r) in enumerate(top_targets.iterrows()):
        sit = r["situation"]
        w(
            f"| {i+1} | **{r['PLAYER_NAME']}** | {r['TEAM_ABBREVIATION']} "
            f"| {sit} | {float(r['PTS']):.1f} | {float(r['REB']):.1f} "
            f"| {float(r['AST']):.1f} | {_pct(r['TS_PCT'])} "
            f"| {_pm(r['NET_RATING'])} | {r['gap_fill']:.1f} | {r['trade_value']:.1f} |"
        )
    w("")

    # ── 5. Trade Scenarios ───────────────────────────────────────
    w("## 5. Proposed Trade Scenarios")
    w("")

    # Build trade scenarios based on target value vs available assets
    # We'll match target trade_value against packages of GSW available players

    avail_players = pd.concat([available, expendable]).to_dict("records")
    avail_by_val = sorted(avail_players, key=lambda x: -x["trade_value"])

    # Helper: find a package that approximately matches target value
    def build_package(target_val: float, exclude_names: set | None = None) -> list[dict]:
        """Greedy package builder — pick players whose combined value ~ target."""
        if exclude_names is None:
            exclude_names = set()
        pkg: list[dict] = []
        remaining = target_val * 0.85  # we don't need to match exactly
        for p in avail_by_val:
            if p["PLAYER_NAME"] in exclude_names:
                continue
            if remaining <= 0:
                break
            if p["trade_value"] <= remaining + 5:
                pkg.append(p)
                remaining -= p["trade_value"]
                exclude_names.add(p["PLAYER_NAME"])
        return pkg

    # Generate scenarios for top targets
    scenarios_written = 0
    used_names: set[str] = set()

    for _, target in top_targets.head(8).iterrows():
        if scenarios_written >= 5:
            break

        target_name = str(target["PLAYER_NAME"])
        target_team = str(target["TEAM_ABBREVIATION"])
        target_ppg = float(target["PTS"])
        target_rpg = float(target["REB"])
        target_apg = float(target["AST"])
        target_ts = float(target["TS_PCT"])
        target_net = float(target["NET_RATING"])
        target_tv = float(target["trade_value"])
        target_sit = str(target["situation"])

        # Skip if we can't build a reasonable package
        pkg = build_package(target_tv, used_names.copy())
        if not pkg:
            continue

        total_pkg_val = sum(p["trade_value"] for p in pkg)
        pkg_names = [p["PLAYER_NAME"] for p in pkg]

        # Mark names as used
        for p in pkg:
            used_names.add(p["PLAYER_NAME"])

        scenarios_written += 1
        letter = chr(64 + scenarios_written)  # A, B, C...

        w(f"### Trade {letter}: Acquire **{target_name}** from {target_team}")
        w("")
        w(f"**{target_team} situation:** {target_sit} ({team_map.get(int(target['TEAM_ID']), {}).get('wins', '?')}-{team_map.get(int(target['TEAM_ID']), {}).get('losses', '?')})")
        w("")
        w("| Warriors Send | Warriors Receive |")
        w("|---|---|")

        send_lines = []
        for p in pkg:
            send_lines.append(f"{p['PLAYER_NAME']} ({float(p['PTS']):.1f} PPG, value {p['trade_value']:.1f})")
        send_lines.append("+ Future draft consideration")

        receive_line = f"**{target_name}** ({target_ppg:.1f} PPG / {target_rpg:.1f} RPG / {target_apg:.1f} APG)"

        for j, sl in enumerate(send_lines):
            rl = receive_line if j == 0 else ""
            w(f"| {sl} | {rl} |")
        w("")

        # Value analysis
        w(f"**Value exchange:** {total_pkg_val:.1f} (outgoing) vs {target_tv:.1f} (incoming)")
        if total_pkg_val >= target_tv * 0.8:
            w(f"  — Fair value range. The {target_team.split()[-1] if ' ' in target_team else target_team} get rotation pieces / youth for their rebuild.")
        else:
            w(f"  — Warriors would likely need to sweeten with a draft pick.")
        w("")

        # Impact projection
        w("**Projected impact:**")
        if target_ppg >= 18:
            w(f"- Scoring: {target_name} immediately becomes the Warriors' **3rd scoring option** at {target_ppg:.1f} PPG")
        elif target_ppg >= 12:
            w(f"- Scoring: Adds {target_ppg:.1f} PPG of **depth scoring**")
        if target_rpg >= 8:
            w(f"- Rebounding: +{target_rpg:.1f} RPG significantly upgrades interior presence")
        if target_net > 0:
            w(f"- On-court impact: {_pm(target_net)} NET RTG — the team gets **better** with him on the court")
        elif target_net > -5:
            w(f"- On-court impact: {_pm(target_net)} NET RTG (on a bad team — expect improvement in better system)")
        if target_ts >= 0.58:
            w(f"- Efficiency: {_pct(target_ts)} TS — won't tank the offense")

        # What GSW loses
        total_outgoing_ppg = sum(float(p["PTS"]) for p in pkg)
        total_outgoing_net = np.mean([float(p["NET_RATING"]) for p in pkg])
        w(f"- Cost: Loses {total_outgoing_ppg:.1f} combined PPG from {', '.join(pkg_names)}")
        w("")

        # Grade
        net_ppg_gain = target_ppg - total_outgoing_ppg
        if net_ppg_gain > 5 and target_ts >= 0.56:
            grade = "A"
        elif net_ppg_gain > 0 and target_ts >= 0.54:
            grade = "B+"
        elif target_net > 0:
            grade = "B"
        else:
            grade = "C+"
        w(f"**Trade grade: {grade}** | Net PPG change: {_pm(net_ppg_gain)}")
        w("")

    # ── 6. Recommended Strategy ──────────────────────────────────
    w("## 6. Recommended Strategy")
    w("")

    # Identify the best trade from our scenarios
    # Use the first scenario as the primary recommendation
    if scenarios_written > 0:
        best = top_targets.iloc[0]
        w(f"### Primary Target: **{best['PLAYER_NAME']}** ({best['TEAM_ABBREVIATION']})")
        w("")
        w(f"- **Why:** {float(best['PTS']):.1f} PPG on {_pct(best['TS_PCT'])} TS, gap-fill score of {best['gap_fill']:.1f}")
        w(f"- **Availability:** {best['situation']} team — motivated seller")
        w(f"- **Fit:** Immediately addresses scoring depth gap and provides {float(best['REB']):.1f} RPG")
        w("")

        if len(top_targets) >= 3:
            backup = top_targets.iloc[2]
            w(f"### Backup Target: **{backup['PLAYER_NAME']}** ({backup['TEAM_ABBREVIATION']})")
            w("")
            w(f"- **Why:** {float(backup['PTS']):.1f} PPG on {_pct(backup['TS_PCT'])} TS")
            w(f"- **Availability:** {backup['situation']} team")
            w("")

    w("### Strategy Notes")
    w("")
    w("1. **Do not trade Curry, Butler, or Draymond** — they are the competitive core")
    w("2. **Prioritize scoring depth** — the 12.1 PPG 3rd-scorer cliff is the biggest fixable issue")
    w("3. **Package expendable contracts** — Combine negative-NET players to match salary for an upgrade")
    w("4. **Protect young assets where possible** — Podziemski and Moody have long-term value")
    w("5. **A 2nd-round pick sweetener** is acceptable; a 1st-round pick only for a clear upgrade")
    w("6. **Deadline urgency is real** — at 29-26, every game matters; the sooner the better")
    w("")

    # ── 7. What Success Looks Like ───────────────────────────────
    w("## 7. What Success Looks Like")
    w("")
    w("| Scenario | Projected Record | Playoff Outcome |")
    w("|---|---|---|")
    w(f"| No trade (status quo) | ~43-39 | Play-in (7-10 seed) |")
    w(f"| Mid-tier upgrade (+3-4 wins) | ~46-36 | Solid playoff seed (6-7) |")
    w(f"| Home-run trade (+5-6 wins) | ~48-34 | Top 6 lock, 1st-round threat |")
    w("")

    w("The difference between standing pat and making a smart trade could be the difference")
    w("between a play-in exit and a genuine playoff run — likely Curry's last realistic shot at contention.")
    w("")

    w("---")
    w(f"*Trade plan generated by NBATracker — based on {season} statistics through {curr_w + curr_l} games.*")
    w("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--season", default="2025-26", help="NBA season (e.g. 2025-26).")
@click.option("--output", "output_path", default=None, help="Output .md file path.")
def main(season: str, output_path: str | None) -> None:
    """Generate a trade deadline plan for the Warriors."""
    if output_path is None:
        safe = season.replace("-", "_")
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", f"warriors_trade_plan_{safe}.md"
        )

    report = build_report(season)

    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nTrade plan written to {out_path}")


if __name__ == "__main__":
    main()
