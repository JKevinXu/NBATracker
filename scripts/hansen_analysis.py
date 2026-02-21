#!/usr/bin/env python3
"""
Comprehensive analysis of Yang Hansen (Portland Trail Blazers)
Generates a detailed Markdown report with traditional stats, advanced metrics,
shooting analysis, on/off impact, lineup data, game-by-game trends,
and ML-driven insights (clustering, consistency, fatigue, value index).
"""

import json, sys, os, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
warnings.filterwarnings("ignore")

CACHE = Path("web/cache")
REPORT = Path("reports/hansen/hansen_analysis_2025_26.md")
PLAYER_NAME = "Yang Hansen"
PLAYER_ID = 1642905
SEASON = "2025-26"

# ── helpers ──────────────────────────────────────────────────────────────
lines = []
def w(s=""): lines.append(s)
def h(n, t): w(f"{'#'*n} {t}"); w()
def p(t): w(t); w()
def blank(): w()
def tbl(*cols): w("| " + " | ".join(cols) + " |"); w("|" + "|".join(["---"]*len(cols)) + "|")
def row(*vals): w("| " + " | ".join(str(v) for v in vals) + " |")

def load(name):
    fp = CACHE / f"{name}.json"
    if not fp.exists():
        print(f"  Warning: {fp} not found")
        return None
    return json.loads(fp.read_text())

def to_df(data, idx=0):
    if data is None: return pd.DataFrame()
    rs = data["resultSets"][idx]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])

# ── load data ────────────────────────────────────────────────────────────
print("Loading data…")
hansen_gl_raw = load("hansen_gamelogs")
hansen_shooting = load("hansen_shooting")
hansen_splits = load("hansen_splits")
por_base_raw = load("por_player_base")
por_adv_raw = load("por_player_adv")
por_hustle_raw = load("por_hustle")
por_tracking_raw = load("por_tracking")
por_lineups_raw = load("por_lineups")
por_on_off_raw = load("por_on_off")
por_gamelogs_raw = load("por_player_gamelogs")
por_team_gl_raw = load("por_gamelog")
por_shot_loc_raw = load("por_shot_locations")

# DataFrames
hansen_gl = to_df(hansen_gl_raw)
por_base = to_df(por_base_raw)
por_adv = to_df(por_adv_raw)
por_hustle = to_df(por_hustle_raw)
por_tracking = to_df(por_tracking_raw)
por_lineups = to_df(por_lineups_raw)
por_gamelogs = to_df(por_gamelogs_raw)
por_team_gl = to_df(por_team_gl_raw)

# On/off: result sets 1=on-court, 2=off-court (idx 0 is overall)
por_on = to_df(por_on_off_raw, 1) if por_on_off_raw else pd.DataFrame()
por_off = to_df(por_on_off_raw, 2) if por_on_off_raw else pd.DataFrame()

# Hansen row in team stats
hansen_base = por_base[por_base["PLAYER_NAME"] == PLAYER_NAME].iloc[0] if len(por_base[por_base["PLAYER_NAME"] == PLAYER_NAME]) > 0 else None
hansen_adv = por_adv[por_adv["PLAYER_NAME"] == PLAYER_NAME].iloc[0] if len(por_adv[por_adv["PLAYER_NAME"] == PLAYER_NAME]) > 0 else None

# ═══════════════════════════════════════════════════════════════════════
# Build Report
# ═══════════════════════════════════════════════════════════════════════
print("Building report…")

h(1, f"Yang Hansen — Portland Trail Blazers | {SEASON} Season Analysis")
p(f"*Generated {datetime.now().strftime('%B %d, %Y')} | Data: stats.nba.com*")

# ── 1. Player Profile ────────────────────────────────────────────────
h(2, "1. Player Profile & Season Overview")

if hansen_base is not None:
    gp = int(hansen_base["GP"])
    mpg = hansen_base["MIN"]
    ppg = hansen_base["PTS"]
    rpg = hansen_base["REB"]
    apg = hansen_base["AST"]
    spg = hansen_base["STL"]
    bpg = hansen_base["BLK"]
    tpg = hansen_base["TOV"]
    fg_pct = hansen_base["FG_PCT"] * 100
    fg3_pct = hansen_base["FG3_PCT"] * 100
    ft_pct = hansen_base["FT_PCT"] * 100
    plus_minus = hansen_base["PLUS_MINUS"]

    tbl("Stat", "Value")
    row("Games Played", gp)
    row("Minutes Per Game", f"{mpg:.1f}")
    row("Points Per Game", f"{ppg:.1f}")
    row("Rebounds Per Game", f"{rpg:.1f}")
    row("Assists Per Game", f"{apg:.1f}")
    row("Steals Per Game", f"{spg:.1f}")
    row("Blocks Per Game", f"{bpg:.1f}")
    row("Turnovers Per Game", f"{tpg:.1f}")
    row("FG%", f"{fg_pct:.1f}%")
    row("3PT%", f"{fg3_pct:.1f}%")
    row("FT%", f"{ft_pct:.1f}%")
    row("+/- Per Game", f"{plus_minus:+.1f}")
    blank()

# Advanced metrics
if hansen_adv is not None:
    h(3, "Advanced Metrics")
    ts = hansen_adv.get("TS_PCT", 0) * 100 if hansen_adv.get("TS_PCT") else 0
    efg = hansen_adv.get("EFG_PCT", 0) * 100 if hansen_adv.get("EFG_PCT") else 0
    usg = hansen_adv.get("USG_PCT", 0) * 100 if hansen_adv.get("USG_PCT") else 0
    off_rtg = hansen_adv.get("OFF_RATING", 0) if hansen_adv.get("OFF_RATING") else 0
    def_rtg = hansen_adv.get("DEF_RATING", 0) if hansen_adv.get("DEF_RATING") else 0
    net_rtg = hansen_adv.get("NET_RATING", 0) if hansen_adv.get("NET_RATING") else 0
    pie = hansen_adv.get("PIE", 0) * 100 if hansen_adv.get("PIE") else 0
    ast_pct = hansen_adv.get("AST_PCT", 0) * 100 if hansen_adv.get("AST_PCT") else 0
    reb_pct = hansen_adv.get("REB_PCT", 0) * 100 if hansen_adv.get("REB_PCT") else 0

    tbl("Metric", "Value", "Context")
    row("True Shooting %", f"{ts:.1f}%", "League avg ~57%")
    row("Effective FG%", f"{efg:.1f}%", "League avg ~53%")
    row("Usage Rate", f"{usg:.1f}%", "Average role ~20%")
    row("Offensive Rating", f"{off_rtg:.1f}", "Points per 100 poss")
    row("Defensive Rating", f"{def_rtg:.1f}", "Points allowed per 100 poss")
    row("Net Rating", f"{net_rtg:+.1f}", "Off - Def rating")
    row("PIE", f"{pie:.1f}%", "Player Impact Estimate")
    row("AST%", f"{ast_pct:.1f}%", "% of team FG assisted while on court")
    row("REB%", f"{reb_pct:.1f}%", "% of available rebounds grabbed")
    blank()

# ── 2. Per-36 & Per-100 Projections ─────────────────────────────────
h(2, "2. Production Rate Projections")
if hansen_base is not None and mpg > 0:
    mult36 = 36.0 / mpg
    p("Projecting Hansen's stats to larger minutes contexts shows his production rate:")
    blank()
    tbl("Stat", "Per Game", "Per 36 Min", "Per 100 Poss")
    # Per 100 poss estimated from off rating and team pace
    pace_est = 100.0  # placeholder
    if hansen_adv is not None and off_rtg > 0:
        # Rough estimate: per100 = per_game * (100 / (pace * mpg / 48))
        pace_est = 100.0
    for stat, val in [("PTS", ppg), ("REB", rpg), ("AST", apg), ("STL", spg), ("BLK", bpg)]:
        per36 = val * mult36
        row(stat, f"{val:.1f}", f"{per36:.1f}", "—")
    blank()
    p(f"*Per-36 multiplier: {mult36:.2f}x (based on {mpg:.1f} MPG)*")

# ── 3. Game Log Trends ──────────────────────────────────────────────
h(2, "3. Game-by-Game Performance Trends")
if len(hansen_gl) > 0:
    # Parse dates
    hansen_gl["GAME_DATE_DT"] = pd.to_datetime(hansen_gl["GAME_DATE"], format="mixed")
    hansen_gl = hansen_gl.sort_values("GAME_DATE_DT")
    hansen_gl["GAME_NUM"] = range(1, len(hansen_gl)+1)

    # Scoring trend
    pts = hansen_gl["PTS"].astype(float)
    mins = hansen_gl["MIN"].astype(float)
    rebs = hansen_gl["REB"].astype(float)
    asts = hansen_gl["AST"].astype(float)

    p(f"**Season scoring line:** {pts.mean():.1f} PPG (σ={pts.std():.1f})")
    p(f"**Career high this season:** {int(pts.max())} pts on {hansen_gl.loc[pts.idxmax(), 'GAME_DATE']} vs {hansen_gl.loc[pts.idxmax(), 'MATCHUP']}")
    blank()

    # Split into halves
    mid = len(hansen_gl) // 2
    first_half = hansen_gl.iloc[:mid]
    second_half = hansen_gl.iloc[mid:]

    p("### First Half vs Second Half")
    tbl("Split", "GP", "MPG", "PPG", "RPG", "APG", "FG%")
    for label, df in [("First Half", first_half), ("Second Half", second_half)]:
        g = len(df)
        m = df["MIN"].astype(float).mean()
        pt = df["PTS"].astype(float).mean()
        rb = df["REB"].astype(float).mean()
        at = df["AST"].astype(float).mean()
        fgm = df["FGM"].astype(float).sum()
        fga = df["FGA"].astype(float).sum()
        fg_p = (fgm/fga*100) if fga > 0 else 0
        row(label, g, f"{m:.1f}", f"{pt:.1f}", f"{rb:.1f}", f"{at:.1f}", f"{fg_p:.1f}%")
    blank()

    # Last 10 games
    last10 = hansen_gl.tail(10)
    h(3, "Last 10 Games Detail")
    tbl("Date", "Matchup", "W/L", "MIN", "PTS", "REB", "AST", "FG", "3PT", "+/-")
    for _, g in last10.iterrows():
        row(
            g["GAME_DATE"],
            g["MATCHUP"],
            g["WL"],
            g["MIN"],
            int(g["PTS"]),
            int(g["REB"]),
            int(g["AST"]),
            f"{int(g['FGM'])}-{int(g['FGA'])}",
            f"{int(g['FG3M'])}-{int(g['FG3A'])}",
            f"{int(g['PLUS_MINUS']):+d}",
        )
    blank()

    # Monthly splits from hansen_splits
    if hansen_splits:
        for rs in hansen_splits["resultSets"]:
            if rs["name"] == "MonthPlayerDashboard":
                month_df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
                if len(month_df) > 0:
                    h(3, "Monthly Splits")
                    tbl("Month", "GP", "MPG", "PPG", "RPG", "APG", "FG%", "3PT%")
                    for _, m in month_df.iterrows():
                        mo = str(m.get("GROUP_VALUE", "?"))
                        row(
                            mo, int(m["GP"]), f"{m['MIN']:.1f}",
                            f"{m['PTS']:.1f}", f"{m['REB']:.1f}", f"{m['AST']:.1f}",
                            f"{m['FG_PCT']*100:.1f}%", f"{m['FG3_PCT']*100:.1f}%"
                        )
                    blank()

# ── 4. Shooting Analysis ────────────────────────────────────────────
h(2, "4. Shooting Profile")
p("*Note: All shooting data is season totals.*")
if hansen_shooting:
    gp_shoot = int(hansen_base["GP"]) if hansen_base is not None else 1
    for rs_item in hansen_shooting["resultSets"]:
        # Shot area
        if rs_item["name"] == "ShotAreaPlayerDashboard":
            area_df = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
            if len(area_df) > 0:
                h(3, "Shot Area Breakdown")
                tbl("Area", "FGM", "FGA", "FG%", "% of Shots")
                total_fga = area_df["FGA"].sum()
                for _, s in area_df.iterrows():
                    pct = s["FGA"]/total_fga*100 if total_fga > 0 else 0
                    row(s["GROUP_VALUE"], int(s['FGM']), int(s['FGA']),
                        f"{s['FG_PCT']*100:.1f}%", f"{pct:.1f}%")
                blank()

        # By distance
        if rs_item["name"] == "Shot5FTPlayerDashboard":
            dist_df = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
            if len(dist_df) > 0:
                h(3, "Shot Distance Distribution")
                tbl("Distance", "FGM", "FGA", "FG%", "% of Shots")
                total_fga = dist_df["FGA"].sum()
                for _, s in dist_df.iterrows():
                    pct = s["FGA"]/total_fga*100 if total_fga > 0 else 0
                    row(s["GROUP_VALUE"], int(s['FGM']), int(s['FGA']),
                        f"{s['FG_PCT']*100:.1f}%", f"{pct:.1f}%")
                blank()

        # Shot type (summary, not detail)
        if rs_item["name"] == "ShotTypeSummaryPlayerDashboard":
            type_df = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
            if len(type_df) > 0:
                h(3, "Shot Type Breakdown")
                tbl("Shot Type", "FGM", "FGA", "FG%")
                for _, s in type_df.iterrows():
                    row(s["GROUP_VALUE"], int(s['FGM']), int(s['FGA']),
                        f"{s['FG_PCT']*100:.1f}%")
                blank()

        # Assisted vs unassisted
        if rs_item["name"] == "AssitedShotPlayerDashboard":
            assist_df = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
            if len(assist_df) > 0:
                h(3, "Assisted vs Unassisted")
                tbl("Type", "FGM", "FGA", "FG%")
                for _, s in assist_df.iterrows():
                    row(s["GROUP_VALUE"], int(s['FGM']), int(s['FGA']),
                        f"{s['FG_PCT']*100:.1f}%")
                blank()

        # Top assisters (who assists Hansen the most)
        if rs_item["name"] == "AssistedBy":
            ab_df = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
            if len(ab_df) > 0:
                h(3, "Top Assisters to Hansen")
                tbl("Player", "Assisted FGM", "FG%")
                top_assisters = ab_df.sort_values("FGM", ascending=False).head(5)
                for _, s in top_assisters.iterrows():
                    row(s["PLAYER_NAME"], int(s['FGM']), f"{s['FG_PCT']*100:.1f}%")
                blank()

# ── 5. Hustle & Tracking ────────────────────────────────────────────
h(2, "5. Hustle & Tracking Data")
hansen_hustle = por_hustle[por_hustle["PLAYER_NAME"] == PLAYER_NAME] if len(por_hustle) > 0 else pd.DataFrame()
hansen_track = por_tracking[por_tracking["PLAYER_NAME"] == PLAYER_NAME] if len(por_tracking) > 0 else pd.DataFrame()

if len(hansen_hustle) > 0:
    hh = hansen_hustle.iloc[0]
    h(3, "Hustle Stats")
    tbl("Metric", "Per Game")
    for col, label in [
        ("CONTESTED_SHOTS", "Contested Shots"),
        ("DEFLECTIONS", "Deflections"),
        ("LOOSE_BALLS_RECOVERED", "Loose Balls Recovered"),
        ("SCREEN_ASSISTS", "Screen Assists"),
        ("CHARGES_DRAWN", "Charges Drawn"),
        ("BOX_OUTS", "Box Outs"),
    ]:
        if col in hh.index:
            row(label, f"{hh[col]:.1f}")
    blank()

if len(hansen_track) > 0:
    ht = hansen_track.iloc[0]
    h(3, "Speed & Distance")
    tbl("Metric", "Value")
    if "AVG_SPEED" in ht.index:
        row("Average Speed", f"{ht['AVG_SPEED']:.2f} mph")
    if "DIST_MILES" in ht.index:
        row("Distance Per Game", f"{ht['DIST_MILES']:.2f} miles")
    if "AVG_SPEED_OFF" in ht.index:
        row("Offensive Speed", f"{ht['AVG_SPEED_OFF']:.2f} mph")
    if "AVG_SPEED_DEF" in ht.index:
        row("Defensive Speed", f"{ht['AVG_SPEED_DEF']:.2f} mph")
    if "DIST_MILES_OFF" in ht.index:
        row("Offensive Distance", f"{ht['DIST_MILES_OFF']:.2f} miles")
    if "DIST_MILES_DEF" in ht.index:
        row("Defensive Distance", f"{ht['DIST_MILES_DEF']:.2f} miles")
    blank()

# ── 6. On/Off Court Impact ──────────────────────────────────────────
h(2, "6. On/Off Court Impact")
if len(por_on) > 0 and len(por_off) > 0:
    h_on = por_on[por_on["VS_PLAYER_NAME"] == PLAYER_NAME]
    h_off = por_off[por_off["VS_PLAYER_NAME"] == PLAYER_NAME]
    if len(h_on) > 0 and len(h_off) > 0:
        on = h_on.iloc[0]
        off = h_off.iloc[0]
        on_net = on.get("NET_RATING", 0)
        off_net = off.get("NET_RATING", 0)
        on_off = on.get("OFF_RATING", 0)
        off_off = off.get("OFF_RATING", 0)
        on_def = on.get("DEF_RATING", 0)
        off_def = off.get("DEF_RATING", 0)
        on_min = on.get("MIN", 0)
        off_min = off.get("MIN", 0)

        tbl("Metric", "On Court", "Off Court", "Swing")
        row("Minutes", f"{on_min:.0f}", f"{off_min:.0f}", "—")
        row("Offensive Rating", f"{on_off:.1f}", f"{off_off:.1f}", f"{on_off - off_off:+.1f}")
        row("Defensive Rating", f"{on_def:.1f}", f"{off_def:.1f}", f"{on_def - off_def:+.1f}")
        row("**Net Rating**", f"**{on_net:+.1f}**", f"**{off_net:+.1f}**", f"**{on_net - off_net:+.1f}**")
        blank()

        net_swing = on_net - off_net
        if net_swing > 3:
            p(f"✅ **Strong positive impact:** The team is {net_swing:.1f} points per 100 possessions better with Hansen on the court.")
        elif net_swing > 0:
            p(f"🔵 **Slightly positive impact:** The team is {net_swing:.1f} points per 100 possessions better with Hansen on court.")
        elif net_swing > -3:
            p(f"🟡 **Neutral impact:** Hansen's on/off swing ({net_swing:+.1f}) is within normal variance.")
        else:
            p(f"⚠️ **Negative swing:** The team is {abs(net_swing):.1f} points per 100 possessions worse with Hansen on court — though this can be influenced by lineup combinations.")
    else:
        p("*On/off data not available for this player.*")
else:
    p("*On/off data not available.*")

# ── 7. Lineup Analysis ──────────────────────────────────────────────
h(2, "7. Lineup Analysis — Hansen's Best & Worst 5-Man Units")
if len(por_lineups) > 0:
    # Filter lineups containing Hansen (listed as "H. Yang" in lineup data)
    hansen_lineups = por_lineups[
        por_lineups["GROUP_NAME"].str.contains("Hansen", case=False) |
        por_lineups["GROUP_NAME"].str.contains("H. Yang", case=False)
    ]
    if len(hansen_lineups) > 0:
        hansen_lineups = hansen_lineups.copy()
        hansen_lineups["MIN_TOTAL"] = hansen_lineups["MIN"].astype(float)
        # Filter to meaningful minutes
        meaningful = hansen_lineups[hansen_lineups["MIN_TOTAL"] >= 5].sort_values("NET_RATING", ascending=False)
        if len(meaningful) == 0:
            meaningful = hansen_lineups.sort_values("NET_RATING", ascending=False)

        h(3, "Best Lineups")
        best = meaningful.head(5)
        tbl("Lineup", "MIN", "OFF RTG", "DEF RTG", "NET RTG")
        for _, l in best.iterrows():
            names = " - ".join([n.split()[-1] for n in l["GROUP_NAME"].split(" - ")])
            row(names, f"{l['MIN']:.0f}", f"{l['OFF_RATING']:.1f}", f"{l['DEF_RATING']:.1f}",
                f"**{l['NET_RATING']:+.1f}**")
        blank()

        h(3, "Worst Lineups")
        worst = meaningful.tail(5)
        tbl("Lineup", "MIN", "OFF RTG", "DEF RTG", "NET RTG")
        for _, l in worst.iterrows():
            names = " - ".join([n.split()[-1] for n in l["GROUP_NAME"].split(" - ")])
            row(names, f"{l['MIN']:.0f}", f"{l['OFF_RATING']:.1f}", f"{l['DEF_RATING']:.1f}",
                f"**{l['NET_RATING']:+.1f}**")
        blank()

        # Partner analysis — which players appear most in Hansen's best lineups?
        h(3, "Best Partners")
        partner_data = {}
        for _, l in meaningful.iterrows():
            players = [n.strip() for n in l["GROUP_NAME"].split(" - ")]
            for pl in players:
                if "Hansen" not in pl and "Yang" not in pl:
                    if pl not in partner_data:
                        partner_data[pl] = {"min": 0, "net_sum": 0, "count": 0}
                    partner_data[pl]["min"] += float(l["MIN"])
                    partner_data[pl]["net_sum"] += float(l["NET_RATING"]) * float(l["MIN"])
                    partner_data[pl]["count"] += 1

        partners = []
        for name, d in partner_data.items():
            if d["min"] > 0:
                partners.append({"name": name, "min": d["min"], "w_net": d["net_sum"]/d["min"], "lineups": d["count"]})
        partners.sort(key=lambda x: x["w_net"], reverse=True)

        tbl("Partner", "Shared MIN", "# Lineups", "Weighted Net RTG")
        for pp in partners[:8]:
            row(pp["name"], f"{pp['min']:.0f}", pp["lineups"], f"{pp['w_net']:+.1f}")
        blank()
    else:
        p("*No lineup data featuring Hansen found.*")
else:
    p("*Lineup data not available.*")

# ── 8. Comparison with Portland Teammates ─────────────────────────
h(2, "8. Comparison with Teammates")
if hansen_base is not None and len(por_base) > 0:
    # Show where Hansen ranks among teammates
    por_ranked = por_base.sort_values("PTS", ascending=False).reset_index(drop=True)
    pts_rank = por_ranked[por_ranked["PLAYER_NAME"] == PLAYER_NAME].index[0] + 1

    p(f"Hansen ranks **#{pts_rank}** in PPG among {len(por_ranked)} Portland players this season.")
    blank()

    # Comparison table — top 8 + Hansen
    top8 = por_ranked.head(8)
    if PLAYER_NAME not in top8["PLAYER_NAME"].values:
        top8 = pd.concat([top8, por_ranked[por_ranked["PLAYER_NAME"] == PLAYER_NAME]])

    tbl("Player", "GP", "MPG", "PPG", "RPG", "APG", "FG%", "3PT%", "+/-")
    for _, r in top8.iterrows():
        name = r["PLAYER_NAME"]
        highlight = " ⭐" if name == PLAYER_NAME else ""
        row(
            f"{name}{highlight}", int(r["GP"]), f"{r['MIN']:.1f}",
            f"{r['PTS']:.1f}", f"{r['REB']:.1f}", f"{r['AST']:.1f}",
            f"{r['FG_PCT']*100:.1f}%", f"{r['FG3_PCT']*100:.1f}%",
            f"{r['PLUS_MINUS']:+.1f}"
        )
    blank()

    # Advanced comparison
    if len(por_adv) > 0:
        h(3, "Advanced Metrics Comparison")
        adv_cols = ["PLAYER_NAME", "GP", "MIN", "OFF_RATING", "DEF_RATING", "NET_RATING", "TS_PCT", "USG_PCT", "PIE"]
        avail = [c for c in adv_cols if c in por_adv.columns]
        adv_top = por_adv[por_adv["PLAYER_NAME"].isin(top8["PLAYER_NAME"].values)][avail].sort_values("NET_RATING", ascending=False)
        if PLAYER_NAME not in adv_top["PLAYER_NAME"].values:
            hansen_row = por_adv[por_adv["PLAYER_NAME"] == PLAYER_NAME][avail]
            adv_top = pd.concat([adv_top, hansen_row])

        tbl("Player", "OFF RTG", "DEF RTG", "NET RTG", "TS%", "USG%", "PIE")
        for _, r in adv_top.iterrows():
            name = r["PLAYER_NAME"]
            highlight = " ⭐" if name == PLAYER_NAME else ""
            ts = r.get("TS_PCT", 0) * 100
            usg = r.get("USG_PCT", 0) * 100
            pie = r.get("PIE", 0) * 100
            row(
                f"{name}{highlight}",
                f"{r.get('OFF_RATING', 0):.1f}", f"{r.get('DEF_RATING', 0):.1f}",
                f"{r.get('NET_RATING', 0):+.1f}",
                f"{ts:.1f}%", f"{usg:.1f}%", f"{pie:.1f}%"
            )
        blank()

# ── 9. ML-Driven Insights ───────────────────────────────────────────
h(2, "9. Machine Learning Insights")

# 9.1 Scoring Consistency
h(3, "9.1 Scoring Consistency Analysis")
if len(hansen_gl) > 0:
    pts = hansen_gl["PTS"].astype(float)
    mean_pts = pts.mean()
    std_pts = pts.std()
    cv = std_pts / mean_pts * 100 if mean_pts > 0 else 0

    # Autocorrelation (streakiness)
    if len(pts) > 5:
        centered = pts - mean_pts
        autocorr_vals = []
        for lag in [1, 2, 3]:
            if len(pts) > lag:
                corr = centered.iloc[lag:].reset_index(drop=True).corr(centered.iloc[:-lag].reset_index(drop=True))
                autocorr_vals.append(corr)
            else:
                autocorr_vals.append(0)
    else:
        autocorr_vals = [0, 0, 0]

    tbl("Metric", "Value", "Interpretation")
    row("Mean PPG", f"{mean_pts:.1f}", "—")
    row("Std Dev", f"{std_pts:.1f}", "Higher = more variable")
    row("CV%", f"{cv:.1f}%", "<50% = consistent, >70% = volatile")
    row("Lag-1 Autocorr", f"{autocorr_vals[0]:.3f}", ">0.3 = streaky, <-0.1 = mean-reverting")
    row("Lag-2 Autocorr", f"{autocorr_vals[1]:.3f}", "—")
    blank()

    if cv > 70:
        p("⚠️ **High variance scorer** — output varies significantly game to game.")
    elif cv > 50:
        p("🟡 **Moderate variance** — some fluctuation, typical for a role player with limited minutes.")
    else:
        p("✅ **Consistent producer** — steady output relative to his role.")

    if autocorr_vals[0] > 0.3:
        p("🔥 **Streaky tendencies** — good performances tend to cluster (positive autocorrelation).")
    elif autocorr_vals[0] < -0.15:
        p("🔄 **Mean-reverting** — tends to bounce back after poor games.")
    else:
        p("➡️ **No significant streakiness** — performance appears game-independent.")
    blank()

# 9.2 Fatigue Analysis
h(3, "9.2 Fatigue & Workload Analysis")
if len(hansen_gl) > 5:
    mins = hansen_gl["MIN"].astype(float).values
    pts_arr = hansen_gl["PTS"].astype(float).values

    # Rolling workload (avg of prev 3 games minutes)
    workload = []
    rest_proxy = []
    y_pts = []
    for i in range(3, len(pts_arr)):
        avg_prev = np.mean(mins[i-3:i])
        workload.append(avg_prev)
        y_pts.append(pts_arr[i])

    if len(workload) > 5:
        from sklearn.linear_model import LinearRegression
        X_fat = np.array(workload).reshape(-1, 1)
        y_fat = np.array(y_pts)
        lr = LinearRegression().fit(X_fat, y_fat)
        coef = lr.coef_[0]
        r2 = lr.score(X_fat, y_fat)

        tbl("Metric", "Value")
        row("Workload Coefficient", f"{coef:+.3f} pts per avg prior minute")
        row("Model R²", f"{r2:.3f}")
        row("Avg Prior-3-Game Minutes", f"{np.mean(workload):.1f}")
        blank()

        if coef < -0.15:
            p(f"⚠️ **Fatigue-sensitive:** Each additional average minute in the prior 3 games costs ~{abs(coef):.2f} points.")
        elif coef > 0.1:
            p(f"💪 **Builds momentum with minutes:** More playing time correlates with better performance ({coef:+.2f} pts/min).")
        else:
            p(f"➡️ **Minimal fatigue effect:** Workload has negligible impact on scoring ({coef:+.3f} pts/min).")
        blank()

# 9.3 Hot/Cold Zone Analysis from game logs
h(3, "9.3 Hot/Cold Streaks")
if len(hansen_gl) > 5:
    pts = hansen_gl["PTS"].astype(float)
    avg = pts.mean()

    # Find longest hot and cold streaks
    hot_streak = cold_streak = 0
    max_hot = max_cold = 0
    for v in pts:
        if v >= avg:
            hot_streak += 1
            cold_streak = 0
            max_hot = max(max_hot, hot_streak)
        else:
            cold_streak += 1
            hot_streak = 0
            max_cold = max(max_cold, cold_streak)

    # 5-game rolling average
    rolling5 = pts.rolling(5).mean()
    peak_5g = rolling5.max()
    trough_5g = rolling5.min()

    tbl("Metric", "Value")
    row("Longest Above-Average Streak", f"{max_hot} games")
    row("Longest Below-Average Streak", f"{max_cold} games")
    row("Peak 5-Game Rolling Avg", f"{peak_5g:.1f} PPG")
    row("Trough 5-Game Rolling Avg", f"{trough_5g:.1f} PPG")
    row("Peak-to-Trough Swing", f"{peak_5g - trough_5g:.1f} PPG")
    blank()

    # Wald-Wolfowitz runs test for scoring
    above = (pts >= avg).astype(int).values
    runs = 1 + np.sum(np.diff(above) != 0)
    n1 = np.sum(above == 1)
    n0 = np.sum(above == 0)
    n = n1 + n0
    if n1 > 0 and n0 > 0:
        mu_r = 1 + 2*n1*n0/n
        var_r = 2*n1*n0*(2*n1*n0 - n) / (n**2 * (n - 1))
        if var_r > 0:
            z_r = (runs - mu_r) / np.sqrt(var_r)
            p_r = 2 * sp_stats.norm.sf(abs(z_r))
            row_label = "Streaky" if z_r < -1.96 else "Clustered" if z_r > 1.96 else "Random"
            p(f"**Runs Test:** z={z_r:.2f}, p={p_r:.3f} → {row_label}")
            if z_r < -1.96:
                p("📊 Scoring patterns show statistically significant streakiness — consider riding hot hands.")
            else:
                p("📊 Scoring patterns are statistically random — no exploitable hot/cold patterns.")
    blank()

# ── 10. Shot Location Comparison ─────────────────────────────────────
# Shot location endpoint has complex nested structure — skip if unavailable
# Hansen's shooting profile is already covered in section 4 from the shooting splits endpoint

# ── 11. Role in Portland's System ────────────────────────────────────
h(2, "11. Role in Portland's System")

# Win/Loss splits from hansen_splits
if hansen_splits:
    for rs in hansen_splits["resultSets"]:
        if rs["name"] == "LocationPlayerDashboard":
            loc_df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            if len(loc_df) > 0:
                h(3, "Home vs Away")
                tbl("Location", "GP", "MPG", "PPG", "RPG", "APG", "FG%")
                for _, r in loc_df.iterrows():
                    row(r["GROUP_VALUE"], int(r["GP"]), f"{r['MIN']:.1f}",
                        f"{r['PTS']:.1f}", f"{r['REB']:.1f}", f"{r['AST']:.1f}",
                        f"{r['FG_PCT']*100:.1f}%")
                blank()

        if rs["name"] == "WinsLossesPlayerDashboard":
            wl_df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            if len(wl_df) > 0:
                h(3, "Wins vs Losses")
                tbl("Result", "GP", "MPG", "PPG", "RPG", "APG", "FG%", "+/-")
                for _, r in wl_df.iterrows():
                    row(r["GROUP_VALUE"], int(r["GP"]), f"{r['MIN']:.1f}",
                        f"{r['PTS']:.1f}", f"{r['REB']:.1f}", f"{r['AST']:.1f}",
                        f"{r['FG_PCT']*100:.1f}%", f"{r['PLUS_MINUS']:+.1f}")
                blank()

        if rs["name"] == "StartingPosition":
            start_df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            if len(start_df) > 0:
                h(3, "Starter vs Bench")
                tbl("Role", "GP", "MPG", "PPG", "RPG", "APG", "FG%")
                for _, r in start_df.iterrows():
                    row(r["GROUP_VALUE"], int(r["GP"]), f"{r['MIN']:.1f}",
                        f"{r['PTS']:.1f}", f"{r['REB']:.1f}", f"{r['AST']:.1f}",
                        f"{r['FG_PCT']*100:.1f}%")
                blank()

# ── 12. Composite Value Assessment ──────────────────────────────────
h(2, "12. Composite Evaluation & Scouting Summary")
p("### Strengths")
strengths = []
concerns = []

if hansen_base is not None:
    # Shooting
    if fg3_pct > 37: strengths.append(f"Solid 3-point shooting ({fg3_pct:.1f}%)")
    elif fg3_pct > 33: strengths.append(f"Adequate 3-point shooting ({fg3_pct:.1f}%)")
    else: concerns.append(f"Below-average 3-point shooting ({fg3_pct:.1f}%)")

    if fg_pct > 47: strengths.append(f"Efficient overall shooting ({fg_pct:.1f}% FG)")
    elif fg_pct < 42: concerns.append(f"Low overall efficiency ({fg_pct:.1f}% FG)")

    # Per-minute production
    if mpg > 0:
        pts_per_min = ppg / mpg
        if pts_per_min > 0.4: strengths.append(f"High per-minute scoring rate ({pts_per_min:.2f} pts/min)")
        elif pts_per_min > 0.3: strengths.append(f"Solid per-minute scoring ({pts_per_min:.2f} pts/min)")

        reb_per_min = rpg / mpg
        if reb_per_min > 0.2: strengths.append(f"Active on the glass ({reb_per_min:.2f} reb/min)")

        ast_per_min = apg / mpg
        if ast_per_min > 0.15: strengths.append(f"Good playmaking rate ({ast_per_min:.2f} ast/min)")

        stl_per_min = spg / mpg
        if stl_per_min > 0.05: strengths.append(f"Disruptive defender ({stl_per_min:.2f} stl/min)")

    if tpg < 1.0: strengths.append(f"Takes care of the ball ({tpg:.1f} TOV/game)")
    elif tpg > 2.5: concerns.append(f"Turnover-prone ({tpg:.1f} TOV/game)")

if hansen_adv is not None:
    if ts > 58: strengths.append(f"Above-average true shooting ({ts:.1f}%)")
    elif ts < 50: concerns.append(f"Below-average true shooting ({ts:.1f}%)")

    if net_rtg > 3: strengths.append(f"Positive net rating ({net_rtg:+.1f})")
    elif net_rtg < -5: concerns.append(f"Negative net rating ({net_rtg:+.1f})")

for s in strengths:
    w(f"- ✅ {s}")
blank()

p("### Areas for Growth")
if not concerns:
    concerns.append("Limited minutes make it hard to identify clear weaknesses")
for c in concerns:
    w(f"- ⚠️ {c}")
blank()

# Overall assessment
h(3, "Overall Assessment")
if hansen_base is not None:
    total_mins = gp * mpg
    p(f"Yang Hansen has played **{gp} games** averaging **{mpg:.1f} minutes** in his rookie season with Portland "
      f"({total_mins:.0f} total minutes). "
      f"His per-game averages of **{ppg:.1f}/{rpg:.1f}/{apg:.1f}** (PTS/REB/AST) reflect his current "
      f"limited role, but the per-minute and efficiency metrics tell a fuller story of his potential impact.")
    blank()

    if mpg < 10:
        p("📋 **Development Phase:** Hansen is in a development role with limited NBA minutes. "
          "The small sample makes definitive conclusions premature, but the efficiency metrics "
          "and per-minute production provide early indicators of his trajectory.")
    elif mpg < 20:
        p("📋 **Rotation Player:** Hansen has earned a consistent rotation spot. "
          "His contributions should be evaluated in the context of his defined role.")
    else:
        p("📋 **Key Contributor:** Hansen has established himself as a significant part of Portland's rotation.")

# ══════════════════════════════════════════════════════════════════════
# Write report
# ══════════════════════════════════════════════════════════════════════
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines))
n_lines = len(lines)
n_chars = sum(len(l) for l in lines)
print(f"\n✅ Report written to {REPORT}")
print(f"   {n_lines} lines, {n_chars:,} characters")
