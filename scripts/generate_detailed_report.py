#!/usr/bin/env python3
"""Generate a comprehensive Warriors detailed report from cached NBA data."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CACHE = Path(__file__).resolve().parent.parent / "web" / "cache"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def load_resultsets(name):
    """Load cached JSON and return dict of {resultset_name: DataFrame}."""
    path = CACHE / f"{name}.json"
    data = json.loads(path.read_text())
    rs_data = data.get("resultSets", data)
    if isinstance(rs_data, list):
        return {rs["name"]: pd.DataFrame(rs["rowSet"], columns=rs["headers"]) for rs in rs_data}
    elif isinstance(rs_data, dict):
        # Single resultSet (e.g. shot_locations)
        return {"main": rs_data}
    return {}


def load_first(name):
    """Load and return just the first DataFrame."""
    d = load_resultsets(name)
    return list(d.values())[0] if d else pd.DataFrame()


def load_shot_locations():
    path = CACHE / "shot_locations.json"
    data = json.loads(path.read_text())
    rs = data["resultSets"]
    info_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "NICKNAME"]
    zones = ["<5ft", "5-9ft", "10-14ft", "15-19ft", "20-24ft", "25-29ft", "30-34ft", "35-39ft", "40+ft"]
    all_cols = list(info_cols)
    for z in zones:
        all_cols.extend([f"{z}_FGM", f"{z}_FGA", f"{z}_FG_PCT"])
    return pd.DataFrame(rs["rowSet"], columns=all_cols), zones


def load_shooting_splits(player_key):
    return load_resultsets(f"shooting_splits_{player_key}")


# ═══════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════

standings = load_first("standings")
gamelog = load_first("gamelog")
splits = load_resultsets("splits")
base = load_first("player_base")
adv = load_first("player_adv")
clutch = load_first("clutch")

tracking_sets = load_resultsets("tracking_speed")
tracking = list(tracking_sets.values())[0]

hustle_sets = load_resultsets("hustle")
hustle = list(hustle_sets.values())[0]

team_shoot = load_resultsets("team_shooting")
shot_loc, ZONES_5FT = load_shot_locations()

PLAYER_KEYS = ["curry", "butler", "green", "kuminga", "hield"]
player_shoots = {k: load_shooting_splits(k) for k in PLAYER_KEYS}

# ═══════════════════════════════════════════════════════════════
# REPORT BUILDER
# ═══════════════════════════════════════════════════════════════

lines = []


def h(level, text):
    lines.append(f'{"#" * level} {text}')
    lines.append("")


def p(text):
    lines.append(text)


def blank():
    lines.append("")


def tbl(*headers):
    p("| " + " | ".join(headers) + " |")
    p("|" + "|".join(["---"] * len(headers)) + "|")


def row(*vals):
    p("| " + " | ".join(str(v) for v in vals) + " |")


def pct_fmt(v, mult=100):
    return f"{v * mult:.1f}%"


def pm_fmt(v):
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"


now = datetime.now().strftime("%B %d, %Y")

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
h(1, "Golden State Warriors — Comprehensive Season Report 2025-26")
p(f"*Generated: {now}*")
blank()
p("---")
blank()

# ═══════════════════════════════════════════════════════════════
# 1. SEASON OVERVIEW
# ═══════════════════════════════════════════════════════════════
h(2, "1. Season Overview")

west = standings[standings["Conference"] == "West"].sort_values("WinPCT", ascending=False).reset_index(drop=True)
gsw = standings[standings["TeamCity"] == "Golden State"].iloc[0]
w, l = int(gsw["WINS"]), int(gsw["LOSSES"])
wpct = gsw["WinPCT"]
ppg = gsw["PointsPG"]
opp_ppg = gsw["OppPointsPG"]
diff = ppg - opp_ppg
seed = int(gsw["PlayoffRank"])
gb = gsw["ConferenceGamesBack"]

tbl("Metric", "Value")
row("**Record**", f"{w}-{l} ({pct_fmt(wpct)})")
row("**Conference Seed**", f"#{seed} West")
row("**Games Back**", gb)
row("**Points Per Game**", f"{ppg:.1f}")
row("**Opponent PPG**", f"{opp_ppg:.1f}")
row("**Point Differential**", pm_fmt(diff))
blank()

recent10 = gamelog.head(10)
recent10_w = int((recent10["WL"] == "W").sum())
last15 = gamelog.head(15)
last15_w = int((last15["WL"] == "W").sum())
p(f"**Recent Form:** Last 10 → {recent10_w}-{10 - recent10_w} | Last 15 → {last15_w}-{15 - last15_w}")
blank()

# Current streak
streak_type = gamelog.iloc[0]["WL"]
streak_count = 0
for _, g in gamelog.iterrows():
    if g["WL"] == streak_type:
        streak_count += 1
    else:
        break
p(f'**Current Streak:** {streak_count} {"Win" if streak_type == "W" else "Loss"}{"es" if streak_count != 1 and streak_type == "L" else "s" if streak_count != 1 else ""}')
blank()

# ═══════════════════════════════════════════════════════════════
# 2. WESTERN CONFERENCE STANDINGS
# ═══════════════════════════════════════════════════════════════
h(2, "2. Western Conference Standings")

tbl("#", "Team", "W", "L", "Win%", "GB", "PPG", "Opp PPG", "Diff")
for i, (_, t) in enumerate(west.iterrows()):
    marker = " **←**" if t["TeamName"] == "Warriors" else ""
    td = t["PointsPG"] - t["OppPointsPG"]
    row(
        i + 1,
        f'{t["TeamCity"]} {t["TeamName"]}{marker}',
        int(t["WINS"]), int(t["LOSSES"]),
        pct_fmt(t["WinPCT"]),
        t["ConferenceGamesBack"],
        f'{t["PointsPG"]:.1f}',
        f'{t["OppPointsPG"]:.1f}',
        pm_fmt(td),
    )
blank()

# ═══════════════════════════════════════════════════════════════
# 3. PLAYER STATISTICS
# ═══════════════════════════════════════════════════════════════
h(2, "3. Player Statistics")

# 3.1 Core stats
h(3, "3.1 Core Stats (Per Game)")
base_sorted = base.sort_values("PTS", ascending=False)
tbl("Player", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%", "+/-")
for _, r in base_sorted.iterrows():
    if r["GP"] >= 5:
        pm = r.get("PLUS_MINUS", 0)
        row(
            r["PLAYER_NAME"], int(r["GP"]), f'{r["MIN"]:.1f}',
            f'**{r["PTS"]:.1f}**', f'{r["REB"]:.1f}', f'{r["AST"]:.1f}',
            f'{r["STL"]:.1f}', f'{r["BLK"]:.1f}', f'{r["TOV"]:.1f}',
            pct_fmt(r["FG_PCT"]), pct_fmt(r["FG3_PCT"]), pct_fmt(r["FT_PCT"]),
            pm_fmt(pm),
        )
blank()

# 3.2 Scoring distribution
h(3, "3.2 Scoring Distribution")
top_scorers = base_sorted[base_sorted["GP"] >= 10].head(8)
total_pts = top_scorers["PTS"].sum()
tbl("Player", "PPG", "Share", "")
for _, r in top_scorers.iterrows():
    share = r["PTS"] / total_pts * 100 if total_pts > 0 else 0
    bar = "█" * int(share / 2) + "░" * max(0, 25 - int(share / 2))
    row(r["PLAYER_NAME"], f'{r["PTS"]:.1f}', f"{share:.1f}%", bar)
blank()

# 3.3 Advanced metrics
h(3, "3.3 Advanced Metrics")
adv_sorted = adv[adv["GP"] >= 10].sort_values("NET_RATING", ascending=False)
tbl("Player", "GP", "OFF RTG", "DEF RTG", "NET RTG", "TS%", "USG%", "PIE", "AST/TO", "EFG%", "Tier")
for _, r in adv_sorted.iterrows():
    net = r["NET_RATING"]
    if net >= 8:
        tier = "🟢 Elite"
    elif net >= 3:
        tier = "🔵 Positive"
    elif net >= -3:
        tier = "🟡 Neutral"
    elif net >= -8:
        tier = "🟠 Negative"
    else:
        tier = "🔴 Struggling"
    row(
        r["PLAYER_NAME"], int(r["GP"]),
        f'{r["OFF_RATING"]:.1f}', f'{r["DEF_RATING"]:.1f}', f'**{net:.1f}**',
        pct_fmt(r.get("TS_PCT", 0)), pct_fmt(r.get("USG_PCT", 0)),
        pct_fmt(r.get("PIE", 0)), f'{r.get("AST_TO", 0):.2f}',
        pct_fmt(r.get("EFG_PCT", 0)), tier,
    )
blank()

# ═══════════════════════════════════════════════════════════════
# 4. SHOOTING ANALYSIS
# ═══════════════════════════════════════════════════════════════
h(2, "4. Shooting Analysis")

# 4.1 Team by area
h(3, "4.1 Team Shot Distribution by Area")
if "ShotAreaTeamDashboard" in team_shoot:
    sa = team_shoot["ShotAreaTeamDashboard"]
    total_fga = sa["FGA"].sum()
    gp_for_area = w + l
    tbl("Zone", "FGM/G", "FGA/G", "FG%", "Volume")
    for _, r in sa.iterrows():
        if r["FGA"] > 0:
            share = r["FGA"] / total_fga * 100
            row(r["GROUP_VALUE"], f'{r["FGM"] / gp_for_area:.1f}', f'{r["FGA"] / gp_for_area:.1f}', f'**{pct_fmt(r["FG_PCT"])}**', f"{share:.1f}%")
    blank()

# 4.2 Team by distance
h(3, "4.2 Team Shot Distribution by Distance")
if "Shot5FTTeamDashboard" in team_shoot:
    s5 = team_shoot["Shot5FTTeamDashboard"]
    gp_for_dist = w + l
    tbl("Distance", "FGM/G", "FGA/G", "FG%")
    for _, r in s5.iterrows():
        if r["FGA"] > 0:
            row(r["GROUP_VALUE"], f'{r["FGM"] / gp_for_dist:.1f}', f'{r["FGA"] / gp_for_dist:.1f}', pct_fmt(r["FG_PCT"]))
    blank()

# 4.3 Player shot zones
h(3, "4.3 Player Shooting by Distance Zone")
p("*FG% (FGA/game) — Minimum 3 FGA/game*")
blank()
tbl("Player", "<5ft", "5-9ft", "10-14ft", "15-19ft", "20-24ft", "25-29ft", "30+ft")
for _, r in shot_loc.iterrows():
    total = sum(r[f"{z}_FGA"] for z in ZONES_5FT)
    if total >= 3:
        def fmt_zone(z):
            fga = r[f"{z}_FGA"]
            pct = r[f"{z}_FG_PCT"]
            if fga < 0.1:
                return "-"
            return f"{pct * 100:.0f}% ({fga:.1f})"

        deep_fga = r["30-34ft_FGA"] + r["35-39ft_FGA"] + r.get("40+ft_FGA", 0)
        deep_fgm = r["30-34ft_FGM"] + r["35-39ft_FGM"] + r.get("40+ft_FGM", 0)
        deep = f"{deep_fgm / deep_fga * 100:.0f}% ({deep_fga:.1f})" if deep_fga >= 0.1 else "-"
        row(
            r["PLAYER_NAME"],
            fmt_zone("<5ft"), fmt_zone("5-9ft"), fmt_zone("10-14ft"),
            fmt_zone("15-19ft"), fmt_zone("20-24ft"), fmt_zone("25-29ft"),
            deep,
        )
blank()

# 4.4 Key player shot area breakdowns
h(3, "4.4 Key Player Shot Area Breakdown")
PLAYER_LABELS = {
    "curry": "Stephen Curry", "butler": "Jimmy Butler III",
    "green": "Draymond Green", "hield": "Buddy Hield", "kuminga": "Jonathan Kuminga",
}
for pkey in PLAYER_KEYS:
    label = PLAYER_LABELS[pkey]
    sh = player_shoots[pkey]

    h(4, label)

    # Shot area
    if "ShotAreaPlayerDashboard" in sh:
        area = sh["ShotAreaPlayerDashboard"]
        total_fga = area["FGA"].sum()
        tbl("Zone", "FGM", "FGA", "FG%", "% of Shots")
        for _, r in area.iterrows():
            if r["FGA"] > 0:
                share = r["FGA"] / total_fga * 100
                row(r["GROUP_VALUE"], int(r["FGM"]), int(r["FGA"]),
                    f'**{pct_fmt(r["FG_PCT"])}**', f"{share:.1f}%")
        blank()

    # Shot type summary
    if "ShotTypeSummaryPlayerDashboard" in sh:
        p("**Shot Type Breakdown:**")
        blank()
        tbl("Shot Type", "FGM", "FGA", "FG%")
        for _, r in sh["ShotTypeSummaryPlayerDashboard"].iterrows():
            if r["FGA"] >= 5:
                row(r["GROUP_VALUE"], int(r["FGM"]), int(r["FGA"]), pct_fmt(r["FG_PCT"]))
        blank()

    # Assisted
    if "AssitedShotPlayerDashboard" in sh:
        ast_df = sh["AssitedShotPlayerDashboard"]
        total_fgm = ast_df["FGM"].sum()
        for _, r in ast_df.iterrows():
            fgm = int(r["FGM"])
            pctv = fgm / total_fgm * 100 if total_fgm > 0 else 0
            p(f'- **{r["GROUP_VALUE"]}** field goals: {fgm} ({pctv:.0f}%)')
        blank()

# ═══════════════════════════════════════════════════════════════
# 5. PLAYER TRACKING
# ═══════════════════════════════════════════════════════════════
h(2, "5. Player Tracking — Speed & Distance")
tracking_sorted = tracking.sort_values("MIN", ascending=False)
tbl("Player", "GP", "MIN", "Miles/G", "Off Miles", "Def Miles", "Speed", "Off Spd", "Def Spd")
for _, r in tracking_sorted.iterrows():
    if r["GP"] >= 10 and r["MIN"] >= 10:
        row(
            r["PLAYER_NAME"], int(r["GP"]), f'{r["MIN"]:.1f}',
            f'**{r["DIST_MILES"]:.2f}**',
            f'{r["DIST_MILES_OFF"]:.2f}', f'{r["DIST_MILES_DEF"]:.2f}',
            f'{r["AVG_SPEED"]:.2f}', f'{r["AVG_SPEED_OFF"]:.2f}', f'{r["AVG_SPEED_DEF"]:.2f}',
        )
blank()

track_f = tracking[tracking["GP"] >= 10]
fastest = track_f.sort_values("AVG_SPEED", ascending=False).iloc[0]
most_miles = track_f.sort_values("DIST_MILES", ascending=False).iloc[0]
fastest_off = track_f.sort_values("AVG_SPEED_OFF", ascending=False).iloc[0]
p(f'- **Fastest Player:** {fastest["PLAYER_NAME"]} ({fastest["AVG_SPEED"]:.2f} mph average)')
p(f'- **Most Distance/Game:** {most_miles["PLAYER_NAME"]} ({most_miles["DIST_MILES"]:.2f} miles/game)')
p(f'- **Fastest on Offense:** {fastest_off["PLAYER_NAME"]} ({fastest_off["AVG_SPEED_OFF"]:.2f} mph)')
blank()

# ═══════════════════════════════════════════════════════════════
# 6. HUSTLE STATS
# ═══════════════════════════════════════════════════════════════
h(2, "6. Hustle & Effort Metrics")
hustle_f = hustle[(hustle["G"] >= 10) & (hustle["MIN"] >= 10)].sort_values("MIN", ascending=False)
tbl("Player", "Contested", "2PT Cont", "3PT Cont", "Deflections", "Screens", "Loose Balls", "Box Outs", "Charges")
for _, r in hustle_f.iterrows():
    row(
        r["PLAYER_NAME"],
        f'**{r["CONTESTED_SHOTS"]:.1f}**',
        f'{r["CONTESTED_SHOTS_2PT"]:.1f}', f'{r["CONTESTED_SHOTS_3PT"]:.1f}',
        f'{r["DEFLECTIONS"]:.1f}', f'{r["SCREEN_ASSISTS"]:.1f}',
        f'{r["LOOSE_BALLS_RECOVERED"]:.1f}', f'{r["BOX_OUTS"]:.1f}',
        f'{r["CHARGES_DRAWN"]:.2f}',
    )
blank()

top_contest = hustle_f.sort_values("CONTESTED_SHOTS", ascending=False).iloc[0]
top_deflect = hustle_f.sort_values("DEFLECTIONS", ascending=False).iloc[0]
top_screen = hustle_f.sort_values("SCREEN_ASSISTS", ascending=False).iloc[0]
top_loose = hustle_f.sort_values("LOOSE_BALLS_RECOVERED", ascending=False).iloc[0]
top_box = hustle_f.sort_values("BOX_OUTS", ascending=False).iloc[0]
p("**Hustle Leaders:**")
p(f'- 🛡️ Contested Shots: **{top_contest["PLAYER_NAME"]}** ({top_contest["CONTESTED_SHOTS"]:.1f}/game)')
p(f'- 🖐️ Deflections: **{top_deflect["PLAYER_NAME"]}** ({top_deflect["DEFLECTIONS"]:.1f}/game)')
p(f'- 📐 Screen Assists: **{top_screen["PLAYER_NAME"]}** ({top_screen["SCREEN_ASSISTS"]:.1f}/game)')
p(f'- 🤾 Loose Balls: **{top_loose["PLAYER_NAME"]}** ({top_loose["LOOSE_BALLS_RECOVERED"]:.1f}/game)')
p(f'- 📦 Box Outs: **{top_box["PLAYER_NAME"]}** ({top_box["BOX_OUTS"]:.1f}/game)')
blank()

# ═══════════════════════════════════════════════════════════════
# 7. CLUTCH PERFORMANCE
# ═══════════════════════════════════════════════════════════════
h(2, "7. Clutch Performance")
p("*Last 5 minutes of 4th quarter / OT, score within 5 points*")
blank()
clutch_sorted = clutch[clutch["GP"] >= 5].sort_values("PTS", ascending=False)
tbl("Player", "GP", "MIN", "PTS", "FG%", "3P%", "FT%", "AST", "REB", "TOV", "+/-")
for _, r in clutch_sorted.iterrows():
    pm = r.get("PLUS_MINUS", 0)
    row(
        r["PLAYER_NAME"], int(r["GP"]), f'{r["MIN"]:.1f}',
        f'**{r["PTS"]:.1f}**',
        pct_fmt(r["FG_PCT"]), pct_fmt(r["FG3_PCT"]), pct_fmt(r["FT_PCT"]),
        f'{r["AST"]:.1f}', f'{r["REB"]:.1f}', f'{r["TOV"]:.1f}',
        pm_fmt(pm),
    )
blank()

# Clutch insights
if len(clutch_sorted) >= 2:
    best_clutch = clutch_sorted.iloc[0]
    p(f'**Top Clutch Performer:** {best_clutch["PLAYER_NAME"]} with {best_clutch["PTS"]:.1f} PPG in clutch situations')
    # Best clutch shooter (min 5 GP, at least some attempts)
    clutch_eff = clutch_sorted[clutch_sorted["FG_PCT"] > 0].sort_values("FG_PCT", ascending=False)
    if len(clutch_eff) > 0:
        best_eff = clutch_eff.iloc[0]
        p(f'**Most Efficient Clutch Shooter:** {best_eff["PLAYER_NAME"]} at {pct_fmt(best_eff["FG_PCT"])} FG%')
    blank()

# ═══════════════════════════════════════════════════════════════
# 8. TEAM SPLITS
# ═══════════════════════════════════════════════════════════════
h(2, "8. Team Performance Splits")

# 8.1 Home vs Away
h(3, "8.1 Home vs Away")
loc_split = splits.get("LocationTeamDashboard")
if loc_split is not None and len(loc_split) > 0:
    tbl("Location", "GP", "W", "L", "Win%", "PTS", "REB", "AST", "FG%", "3P%", "+/-")
    for _, r in loc_split.iterrows():
        pm = r.get("PLUS_MINUS", 0)
        row(
            f'**{r["GROUP_VALUE"]}**', int(r["GP"]), int(r["W"]), int(r["L"]),
            pct_fmt(r["W_PCT"]), f'{r["PTS"]:.1f}', f'{r["REB"]:.1f}', f'{r["AST"]:.1f}',
            pct_fmt(r["FG_PCT"]), pct_fmt(r["FG3_PCT"]), pm_fmt(pm),
        )
    blank()

# 8.2 Wins vs Losses
h(3, "8.2 Wins vs Losses Profile")
out_split = splits.get("WinsLossesTeamDashboard")
if out_split is not None and len(out_split) > 0:
    tbl("Outcome", "GP", "PTS", "REB", "AST", "FG%", "3P%", "TOV", "STL", "+/-")
    for _, r in out_split.iterrows():
        pm = r.get("PLUS_MINUS", 0)
        row(
            f'**{r["GROUP_VALUE"]}**', int(r["GP"]),
            f'{r["PTS"]:.1f}', f'{r["REB"]:.1f}', f'{r["AST"]:.1f}',
            pct_fmt(r["FG_PCT"]), pct_fmt(r["FG3_PCT"]),
            f'{r["TOV"]:.1f}', f'{r["STL"]:.1f}', pm_fmt(pm),
        )
    blank()

# 8.3 Monthly
h(3, "8.3 Monthly Breakdown")
mo_split = splits.get("MonthTeamDashboard")
if mo_split is not None and len(mo_split) > 0:
    tbl("Month", "GP", "W", "L", "Win%", "PTS", "FG%", "3P%", "+/-")
    for _, r in mo_split.iterrows():
        pm = r.get("PLUS_MINUS", 0)
        row(
            r["GROUP_VALUE"], int(r["GP"]), int(r["W"]), int(r["L"]),
            pct_fmt(r["W_PCT"]), f'{r["PTS"]:.1f}',
            pct_fmt(r["FG_PCT"]), pct_fmt(r["FG3_PCT"]), pm_fmt(pm),
        )
    blank()

# 8.4 Days Rest
h(3, "8.4 Performance by Days Rest")
rest_split = splits.get("DaysRestTeamDashboard")
if rest_split is not None and len(rest_split) > 0:
    tbl("Rest Days", "GP", "W", "L", "Win%", "PTS", "FG%", "+/-")
    for _, r in rest_split.iterrows():
        pm = r.get("PLUS_MINUS", 0)
        row(
            r["GROUP_VALUE"], int(r["GP"]), int(r["W"]), int(r["L"]),
            pct_fmt(r["W_PCT"]), f'{r["PTS"]:.1f}',
            pct_fmt(r["FG_PCT"]), pm_fmt(pm),
        )
    blank()

# ═══════════════════════════════════════════════════════════════
# 9. LAST 15 GAMES
# ═══════════════════════════════════════════════════════════════
h(2, "9. Last 15 Games")
last15 = gamelog.head(15)
tbl("Date", "Matchup", "W/L", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "+/-")
for _, g in last15.iterrows():
    pm = g.get("PLUS_MINUS", 0)
    wl = g["WL"]
    row(
        g["GAME_DATE"], g["MATCHUP"], f"**{wl}**",
        int(g["PTS"]), int(g["REB"]), int(g["AST"]),
        int(g["STL"]), int(g["BLK"]), int(g["TOV"]),
        pct_fmt(g["FG_PCT"]), pct_fmt(g["FG3_PCT"]),
        pm_fmt(pm),
    )
blank()

# ═══════════════════════════════════════════════════════════════
# 10. KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════
h(2, "10. Key Insights & Takeaways")

# 10.1 Team leaders
h(3, "10.1 Statistical Leaders")
pts_leader = base_sorted.iloc[0]
ast_leader = base.sort_values("AST", ascending=False).iloc[0]
reb_leader = base.sort_values("REB", ascending=False).iloc[0]
stl_leader = base.sort_values("STL", ascending=False).iloc[0]
blk_leader = base.sort_values("BLK", ascending=False).iloc[0]

tbl("Category", "Player", "Value")
row("Scoring", pts_leader["PLAYER_NAME"], f'{pts_leader["PTS"]:.1f} PPG')
row("Assists", ast_leader["PLAYER_NAME"], f'{ast_leader["AST"]:.1f} APG')
row("Rebounds", reb_leader["PLAYER_NAME"], f'{reb_leader["REB"]:.1f} RPG')
row("Steals", stl_leader["PLAYER_NAME"], f'{stl_leader["STL"]:.1f} SPG')
row("Blocks", blk_leader["PLAYER_NAME"], f'{blk_leader["BLK"]:.1f} BPG')
if len(adv_sorted) > 0:
    best_net = adv_sorted.iloc[0]
    row("Net Rating", best_net["PLAYER_NAME"], pm_fmt(best_net["NET_RATING"]))
blank()

# 10.2 Shooting
h(3, "10.2 Shooting Insights")

if "ShotAreaTeamDashboard" in team_shoot:
    team_areas = team_shoot["ShotAreaTeamDashboard"]
    gp_team = w + l  # games played
    ab3 = team_areas[team_areas["GROUP_VALUE"] == "Above the Break 3"]
    if len(ab3) > 0:
        p(f'- **Team Above-Break 3PT:** {pct_fmt(ab3.iloc[0]["FG_PCT"])} ({ab3.iloc[0]["FGA"] / gp_team:.1f} FGA/game)')
    ra = team_areas[team_areas["GROUP_VALUE"] == "Restricted Area"]
    if len(ra) > 0:
        p(f'- **Restricted Area:** {pct_fmt(ra.iloc[0]["FG_PCT"])} ({ra.iloc[0]["FGA"] / gp_team:.1f} FGA/game)')
    mr = team_areas[team_areas["GROUP_VALUE"] == "Mid-Range"]
    if len(mr) > 0:
        p(f'- **Mid-Range:** {pct_fmt(mr.iloc[0]["FG_PCT"])} ({mr.iloc[0]["FGA"] / gp_team:.1f} FGA/game)')
    # Corner 3s
    lc = team_areas[team_areas["GROUP_VALUE"] == "Left Corner 3"]
    rc = team_areas[team_areas["GROUP_VALUE"] == "Right Corner 3"]
    if len(lc) > 0 and len(rc) > 0:
        combined_fgm = lc.iloc[0]["FGM"] + rc.iloc[0]["FGM"]
        combined_fga = lc.iloc[0]["FGA"] + rc.iloc[0]["FGA"]
        if combined_fga > 0:
            p(f'- **Corner 3s:** {combined_fgm / combined_fga * 100:.1f}% ({combined_fga / gp_team:.1f} FGA/game, L: {pct_fmt(lc.iloc[0]["FG_PCT"])}, R: {pct_fmt(rc.iloc[0]["FG_PCT"])})')
blank()

# Curry analysis
if "ShotAreaPlayerDashboard" in player_shoots["curry"]:
    curry_area = player_shoots["curry"]["ShotAreaPlayerDashboard"]
    p("**Stephen Curry Shooting Profile:**")
    curry_ab3 = curry_area[curry_area["GROUP_VALUE"] == "Above the Break 3"]
    if len(curry_ab3) > 0:
        p(f'- Above-Break 3: {pct_fmt(curry_ab3.iloc[0]["FG_PCT"])} on {int(curry_ab3.iloc[0]["FGA"])} attempts')
    curry_lc = curry_area[curry_area["GROUP_VALUE"] == "Left Corner 3"]
    curry_rc = curry_area[curry_area["GROUP_VALUE"] == "Right Corner 3"]
    if len(curry_lc) > 0 and len(curry_rc) > 0:
        p(f'- Corner 3s: Left {pct_fmt(curry_lc.iloc[0]["FG_PCT"])}, Right {pct_fmt(curry_rc.iloc[0]["FG_PCT"])}')
    curry_mr = curry_area[curry_area["GROUP_VALUE"] == "Mid-Range"]
    if len(curry_mr) > 0:
        p(f'- Mid-Range: {pct_fmt(curry_mr.iloc[0]["FG_PCT"])} on {int(curry_mr.iloc[0]["FGA"])} attempts')
    curry_ra = curry_area[curry_area["GROUP_VALUE"] == "Restricted Area"]
    if len(curry_ra) > 0:
        p(f'- Restricted Area: {pct_fmt(curry_ra.iloc[0]["FG_PCT"])} on {int(curry_ra.iloc[0]["FGA"])} attempts')
    blank()

# 10.3 Effort
h(3, "10.3 Effort & Defense Observations")
p(f'- **{top_contest["PLAYER_NAME"]}** anchors interior defense with {top_contest["CONTESTED_SHOTS"]:.1f} contested shots/game')
p(f'- **{top_deflect["PLAYER_NAME"]}** disrupts passing lanes with {top_deflect["DEFLECTIONS"]:.1f} deflections/game')
p(f'- **{top_screen["PLAYER_NAME"]}** powers the motion offense with {top_screen["SCREEN_ASSISTS"]:.1f} screen assists/game')
p(f'- **{fastest["PLAYER_NAME"]}** is the fastest player at {fastest["AVG_SPEED"]:.2f} mph, creating pace advantages')
blank()

# 10.4 Strengths
h(3, "10.4 Strengths")
strengths = []
if wpct >= 0.5:
    strengths.append(f"- Winning record ({w}-{l}, {pct_fmt(wpct)}) in playoff contention")
if loc_split is not None:
    home = loc_split[loc_split["GROUP_VALUE"] == "Home"]
    if len(home) > 0 and home.iloc[0]["W_PCT"] >= 0.6:
        strengths.append(f'- Dominant at Chase Center ({pct_fmt(home.iloc[0]["W_PCT"])} home win rate)')
if "ShotAreaPlayerDashboard" in player_shoots["curry"]:
    cab3 = player_shoots["curry"]["ShotAreaPlayerDashboard"]
    cab3_row = cab3[cab3["GROUP_VALUE"] == "Above the Break 3"]
    if len(cab3_row) > 0 and cab3_row.iloc[0]["FG_PCT"] >= 0.36:
        strengths.append(f'- Curry remains an elite 3-point threat ({pct_fmt(cab3_row.iloc[0]["FG_PCT"])} from above the break)')
if top_screen["SCREEN_ASSISTS"] >= 2.0:
    strengths.append(f'- Elite motion offense ({top_screen["PLAYER_NAME"]} generates {top_screen["SCREEN_ASSISTS"]:.1f} screen assists/game)')

for s in strengths:
    p(s)
if not strengths:
    p("- Team is building cohesion with new roster additions")
blank()

# 10.5 Areas of concern
h(3, "10.5 Areas of Concern")
concerns = []
if diff < 0:
    concerns.append(f"- **Negative point differential** ({pm_fmt(diff)}) — close losses outweighing blowout wins")
if loc_split is not None:
    away = loc_split[loc_split["GROUP_VALUE"] == "Road"]
    if len(away) > 0 and away.iloc[0]["W_PCT"] < 0.4:
        concerns.append(f'- **Road struggles:** {pct_fmt(away.iloc[0]["W_PCT"])} win rate away from home')
# Check mid-range volume
if "ShotAreaTeamDashboard" in team_shoot:
    mr_row = team_shoot["ShotAreaTeamDashboard"]
    mr_row = mr_row[mr_row["GROUP_VALUE"] == "Mid-Range"]
    if len(mr_row) > 0:
        mr_per_game = mr_row.iloc[0]["FGA"] / (w + l)
        if mr_per_game > 6:
            concerns.append(f'- **High mid-range volume** ({mr_per_game:.1f} FGA/game) — could shift to more efficient shots')
# Check recent form
if recent10_w <= 4:
    concerns.append(f"- **Recent skid:** {recent10_w}-{10 - recent10_w} in last 10 games")
# Turnovers
avg_tov_team = base[base["GP"] >= 20]["TOV"].mean()
if avg_tov_team > 2.0:
    concerns.append(f"- Elevated turnover rate among rotation players ({avg_tov_team:.1f} avg)")

for c in concerns:
    p(c)
if not concerns:
    p("- No major red flags identified")
blank()

# 10.6 Roster construction
h(3, "10.6 Roster Composition Analysis")

# Classify players by role based on stats
starters = base[base["MIN"] >= 25].sort_values("MIN", ascending=False)
rotation = base[(base["MIN"] >= 15) & (base["MIN"] < 25) & (base["GP"] >= 20)].sort_values("MIN", ascending=False)
bench = base[(base["MIN"] >= 5) & (base["MIN"] < 15) & (base["GP"] >= 15)].sort_values("MIN", ascending=False)

p("**Starters (25+ MPG):**")
for _, r in starters.iterrows():
    p(f'- {r["PLAYER_NAME"]}: {r["PTS"]:.1f} PTS / {r["REB"]:.1f} REB / {r["AST"]:.1f} AST ({r["MIN"]:.1f} min)')
blank()

p("**Key Rotation (15-25 MPG):**")
for _, r in rotation.iterrows():
    p(f'- {r["PLAYER_NAME"]}: {r["PTS"]:.1f} PTS / {r["REB"]:.1f} REB / {r["AST"]:.1f} AST ({r["MIN"]:.1f} min)')
blank()

if len(bench) > 0:
    p("**Bench Contributors (<15 MPG):**")
    for _, r in bench.iterrows():
        p(f'- {r["PLAYER_NAME"]}: {r["PTS"]:.1f} PTS / {r["REB"]:.1f} REB / {r["AST"]:.1f} AST ({r["MIN"]:.1f} min)')
    blank()

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
p("---")
p(f"*Data source: NBA.com/stats via stats.nba.com | Season: 2025-26 | Report generated: {now}*")

# Write report
report = "\n".join(lines)
output = REPORTS / "warriors_detailed_report_2025_26.md"
output.write_text(report)
print(f"✅ Report written to {output}")
print(f"   {len(lines)} lines, {len(report):,} characters")
