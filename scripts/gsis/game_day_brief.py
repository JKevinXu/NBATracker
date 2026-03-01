"""
GSIS Phase 6 — Game-Day Intelligence Brief
Integrates all five models (M1–M5) into a single actionable pre-game report.
Now team-configurable: pass --team LAL to run for the Lakers.
"""

import json, os, sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gsis.team_config import get_team, set_team, load_cache

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"
GOLD, BLUE, GREEN, RED, WHITE = "#FFC72C", "#1D428A", "#2ecc71", "#e74c3c", "#e8e8e8"

plt.rcParams.update({
    "figure.facecolor": "#0f1923", "axes.facecolor": "#0f1923",
    "axes.edgecolor": "#2a3f52", "text.color": WHITE,
    "axes.labelcolor": WHITE, "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0", "grid.color": "#1e3044",
    "grid.alpha": 0.5, "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})


def _load(name):
    return load_cache(name)

def _rs_to_df(data, idx=0):
    rs = data.get("resultSets", data)
    return pd.DataFrame(rs[idx]["rowSet"], columns=rs[idx]["headers"])


# ══════════════════════════════════════════════════════════════════
# DETECT NEXT GAME
# ══════════════════════════════════════════════════════════════════

def get_next_opponent():
    """Determine the next opponent from the schedule / game log.
    Since we can't know the actual future schedule, we use the most
    recent opponent as a stand-in and generate the brief for that matchup.
    """
    gl = _load("gamelog")
    df = _rs_to_df(gl)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")
    last = df.iloc[-1]
    matchup = last["MATCHUP"]

    # Parse opponent
    if " vs. " in matchup:
        opp = matchup.split(" vs. ")[-1].strip()
    else:
        opp = matchup.split(" @ ")[-1].strip()

    home = " vs. " in matchup
    return {
        "opponent": opp,
        "date": last["GAME_DATE"].strftime("%b %d, %Y"),
        "home": home,
        "last_game_date": last["GAME_DATE"],
    }


# ══════════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════

def run_all_models():
    """Execute each GSIS model and collect results."""
    from scripts.gsis.pregame_model import run as run_m1
    from scripts.gsis.fatigue_manager import run as run_m5
    from scripts.gsis.opponent_cluster import run as run_m3
    from scripts.gsis.player_forecast import run as run_m4
    from scripts.gsis.lineup_optimizer import run as run_m2

    team = get_team()
    img_dir = ROOT / "reports" / "game_briefs" / "figures"
    os.makedirs(img_dir, exist_ok=True)

    print("=" * 60)
    print(f"GSIS — Game-Day Intelligence Brief Generator ({team})")
    print("=" * 60)
    print()

    # Each model returns its key results
    print("Phase 1: Pre-Game Win Predictor")
    m1 = run_m1(img_dir)               # returns results dict
    print()

    print("Phase 2: Fatigue Manager")
    fat_df, snapshot = run_m5()         # returns (DataFrame, latest_snapshot)
    # Build a dict keyed by player for the brief
    from scripts.gsis.fatigue_manager import minutes_recommendation
    m5 = {}
    if snapshot is not None and isinstance(snapshot, pd.DataFrame):
        for _, row in snapshot.iterrows():
            name = row.get("PLAYER_NAME", "")
            fi = float(row.get("FATIGUE_INDEX", 0))
            zone = ("Red" if fi > 70 else "Orange" if fi > 55
                    else "Yellow" if fi > 40 else "Green")
            avg_min = float(row.get("SEASON_AVG_MIN", 25))
            rec = minutes_recommendation(fi, avg_min)
            m5[name] = {
                "fatigue_index": fi,
                "zone": zone,
                "recommendation": rec,
            }
    print()

    print("Phase 3: Opponent Archetype Classifier")
    profiles, strategies = run_m3()     # returns (profiles_df, strategies)
    m3 = {"profiles": profiles, "strategies": strategies}
    print()

    print("Phase 4: Player Performance Forecaster")
    m4 = run_m4()                       # returns results dict
    print()

    print("Phase 5: Lineup Optimizer")
    m2 = run_m2()                       # returns lineups_scored list
    print()

    return {
        "m1_pregame": m1,
        "m5_fatigue": m5,
        "m3_opponent": m3,
        "m4_forecast": m4,
        "m2_lineup": m2,
    }


# ══════════════════════════════════════════════════════════════════
# GAME-DAY BRIEF
# ══════════════════════════════════════════════════════════════════

def generate_brief(models_output, img_dir, report_path):
    """Generate the unified Game-Day Intelligence Brief."""
    team = get_team()
    game_info = get_next_opponent()
    opp = game_info["opponent"]
    date = game_info["date"]
    venue = "Home" if game_info["home"] else "Away"

    md = []
    p = md.append

    p(f"# 🏀 GSIS Game-Day Intelligence Brief")
    p(f"## {team} {'vs.' if game_info['home'] else '@'} {opp} — {date}")
    p("")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} | System: Game Strategy Intelligence System (GSIS)*")
    p(f"*Team: {team} | Venue: {venue} | Models: 5 interconnected ML systems*")
    p("")
    p("---")
    p("")

    # ══════════ EXECUTIVE SUMMARY ══════════
    p("## 📋 Executive Summary")
    p("")

    # Build priorities from model outputs
    priorities = []

    # Fatigue alerts
    m5_data = models_output.get("m5_fatigue")
    high_fatigue_players = []
    if m5_data and isinstance(m5_data, dict):
        for player, info in m5_data.items():
            if isinstance(info, dict) and info.get("fatigue_index", 0) > 55:
                high_fatigue_players.append((player, info["fatigue_index"]))
    if high_fatigue_players:
        names = ", ".join([p[0].split()[-1] for p in high_fatigue_players[:3]])
        priorities.append(f"⚠️ **Monitor fatigue** for {names} — consider minutes restrictions")

    # Player forecasts
    m4_data = models_output.get("m4_forecast")
    hot_players = []
    cold_players = []
    if m4_data and isinstance(m4_data, dict):
        for player, stats in m4_data.items():
            if isinstance(stats, dict) and "PTS" in stats:
                pts_info = stats["PTS"]
                if pts_info.get("trend") == "↑":
                    hot_players.append(player)
                elif pts_info.get("trend") == "↓":
                    cold_players.append(player)

    if hot_players:
        names = ", ".join([p.split()[-1] for p in hot_players[:3]])
        priorities.append(f"🔥 **Ride the hot hand**: {names} trending above season averages")
    if cold_players:
        names = ", ".join([p.split()[-1] for p in cold_players[:3]])
        priorities.append(f"📉 **Cold streak watch**: {names} — adjust expectations / increase touches")

    # Lineup recommendation
    m2_data = models_output.get("m2_lineup")
    if m2_data and len(m2_data) > 0:
        best_net = m2_data[0]["net_rating"]
        if best_net > 10:
            priorities.append(f"🏀 **Optimal lineup available**: projected +{best_net:.1f} net rating")
        elif best_net < 0:
            priorities.append(f"⚠️ **Lineup challenge**: best projected lineup is {best_net:+.1f} net rating")

    if not priorities:
        priorities.append("📊 All systems nominal — proceed with standard game plan")

    p("**Top Game-Day Priorities:**")
    p("")
    for i, pri in enumerate(priorities, 1):
        p(f"{i}. {pri}")
    p("")
    p("---")
    p("")

    # ══════════ M1: WIN PROBABILITY ══════════
    p("## 1. Pre-Game Win Probability (M1)")
    p("")
    p("*Full analysis: [pregame_win_predictor.md](pregame_win_predictor.md)*")
    p("")
    p("The stacked ensemble (XGBoost + LightGBM + Logistic Regression) provides a calibrated")
    p("win probability using only pre-game information (recent form, rest, opponent quality, etc.).")
    p("")
    p("![Win Probability Factors](figures/pregame_shap_summary.png)")
    p("")

    # ══════════ M3: OPPONENT SCOUTING ══════════
    p("## 2. Opponent Scouting Report (M3)")
    p("")
    p("*Full analysis: [opponent_scouting.md](opponent_scouting.md)*")
    p("")
    p(f"**Opponent: {opp}**")
    p("")
    p("The K-Means archetype classifier has categorized all 30 teams by playing style.")
    p("See the full scouting report for this opponent's archetype, profile, and recommended")
    p("counter-strategies.")
    p("")
    p("![Archetype Map](figures/opponent_cluster_map.png)")
    p("")

    # ══════════ M2: LINEUP ══════════
    p("## 3. Lineup Recommendations (M2)")
    p("")
    p("*Full analysis: [lineup_recommendations.md](lineup_recommendations.md)*")
    p("")
    if m2_data and len(m2_data) >= 3:
        p("### 🏀 Recommended Starting Lineup")
        p("")
        best = m2_data[0]
        p(f"**{' — '.join(best['players'])}**")
        p(f"- Predicted Net Rating: **{best['net_rating']:+.1f}**")
        p(f"- Off: {best['off_rating']:.1f} | Def: {best['def_rating']:.1f}")
        p("")

        # Best defensive
        by_def = sorted(m2_data[:30], key=lambda x: x["def_rating"])
        p("### 🛡️ Best Defensive Lineup")
        p("")
        bd = by_def[0]
        p(f"**{' — '.join(bd['players'])}**")
        p(f"- Def Rating: **{bd['def_rating']:.1f}** | Net: {bd['net_rating']:+.1f}")
        p("")
    p("![Lineup Rankings](figures/lineup_rankings.png)")
    p("")

    # ══════════ M4: PLAYER FORECASTS ══════════
    p("## 4. Player Performance Forecasts (M4)")
    p("")
    p("*Full analysis: [player_forecasts.md](player_forecasts.md)*")
    p("")

    if m4_data and isinstance(m4_data, dict):
        p("| Player | PTS Forecast | REB | AST | Trend |")
        p("|---|---|---|---|---|")
        sorted_players = sorted(
            m4_data.items(),
            key=lambda x: x[1].get("PTS", {}).get("prediction", 0) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        for player, stats in sorted_players[:8]:
            if not isinstance(stats, dict) or "PTS" not in stats:
                continue
            pts = stats["PTS"]
            reb = stats.get("REB", {})
            ast = stats.get("AST", {})
            p(f"| {player} | **{pts['prediction']:.1f}** [{pts.get('lo_80',0):.0f}–{pts.get('hi_80',0):.0f}] "
              f"| {reb.get('prediction', 0):.1f} | {ast.get('prediction', 0):.1f} | {pts.get('trend', '→')} |")
        p("")

    p("![Player Forecasts](figures/player_forecasts.png)")
    p("")

    # ══════════ M5: FATIGUE ══════════
    p("## 5. Fatigue & Load Management (M5)")
    p("")
    p("*Full analysis: [fatigue_dashboard.md](fatigue_dashboard.md)*")
    p("")

    if m5_data and isinstance(m5_data, dict):
        # Show any fatigue alerts
        alerts = []
        for player, info in m5_data.items():
            if isinstance(info, dict):
                fi = info.get("fatigue_index", 0)
                zone = info.get("zone", "")
                rec = info.get("recommendation", "")
                if fi > 40:
                    emoji = "🔴" if fi > 70 else "🟠" if fi > 55 else "🟡"
                    alerts.append((player, fi, zone, rec, emoji))

        if alerts:
            alerts.sort(key=lambda x: -x[1])
            p("| Player | Fatigue Index | Zone | Recommendation |")
            p("|---|---|---|---|")
            for player, fi, zone, rec, emoji in alerts:
                p(f"| {player} | {emoji} **{fi:.0f}** | {zone} | {rec} |")
            p("")
        else:
            p("✅ All players in the green zone. No fatigue alerts.")
            p("")

    p("![Fatigue Dashboard](figures/fatigue_dashboard.png)")
    p("")

    # ══════════ TACTICAL TAKEAWAYS ══════════
    p("## 6. Tactical Takeaways")
    p("")
    p("Based on the integrated GSIS analysis:")
    p("")
    takeaways = []

    # From forecasts
    if hot_players:
        top_hot = hot_players[0]
        if m4_data and top_hot in m4_data and "PTS" in m4_data[top_hot]:
            pred = m4_data[top_hot]["PTS"]["prediction"]
            takeaways.append(f"**Feed {top_hot.split()[-1]}** — forecasted for {pred:.0f} PTS, trending above season average")

    if high_fatigue_players:
        top_tired = high_fatigue_players[0]
        takeaways.append(f"**Manage {top_tired[0].split()[-1]}'s minutes** — fatigue index {top_tired[1]:.0f}, "
                         f"consider <25 min")

    if m2_data and len(m2_data) > 0:
        best_closing = m2_data[0]
        closers = ", ".join([p.split()[-1] for p in best_closing["players"]])
        takeaways.append(f"**Closing lineup**: {closers} (Net: {best_closing['net_rating']:+.1f})")

    if not takeaways:
        takeaways.append("Execute standard game plan — no significant deviations recommended")

    for i, t in enumerate(takeaways, 1):
        p(f"{i}. {t}")
    p("")

    p("---")
    p("")
    p("## 7. Glossary")
    p("")
    p("| Term | Definition |")
    p("|---|---|")
    p("| Net Rating | Points scored minus points allowed per 100 possessions |")
    p("| Off Rating | Points scored per 100 possessions |")
    p("| Def Rating | Points allowed per 100 possessions (lower = better) |")
    p("| Fatigue Index | 0–100 scale (0=fresh, 100=exhausted) based on minutes, rest, age |")
    p("| SHAP | Feature importance method showing each factor's contribution to prediction |")
    p("| Archetype | Team playing-style cluster from K-Means algorithm |")
    p("| Stacked Ensemble | Combining multiple ML models via a meta-learner for better predictions |")
    p("| Quantile Regression | Predicts ranges (80% CI) instead of single point estimates |")
    p("| 80% CI | 80% prediction interval — true value falls in this range 80% of the time |")
    p("| L5/L10 | Rolling average over last 5 or 10 games |")
    p("")
    p("---")
    p("")
    p(f"*GSIS v1.0 — {datetime.now().strftime('%B %d, %Y')} | 5 Models | "
      f"{datetime.now().strftime('%H:%M')} | {team}*")

    report_path.write_text("\n".join(md))
    print(f"  📄 Game-Day Brief: {report_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(team=None):
    if team:
        set_team(team)

    team_abbrev = get_team()
    img_dir = ROOT / "reports" / "game_briefs" / "figures"
    report_path = ROOT / "reports" / "game_briefs" / f"game_day_brief_{team_abbrev.lower()}.md"
    os.makedirs(img_dir, exist_ok=True)

    models_output = run_all_models()

    print("=" * 60)
    print(f"Phase 6: Generating unified Game-Day Brief ({team_abbrev})")
    print("=" * 60)
    generate_brief(models_output, img_dir, report_path)
    print()
    print(f"✅ GSIS complete — all 5 models + unified brief generated for {team_abbrev}.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

    # Parse --team argument
    team = "GSW"  # default
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--team" and i < len(sys.argv):
            team = sys.argv[i + 1].upper()
        elif arg.startswith("--team="):
            team = arg.split("=")[1].upper()
        elif len(arg) == 3 and arg.isalpha():
            team = arg.upper()

    run(team)
