"""
GSIS Model M5 — Player Load & Fatigue Manager
Computes per-player fatigue index entering every game,
correlates fatigue with performance drops, and generates
minutes-management recommendations + visualizations.
"""

import json, os, warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scripts.gsis.team_config import get_team, load_cache

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"

GOLD = "#FFC72C"
BLUE = "#1D428A"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#e67e22"
WHITE = "#e8e8e8"

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
    if isinstance(rs, list):
        return pd.DataFrame(rs[idx]["rowSet"], columns=rs[idx]["headers"])
    raise ValueError(f"Unexpected resultSets type: {type(rs)}")


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_player_data():
    """Load and merge player game logs, ages, and tracking data."""
    # Player game logs (per-game)
    team = get_team()
    pg = _load("player_gamelogs")
    df = _rs_to_df(pg)
    df = df[df["TEAM_ABBREVIATION"] == team].copy()
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0)
    df["REB"] = pd.to_numeric(df["REB"], errors="coerce").fillna(0)
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0)
    df["TOV"] = pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
    df["STL"] = pd.to_numeric(df["STL"], errors="coerce").fillna(0)
    df["BLK"] = pd.to_numeric(df["BLK"], errors="coerce").fillna(0)
    df["FG_PCT"] = pd.to_numeric(df["FG_PCT"], errors="coerce").fillna(0)
    df["FGA"] = pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
    df["FGM"] = pd.to_numeric(df["FGM"], errors="coerce").fillna(0)
    df["FTA"] = pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
    df["PLUS_MINUS"] = pd.to_numeric(df["PLUS_MINUS"], errors="coerce").fillna(0)
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE_DT"]).reset_index(drop=True)

    # Player ages + season averages
    try:
        lb = _load("league_base")
        base = _rs_to_df(lb)
        base = base[base["TEAM_ABBREVIATION"] == team].copy()
    except FileNotFoundError:
        # Build from player gamelogs if league_base unavailable
        base = df.groupby("PLAYER_NAME").agg(
            GP=("GAME_DATE_DT", "count"),
            MIN=("MIN", "mean"),
            PTS=("PTS", "mean"),
        ).reset_index()
        base["AGE"] = 25  # default if unknown
        base["TEAM_ABBREVIATION"] = team
    age_map = dict(zip(base["PLAYER_NAME"], pd.to_numeric(base["AGE"], errors="coerce")))
    season_min_map = dict(zip(base["PLAYER_NAME"], pd.to_numeric(base["MIN"], errors="coerce")))
    season_pts_map = dict(zip(base["PLAYER_NAME"], pd.to_numeric(base["PTS"], errors="coerce")))
    gp_map = dict(zip(base["PLAYER_NAME"], pd.to_numeric(base["GP"], errors="coerce")))

    df["AGE"] = df["PLAYER_NAME"].map(age_map)
    df["SEASON_AVG_MIN"] = df["PLAYER_NAME"].map(season_min_map)
    df["SEASON_AVG_PTS"] = df["PLAYER_NAME"].map(season_pts_map)
    df["GP_SEASON"] = df["PLAYER_NAME"].map(gp_map)

    # Tracking data (speed/distance)
    try:
        ts = _load("tracking_speed")
        track = _rs_to_df(ts)
        track = track[track["TEAM_ABBREVIATION"] == team].copy()
        speed_map = dict(zip(track["PLAYER_NAME"],
                             pd.to_numeric(track["AVG_SPEED"], errors="coerce")))
        dist_map = dict(zip(track["PLAYER_NAME"],
                            pd.to_numeric(track["DIST_MILES"], errors="coerce")))
        df["AVG_SPEED"] = df["PLAYER_NAME"].map(speed_map).fillna(0)
        df["DIST_MILES"] = df["PLAYER_NAME"].map(dist_map).fillna(0)
    except Exception:
        df["AVG_SPEED"] = 0
        df["DIST_MILES"] = 0

    return df, age_map, season_min_map, season_pts_map, gp_map


# ══════════════════════════════════════════════════════════════════
# FATIGUE INDEX COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_fatigue(df):
    """
    Compute per-player per-game fatigue index (0–100).
    Components:
      1. MINUTES_LOAD    — L3 avg minutes / season avg minutes
      2. REST_PENALTY    — days since last game this player played
      3. SCHEDULE_DENSITY — games played in last 7 days
      4. AGE_FACTOR      — older players fatigue faster
      5. CUMULATIVE_LOAD — total minutes played in season so far
    """
    records = []
    for player, grp in df.groupby("PLAYER_NAME"):
        grp = grp.sort_values("GAME_DATE_DT").reset_index(drop=True)
        age = grp["AGE"].iloc[0] if not grp["AGE"].isna().all() else 25
        season_avg_min = grp["SEASON_AVG_MIN"].iloc[0] if not grp["SEASON_AVG_MIN"].isna().all() else 20
        season_avg_pts = grp["SEASON_AVG_PTS"].iloc[0] if not grp["SEASON_AVG_PTS"].isna().all() else 10

        for i in range(len(grp)):
            row = grp.iloc[i]
            prior = grp.iloc[:i]

            # 1. Minutes load: L3 avg minutes / season avg
            l3_mins = prior["MIN"].tail(3).mean() if len(prior) >= 1 else season_avg_min
            minutes_load = (l3_mins / max(season_avg_min, 1)) if season_avg_min > 0 else 1.0
            minutes_load = min(minutes_load, 2.0)

            # 2. Rest penalty
            if i > 0:
                days_rest = (row["GAME_DATE_DT"] - grp.iloc[i-1]["GAME_DATE_DT"]).days
            else:
                days_rest = 3  # assume rested for first game
            if days_rest <= 1:
                rest_penalty = 1.0  # back-to-back
            elif days_rest == 2:
                rest_penalty = 0.5
            elif days_rest == 3:
                rest_penalty = 0.2
            else:
                rest_penalty = 0.0

            # 3. Schedule density: games in last 7 days
            week_ago = row["GAME_DATE_DT"] - timedelta(days=7)
            games_in_week = len(prior[prior["GAME_DATE_DT"] >= week_ago])
            schedule_density = min(games_in_week / 4.0, 1.0)

            # 4. Age factor
            age_factor = max(0, (age - 28) / 12.0)
            age_factor = min(age_factor, 1.0)

            # 5. Cumulative load: fraction of max possible minutes played
            cum_minutes = prior["MIN"].sum()
            max_possible = len(prior) * 48  # 48 min per game max
            cum_load = (cum_minutes / max(max_possible, 1)) if max_possible > 0 else 0

            # Weighted combination (weights calibrated by design)
            raw_fatigue = (
                0.30 * minutes_load +
                0.25 * rest_penalty +
                0.20 * schedule_density +
                0.15 * age_factor +
                0.10 * cum_load
            )
            # Scale to 0–100
            fatigue_index = min(raw_fatigue * 100, 100)

            # Performance delta vs season average
            pts_delta = row["PTS"] - season_avg_pts if season_avg_pts > 0 else 0
            min_delta = row["MIN"] - season_avg_min if season_avg_min > 0 else 0

            records.append({
                "PLAYER_NAME": player,
                "GAME_DATE": row["GAME_DATE_DT"],
                "MATCHUP": row["MATCHUP"],
                "WL": row["WL"],
                "MIN": row["MIN"],
                "PTS": row["PTS"],
                "FG_PCT": row["FG_PCT"],
                "PLUS_MINUS": row["PLUS_MINUS"],
                "AGE": age,
                "SEASON_AVG_MIN": season_avg_min,
                "SEASON_AVG_PTS": season_avg_pts,
                "FATIGUE_INDEX": round(fatigue_index, 1),
                "MINUTES_LOAD": round(minutes_load, 3),
                "REST_PENALTY": round(rest_penalty, 3),
                "SCHEDULE_DENSITY": round(schedule_density, 3),
                "AGE_FACTOR": round(age_factor, 3),
                "CUM_LOAD": round(cum_load, 3),
                "PTS_DELTA": round(pts_delta, 1),
                "MIN_DELTA": round(min_delta, 1),
                "REST_DAYS": days_rest,
                "L3_AVG_MIN": round(l3_mins, 1),
                "GAMES_IN_WEEK": games_in_week,
            })

    return pd.DataFrame(records)


def calibrate_weights(fat_df):
    """
    Use Ridge Regression to learn which fatigue components
    best predict performance drops.
    Returns learned coefficients and R² score.
    """
    # Only use rotation players (≥15 GP)
    rotation = fat_df[fat_df.groupby("PLAYER_NAME")["PLAYER_NAME"].transform("count") >= 15].copy()
    if len(rotation) < 20:
        return None, None

    X = rotation[["MINUTES_LOAD", "REST_PENALTY", "SCHEDULE_DENSITY",
                   "AGE_FACTOR", "CUM_LOAD"]].values
    y = rotation["PTS_DELTA"].values

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    r2 = ridge.score(X, y)

    coef_names = ["Minutes Load", "Rest Penalty", "Schedule Density",
                  "Age Factor", "Cumulative Load"]
    coefficients = dict(zip(coef_names, ridge.coef_))

    return coefficients, r2


# ══════════════════════════════════════════════════════════════════
# CURRENT FATIGUE SNAPSHOT
# ══════════════════════════════════════════════════════════════════

def current_fatigue_snapshot(fat_df):
    """Get the most recent fatigue index for each player."""
    latest = fat_df.sort_values("GAME_DATE").groupby("PLAYER_NAME").tail(1)
    latest = latest.sort_values("FATIGUE_INDEX", ascending=False)
    return latest


def minutes_recommendation(fatigue, season_avg_min):
    """Generate a minutes recommendation based on fatigue level."""
    if fatigue > 70:
        return f"🔴 REST or hard cap at {max(int(season_avg_min * 0.6), 10)} min"
    elif fatigue > 55:
        return f"🟠 Limit to {int(season_avg_min * 0.8)} min"
    elif fatigue > 40:
        return f"🟡 Normal ({int(season_avg_min)} min)"
    else:
        return f"🟢 Full workload ({int(season_avg_min + 3)} min max)"


# ══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════

def plot_fatigue_dashboard(snapshot, img_dir):
    """Current fatigue levels for all rotation players."""
    rot = snapshot[snapshot["SEASON_AVG_MIN"] >= 10].copy()
    rot = rot.sort_values("FATIGUE_INDEX", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(rot) * 0.45)))
    colors = []
    for f in rot["FATIGUE_INDEX"]:
        if f > 70:
            colors.append(RED)
        elif f > 55:
            colors.append(ORANGE)
        elif f > 40:
            colors.append(GOLD)
        else:
            colors.append(GREEN)

    bars = ax.barh(range(len(rot)), rot["FATIGUE_INDEX"].values,
                   color=colors, edgecolor="white", linewidth=0.3, height=0.7)

    names = rot["PLAYER_NAME"].values
    ages = rot["AGE"].values
    for i, (bar, name, age, fi) in enumerate(zip(bars, names, ages, rot["FATIGUE_INDEX"].values)):
        label = f"{name} (age {int(age)})"
        ax.text(0.5, i, label, va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(bar.get_width() + 1, i, f"{fi:.0f}", va="center", fontsize=11,
                color="white", fontweight="bold")

    # Zone backgrounds
    ax.axvline(40, color=GREEN, linestyle="--", alpha=0.3)
    ax.axvline(55, color=GOLD, linestyle="--", alpha=0.3)
    ax.axvline(70, color=RED, linestyle="--", alpha=0.3)
    ax.text(20, len(rot) - 0.3, "Low", color=GREEN, fontsize=9, alpha=0.6)
    ax.text(46, len(rot) - 0.3, "Mod", color=GOLD, fontsize=9, alpha=0.6)
    ax.text(61, len(rot) - 0.3, "High", color=ORANGE, fontsize=9, alpha=0.6)
    ax.text(75, len(rot) - 0.3, "Critical", color=RED, fontsize=9, alpha=0.6)

    ax.set_yticks([])
    ax.set_xlabel("Fatigue Index (0–100)")
    ax.set_title("Current Player Fatigue Levels")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(img_dir / "fatigue_dashboard.png", dpi=150)
    plt.close(fig)


def plot_fatigue_vs_performance(fat_df, img_dir):
    """Scatter plot: fatigue index vs points delta for rotation players."""
    rot = fat_df[fat_df["SEASON_AVG_MIN"] >= 13].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: fatigue vs PTS delta
    ax = axes[0]
    for player, grp in rot.groupby("PLAYER_NAME"):
        ax.scatter(grp["FATIGUE_INDEX"], grp["PTS_DELTA"], alpha=0.4, s=20,
                   label=player if len(grp) > 20 else None)
    # Trend line
    if len(rot) > 10:
        z = np.polyfit(rot["FATIGUE_INDEX"], rot["PTS_DELTA"], 1)
        x_line = np.linspace(rot["FATIGUE_INDEX"].min(), rot["FATIGUE_INDEX"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), color=RED, linewidth=2.5,
                label=f"Trend (slope={z[0]:.2f})")
    ax.axhline(0, color="white", linestyle="--", alpha=0.3)
    ax.set_xlabel("Fatigue Index")
    ax.set_ylabel("Points vs Season Average")
    ax.set_title("Fatigue Impact on Scoring")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # Right: fatigue vs plus/minus
    ax2 = axes[1]
    for player, grp in rot.groupby("PLAYER_NAME"):
        ax2.scatter(grp["FATIGUE_INDEX"], grp["PLUS_MINUS"], alpha=0.4, s=20)
    if len(rot) > 10:
        z2 = np.polyfit(rot["FATIGUE_INDEX"], rot["PLUS_MINUS"], 1)
        ax2.plot(x_line, np.polyval(z2, x_line), color=RED, linewidth=2.5,
                 label=f"Trend (slope={z2[0]:.2f})")
    ax2.axhline(0, color="white", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Fatigue Index")
    ax2.set_ylabel("Plus/Minus")
    ax2.set_title("Fatigue Impact on +/-")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(img_dir / "fatigue_vs_performance.png", dpi=150)
    plt.close(fig)


def plot_fatigue_timeline(fat_df, img_dir):
    """Fatigue timeline for top 5 players across the season."""
    top_players = fat_df.groupby("PLAYER_NAME")["MIN"].mean().nlargest(5).index.tolist()
    fig, axes = plt.subplots(len(top_players), 1, figsize=(14, 3 * len(top_players)),
                             sharex=True)
    if len(top_players) == 1:
        axes = [axes]

    for ax, player in zip(axes, top_players):
        grp = fat_df[fat_df["PLAYER_NAME"] == player].sort_values("GAME_DATE")
        x = range(len(grp))
        fi = grp["FATIGUE_INDEX"].values

        # Color bars by level
        colors = [RED if f > 70 else ORANGE if f > 55 else GOLD if f > 40 else GREEN for f in fi]
        ax.bar(x, fi, color=colors, width=0.8, alpha=0.8)

        # Rolling average
        roll = pd.Series(fi).rolling(5, min_periods=1).mean()
        ax.plot(x, roll, color=WHITE, linewidth=2, label="5-game avg")

        # Threshold lines
        ax.axhline(70, color=RED, linestyle="--", alpha=0.3)
        ax.axhline(55, color=ORANGE, linestyle="--", alpha=0.3)

        ax.set_ylabel("Fatigue")
        ax.set_title(f"{player} (age {int(grp['AGE'].iloc[0])})", fontsize=12)
        ax.set_ylim(0, 100)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Game Number (chronological)")
    plt.tight_layout()
    fig.savefig(img_dir / "fatigue_timeline.png", dpi=150)
    plt.close(fig)


def plot_rest_impact(fat_df, img_dir):
    """Bar chart: performance by rest days and age group."""
    rot = fat_df[fat_df["SEASON_AVG_MIN"] >= 13].copy()
    rot["AGE_GROUP"] = rot["AGE"].apply(
        lambda a: "30+" if a >= 30 else "Under 30"
    )
    rot["REST_BUCKET"] = rot["REST_DAYS"].apply(
        lambda d: "B2B (0-1)" if d <= 1 else "Normal (2)" if d == 2 else "Rested (3+)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PTS delta by rest bucket
    ax = axes[0]
    order = ["B2B (0-1)", "Normal (2)", "Rested (3+)"]
    for i, bucket in enumerate(order):
        sub = rot[rot["REST_BUCKET"] == bucket]
        young = sub[sub["AGE_GROUP"] == "Under 30"]["PTS_DELTA"]
        old = sub[sub["AGE_GROUP"] == "30+"]["PTS_DELTA"]
        ax.bar(i * 3, young.mean() if len(young) > 0 else 0,
               color=BLUE, width=0.8, label="Under 30" if i == 0 else "")
        ax.bar(i * 3 + 1, old.mean() if len(old) > 0 else 0,
               color=RED, width=0.8, label="30+" if i == 0 else "")
        ax.text(i * 3 + 0.5, -0.5, bucket, ha="center", fontsize=9,
                color="white", transform=ax.get_xaxis_transform())
    ax.axhline(0, color="white", linestyle="--", alpha=0.3)
    ax.set_ylabel("Points vs Season Average")
    ax.set_title("Rest Impact on Scoring by Age Group")
    ax.set_xticks([])
    ax.legend(fontsize=9)

    # Right: +/- by rest bucket
    ax2 = axes[1]
    for i, bucket in enumerate(order):
        sub = rot[rot["REST_BUCKET"] == bucket]
        young = sub[sub["AGE_GROUP"] == "Under 30"]["PLUS_MINUS"]
        old = sub[sub["AGE_GROUP"] == "30+"]["PLUS_MINUS"]
        ax2.bar(i * 3, young.mean() if len(young) > 0 else 0,
                color=BLUE, width=0.8, label="Under 30" if i == 0 else "")
        ax2.bar(i * 3 + 1, old.mean() if len(old) > 0 else 0,
                color=RED, width=0.8, label="30+" if i == 0 else "")
        ax2.text(i * 3 + 0.5, -0.5, bucket, ha="center", fontsize=9,
                 color="white", transform=ax2.get_xaxis_transform())
    ax2.axhline(0, color="white", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Plus/Minus")
    ax2.set_title("Rest Impact on +/- by Age Group")
    ax2.set_xticks([])
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(img_dir / "fatigue_rest_impact.png", dpi=150)
    plt.close(fig)


def plot_component_breakdown(snapshot, img_dir):
    """Stacked bar showing fatigue component breakdown per player."""
    rot = snapshot[snapshot["SEASON_AVG_MIN"] >= 10].copy()
    rot = rot.sort_values("FATIGUE_INDEX", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(rot) * 0.4)))
    y = range(len(rot))
    components = [
        ("MINUTES_LOAD", 0.30, "Minutes Load", BLUE),
        ("REST_PENALTY", 0.25, "Rest Penalty", RED),
        ("SCHEDULE_DENSITY", 0.20, "Schedule Density", ORANGE),
        ("AGE_FACTOR", 0.15, "Age Factor", "#9b59b6"),
        ("CUM_LOAD", 0.10, "Cumulative Load", GREEN),
    ]
    left = np.zeros(len(rot))
    for col, weight, label, color in components:
        vals = rot[col].values * weight * 100
        ax.barh(y, vals, left=left, color=color, label=label,
                edgecolor="white", linewidth=0.2, height=0.7)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({fi:.0f})" for n, fi in
                        zip(rot["PLAYER_NAME"].values, rot["FATIGUE_INDEX"].values)],
                       fontsize=9)
    ax.set_xlabel("Fatigue Index (0–100)")
    ax.set_title("Fatigue Component Breakdown")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(img_dir / "fatigue_components.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_report(fat_df, snapshot, coefficients, r2, img_dir, report_path):
    """Generate the fatigue management markdown report."""
    md = []
    p = md.append

    p("# Player Load & Fatigue Management Dashboard")
    p("")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Model: Weighted Fatigue Index + Ridge Regression Calibration*")
    p("")
    p("This dashboard tracks player fatigue levels across the season, identifies fatigue-related")
    p("performance declines, and provides data-driven minutes management recommendations.")
    p("")
    p("---")
    p("")

    # ── Current Fatigue Status ──
    p("## 1. Current Fatigue Status")
    p("")
    p("**What is the Fatigue Index?** A composite score (0–100) computed from five factors entering")
    p("each game: recent minutes load, rest days, schedule density, player age, and cumulative season")
    p("workload. Higher = more fatigued. The index uses only information available *before* tip-off.")
    p("")
    p("**How to read this chart:** Each bar represents one player's fatigue level entering their most")
    p("recent game. Green = fresh (< 40), yellow = moderate (40–55), orange = elevated (55–70),")
    p("red = critical (> 70). Player age is shown in parentheses.")
    p("")
    p("![Fatigue Dashboard](figures/fatigue_dashboard.png)")
    p("")

    # Table
    rot = snapshot[snapshot["SEASON_AVG_MIN"] >= 10].sort_values("FATIGUE_INDEX", ascending=False)
    p("| Player | Age | Fatigue | L3 Avg Min | Rest Days | Recommendation |")
    p("|---|---|---|---|---|---|")
    for _, r in rot.iterrows():
        rec = minutes_recommendation(r["FATIGUE_INDEX"], r["SEASON_AVG_MIN"])
        p(f"| {r['PLAYER_NAME']} | {int(r['AGE'])} | **{r['FATIGUE_INDEX']:.0f}** | "
          f"{r['L3_AVG_MIN']:.1f} | {int(r['REST_DAYS'])} | {rec} |")
    p("")

    # ── Component Breakdown ──
    p("## 2. Fatigue Component Breakdown")
    p("")
    p("**What is this chart?** Each player's fatigue index is decomposed into five weighted components.")
    p("This shows *why* a player is fatigued — whether it's heavy minutes, back-to-backs, age, or")
    p("cumulative season workload.")
    p("")
    p("**The five components:**")
    p("- **Minutes Load (30%):** Recent 3-game average minutes divided by season average. >1.0 means playing heavier than usual.")
    p("- **Rest Penalty (25%):** Days since last game. Back-to-backs (0–1 days) get full penalty; 3+ days rest = no penalty.")
    p("- **Schedule Density (20%):** Number of games played in last 7 days. More games = more fatigue.")
    p("- **Age Factor (15%):** Players over 28 accumulate fatigue faster, scaling linearly to age 40.")
    p("- **Cumulative Load (10%):** Total minutes played this season as a fraction of maximum possible.")
    p("")
    p("![Fatigue Components](figures/fatigue_components.png)")
    p("")

    # ── Fatigue Timeline ──
    p("## 3. Season Fatigue Timeline")
    p("")
    p("**How to read this chart:** Each panel tracks one player's fatigue index across every game of")
    p("the season. Bar color indicates severity (green/yellow/orange/red). The white line is a 5-game")
    p("rolling average. Look for upward trends that indicate accumulating fatigue, and spikes that")
    p("correspond to back-to-backs or dense schedule stretches.")
    p("")
    p("![Fatigue Timeline](figures/fatigue_timeline.png)")
    p("")

    # ── Fatigue vs Performance ──
    p("## 4. Fatigue Impact on Performance")
    p("")
    p("**What is this chart?** Each dot is one player-game. X-axis is the fatigue index *entering*")
    p("that game; Y-axis is how the player performed relative to their season average (positive =")
    p("above average, negative = below). The red trend line shows the overall relationship.")
    p("")
    p("**What to look for:** A downward-sloping trend line confirms that higher fatigue predicts")
    p("worse performance. A steeper slope = stronger fatigue effect. If the slope is near zero,")
    p("fatigue has minimal impact on this metric.")
    p("")
    p("![Fatigue vs Performance](figures/fatigue_vs_performance.png)")
    p("")

    # Ridge regression results
    if coefficients:
        p("### 4.1 Ridge Regression Calibration")
        p("")
        p(f"A Ridge Regression model (R² = {r2:.3f}) quantifies which fatigue components most predict")
        p("scoring performance drops:")
        p("")
        p("| Fatigue Component | Coefficient | Interpretation |")
        p("|---|---|---|")
        for comp, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "↓ scoring" if coef < 0 else "↑ scoring"
            p(f"| {comp} | {coef:+.3f} | {direction} per unit increase |")
        p("")
        p(f"> *R² = {r2:.3f} means the fatigue model explains {r2*100:.1f}% of the variance in scoring*")
        p(f"> *deviations from season averages.*")
        p("")

    # ── Rest Day Impact ──
    p("## 5. Rest Day Impact by Age Group")
    p("")
    p("**What is this chart?** Performance split by rest days (B2B vs normal vs well-rested) and age")
    p("group (under 30 vs 30+). This reveals whether veteran players need more rest than younger ones.")
    p("")
    p("**What to look for:** If the 30+ bars are significantly lower on B2Bs compared to rested games,")
    p("the coaching staff should prioritize resting veterans on back-to-backs.")
    p("")
    p("![Rest Impact](figures/fatigue_rest_impact.png)")
    p("")

    # ── Key Recommendations ──
    p("## 6. Load Management Recommendations")
    p("")

    # Find high-fatigue players
    high_fat = rot[rot["FATIGUE_INDEX"] > 55]
    if len(high_fat) > 0:
        p("### ⚠️ Immediate Attention Required")
        p("")
        for _, r in high_fat.iterrows():
            rec = minutes_recommendation(r["FATIGUE_INDEX"], r["SEASON_AVG_MIN"])
            p(f"- **{r['PLAYER_NAME']}** (age {int(r['AGE'])}, fatigue: {r['FATIGUE_INDEX']:.0f}) — {rec}")
        p("")

    # Age-based recommendations
    veterans = rot[rot["AGE"] >= 33].sort_values("AGE", ascending=False)
    if len(veterans) > 0:
        p("### 🧓 Veteran Management Plan")
        p("")
        for _, r in veterans.iterrows():
            avg = r["SEASON_AVG_MIN"]
            p(f"- **{r['PLAYER_NAME']}** (age {int(r['AGE'])}, {avg:.1f} mpg): "
              f"Consider capping at {int(avg * 0.85)} min on B2Bs; "
              f"rest 1 in every 5 B2B games")
        p("")

    # General
    p("### 📋 General Principles")
    p("")
    p("1. **Back-to-back rule:** No player over 33 should exceed 85% of their season average minutes on B2Bs")
    p("2. **3-in-5 rule:** When the team plays 3 games in 5 days, rest at least one starter per game")
    p("3. **Fatigue threshold:** Any player crossing fatigue index 70 should be considered for a rest game")
    p("4. **Monitor trend:** If a player's 5-game rolling fatigue average exceeds 60, proactively reduce minutes")
    p("")

    p("---")
    p("")
    p("## Appendix: Fatigue Index Formula")
    p("")
    p("```")
    p("FATIGUE_INDEX = 30% × MINUTES_LOAD")
    p("             + 25% × REST_PENALTY")
    p("             + 20% × SCHEDULE_DENSITY")
    p("             + 15% × AGE_FACTOR")
    p("             + 10% × CUMULATIVE_LOAD")
    p("")
    p("Where:")
    p("  MINUTES_LOAD     = (L3 avg minutes) / (season avg minutes)    [0–2]")
    p("  REST_PENALTY     = 1.0 if B2B, 0.5 if 1-day, 0.2 if 2-day   [0–1]")
    p("  SCHEDULE_DENSITY = (games in last 7 days) / 4                 [0–1]")
    p("  AGE_FACTOR       = max(0, (age − 28) / 12)                   [0–1]")
    p("  CUMULATIVE_LOAD  = (total min played) / (GP × 48)            [0–1]")
    p("```")
    p("")
    p("---")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Data: stats.nba.com 2025-26*")

    report_path.write_text("\n".join(md))
    print(f"  📄 Report: {report_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(img_dir=None, report_path=None):
    """Full pipeline: load → compute → visualize → report."""
    if img_dir is None:
        img_dir = ROOT / "reports" / "game_briefs" / "figures"
    if report_path is None:
        report_path = ROOT / "reports" / "game_briefs" / "fatigue_dashboard.md"

    img_dir = Path(img_dir)
    report_path = Path(report_path)
    os.makedirs(img_dir, exist_ok=True)

    print("M5 — Player Load & Fatigue Manager")
    print("  Loading player data …")
    df, age_map, season_min_map, season_pts_map, gp_map = load_player_data()
    print(f"  {len(df)} player-game rows loaded")

    print("  Computing fatigue index …")
    fat_df = compute_fatigue(df)
    print(f"  {len(fat_df)} fatigue records computed")

    print("  Calibrating weights via Ridge Regression …")
    coefficients, r2 = calibrate_weights(fat_df)
    if r2 is not None:
        print(f"  R² = {r2:.3f}")

    snapshot = current_fatigue_snapshot(fat_df)
    print(f"  Current snapshot: {len(snapshot)} players")

    print("  Generating visualizations …")
    plot_fatigue_dashboard(snapshot, img_dir)
    plot_fatigue_vs_performance(fat_df, img_dir)
    plot_fatigue_timeline(fat_df, img_dir)
    plot_rest_impact(fat_df, img_dir)
    plot_component_breakdown(snapshot, img_dir)

    print("  Generating report …")
    generate_report(fat_df, snapshot, coefficients, r2, img_dir, report_path)

    print("  ✅ Fatigue Manager complete.")
    return fat_df, snapshot


if __name__ == "__main__":
    run()
