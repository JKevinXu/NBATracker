"""
GSIS Model M4 — Player Performance Forecaster
Per-player XGBoost regressors to predict next-game PTS, REB, AST, FG%.
Includes quantile regression for 80% prediction intervals.
"""

import json, os, warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gsis.team_config import get_team, load_cache

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"
GOLD, BLUE, GREEN, RED, WHITE = "#FFC72C", "#1D428A", "#2ecc71", "#e74c3c", "#e8e8e8"
ORANGE, PURPLE = "#e67e22", "#9b59b6"

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
# DATA & FEATURES
# ══════════════════════════════════════════════════════════════════

STAT_COLS = ["PTS", "REB", "AST", "FG_PCT"]
NUM_COLS = ["MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "TOV", "STL", "BLK", "PTS", "PLUS_MINUS"]

def load_player_games():
    """Load per-game player data for rotation players."""
    team = get_team()
    pg = _load("player_gamelogs")
    df = _rs_to_df(pg)
    df = df[df["TEAM_ABBREVIATION"] == team].copy()
    df["DATE"] = pd.to_datetime(df["GAME_DATE"])
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Ages & GP
    try:
        lb = _load("league_base")
        base = _rs_to_df(lb)
        base = base[base["TEAM_ABBREVIATION"] == team]
        age_map = dict(zip(base["PLAYER_NAME"], pd.to_numeric(base["AGE"], errors="coerce")))
        gp_map = dict(zip(base["PLAYER_NAME"], pd.to_numeric(base["GP"], errors="coerce")))
    except FileNotFoundError:
        # Derive from player gamelogs
        gp_map = df.groupby("PLAYER_NAME")["GAME_ID"].nunique().to_dict()
        age_map = {n: 25 for n in gp_map}

    df["AGE"] = df["PLAYER_NAME"].map(age_map).fillna(25)
    df["GP_SEASON"] = df["PLAYER_NAME"].map(gp_map).fillna(0)

    # Filter to rotation players (≥15 GP)
    rotation = [p for p, gp in gp_map.items() if gp >= 15]
    df = df[df["PLAYER_NAME"].isin(rotation)].copy()
    df = df.sort_values(["PLAYER_NAME", "DATE"]).reset_index(drop=True)

    # Opponent abbreviation
    def _opp(m):
        return m.split(" vs. ")[-1].strip() if " vs. " in m else m.split(" @ ")[-1].strip()
    df["OPP"] = df["MATCHUP"].apply(_opp)
    df["HOME"] = df["MATCHUP"].apply(lambda m: 1 if " vs. " in m else 0)

    return df, rotation


def build_player_features(df, player, stat):
    """Build feature matrix for one player and one stat."""
    grp = df[df["PLAYER_NAME"] == player].sort_values("DATE").reset_index(drop=True)
    if len(grp) < 8:
        return None, None, None

    target = grp[stat].values
    # Rolling features (shifted so game N uses only games 0..N-1)
    features = pd.DataFrame(index=grp.index)
    features["L3"] = grp[stat].shift(1).rolling(3, min_periods=1).mean()
    features["L5"] = grp[stat].shift(1).rolling(5, min_periods=1).mean()
    features["L10"] = grp[stat].shift(1).rolling(10, min_periods=1).mean()
    features["SEASON_AVG"] = grp[stat].shift(1).expanding().mean()
    features["L3_STD"] = grp[stat].shift(1).rolling(3, min_periods=1).std().fillna(0)
    features["TREND"] = features["L3"] - features["L10"]  # positive = trending up

    # Context
    features["HOME"] = grp["HOME"].values
    features["MIN_L3"] = grp["MIN"].shift(1).rolling(3, min_periods=1).mean()
    features["GAME_NUM"] = range(len(grp))
    features["AGE"] = grp["AGE"].values

    # Rest days
    dates = grp["DATE"].values
    rest = [3]  # first game
    for i in range(1, len(dates)):
        delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])).days
        rest.append(min(delta, 5))
    features["REST_DAYS"] = rest

    # Fill NaN from first few games
    features = features.fillna(0)

    return features, target, grp


# ══════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════

def train_forecast_model(features, target):
    """Train XGBoost regressor with time-series CV."""
    X = features.values.astype(float)
    y = target.astype(float)

    model = xgb.XGBRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42,
    )
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    cv_mae = -scores.mean()

    # Fit on all data
    model.fit(X, y)

    # Quantile models for 80% intervals
    model_lo = xgb.XGBRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        objective="reg:quantileerror", quantile_alpha=0.10,
        subsample=0.8, reg_alpha=1.0, reg_lambda=3.0, random_state=42,
    )
    model_hi = xgb.XGBRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        objective="reg:quantileerror", quantile_alpha=0.90,
        subsample=0.8, reg_alpha=1.0, reg_lambda=3.0, random_state=42,
    )
    model_lo.fit(X, y)
    model_hi.fit(X, y)

    return model, model_lo, model_hi, cv_mae


def forecast_next_game(model, model_lo, model_hi, features):
    """Predict the next game using the last row of features."""
    X_last = features.iloc[[-1]].values.astype(float)
    pred = float(model.predict(X_last)[0])
    lo = float(model_lo.predict(X_last)[0])
    hi = float(model_hi.predict(X_last)[0])
    return pred, lo, hi


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_forecaster():
    """Train models for all rotation players × all stats."""
    df, rotation = load_player_games()
    results = {}

    for player in rotation:
        results[player] = {}
        for stat in STAT_COLS:
            features, target, grp = build_player_features(df, player, stat)
            if features is None:
                continue
            model, model_lo, model_hi, cv_mae = train_forecast_model(features, target)
            pred, lo, hi = forecast_next_game(model, model_lo, model_hi, features)

            season_avg = grp[stat].mean()
            l5_avg = grp[stat].tail(5).mean()
            trend = "↑" if l5_avg > season_avg * 1.05 else "↓" if l5_avg < season_avg * 0.95 else "→"

            results[player][stat] = {
                "prediction": round(pred, 1),
                "lo_80": round(max(lo, 0), 1),
                "hi_80": round(hi, 1),
                "season_avg": round(season_avg, 1),
                "l5_avg": round(l5_avg, 1),
                "cv_mae": round(cv_mae, 2),
                "trend": trend,
                "n_games": len(grp),
            }

    return results, df


# ══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════

def plot_forecast_cards(results, img_dir):
    """Forecast cards for top players."""
    # Sort players by predicted PTS
    players_with_pts = [(p, r["PTS"]["prediction"]) for p, r in results.items() if "PTS" in r]
    players_with_pts.sort(key=lambda x: -x[1])
    top = [p for p, _ in players_with_pts[:8]]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, player in zip(axes, top):
        r = results[player]
        stats_data = []
        labels = []
        for stat in STAT_COLS:
            if stat in r:
                s = r[stat]
                stats_data.append(s)
                labels.append(stat.replace("_PCT", "%"))

        if not stats_data:
            ax.set_visible(False)
            continue

        x = range(len(labels))
        preds = [s["prediction"] for s in stats_data]
        lo = [s["lo_80"] for s in stats_data]
        hi = [s["hi_80"] for s in stats_data]
        season = [s["season_avg"] for s in stats_data]
        trends = [s["trend"] for s in stats_data]

        # Bars for prediction
        bars = ax.bar(x, preds, color=GOLD, width=0.5, alpha=0.9, label="Forecast")
        # Error bars for 80% CI
        for i in range(len(x)):
            ax.plot([i, i], [lo[i], hi[i]], color=WHITE, linewidth=2, alpha=0.7)
            ax.plot([i-0.1, i+0.1], [lo[i], lo[i]], color=WHITE, linewidth=1.5, alpha=0.7)
            ax.plot([i-0.1, i+0.1], [hi[i], hi[i]], color=WHITE, linewidth=1.5, alpha=0.7)

        # Season avg markers
        ax.scatter(x, season, color=BLUE, s=60, zorder=5, marker="D", label="Season Avg")

        ax.set_xticks(x)
        xlabels = [f"{l}\n{trends[i]}" for i, l in enumerate(labels)]
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_title(player, fontsize=11)
        ax.set_ylim(bottom=0)

        # Value labels
        for i, (bar, pred_val) in enumerate(zip(bars, preds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{pred_val:.1f}", ha="center", fontsize=9, color=GOLD, fontweight="bold")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Next-Game Player Forecasts (with 80% Prediction Intervals)",
                 fontsize=15, fontweight="bold", color=WHITE)
    plt.tight_layout()
    fig.savefig(img_dir / "player_forecasts.png", dpi=150)
    plt.close(fig)


def plot_forecast_accuracy(results, img_dir):
    """Show cross-validated MAE for each player-stat model."""
    records = []
    for player, stats in results.items():
        for stat, info in stats.items():
            records.append({
                "Player": player.split()[-1],  # last name
                "Stat": stat.replace("_PCT", "%"),
                "MAE": info["cv_mae"],
                "Season Avg": info["season_avg"],
            })
    acc_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6))
    stats_list = ["PTS", "REB", "AST", "FG%"]
    colors = [GOLD, BLUE, GREEN, PURPLE]

    for i, stat in enumerate(stats_list):
        sub = acc_df[acc_df["Stat"] == stat].sort_values("MAE")
        y_pos = np.arange(len(sub)) + i * 0.2
        ax.barh(y_pos * 4 + i, sub["MAE"], height=0.8, color=colors[i],
                alpha=0.8, label=stat)
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(row["MAE"] + 0.1, j * 4 + i, f'{row["MAE"]:.1f}',
                    va="center", fontsize=8, color=WHITE)

    # Only label with player names at their mid-points
    players = acc_df[acc_df["Stat"] == "PTS"].sort_values("MAE")["Player"]
    ax.set_yticks(np.arange(len(players)) * 4 + 1.5)
    ax.set_yticklabels(players, fontsize=9)
    ax.set_xlabel("Cross-Validated MAE")
    ax.set_title("Forecast Model Accuracy (Lower = Better)")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(img_dir / "forecast_accuracy.png", dpi=150)
    plt.close(fig)


def plot_trend_tracker(results, df, img_dir):
    """Show L5 trend vs season average for key players."""
    players_with_pts = [(p, r.get("PTS", {}).get("prediction", 0)) for p, r in results.items()]
    players_with_pts.sort(key=lambda x: -x[1])
    top = [p for p, _ in players_with_pts[:6]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, player in zip(axes, top):
        grp = df[df["PLAYER_NAME"] == player].sort_values("DATE")
        pts = grp["PTS"].values
        x = range(len(pts))
        season_avg = np.mean(pts)

        ax.bar(x, pts, color=[GREEN if p >= season_avg else RED for p in pts],
               alpha=0.6, width=0.8)
        # Rolling average
        roll = pd.Series(pts).rolling(5, min_periods=1).mean()
        ax.plot(x, roll, color=GOLD, linewidth=2.5, label="L5 rolling avg")
        ax.axhline(season_avg, color=WHITE, linestyle="--", alpha=0.4,
                   label=f"Season avg: {season_avg:.1f}")

        # Forecast
        r = results.get(player, {}).get("PTS", {})
        if r:
            next_x = len(pts)
            ax.bar(next_x, r["prediction"], color=GOLD, alpha=0.9, width=0.8)
            ax.plot([next_x, next_x], [r["lo_80"], r["hi_80"]],
                    color=WHITE, linewidth=3, alpha=0.8)
            ax.text(next_x, r["prediction"] + 1, f"{r['prediction']:.0f}",
                    ha="center", fontsize=10, color=GOLD, fontweight="bold")

        ax.set_title(f"{player} {r.get('trend', '')}", fontsize=11)
        ax.set_ylim(bottom=0)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Scoring Trend + Next-Game Forecast", fontsize=14,
                 fontweight="bold", color=WHITE)
    plt.tight_layout()
    fig.savefig(img_dir / "forecast_trends.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════

def generate_report(results, img_dir, report_path):
    md = []
    p = md.append

    p("# Player Performance Forecaster")
    p("")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Model: Per-Player XGBoost Regressors with Quantile Intervals*")
    p("")
    p("Individual XGBoost models trained on each player's game-by-game history predict their next-game")
    p("stat line. Features include rolling averages (L3/L5/L10), trends, rest days, home/away, and")
    p("game context. Quantile regression provides 80% prediction intervals.")
    p("")
    p("---")
    p("")

    # ── Forecast Cards ──
    p("## 1. Next-Game Forecasts")
    p("")
    p("**How to read this chart:** Each card shows one player's predicted stats for the next game.")
    p("Gold bars = point forecast. White error bars = 80% prediction interval (the true value will fall")
    p("in this range 80% of the time). Blue diamonds = season average for comparison. Arrows below each")
    p("stat show the trend (↑ = L5 avg > season avg, ↓ = below, → = stable).")
    p("")
    p("![Player Forecasts](figures/player_forecasts.png)")
    p("")

    # Forecast table
    players_sorted = sorted(results.keys(),
                            key=lambda pl: results[pl].get("PTS", {}).get("prediction", 0),
                            reverse=True)
    p("| Player | PTS | REB | AST | FG% | PTS Trend |")
    p("|---|---|---|---|---|---|")
    for player in players_sorted:
        r = results[player]
        pts = r.get("PTS", {})
        reb = r.get("REB", {})
        ast = r.get("AST", {})
        fg = r.get("FG_PCT", {})
        pts_str = f"**{pts.get('prediction',0):.1f}** [{pts.get('lo_80',0):.0f}–{pts.get('hi_80',0):.0f}]" if pts else "—"
        reb_str = f"{reb.get('prediction',0):.1f} [{reb.get('lo_80',0):.0f}–{reb.get('hi_80',0):.0f}]" if reb else "—"
        ast_str = f"{ast.get('prediction',0):.1f} [{ast.get('lo_80',0):.0f}–{ast.get('hi_80',0):.0f}]" if ast else "—"
        fg_str = f"{fg.get('prediction',0)*100:.1f}%" if fg else "—"
        trend = pts.get("trend", "→")
        p(f"| {player} | {pts_str} | {reb_str} | {ast_str} | {fg_str} | {trend} |")
    p("")

    # ── Trend Tracker ──
    p("## 2. Scoring Trends + Forecast")
    p("")
    p("**How to read this chart:** Each panel tracks a player's game-by-game scoring (green = above")
    p("average, red = below). The gold line is the 5-game rolling average. The final gold bar with")
    p("error bar is the forecast for the *next* game. Compare the forecast to the season average")
    p("(dashed line) to see if the player is trending up or down.")
    p("")
    p("![Forecast Trends](figures/forecast_trends.png)")
    p("")

    # ── Model Accuracy ──
    p("## 3. Model Accuracy")
    p("")
    p("**How to read this chart:** Cross-validated Mean Absolute Error (MAE) for each player-stat model.")
    p("Lower bars = more accurate predictions. For context, a PTS MAE of 6.0 means the model's")
    p("predictions are off by ±6 points on average. For FG%, divide by 100 (MAE of 0.08 = ±8% FG%).")
    p("")
    p("![Forecast Accuracy](figures/forecast_accuracy.png)")
    p("")

    # ── Alerts ──
    p("## 4. Trend Alerts")
    p("")
    up = [(pl, r["PTS"]) for pl, r in results.items()
          if "PTS" in r and r["PTS"]["trend"] == "↑"]
    down = [(pl, r["PTS"]) for pl, r in results.items()
            if "PTS" in r and r["PTS"]["trend"] == "↓"]

    if up:
        p("### 🔥 Hot Streaks (L5 > Season Average)")
        p("")
        for pl, info in sorted(up, key=lambda x: -(x[1]["l5_avg"] - x[1]["season_avg"])):
            delta = info["l5_avg"] - info["season_avg"]
            p(f"- **{pl}**: L5 avg {info['l5_avg']:.1f} vs season {info['season_avg']:.1f} "
              f"(**+{delta:.1f}**) — forecast: {info['prediction']:.1f}")
        p("")

    if down:
        p("### ⚠️ Cold Streaks (L5 < Season Average)")
        p("")
        for pl, info in sorted(down, key=lambda x: x[1]["l5_avg"] - x[1]["season_avg"]):
            delta = info["l5_avg"] - info["season_avg"]
            p(f"- **{pl}**: L5 avg {info['l5_avg']:.1f} vs season {info['season_avg']:.1f} "
              f"(**{delta:.1f}**) — forecast: {info['prediction']:.1f}")
        p("")

    p("---")
    p(f"*Models: {len(results)} players × {len(STAT_COLS)} stats = {len(results)*len(STAT_COLS)} individual XGBoost models*")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Data: stats.nba.com 2025-26*")

    report_path.write_text("\n".join(md))
    print(f"  📄 Report: {report_path}")


# ══════════════════════════════════════════════════════════════════

def run(img_dir=None, report_path=None):
    if img_dir is None:
        img_dir = ROOT / "reports" / "game_briefs" / "figures"
    if report_path is None:
        report_path = ROOT / "reports" / "game_briefs" / "player_forecasts.md"
    img_dir, report_path = Path(img_dir), Path(report_path)
    os.makedirs(img_dir, exist_ok=True)

    print("M4 — Player Performance Forecaster")
    print("  Loading player data …")
    results, df = run_forecaster()
    n_models = sum(len(v) for v in results.values())
    print(f"  Trained {n_models} models for {len(results)} players")

    print("  Generating visualizations …")
    plot_forecast_cards(results, img_dir)
    plot_forecast_accuracy(results, img_dir)
    plot_trend_tracker(results, df, img_dir)

    print("  Generating report …")
    generate_report(results, img_dir, report_path)
    print("  ✅ Player Forecaster complete.")
    return results


if __name__ == "__main__":
    run()
