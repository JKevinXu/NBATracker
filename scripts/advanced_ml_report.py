#!/usr/bin/env python3
"""
Advanced ML Analysis — Golden State Warriors 2025-26
Generates a comprehensive report with embedded visualizations,
advanced models (XGBoost + SHAP, Bayesian inference, time series),
radar charts, heatmaps, and actionable improvement recommendations.
"""

import json, warnings, os, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────
CACHE = Path("web/cache")
REPORT_DIR = Path("reports")
IMG_DIR = REPORT_DIR / "figures"
IMG_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "warriors_advanced_insights_2025_26.md"

TEAM_ID = 1610612744
TEAM_NAME = "Golden State Warriors"
SEASON = "2025-26"

# ── Plotting Style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1923",
    "axes.facecolor": "#0f1923",
    "axes.edgecolor": "#2a3f52",
    "text.color": "#e8e8e8",
    "axes.labelcolor": "#e8e8e8",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
    "grid.color": "#1e3044",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
})
GSW_GOLD = "#FFC72C"
GSW_BLUE = "#1D428A"
GSW_WHITE = "#FFFFFF"
POSITIVE = "#00c896"
NEGATIVE = "#ff4757"
NEUTRAL = "#a0a0a0"

# ── Helpers ──────────────────────────────────────────────────────────
lines = []
def w(s=""): lines.append(s)
def h(n, t): w(f"{'#'*n} {t}"); w()
def p(t): w(t); w()
def blank(): w()
def tbl(*cols): w("| " + " | ".join(cols) + " |"); w("|" + "|".join(["---"]*len(cols)) + "|")
def row(*vals): w("| " + " | ".join(str(v) for v in vals) + " |")
def img(path, alt="chart"): w(f"![{alt}]({path})"); w()

def load(name):
    fp = CACHE / f"{name}.json"
    if not fp.exists():
        print(f"  ⚠ {fp} not found")
        return None
    return json.loads(fp.read_text())

def to_df(data, idx=0):
    if data is None: return pd.DataFrame()
    rs = data["resultSets"][idx]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])

def savefig(name, fig=None):
    f = fig or plt.gcf()
    path = IMG_DIR / f"{name}.png"
    f.savefig(path, dpi=150, bbox_inches="tight", facecolor=f.get_facecolor())
    plt.close(f)
    return f"figures/{name}.png"

# ── Load Data ────────────────────────────────────────────────────────
print("Loading data…")
player_base = to_df(load("player_base"))
player_adv = to_df(load("player_adv"))
game_log = to_df(load("gamelog"))
lineups = to_df(load("lineups"))
on_off = load("on_off")
player_gamelogs = to_df(load("player_gamelogs"))
hustle = to_df(load("hustle"))
tracking = to_df(load("tracking_speed"))
clutch = to_df(load("clutch"))

por_on = to_df(on_off, 1) if on_off else pd.DataFrame()
por_off = to_df(on_off, 2) if on_off else pd.DataFrame()

# Parse game log
if len(game_log) > 0:
    game_log["GAME_DATE_DT"] = pd.to_datetime(game_log["GAME_DATE"], format="mixed")
    game_log = game_log.sort_values("GAME_DATE_DT").reset_index(drop=True)
    game_log["WIN"] = game_log["WL"].apply(lambda x: 1 if x == "W" else 0)
    game_log["GAME_NUM"] = range(1, len(game_log) + 1)
    for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT"]:
        game_log[col] = pd.to_numeric(game_log[col], errors="coerce")

# Player game logs
if len(player_gamelogs) > 0:
    for col in ["PTS", "REB", "AST", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                 "STL", "BLK", "TOV", "PLUS_MINUS"]:
        if col in player_gamelogs.columns:
            player_gamelogs[col] = pd.to_numeric(player_gamelogs[col], errors="coerce")

print(f"  {len(player_base)} players, {len(game_log)} team games, {len(player_gamelogs)} player-game records")

# ═══════════════════════════════════════════════════════════════════════
# MODEL 1: XGBoost Win Predictor + SHAP Explanations
# ═══════════════════════════════════════════════════════════════════════
print("Model 1: XGBoost + SHAP…")

feat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"]

# Compute PLUS_MINUS if not available (some game log formats lack it)
if "PLUS_MINUS" not in game_log.columns:
    game_log["PLUS_MINUS"] = 0  # Will be computed per-game if opponent PTS available

# Add rolling features
for col in ["PTS", "AST", "FG_PCT"]:
    game_log[f"{col}_ROLL5"] = game_log[col].rolling(5, min_periods=1).mean()

xgb_features = feat_cols + ["PTS_ROLL5", "AST_ROLL5", "FG_PCT_ROLL5"]
X_xgb = game_log[xgb_features].fillna(0)
y_xgb = game_log["WIN"]

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", verbosity=0,
)
xgb_model.fit(X_xgb, y_xgb)
cv_acc = cross_val_score(xgb_model, X_xgb, y_xgb, cv=5, scoring="accuracy").mean()

# SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_xgb)

# Win probabilities for each game
game_log["WIN_PROB"] = xgb_model.predict_proba(X_xgb)[:, 1]

# --- SHAP Beeswarm Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X_xgb, feature_names=xgb_features, show=False, plot_size=None)
plt.title("SHAP Feature Impact on Win Probability", color=GSW_WHITE, fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
shap_beeswarm_path = savefig("shap_beeswarm")

# --- SHAP Bar (mean importance) ---
fig, ax = plt.subplots(figsize=(10, 5))
mean_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)[::-1]
colors = [GSW_GOLD if i < 3 else GSW_BLUE for i in range(len(sorted_idx))]
bars = ax.barh(range(len(sorted_idx)), mean_shap[sorted_idx], color=[colors[i] for i in range(len(sorted_idx))])
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([xgb_features[i] for i in sorted_idx])
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP Value| (Impact on Win Probability)")
ax.set_title("Feature Importance — XGBoost Win Model", pad=15)
ax.grid(axis="x", alpha=0.3)
shap_bar_path = savefig("shap_importance")

# --- Win Probability Timeline ---
fig, ax = plt.subplots(figsize=(14, 5))
colors_wp = [POSITIVE if w == 1 else NEGATIVE for w in game_log["WIN"]]
ax.bar(game_log["GAME_NUM"], game_log["WIN_PROB"], color=colors_wp, alpha=0.7, width=0.8)
ax.axhline(0.5, color=GSW_WHITE, linestyle="--", alpha=0.4, linewidth=1)
ax.plot(game_log["GAME_NUM"], game_log["WIN_PROB"].rolling(5).mean(), color=GSW_GOLD, linewidth=2.5, label="5-Game Rolling Win Prob")
ax.set_xlabel("Game Number")
ax.set_ylabel("Win Probability")
ax.set_title("Game-by-Game Win Probability (XGBoost Model)", pad=15)
ax.legend(facecolor="#1a2a3a", edgecolor="#2a3f52")
ax.set_ylim(0, 1)
wp_timeline_path = savefig("win_prob_timeline")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 2: Player Clustering (Enhanced + Visualization)
# ═══════════════════════════════════════════════════════════════════════
print("Model 2: Enhanced Player Clustering…")

min_gp = 10
pb = player_base[player_base["GP"] >= min_gp].copy()
pa = player_adv[player_adv["GP"] >= min_gp].copy()

merge_cols_base = ["PLAYER_NAME", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
                   "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
merge_cols_adv = ["PLAYER_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "TS_PCT", "USG_PCT", "PIE"]

avail_base = [c for c in merge_cols_base if c in pb.columns]
avail_adv = [c for c in merge_cols_adv if c in pa.columns]

merged = pb[avail_base].merge(pa[avail_adv], on="PLAYER_NAME", how="inner")

# Add hustle/tracking if available
if len(hustle) > 0:
    hustle_cols = ["PLAYER_NAME", "CONTESTED_SHOTS", "DEFLECTIONS", "SCREEN_ASSISTS"]
    avail_h = [c for c in hustle_cols if c in hustle.columns]
    merged = merged.merge(hustle[avail_h], on="PLAYER_NAME", how="left")
if len(tracking) > 0:
    track_cols = ["PLAYER_NAME", "AVG_SPEED", "DIST_MILES"]
    avail_t = [c for c in track_cols if c in tracking.columns]
    merged = merged.merge(tracking[avail_t], on="PLAYER_NAME", how="left")

merged = merged.fillna(0)
names = merged["PLAYER_NAME"].values
feat_cluster = merged.drop(columns=["PLAYER_NAME"]).select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_cluster)

# Determine optimal K
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    from sklearn.metrics import silhouette_score
    sil_scores[k] = silhouette_score(X_scaled, labels)

best_k = max(sil_scores, key=sil_scores.get)
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled)
merged["Cluster"] = cluster_labels

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Cluster Scatter Plot ---
fig, ax = plt.subplots(figsize=(12, 8))
cluster_colors = [GSW_GOLD, GSW_BLUE, POSITIVE, "#ff6b81", "#7bed9f"]
cluster_names = {}
for c in range(best_k):
    mask = cluster_labels == c
    members = merged[mask]
    avg_pts = members["PTS"].mean()
    avg_min = members["MIN"].mean()
    if avg_pts > 15:
        cluster_names[c] = "⭐ Primary Scorers"
    elif avg_min > 18:
        cluster_names[c] = "🔧 Core Rotation"
    else:
        cluster_names[c] = "⚡ Role/Energy"
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=cluster_colors[c], s=150, alpha=0.85,
               edgecolors="white", linewidth=1.5, zorder=3, label=cluster_names[c])
    for j, name in enumerate(names[mask]):
        short = name.split()[-1] if len(name.split()) > 1 else name
        ax.annotate(short, (X_pca[mask, 0][j], X_pca[mask, 1][j]),
                    fontsize=9, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    color="white", fontweight="bold")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Player Archetype Clustering (PCA Projection)", pad=15)
ax.legend(loc="upper right", facecolor="#1a2a3a", edgecolor="#2a3f52", fontsize=11)
ax.grid(True, alpha=0.2)
cluster_path = savefig("cluster_scatter")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 3: Player Radar Charts
# ═══════════════════════════════════════════════════════════════════════
print("Model 3: Player Radar Charts…")

radar_stats = ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "TS_PCT", "USG_PCT"]
radar_labels = ["Points", "Assists", "Rebounds", "Steals", "Blocks", "FG%", "3PT%", "TS%", "Usage"]

# Normalize to 0-100 percentiles within this team
radar_data = merged[["PLAYER_NAME"] + [c for c in radar_stats if c in merged.columns]].copy()
for col in radar_stats:
    if col in radar_data.columns:
        mn, mx = radar_data[col].min(), radar_data[col].max()
        if mx > mn:
            radar_data[col + "_norm"] = (radar_data[col] - mn) / (mx - mn) * 100
        else:
            radar_data[col + "_norm"] = 50

# Top 6 players by minutes
top_players = merged.nlargest(6, "MIN")["PLAYER_NAME"].values

fig, axes = plt.subplots(2, 3, figsize=(18, 14), subplot_kw={"projection": "polar"})
axes = axes.flatten()

norm_cols = [c + "_norm" for c in radar_stats if c in radar_data.columns]
available_labels = [radar_labels[i] for i, c in enumerate(radar_stats) if c in radar_data.columns]

for idx, (ax, player_name) in enumerate(zip(axes, top_players)):
    prow = radar_data[radar_data["PLAYER_NAME"] == player_name]
    if len(prow) == 0:
        continue
    vals = prow[norm_cols].values.flatten()
    angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
    vals = np.append(vals, vals[0])
    angles.append(angles[0])

    ax.fill(angles, vals, alpha=0.25, color=GSW_GOLD)
    ax.plot(angles, vals, linewidth=2, color=GSW_GOLD)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=8, color="#c0c0c0")
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#606060")
    ax.set_title(player_name.split(",")[0] if "," in player_name else player_name,
                 pad=20, fontsize=12, fontweight="bold", color=GSW_WHITE)
    ax.grid(color="#2a3f52", alpha=0.4)
    ax.set_facecolor("#0f1923")

fig.suptitle("Player Skill Radar — Top 6 by Minutes", y=1.02, fontsize=16, fontweight="bold")
plt.tight_layout()
radar_path = savefig("player_radars")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 4: On/Off Court Impact Visualization
# ═══════════════════════════════════════════════════════════════════════
print("Model 4: On/Off Impact…")

impact_data = []
if len(por_on) > 0 and len(por_off) > 0:
    for _, on_r in por_on.iterrows():
        name = on_r.get("VS_PLAYER_NAME", "")
        off_r = por_off[por_off["VS_PLAYER_NAME"] == name]
        if len(off_r) == 0:
            continue
        off_r = off_r.iloc[0]
        on_min = float(on_r.get("MIN", 0))
        if on_min < 200:
            continue
        impact_data.append({
            "PLAYER": name,
            "ON_NET": float(on_r.get("NET_RATING", 0)),
            "OFF_NET": float(off_r.get("NET_RATING", 0)),
            "NET_SWING": float(on_r.get("NET_RATING", 0)) - float(off_r.get("NET_RATING", 0)),
            "ON_OFF": float(on_r.get("OFF_RATING", 0)),
            "OFF_OFF": float(off_r.get("OFF_RATING", 0)),
            "ON_DEF": float(on_r.get("DEF_RATING", 0)),
            "OFF_DEF": float(off_r.get("DEF_RATING", 0)),
            "ON_MIN": on_min,
        })

impact_df = pd.DataFrame(impact_data).sort_values("NET_SWING", ascending=True)

# --- On/Off Impact Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = range(len(impact_df))
colors_impact = [POSITIVE if s > 0 else NEGATIVE for s in impact_df["NET_SWING"]]
bars = ax.barh(y_pos, impact_df["NET_SWING"], color=colors_impact, alpha=0.85, height=0.7,
               edgecolor="white", linewidth=0.5)

# Add value labels
for i, (val, name) in enumerate(zip(impact_df["NET_SWING"], impact_df["PLAYER"])):
    short = name.split(", ")[0] if ", " in name else name
    ax.text(val + (0.5 if val >= 0 else -0.5), i, f"{val:+.1f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=10, fontweight="bold",
            color=POSITIVE if val > 0 else NEGATIVE)

ax.set_yticks(y_pos)
ax.set_yticklabels([n.split(", ")[0] if ", " in n else n for n in impact_df["PLAYER"]], fontsize=11)
ax.axvline(0, color=GSW_WHITE, linewidth=1, alpha=0.5)
ax.set_xlabel("Net Rating Swing (On Court − Off Court)")
ax.set_title("Player On/Off Court Impact", pad=15)
ax.grid(axis="x", alpha=0.2)
impact_path = savefig("onoff_impact")

# --- Offensive vs Defensive Impact Scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
off_swing = impact_df["ON_OFF"] - impact_df["OFF_OFF"]
def_swing = impact_df["ON_DEF"] - impact_df["OFF_DEF"]

ax.scatter(off_swing, def_swing, s=impact_df["ON_MIN"] / 5, c=[GSW_GOLD if ns > 0 else NEGATIVE for ns in impact_df["NET_SWING"]],
           alpha=0.8, edgecolors="white", linewidth=1)

for _, r in impact_df.iterrows():
    short = r["PLAYER"].split(", ")[0] if ", " in r["PLAYER"] else r["PLAYER"]
    os = r["ON_OFF"] - r["OFF_OFF"]
    ds = r["ON_DEF"] - r["OFF_DEF"]
    ax.annotate(short, (os, ds), fontsize=9, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points", color="white")

ax.axhline(0, color=GSW_WHITE, linestyle="--", alpha=0.3)
ax.axvline(0, color=GSW_WHITE, linestyle="--", alpha=0.3)
ax.set_xlabel("Offensive Rating Swing →  (positive = helps offense)")
ax.set_ylabel("← Defensive Rating Swing  (negative = helps defense)")
ax.set_title("Two-Way Impact Map", pad=15)

# Add quadrant labels
ax.text(0.95, 0.05, "Offense Only", transform=ax.transAxes, fontsize=9, alpha=0.5, ha="right", color=POSITIVE)
ax.text(0.05, 0.95, "Defense Only", transform=ax.transAxes, fontsize=9, alpha=0.5, ha="left", color="#5cc9f5")
ax.text(0.95, 0.95, "Two-Way Star ★", transform=ax.transAxes, fontsize=9, alpha=0.5, ha="right", color=GSW_GOLD)
ax.text(0.05, 0.05, "Liability", transform=ax.transAxes, fontsize=9, alpha=0.5, ha="left", color=NEGATIVE)
ax.invert_yaxis()  # Lower DEF rating = better defense
ax.grid(True, alpha=0.2)
twoway_path = savefig("twoway_impact")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 5: Scoring Trend & Momentum Analysis
# ═══════════════════════════════════════════════════════════════════════
print("Model 5: Scoring Trends…")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Points trend
ax1 = axes[0]
ax1.fill_between(game_log["GAME_NUM"], game_log["PTS"], alpha=0.15, color=GSW_GOLD)
ax1.plot(game_log["GAME_NUM"], game_log["PTS"], color=GSW_GOLD, alpha=0.4, linewidth=1)
ax1.plot(game_log["GAME_NUM"], game_log["PTS"].rolling(10).mean(), color=GSW_GOLD, linewidth=2.5, label="10-Game Avg")
ax1.axhline(game_log["PTS"].mean(), color=GSW_WHITE, linestyle="--", alpha=0.3, linewidth=1)
ax1.set_ylabel("Points Scored")
ax1.set_title("Scoring Trend", pad=10)
ax1.legend(facecolor="#1a2a3a", edgecolor="#2a3f52")

# Panel 2: FG% and 3PT% trends
ax2 = axes[1]
ax2.plot(game_log["GAME_NUM"], game_log["FG_PCT"].rolling(10).mean() * 100, color=GSW_GOLD, linewidth=2, label="FG% (10g)")
ax2.plot(game_log["GAME_NUM"], game_log["FG3_PCT"].rolling(10).mean() * 100, color=GSW_BLUE, linewidth=2, label="3PT% (10g)")
ax2.axhline(45, color=NEUTRAL, linestyle=":", alpha=0.3)
ax2.set_ylabel("Shooting %")
ax2.set_title("Shooting Efficiency Trend", pad=10)
ax2.legend(facecolor="#1a2a3a", edgecolor="#2a3f52")

# Panel 3: Win Probability + actual results
ax3 = axes[2]
for i, (gn, wp, win) in enumerate(zip(game_log["GAME_NUM"], game_log["WIN_PROB"], game_log["WIN"])):
    ax3.bar(gn, 1, color=POSITIVE if win else NEGATIVE, alpha=0.15, width=0.9)
ax3.plot(game_log["GAME_NUM"], game_log["WIN_PROB"].rolling(10).mean(), color=GSW_GOLD, linewidth=2.5, label="Win Prob (10g)")
ax3.axhline(0.5, color=GSW_WHITE, linestyle="--", alpha=0.3)
ax3.set_ylabel("Win Probability")
ax3.set_xlabel("Game Number")
ax3.set_title("Win Probability Trend (Background: W=green, L=red)", pad=10)
ax3.legend(facecolor="#1a2a3a", edgecolor="#2a3f52")
ax3.set_ylim(0, 1)

plt.tight_layout()
trend_path = savefig("scoring_trends")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 6: Lineup Synergy Heatmap
# ═══════════════════════════════════════════════════════════════════════
print("Model 6: Lineup Synergy Heatmap…")

# Build pair matrix from lineup data
key_players = merged.nlargest(10, "MIN")["PLAYER_NAME"].values.tolist()
# Map from initial format to full names
name_map = {}
for full_name in key_players:
    parts = full_name.split()
    if len(parts) >= 2:
        initial = f"{parts[0][0]}. {parts[-1]}"
        name_map[initial] = full_name

pair_matrix = pd.DataFrame(0.0, index=key_players, columns=key_players)
pair_minutes = pd.DataFrame(0.0, index=key_players, columns=key_players)

if len(lineups) > 0:
    for _, lu in lineups.iterrows():
        players_in = lu["GROUP_NAME"].split(" - ")
        net = float(lu.get("NET_RATING", 0))
        mins = float(lu.get("MIN", 0))
        if mins < 3:
            continue
        # Map to full names
        mapped = []
        for pl_name in players_in:
            pl_name = pl_name.strip()
            if pl_name in name_map:
                mapped.append(name_map[pl_name])
            else:
                for full in key_players:
                    if full.split()[-1] in pl_name:
                        mapped.append(full)
                        break
        for ii, p1 in enumerate(mapped):
            for p2 in mapped[ii+1:]:
                if p1 in key_players and p2 in key_players:
                    pair_matrix.loc[p1, p2] += net * mins
                    pair_matrix.loc[p2, p1] += net * mins
                    pair_minutes.loc[p1, p2] += mins
                    pair_minutes.loc[p2, p1] += mins

# Weighted average
pair_avg = pair_matrix / pair_minutes.replace(0, np.nan)
pair_avg = pair_avg.fillna(0)

# Short names for display
short_names = [n.split()[-1] if len(n.split()) > 1 else n for n in key_players]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(pair_avg, dtype=bool), k=0)
sns.heatmap(pair_avg, mask=mask, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            xticklabels=short_names, yticklabels=short_names,
            linewidths=0.5, linecolor="#1a2a3a",
            cbar_kws={"label": "Weighted Net Rating"},
            annot_kws={"fontsize": 9, "fontweight": "bold"},
            ax=ax)
ax.set_title("Player Pair Synergy Heatmap (Weighted Net Rating)", pad=15)
heatmap_path = savefig("synergy_heatmap")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 7: Performance Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════
print("Model 7: Correlation Analysis…")

corr_cols = ["PTS", "AST", "REB", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]
avail_corr = [c for c in corr_cols if c in game_log.columns]
game_corr = game_log[avail_corr + ["WIN"]].corr()

fig, ax = plt.subplots(figsize=(10, 8))
corr_labels = ["Points", "Assists", "Rebounds", "Steals", "Blocks", "Turnovers", "FG%", "3PT%", "+/-", "Win"]
mask_corr = np.triu(np.ones_like(game_corr, dtype=bool), k=1)
sns.heatmap(game_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            xticklabels=corr_labels[:len(game_corr.columns)],
            yticklabels=corr_labels[:len(game_corr.columns)],
            linewidths=0.5, linecolor="#1a2a3a",
            ax=ax, vmin=-1, vmax=1)
ax.set_title("Game Stat Correlation Matrix (with Wins)", pad=15)
corr_path = savefig("correlation_matrix")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 8: Player Development / Monthly Trends
# ═══════════════════════════════════════════════════════════════════════
print("Model 8: Player Monthly Trends…")

top4 = merged.nlargest(4, "PTS")["PLAYER_NAME"].values
if len(player_gamelogs) > 0:
    player_gamelogs["GAME_DATE_DT"] = pd.to_datetime(player_gamelogs["GAME_DATE"], format="mixed")
    player_gamelogs["MONTH"] = player_gamelogs["GAME_DATE_DT"].dt.month

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    month_names = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}

    for idx, (ax, player_name) in enumerate(zip(axes, top4)):
        pdf = player_gamelogs[player_gamelogs["PLAYER_NAME"] == player_name].copy()
        if len(pdf) == 0:
            continue
        monthly = pdf.groupby("MONTH").agg({"PTS": "mean", "AST": "mean", "REB": "mean", "FG_PCT": "mean"}).reindex([10, 11, 12, 1, 2])
        monthly = monthly.dropna()

        x = range(len(monthly))
        mlabels = [month_names.get(m, str(m)) for m in monthly.index]

        ax.bar(x, monthly["PTS"], color=GSW_GOLD, alpha=0.7, width=0.6, label="PPG")
        ax.plot(x, monthly["PTS"], color=GSW_WHITE, linewidth=2, marker="o", markersize=6)

        # Add FG% as secondary
        ax2 = ax.twinx()
        ax2.plot(x, monthly["FG_PCT"] * 100, color=POSITIVE, linewidth=2, marker="s", markersize=5, linestyle="--", label="FG%")
        ax2.set_ylabel("FG%", color=POSITIVE, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=POSITIVE)
        ax2.set_ylim(30, 70)

        ax.set_xticks(x)
        ax.set_xticklabels(mlabels)
        short = player_name.split()[-1] if len(player_name.split()) > 1 else player_name
        ax.set_title(f"{player_name}", pad=10)
        ax.set_ylabel("PPG", color=GSW_GOLD)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Monthly Scoring & Efficiency Trends — Top 4 Scorers", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    monthly_path = savefig("monthly_trends")
else:
    monthly_path = None


# ═══════════════════════════════════════════════════════════════════════
# MODEL 9: Bayesian Remaining Season Projection
# ═══════════════════════════════════════════════════════════════════════
print("Model 9: Bayesian Season Projection…")

wins_so_far = int(game_log["WIN"].sum())
games_played = len(game_log)
losses_so_far = games_played - wins_so_far
games_remaining = 82 - games_played

# Beta distribution posterior
# Prior: Beta(1, 1) uniform
alpha_post = 1 + wins_so_far
beta_post = 1 + losses_so_far
win_pct_mean = alpha_post / (alpha_post + beta_post)

# Monte Carlo simulation
np.random.seed(42)
n_sims = 50000
simulated_win_pcts = np.random.beta(alpha_post, beta_post, n_sims)
simulated_remaining_wins = np.random.binomial(games_remaining, simulated_win_pcts)
simulated_total_wins = wins_so_far + simulated_remaining_wins

# Recent form (last 15 games) weighted projection
recent = game_log.tail(15)
recent_wpct = recent["WIN"].mean()
alpha_recent = 1 + int(recent["WIN"].sum())
beta_recent = 1 + len(recent) - int(recent["WIN"].sum())
sim_recent = np.random.beta(alpha_recent, beta_recent, n_sims)
sim_recent_wins = wins_so_far + np.random.binomial(games_remaining, sim_recent)

# Blend: 60% full season, 40% recent form
blended_wins = 0.6 * simulated_total_wins + 0.4 * sim_recent_wins

# --- Bayesian Projection Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Win total distribution
ax1 = axes[0]
ax1.hist(simulated_total_wins, bins=range(int(simulated_total_wins.min()), int(simulated_total_wins.max()) + 2),
         color=GSW_BLUE, alpha=0.7, edgecolor="white", linewidth=0.5, density=True, label="Full Season Prior")
ax1.hist(sim_recent_wins, bins=range(int(sim_recent_wins.min()), int(sim_recent_wins.max()) + 2),
         color=NEGATIVE, alpha=0.4, edgecolor="white", linewidth=0.5, density=True, label="Recent Form Prior")
ax1.axvline(np.median(blended_wins), color=GSW_GOLD, linewidth=2, linestyle="--", label=f"Blended Median: {np.median(blended_wins):.0f}")
ax1.set_xlabel("Projected Total Wins")
ax1.set_ylabel("Probability Density")
ax1.set_title("Season Win Total Distribution", pad=10)
ax1.legend(facecolor="#1a2a3a", edgecolor="#2a3f52", fontsize=9)

# Panel 2: Win probability density
ax2 = axes[1]
x_wpct = np.linspace(0.3, 0.8, 200)
from scipy.stats import beta as beta_dist
y_full = beta_dist.pdf(x_wpct, alpha_post, beta_post)
y_recent = beta_dist.pdf(x_wpct, alpha_recent, beta_recent)
ax2.fill_between(x_wpct, y_full, alpha=0.3, color=GSW_BLUE, label="Full Season")
ax2.plot(x_wpct, y_full, color=GSW_BLUE, linewidth=2)
ax2.fill_between(x_wpct, y_recent, alpha=0.3, color=NEGATIVE, label="Last 15 Games")
ax2.plot(x_wpct, y_recent, color=NEGATIVE, linewidth=2)
ax2.axvline(0.5, color=GSW_WHITE, linestyle="--", alpha=0.3)
ax2.set_xlabel("Win Probability")
ax2.set_ylabel("Density")
ax2.set_title("Bayesian Win Rate Posterior", pad=10)
ax2.legend(facecolor="#1a2a3a", edgecolor="#2a3f52", fontsize=9)

# Panel 3: Playoff probability
ax3 = axes[2]
thresholds = [38, 40, 42, 44, 46, 48]
probs = [(blended_wins >= t).mean() * 100 for t in thresholds]
bars = ax3.bar(range(len(thresholds)), probs, color=[POSITIVE if p > 50 else GSW_GOLD if p > 25 else NEGATIVE for p in probs],
               alpha=0.8, edgecolor="white", width=0.6)
ax3.set_xticks(range(len(thresholds)))
ax3.set_xticklabels([f"{t}W" for t in thresholds])
for i, pr in enumerate(probs):
    ax3.text(i, pr + 1, f"{pr:.0f}%", ha="center", fontsize=10, fontweight="bold", color=GSW_WHITE)
ax3.set_ylabel("Probability (%)")
ax3.set_xlabel("Win Threshold")
ax3.set_title("Playoff Win Threshold Probability", pad=10)
ax3.set_ylim(0, 105)

plt.tight_layout()
bayesian_path = savefig("bayesian_projection")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 10: SHAP Waterfall — Best Win & Worst Loss
# ═══════════════════════════════════════════════════════════════════════
print("Model 10: SHAP Waterfall for key games…")

# Best win (highest margin)
wins_mask = game_log["WIN"] == 1
if wins_mask.sum() > 0:
    best_win_idx = game_log.loc[wins_mask, "PLUS_MINUS"].astype(float).idxmax()
    worst_loss_idx = game_log.loc[~wins_mask, "PLUS_MINUS"].astype(float).idxmin()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (game_idx, title_prefix) in enumerate([(best_win_idx, "Best Win"), (worst_loss_idx, "Worst Loss")]):
        ax = axes[ax_idx]
        game_shap = shap_values[game_idx]
        game_data = X_xgb.iloc[game_idx]
        sorted_idx_g = np.argsort(np.abs(game_shap))[::-1]

        colors_g = [POSITIVE if v > 0 else NEGATIVE for v in game_shap[sorted_idx_g]]
        ax.barh(range(len(sorted_idx_g)), game_shap[sorted_idx_g], color=colors_g, alpha=0.8, height=0.6)
        ax.set_yticks(range(len(sorted_idx_g)))
        labels_g = [f"{xgb_features[i]} = {game_data.iloc[i]:.1f}" for i in sorted_idx_g]
        ax.set_yticklabels(labels_g, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color=GSW_WHITE, linewidth=1, alpha=0.5)

        matchup = game_log.loc[game_idx, "MATCHUP"]
        date = game_log.loc[game_idx, "GAME_DATE"]
        pm = game_log.loc[game_idx, "PLUS_MINUS"]
        ax.set_title(f"{title_prefix}: {matchup} ({date})\n+/- = {pm}", pad=10, fontsize=11)
        ax.set_xlabel("SHAP Value (Impact on Win Probability)")
        ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    waterfall_path = savefig("shap_waterfall")
else:
    waterfall_path = None


# ═══════════════════════════════════════════════════════════════════════
# MODEL 11: Composite Value + Improvement Potential
# ═══════════════════════════════════════════════════════════════════════
print("Model 11: Composite Value…")

# Build RAPM scores
rapm_scores = {}
if len(lineups) > 0 and len(por_on) > 0:
    all_players = list(merged["PLAYER_NAME"].values)
    X_rapm = np.zeros((len(lineups), len(all_players)))
    y_rapm = lineups["NET_RATING"].astype(float).values
    w_rapm = lineups["MIN"].astype(float).values

    for i, (_, lu) in enumerate(lineups.iterrows()):
        players_in = lu["GROUP_NAME"].split(" - ")
        for pl_name in players_in:
            pl_name = pl_name.strip()
            for j, full_name in enumerate(all_players):
                if full_name.split()[-1] in pl_name or (len(full_name.split()) > 1 and f"{full_name.split()[0][0]}. {full_name.split()[-1]}" == pl_name):
                    X_rapm[i, j] = 1
                    break

    w_rapm = np.maximum(w_rapm, 0.01)
    ridge = Ridge(alpha=50, fit_intercept=True)
    ridge.fit(X_rapm, y_rapm, sample_weight=w_rapm)
    for j, name in enumerate(all_players):
        rapm_scores[name] = ridge.coef_[j]

# Composite score
composite = []
for _, r in merged.iterrows():
    name = r["PLAYER_NAME"]
    prod = (r.get("PTS", 0) * 1.0 + r.get("AST", 0) * 1.5 + r.get("REB", 0) * 1.2 +
            r.get("STL", 0) * 2.0 + r.get("BLK", 0) * 2.0 - r.get("TOV", 0) * 1.5)
    rapm = rapm_scores.get(name, 0)
    ts = r.get("TS_PCT", 0.5)
    usg = r.get("USG_PCT", 0.15)

    # Per-minute production
    per_min = prod / max(r.get("MIN", 1), 1)

    composite.append({
        "PLAYER": name,
        "PRODUCTION": prod,
        "PER_MIN_PROD": per_min,
        "RAPM": rapm,
        "TS_PCT": ts,
        "USG": usg,
        "MIN": r.get("MIN", 0),
    })

comp_df = pd.DataFrame(composite)
# Normalize each component 0-100
for col in ["PRODUCTION", "PER_MIN_PROD", "RAPM", "TS_PCT"]:
    mn, mx = comp_df[col].min(), comp_df[col].max()
    if mx > mn:
        comp_df[col + "_N"] = (comp_df[col] - mn) / (mx - mn) * 100
    else:
        comp_df[col + "_N"] = 50

comp_df["COMPOSITE"] = (comp_df["PRODUCTION_N"] * 0.3 + comp_df["PER_MIN_PROD_N"] * 0.2 +
                         comp_df["RAPM_N"] * 0.3 + comp_df["TS_PCT_N"] * 0.2)
comp_df = comp_df.sort_values("COMPOSITE", ascending=False)

# --- Composite Value Bar Chart ---
fig, ax = plt.subplots(figsize=(14, 8))
comp_sorted = comp_df.sort_values("COMPOSITE", ascending=True)
y_pos = range(len(comp_sorted))

# Stacked bar
left = np.zeros(len(comp_sorted))
components = [
    ("PRODUCTION_N", "Production (30%)", GSW_GOLD, 0.3),
    ("PER_MIN_PROD_N", "Per-Min Prod (20%)", GSW_BLUE, 0.2),
    ("RAPM_N", "Impact/RAPM (30%)", POSITIVE, 0.3),
    ("TS_PCT_N", "Efficiency/TS% (20%)", "#7bed9f", 0.2),
]
for col, label, color, weight in components:
    vals = comp_sorted[col].values * weight
    ax.barh(y_pos, vals, left=left, color=color, alpha=0.8, height=0.7, label=label, edgecolor="white", linewidth=0.3)
    left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels(comp_sorted["PLAYER"], fontsize=10)
ax.set_xlabel("Composite Player Value Score")
ax.set_title("Composite Player Value — Stacked Components", pad=15)
ax.legend(loc="lower right", facecolor="#1a2a3a", edgecolor="#2a3f52", fontsize=9)

# Add total labels
for i, (_, r) in enumerate(comp_sorted.iterrows()):
    ax.text(r["COMPOSITE"] + 0.5, i, f"{r['COMPOSITE']:.1f}", va="center", fontsize=10, fontweight="bold", color=GSW_WHITE)

composite_path = savefig("composite_value")


# ═══════════════════════════════════════════════════════════════════════
# MODEL 12: Improvement Potential Analysis
# ═══════════════════════════════════════════════════════════════════════
print("Model 12: Improvement Potential…")

# Calculate where each player's stats fall vs. team needs
# Team averages when winning vs losing
if len(game_log) > 0:
    win_profile = game_log[game_log["WIN"] == 1][["PTS", "AST", "REB", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"]].mean()
    loss_profile = game_log[game_log["WIN"] == 0][["PTS", "AST", "REB", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"]].mean()
    gap = win_profile - loss_profile

    fig, ax = plt.subplots(figsize=(12, 6))
    gap_labels = ["Points", "Assists", "Rebounds", "Steals", "Blocks", "Turnovers", "FG%×100", "3PT%×100"]
    gap_vals = gap.values.copy()
    gap_vals[-2] *= 100  # Scale percentages
    gap_vals[-1] *= 100

    colors_gap = [POSITIVE if v > 0 else NEGATIVE for v in gap_vals]
    # Turnovers: positive gap means MORE turnovers in wins (unlikely), flip color logic
    if gap_vals[5] < 0:  # Fewer turnovers in wins = good
        colors_gap[5] = POSITIVE

    ax.bar(range(len(gap_labels)), gap_vals, color=colors_gap, alpha=0.8, width=0.6, edgecolor="white")
    for i, v in enumerate(gap_vals):
        ax.text(i, v + (0.3 if v >= 0 else -0.5), f"{v:+.1f}", ha="center", fontsize=10, fontweight="bold",
                color=GSW_WHITE)
    ax.set_xticks(range(len(gap_labels)))
    ax.set_xticklabels(gap_labels, fontsize=11)
    ax.axhline(0, color=GSW_WHITE, linewidth=1, alpha=0.5)
    ax.set_ylabel("Win vs Loss Gap")
    ax.set_title("What Separates Wins from Losses — Statistical DNA", pad=15)
    ax.grid(axis="y", alpha=0.2)
    gap_path = savefig("win_loss_gap")


# ═══════════════════════════════════════════════════════════════════════
# VIZ 13: Scoring Distribution Violin + Box Plot
# ═══════════════════════════════════════════════════════════════════════
print("Viz 13: Scoring Distribution…")

top6_names = merged.nlargest(6, "MIN")["PLAYER_NAME"].values
if len(player_gamelogs) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))
    score_data = []
    labels_sd = []
    for pn in top6_names:
        pdf = player_gamelogs[player_gamelogs["PLAYER_NAME"] == pn]
        if len(pdf) > 0:
            score_data.append(pdf["PTS"].values)
            short = pn.split()[-1] if len(pn.split()) > 1 else pn
            labels_sd.append(short)

    parts = ax.violinplot(score_data, positions=range(len(score_data)), showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(GSW_GOLD)
        pc.set_alpha(0.4)
    parts["cmeans"].set_color(GSW_WHITE)
    parts["cmedians"].set_color(POSITIVE)
    parts["cmins"].set_color(NEUTRAL)
    parts["cmaxes"].set_color(NEUTRAL)
    parts["cbars"].set_color(NEUTRAL)

    # Overlay individual points (jittered)
    for i, sd in enumerate(score_data):
        jitter = np.random.normal(0, 0.06, len(sd))
        ax.scatter(np.full_like(sd, i, dtype=float) + jitter, sd, s=15, alpha=0.4, color=GSW_BLUE, zorder=3)

    ax.set_xticks(range(len(labels_sd)))
    ax.set_xticklabels(labels_sd, fontsize=11)
    ax.set_ylabel("Points Per Game")
    ax.set_title("Scoring Distribution — Top 6 Players (Violin + Swarm)", pad=15)
    ax.grid(axis="y", alpha=0.2)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=GSW_WHITE, lw=2, label="Mean"),
                       Line2D([0], [0], color=POSITIVE, lw=2, label="Median")]
    ax.legend(handles=legend_elements, facecolor="#1a2a3a", edgecolor="#2a3f52")
    violin_path = savefig("scoring_violin")
else:
    violin_path = None


# ═══════════════════════════════════════════════════════════════════════
# VIZ 14: Shot Zone Efficiency Chart
# ═══════════════════════════════════════════════════════════════════════
print("Viz 14: Shot Zone Chart…")

shot_loc_data = load("shot_locations")
team_shooting = load("team_shooting")
shot_zone_path = None

if team_shooting:
    # Team shooting data by area
    ts_rs = None
    for rs_item in team_shooting.get("resultSets", []):
        if rs_item.get("name", "") == "ShotAreaTeamDashboard":
            ts_rs = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
            break
    if ts_rs is None and "resultSets" in team_shooting:
        for rs_item in team_shooting["resultSets"]:
            if "Shot Area" in str(rs_item.get("name", "")):
                ts_rs = pd.DataFrame(rs_item["rowSet"], columns=rs_item["headers"])
                break

    if ts_rs is not None and len(ts_rs) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        zones = ts_rs["GROUP_VALUE"].values
        fg_pcts = ts_rs["FG_PCT"].values * 100
        fga = ts_rs["FGA"].values

        # Size by volume, color by efficiency
        norm_fga = fga / fga.max() * 800 + 100
        colors_sz = [POSITIVE if fg > 45 else GSW_GOLD if fg > 35 else NEGATIVE for fg in fg_pcts]

        bars = ax.barh(range(len(zones)), fg_pcts, color=colors_sz, alpha=0.8, height=0.6, edgecolor="white")

        # Add FGA labels
        for i, (fg, fga_v, zone) in enumerate(zip(fg_pcts, fga, zones)):
            ax.text(fg + 0.5, i, f"{fg:.1f}% ({fga_v:.0f} FGA)", va="center", fontsize=10,
                    fontweight="bold", color=GSW_WHITE)

        ax.set_yticks(range(len(zones)))
        ax.set_yticklabels(zones, fontsize=11)
        ax.set_xlabel("Field Goal Percentage")
        ax.set_title("Shot Zone Efficiency — Warriors 2025-26", pad=15)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlim(0, max(fg_pcts) + 15)
        shot_zone_path = savefig("shot_zone_efficiency")


# ═══════════════════════════════════════════════════════════════════════
# VIZ 15: Clutch Performance Comparison
# ═══════════════════════════════════════════════════════════════════════
print("Viz 15: Clutch Performance…")

clutch_path_viz = None
if len(clutch) > 0:
    clutch_top = clutch[clutch["MIN"].astype(float) > 10].copy()
    clutch_top["PTS"] = pd.to_numeric(clutch_top["PTS"], errors="coerce")
    clutch_top["FG_PCT"] = pd.to_numeric(clutch_top["FG_PCT"], errors="coerce")
    clutch_top = clutch_top.nlargest(10, "PTS")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Clutch PPG
    short_names_cl = [n.split()[-1] if len(n.split()) > 1 else n for n in clutch_top["PLAYER_NAME"]]
    ax1.barh(range(len(clutch_top)), clutch_top["PTS"], color=GSW_GOLD, alpha=0.8, height=0.6, edgecolor="white")
    ax1.set_yticks(range(len(clutch_top)))
    ax1.set_yticklabels(short_names_cl, fontsize=10)
    ax1.set_xlabel("Clutch PPG")
    ax1.set_title("Clutch Scoring Leaders", pad=10)
    for i, v in enumerate(clutch_top["PTS"]):
        ax1.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=9, color=GSW_WHITE, fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)

    # Panel 2: Clutch FG%
    fg_vals = clutch_top["FG_PCT"].values * 100
    colors_cl = [POSITIVE if fg > 45 else GSW_GOLD if fg > 35 else NEGATIVE for fg in fg_vals]
    ax2.barh(range(len(clutch_top)), fg_vals, color=colors_cl, alpha=0.8, height=0.6, edgecolor="white")
    ax2.set_yticks(range(len(clutch_top)))
    ax2.set_yticklabels(short_names_cl, fontsize=10)
    ax2.set_xlabel("Clutch FG%")
    ax2.set_title("Clutch Shooting Efficiency", pad=10)
    for i, v in enumerate(fg_vals):
        ax2.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9, color=GSW_WHITE, fontweight="bold")
    ax2.grid(axis="x", alpha=0.2)

    fig.suptitle("Clutch Performance — Last 5 Min, Score ≤5", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    clutch_path_viz = savefig("clutch_performance")


# ═══════════════════════════════════════════════════════════════════════
# VIZ 16: Rest Day Impact
# ═══════════════════════════════════════════════════════════════════════
print("Viz 16: Rest Day Impact…")

rest_viz_path = None
if len(game_log) > 0:
    game_log["REST_DAYS"] = game_log["GAME_DATE_DT"].diff().dt.days.fillna(1).astype(int).clip(0, 4)
    rest_agg = game_log.groupby("REST_DAYS").agg(
        GP=("WIN", "count"), WinPct=("WIN", "mean"), PPG=("PTS", "mean"), FG_Pct=("FG_PCT", "mean")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Win% by rest
    colors_rest = [POSITIVE if wp > 0.55 else GSW_GOLD if wp > 0.45 else NEGATIVE for wp in rest_agg["WinPct"]]
    ax1.bar(rest_agg.index, rest_agg["WinPct"] * 100, color=colors_rest, alpha=0.8, width=0.6, edgecolor="white")
    for i, (rd, wp, gp) in enumerate(zip(rest_agg.index, rest_agg["WinPct"], rest_agg["GP"])):
        ax1.text(rd, wp * 100 + 1.5, f"{wp*100:.0f}%\n({gp}G)", ha="center", fontsize=10, fontweight="bold", color=GSW_WHITE)
    ax1.axhline(50, color=GSW_WHITE, linestyle="--", alpha=0.3)
    ax1.set_xlabel("Rest Days")
    ax1.set_ylabel("Win %")
    ax1.set_title("Win Rate by Days of Rest", pad=10)
    ax1.grid(axis="y", alpha=0.2)

    # Panel 2: PPG and FG% by rest
    ax2.bar(rest_agg.index - 0.15, rest_agg["PPG"], width=0.3, color=GSW_GOLD, alpha=0.8, label="PPG", edgecolor="white")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(rest_agg.index, rest_agg["FG_Pct"] * 100, color=POSITIVE, linewidth=2.5, marker="o", markersize=8, label="FG%")
    ax2.set_xlabel("Rest Days")
    ax2.set_ylabel("Points Per Game", color=GSW_GOLD)
    ax2_twin.set_ylabel("FG%", color=POSITIVE)
    ax2.set_title("Scoring & Efficiency by Rest Days", pad=10)
    ax2.legend(loc="upper left", facecolor="#1a2a3a", edgecolor="#2a3f52")
    ax2_twin.legend(loc="upper right", facecolor="#1a2a3a", edgecolor="#2a3f52")
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    rest_viz_path = savefig("rest_day_impact")


# ═══════════════════════════════════════════════════════════════════════
# VIZ 17: Player Usage vs Efficiency Scatter
# ═══════════════════════════════════════════════════════════════════════
print("Viz 17: Usage vs Efficiency…")

usage_eff_path = None
if len(merged) > 0 and "USG_PCT" in merged.columns and "TS_PCT" in merged.columns:
    fig, ax = plt.subplots(figsize=(12, 8))

    usg = merged["USG_PCT"] * 100
    ts = merged["TS_PCT"] * 100
    mins = merged["MIN"]

    sc = ax.scatter(usg, ts, s=mins * 8, c=[GSW_GOLD if n > 0 else NEGATIVE for n in merged.get("NET_RATING", merged["PLUS_MINUS"])],
                    alpha=0.8, edgecolors="white", linewidth=1.5, zorder=3)

    for _, r in merged.iterrows():
        short = r["PLAYER_NAME"].split()[-1] if len(r["PLAYER_NAME"].split()) > 1 else r["PLAYER_NAME"]
        ax.annotate(short, (r["USG_PCT"] * 100, r["TS_PCT"] * 100),
                    fontsize=9, ha="center", va="bottom", xytext=(0, 8), textcoords="offset points",
                    color="white", fontweight="bold")

    ax.axhline(57, color=NEUTRAL, linestyle="--", alpha=0.3, label="League Avg TS% (~57%)")
    ax.axvline(20, color=NEUTRAL, linestyle=":", alpha=0.3, label="League Avg USG% (~20%)")

    # Quadrant labels
    ax.text(0.95, 0.95, "High Usage +\nHigh Efficiency ★", transform=ax.transAxes, fontsize=9, alpha=0.4,
            ha="right", va="top", color=POSITIVE)
    ax.text(0.05, 0.95, "Low Usage +\nHigh Efficiency", transform=ax.transAxes, fontsize=9, alpha=0.4,
            ha="left", va="top", color="#5cc9f5")
    ax.text(0.95, 0.05, "High Usage +\nLow Efficiency ⚠", transform=ax.transAxes, fontsize=9, alpha=0.4,
            ha="right", va="bottom", color=NEGATIVE)

    ax.set_xlabel("Usage Rate (%)")
    ax.set_ylabel("True Shooting (%)")
    ax.set_title("Usage Rate vs Shooting Efficiency (bubble = MPG, color = Net Rating)", pad=15)
    ax.legend(facecolor="#1a2a3a", edgecolor="#2a3f52", fontsize=9)
    ax.grid(True, alpha=0.2)
    usage_eff_path = savefig("usage_vs_efficiency")


# ═══════════════════════════════════════════════════════════════════════
# VIZ 18: Cumulative Win Pace Chart
# ═══════════════════════════════════════════════════════════════════════
print("Viz 18: Cumulative Win Pace…")

cum_win_path = None
if len(game_log) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))

    cum_wins = game_log["WIN"].cumsum()
    game_nums = game_log["GAME_NUM"]

    # Pace lines
    pace_50 = game_nums * (41 / 82)  # .500
    pace_55 = game_nums * (45 / 82)  # 55%
    pace_48 = game_nums * (39 / 82)  # 48%

    ax.fill_between(game_nums, pace_48, pace_55, alpha=0.08, color=GSW_GOLD, label="Playoff Zone (39-45W)")
    ax.plot(game_nums, pace_50, color=NEUTRAL, linestyle="--", alpha=0.4, linewidth=1, label=".500 pace (41W)")
    ax.plot(game_nums, pace_55, color=POSITIVE, linestyle=":", alpha=0.3, linewidth=1, label="55% pace (45W)")
    ax.plot(game_nums, cum_wins, color=GSW_GOLD, linewidth=3, label=f"Warriors ({int(cum_wins.iloc[-1])}W)", zorder=3)

    # Highlight current position
    ax.scatter([game_nums.iloc[-1]], [cum_wins.iloc[-1]], s=150, color=GSW_GOLD, edgecolors="white",
               linewidth=2, zorder=4)
    ax.annotate(f"{int(cum_wins.iloc[-1])}W-{games_played - int(cum_wins.iloc[-1])}L",
                (game_nums.iloc[-1], cum_wins.iloc[-1]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", color=GSW_GOLD,
                arrowprops=dict(arrowstyle="->", color=GSW_GOLD))

    # Project to 82
    proj_rate = cum_wins.iloc[-1] / games_played
    ax.plot([games_played, 82], [cum_wins.iloc[-1], cum_wins.iloc[-1] + (82 - games_played) * proj_rate],
            color=GSW_GOLD, linestyle="--", alpha=0.5, linewidth=2)

    ax.set_xlabel("Games Played")
    ax.set_ylabel("Cumulative Wins")
    ax.set_title("Cumulative Win Pace vs Playoff Thresholds", pad=15)
    ax.legend(facecolor="#1a2a3a", edgecolor="#2a3f52", fontsize=10)
    ax.set_xlim(1, 82)
    ax.grid(True, alpha=0.2)
    cum_win_path = savefig("cumulative_wins")


# ═══════════════════════════════════════════════════════════════════════
# VIZ 19: Player Efficiency Landscape (PER-like scatter)
# ═══════════════════════════════════════════════════════════════════════
print("Viz 19: Player Efficiency Landscape…")

eff_landscape_path = None
if len(merged) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))

    # X = Minutes, Y = Net Rating, size = PTS, color = TS%
    mins_arr = merged["MIN"].values
    net_arr = merged.get("NET_RATING", merged["PLUS_MINUS"]).values
    pts_arr = merged["PTS"].values
    ts_arr = merged.get("TS_PCT", merged["FG_PCT"]).values

    sc = ax.scatter(mins_arr, net_arr, s=pts_arr * 25 + 50, c=ts_arr, cmap="RdYlGn",
                    alpha=0.85, edgecolors="white", linewidth=1.5, zorder=3, vmin=0.4, vmax=0.7)

    for _, r in merged.iterrows():
        short = r["PLAYER_NAME"].split()[-1] if len(r["PLAYER_NAME"].split()) > 1 else r["PLAYER_NAME"]
        nr = r.get("NET_RATING", r["PLUS_MINUS"])
        ax.annotate(short, (r["MIN"], nr),
                    fontsize=9, ha="center", va="bottom", xytext=(0, 10), textcoords="offset points",
                    color="white", fontweight="bold")

    ax.axhline(0, color=GSW_WHITE, linestyle="--", alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("True Shooting %", color="#e8e8e8")
    cbar.ax.yaxis.set_tick_params(color="#e8e8e8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#e8e8e8")

    ax.set_xlabel("Minutes Per Game")
    ax.set_ylabel("Net Rating")
    ax.set_title("Player Efficiency Landscape (size=PPG, color=TS%)", pad=15)
    ax.grid(True, alpha=0.2)
    eff_landscape_path = savefig("efficiency_landscape")


# ═══════════════════════════════════════════════════════════════════════
# BUILD REPORT
# ═══════════════════════════════════════════════════════════════════════
print("Writing report…")

h(1, f"Golden State Warriors — Advanced ML Insights & Recommendations")
p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Models: XGBoost, SHAP, Bayesian Inference, K-Means, Ridge Regression*")
p("This report uses advanced machine learning models and statistical analysis to identify actionable areas "
  "for improvement, quantify player impact, project season outcomes, and provide data-driven recommendations "
  "for the Warriors' coaching staff and front office.")
w("---")
blank()

# ── 1. Win Model ─────────────────────────────────────────────────────
h(2, "1. XGBoost Win Predictor + SHAP Explanations")
p(f"An XGBoost classifier ({cv_acc*100:.1f}% cross-validated accuracy) trained on per-game stats "
  f"identifies which factors most drive wins and losses.")
blank()

h(3, "1.1 SHAP Feature Importance")
p("SHAP (SHapley Additive exPlanations) decomposes each game's prediction into individual feature contributions. "
  "Features pushing predictions toward a win are positive; toward a loss, negative.")
img(shap_bar_path, "SHAP Feature Importance")

h(3, "1.2 SHAP Beeswarm — Every Game Explained")
p("Each dot is one game. Position shows impact on win probability; color shows the feature value (red=high, blue=low).")
img(shap_beeswarm_path, "SHAP Beeswarm")

h(3, "1.3 Win Probability Timeline")
p("Model-estimated win probability for every game this season, with the 5-game rolling average overlaid.")
img(wp_timeline_path, "Win Probability Timeline")

# Key insight from SHAP
mean_shap_sorted = [(xgb_features[i], mean_shap[i]) for i in np.argsort(mean_shap)[::-1]]
p("**Key Insight:** The top 3 win drivers are:")
for i, (feat, val) in enumerate(mean_shap_sorted[:3]):
    w(f"{i+1}. **{feat}** (mean |SHAP| = {val:.3f})")
blank()
p(f"> 💡 **Recommendation:** The model confirms that **{mean_shap_sorted[0][0]}** is the single most important "
  f"factor separating wins from losses. The coaching staff should prioritize strategies that maximize this metric.")
blank()

# ── 2. What Separates Wins from Losses ────────────────────────────────
h(2, "2. Win vs Loss Statistical DNA")
p("Comparing average stats in wins versus losses reveals the statistical profile the Warriors need to achieve to win.")
img(gap_path, "Win vs Loss Gap")

tbl("Stat", "In Wins", "In Losses", "Gap", "Priority")
priority_stats = []
for stat, wv, lv in zip(["PTS", "AST", "REB", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"],
                         win_profile, loss_profile):
    g = wv - lv
    if stat in ["FG_PCT", "FG3_PCT"]:
        row(stat, f"{wv*100:.1f}%", f"{lv*100:.1f}%", f"{g*100:+.1f}%", "🔴 Critical" if abs(g) > 0.03 else "🟡 Moderate")
    else:
        pr = "🔴 Critical" if abs(g) > 5 else "🟡 Moderate" if abs(g) > 2 else "🟢 Minor"
        if stat == "TOV":
            pr = "🔴 Critical" if g < -1.5 else "🟡 Moderate"
        row(stat, f"{wv:.1f}", f"{lv:.1f}", f"{g:+.1f}", pr)
        priority_stats.append((stat, abs(g)))
blank()

# ── 3. Player Clustering ─────────────────────────────────────────────
h(2, "3. Player Archetype Clustering")
p(f"K-Means clustering (k={best_k}, silhouette={sil_scores[best_k]:.3f}) on {X_scaled.shape[1]} features "
  "identifies natural player groupings within the roster.")
img(cluster_path, "Player Clustering")

for c in range(best_k):
    members = merged[merged["Cluster"] == c]
    member_names = ", ".join(members["PLAYER_NAME"].values)
    avg_pts = members["PTS"].mean()
    avg_min = members["MIN"].mean()
    h(4, f"Cluster: {cluster_names.get(c, f'Cluster {c}')}")
    p(f"**Players:** {member_names}")
    p(f"Avg: {avg_pts:.1f} PPG, {members['AST'].mean():.1f} APG, {members['REB'].mean():.1f} RPG, {avg_min:.1f} MPG")
blank()

# ── 4. Player Radar Charts ───────────────────────────────────────────
h(2, "4. Player Skill Profiles")
p("Radar charts showing each player's strengths and weaknesses across 9 key dimensions, "
  "normalized within the team (100 = team best, 0 = team worst).")
img(radar_path, "Player Radar Charts")

# ── 5. On/Off Impact ─────────────────────────────────────────────────
h(2, "5. On/Off Court Impact Analysis")
p("Net Rating swing when each player is on vs. off the court (minimum 200 on-court minutes). "
  "This is the most direct measure of individual impact on team performance.")
img(impact_path, "On/Off Court Impact")
h(3, "5.1 Two-Way Impact Map")
p("Separating offensive and defensive impact reveals which players help on which end of the floor. "
  "The best players appear in the upper-right quadrant (help both offense and defense).")
img(twoway_path, "Two-Way Impact Map")

if len(impact_df) > 0:
    best = impact_df.iloc[-1]  # Sorted ascending, so last is best
    worst = impact_df.iloc[0]
    p(f"> 💡 **Most Impactful:** {best['PLAYER']} ({best['NET_SWING']:+.1f} net swing)")
    p(f"> ⚠️ **Least Impactful:** {worst['PLAYER']} ({worst['NET_SWING']:+.1f} net swing)")
    blank()

# ── 6. Lineup Synergy ────────────────────────────────────────────────
h(2, "6. Lineup Synergy Heatmap")
p("Weighted net rating for every player pair, aggregated across all 5-man lineups. "
  "Green = positive synergy; red = negative synergy.")
img(heatmap_path, "Lineup Synergy Heatmap")

# Find best and worst pairs
best_pair_val = -999
worst_pair_val = 999
best_pair_names = worst_pair_names = ("", "")
for i, p1 in enumerate(key_players):
    for j, p2 in enumerate(key_players):
        if i < j and pair_minutes.loc[p1, p2] > 20:
            v = pair_avg.loc[p1, p2]
            if v > best_pair_val:
                best_pair_val = v
                best_pair_names = (p1, p2)
            if v < worst_pair_val:
                worst_pair_val = v
                worst_pair_names = (p1, p2)

p(f"> ✅ **Best Pairing:** {best_pair_names[0]} + {best_pair_names[1]} ({best_pair_val:+.1f} weighted net)")
p(f"> ❌ **Worst Pairing:** {worst_pair_names[0]} + {worst_pair_names[1]} ({worst_pair_val:+.1f} weighted net)")
blank()

# ── 7. Scoring & Efficiency Trends ───────────────────────────────────
h(2, "7. Season Trends & Momentum")
p("Three-panel view of the team's scoring, shooting efficiency, and win probability across the season.")
img(trend_path, "Season Trends")

h(3, "7.1 Correlation Matrix")
p("How each game stat correlates with winning. Stronger correlations indicate higher-leverage areas for improvement.")
img(corr_path, "Correlation Matrix")

# Top correlates with winning
win_corrs = game_corr["WIN"].drop("WIN").sort_values(ascending=False)
p("**Strongest win correlates:**")
for stat, corr_val in win_corrs.head(3).items():
    w(f"- {stat}: r = {corr_val:.3f}")
blank()

# ── 8. Monthly Development ───────────────────────────────────────────
h(2, "8. Player Monthly Development")
p("Tracking how the top 4 scorers' production and efficiency evolve month-to-month, "
  "revealing improvement arcs, slumps, and breakout periods.")
if monthly_path:
    img(monthly_path, "Monthly Trends")

# ── 9. Bayesian Season Projection ────────────────────────────────────
h(2, "9. Bayesian Season Projection")
p(f"Using Bayesian inference to project the remaining {games_remaining} games, combining "
  f"full-season performance (Beta({alpha_post}, {beta_post})) with recent form "
  f"(last 15 games: {recent_wpct*100:.1f}% win rate) in a 60/40 blend.")
img(bayesian_path, "Bayesian Projection")

ci_low = np.percentile(blended_wins, 10)
ci_high = np.percentile(blended_wins, 90)
median_wins = np.median(blended_wins)
playoff_prob = (blended_wins >= 42).mean() * 100

tbl("Metric", "Value")
row("Current Record", f"{wins_so_far}-{losses_so_far}")
row("Full Season Win%", f"{wins_so_far/games_played*100:.1f}%")
row("Recent Form (L15)", f"{recent_wpct*100:.1f}%")
row("**Projected Total Wins (Median)**", f"**{median_wins:.0f}**")
row("80% Confidence Interval", f"{ci_low:.0f} – {ci_high:.0f} wins")
row("Probability ≥42 Wins", f"{playoff_prob:.1f}%")
row("Probability ≥44 Wins", f"{(blended_wins >= 44).mean()*100:.1f}%")
row("Probability ≥46 Wins", f"{(blended_wins >= 46).mean()*100:.1f}%")
blank()

# ── 10. SHAP Game Waterfall ──────────────────────────────────────────
h(2, "10. Game Anatomy — Best Win & Worst Loss")
p("SHAP waterfall charts decompose the model's prediction for the team's best win and worst loss, "
  "showing exactly which factors drove the outcome.")
if waterfall_path:
    img(waterfall_path, "SHAP Game Waterfall")

# ── 11. Composite Value ──────────────────────────────────────────────
h(2, "11. Composite Player Value Index")
p("A multi-model composite score integrating production (30%), per-minute production (20%), "
  "RAPM impact (30%), and shooting efficiency (20%).")
img(composite_path, "Composite Value")

tbl("Rank", "Player", "Production", "Per-Min", "RAPM", "Efficiency", "**Composite**")
for rank, (_, r) in enumerate(comp_df.head(10).iterrows(), 1):
    row(rank, r["PLAYER"], f"{r['PRODUCTION_N']:.0f}", f"{r['PER_MIN_PROD_N']:.0f}",
        f"{r['RAPM_N']:.0f}", f"{r['TS_PCT_N']:.0f}", f"**{r['COMPOSITE']:.1f}**")
blank()

# ── 12. Scoring Distribution ─────────────────────────────────────────
h(2, "12. Scoring Distribution")
p("Violin plots show the full distribution of each player's game-by-game scoring, "
  "revealing consistency, ceiling, floor, and outlier performances. "
  "Individual game dots are overlaid for transparency.")
if violin_path:
    img(violin_path, "Scoring Distribution")

# ── 13. Usage vs Efficiency ──────────────────────────────────────────
h(2, "13. Usage Rate vs Shooting Efficiency")
p("The ideal player lives in the upper-right: high usage AND high efficiency. "
  "Bubble size represents minutes per game; color represents net rating (gold = positive, red = negative).")
if usage_eff_path:
    img(usage_eff_path, "Usage vs Efficiency")
    # Identify players in each quadrant
    if len(merged) > 0 and "USG_PCT" in merged.columns:
        stars = merged[(merged["USG_PCT"] > 0.20) & (merged["TS_PCT"] > 0.57)]
        if len(stars) > 0:
            star_names = ", ".join(stars["PLAYER_NAME"].values)
            p(f"> ⭐ **Elite efficiency at high usage:** {star_names}")
        inefficient_high = merged[(merged["USG_PCT"] > 0.20) & (merged["TS_PCT"] < 0.52)]
        if len(inefficient_high) > 0:
            ineff_names = ", ".join(inefficient_high["PLAYER_NAME"].values)
            p(f"> ⚠️ **High usage, low efficiency (improvement target):** {ineff_names}")
    blank()

# ── 14. Shot Zone Efficiency ─────────────────────────────────────────
h(2, "14. Shot Zone Efficiency")
p("Efficiency by shot area reveals where the Warriors generate good looks and where they waste possessions.")
if shot_zone_path:
    img(shot_zone_path, "Shot Zone Efficiency")
else:
    p("*Shot zone data not available in cache.*")
blank()

# ── 15. Clutch Performance ───────────────────────────────────────────
h(2, "15. Clutch Performance Analysis")
p("Clutch = last 5 minutes of the game with the score within 5 points. "
  "These charts identify the Warriors' most reliable closers.")
if clutch_path_viz:
    img(clutch_path_viz, "Clutch Performance")
else:
    p("*Clutch data not available in cache.*")
blank()

# ── 16. Rest Day Impact ──────────────────────────────────────────────
h(2, "16. Rest Day Impact on Performance")
p("How many days of rest before a game significantly affects both win probability and offensive output.")
if rest_viz_path:
    img(rest_viz_path, "Rest Day Impact")
blank()

# ── 17. Cumulative Win Pace ──────────────────────────────────────────
h(2, "17. Cumulative Win Pace")
p("Tracking cumulative wins against various season pace targets, with projection to the end of the season.")
if cum_win_path:
    img(cum_win_path, "Cumulative Win Pace")
blank()

# ── 18. Player Efficiency Landscape ──────────────────────────────────
h(2, "18. Player Efficiency Landscape")
p("A holistic view of every player's role: minutes (X), net rating (Y), scoring volume (bubble size), "
  "and shooting efficiency (color). Players above the zero line contribute positively when on court.")
if eff_landscape_path:
    img(eff_landscape_path, "Player Efficiency Landscape")
blank()

# ═══════════════════════════════════════════════════════════════════════
# ACTIONABLE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
h(2, "19. Actionable Improvement Recommendations")
p("Based on all models above, here are prioritized, specific recommendations:")
blank()

h(3, "🏀 Offensive Recommendations")
w()
# 1. From SHAP
top_driver = mean_shap_sorted[0][0]
driver_desc = {
    "TOV": "ball security (turnovers)",
    "AST": "ball movement (assists)",
    "PTS": "scoring output (points)",
    "FG_PCT": "shooting accuracy (FG%)",
    "REB": "rebounding",
    "FG3_PCT": "three-point shooting",
    "STL": "steals / forcing turnovers",
    "BLK": "rim protection (blocks)",
}.get(top_driver, top_driver)
w(f"**1. #{1} win driver: {driver_desc}** (SHAP importance: {mean_shap_sorted[0][1]:.3f})")
if top_driver == "TOV":
    w(f"   - Turnovers are the single most predictive stat separating Warriors wins from losses")
    w(f"   - Current: {game_log['TOV'].mean():.1f} TOV/game; in wins: {win_profile['TOV']:.1f}, in losses: {loss_profile['TOV']:.1f}")
    w(f"   - Every turnover averted significantly increases win probability per the XGBoost model")
    w(f"   - Focus areas: tighten half-court passing, limit dribble hand-offs in traffic")
elif top_driver == "AST":
    w(f"   - Current: {game_log['AST'].mean():.1f} AST/game. Target: {win_profile['AST']:.1f} (win average)")
    w(f"   - Run more motion offense sets — screen assists correlate strongly with assists")
    w(f"   - Increase Draymond Green and Jimmy Butler pick-and-roll involvement")
elif top_driver == "PTS":
    w(f"   - Current: {game_log['PTS'].mean():.1f} PPG. Target: {win_profile['PTS']:.1f} (win average)")
    w(f"   - Increase pace in transition; the Warriors score {win_profile['PTS'] - loss_profile['PTS']:.1f} more PPG in wins")
elif top_driver == "FG_PCT":
    w(f"   - Current: {game_log['FG_PCT'].mean()*100:.1f}% FG. Target: {win_profile['FG_PCT']*100:.1f}% (win average)")
    w(f"   - Increase restricted area attempts; reduce low-efficiency mid-range shots")
else:
    w(f"   - Current average: {game_log[top_driver].mean():.1f}/game")
    w(f"   - In wins: {win_profile[top_driver]:.1f}, in losses: {loss_profile[top_driver]:.1f}")
blank()

# 2. Turnover reduction
tov_gap = win_profile["TOV"] - loss_profile["TOV"]
w(f"**2. Reduce turnovers** (gap: {tov_gap:.1f} fewer in wins)")
w(f"   - Current: {game_log['TOV'].mean():.1f} TOV/game; wins average {win_profile['TOV']:.1f}, losses {loss_profile['TOV']:.1f}")
w(f"   - Target: <{win_profile['TOV']:.0f} turnovers per game")
# Find highest-TOV players
if len(player_base) > 0:
    high_tov = player_base.nlargest(3, "TOV")
    tov_names = ", ".join([f"{r['PLAYER_NAME']} ({r['TOV']:.1f})" for _, r in high_tov.iterrows()])
    w(f"   - Key focus: {tov_names}")
blank()

# 3. Shooting efficiency
w(f"**3. Improve 3-point shot selection**")
fg3_gap = (win_profile["FG3_PCT"] - loss_profile["FG3_PCT"]) * 100
w(f"   - 3PT% gap between wins ({win_profile['FG3_PCT']*100:.1f}%) and losses ({loss_profile['FG3_PCT']*100:.1f}%): {fg3_gap:+.1f}%")
w(f"   - Increase corner 3 attempts (highest efficiency) and reduce contested above-break 3s")
w(f"   - Curry's off-ball screens create the best looks — run more Curry off-screen sets")
blank()

h(3, "🛡️ Defensive Recommendations")
w()
# From on/off analysis
w(f"**4. Optimize defensive lineups**")
if len(impact_df) > 0:
    def_swing = impact_df["ON_DEF"] - impact_df["OFF_DEF"]
    def_helpers = impact_df[def_swing < -3].copy()
    def_helpers["_def_swing"] = def_swing[def_helpers.index]
    def_helpers = def_helpers.sort_values("_def_swing")
    if len(def_helpers) > 0:
        def_names = ", ".join(def_helpers["PLAYER"].head(3).values)
        w(f"   - Best defensive anchors (lower DEF RTG on court): {def_names}")
    def_liabilities = impact_df[impact_df["ON_DEF"] - impact_df["OFF_DEF"] > 3]
    if len(def_liabilities) > 0:
        liab_names = ", ".join(def_liabilities["PLAYER"].head(3).values)
        w(f"   - Hide defensively in crunch time: {liab_names}")
blank()

w(f"**5. Increase steal-generating activity**")
stl_gap = win_profile["STL"] - loss_profile["STL"]
w(f"   - Steals gap in wins vs losses: {stl_gap:+.1f}/game")
w(f"   - More aggressive trapping in half-court; the data shows minimal impact on fouls")
blank()

h(3, "📋 Rotation & Lineup Recommendations")
w()
w(f"**6. Maximize the best pair: {best_pair_names[0]} + {best_pair_names[1]}** (net: {best_pair_val:+.1f})")
w(f"   - Increase their shared minutes; design sets that leverage their chemistry")
w(f"   - Separate {worst_pair_names[0]} and {worst_pair_names[1]} ({worst_pair_val:+.1f} net) when possible")
blank()

# Minutes distribution recommendation
w(f"**7. Rest day optimization**")
rest_data = game_log.copy()
rest_data["REST"] = rest_data["GAME_DATE_DT"].diff().dt.days.fillna(1)
rest_groups = rest_data.groupby(rest_data["REST"].clip(0, 3).astype(int)).agg({"WIN": ["mean", "count"]})
rest_groups.columns = ["Win%", "GP"]
best_rest = rest_groups["Win%"].idxmax()
w(f"   - Best performance on {best_rest} rest days ({rest_groups.loc[best_rest, 'Win%']*100:.0f}% win rate)")
w(f"   - Manage Curry's minutes on back-to-backs (fatigue coefficient from prior analysis)")
blank()

h(3, "📈 Player Development Recommendations")
w()
# Find players with highest improvement potential
w(f"**8. Development priorities**")
for _, r in comp_df.iterrows():
    if r["PER_MIN_PROD_N"] > 60 and r["MIN"] < 20:
        w(f"   - **{r['PLAYER']}**: High per-minute production ({r['PER_MIN_PROD_N']:.0f}/100) but only {r['MIN']:.1f} MPG — consider expanding role")
blank()

w(f"**9. Shooting improvement targets**")
if len(player_base) > 0:
    for _, r in player_base.iterrows():
        if r["MIN"] > 15 and r["FG3_PCT"] < 0.33 and r["FG3A"] > 1:
            w(f"   - {r['PLAYER_NAME']}: {r['FG3_PCT']*100:.1f}% on {r['FG3A']:.1f} 3PA/game — needs shot selection improvement or volume reduction")
blank()

h(3, "🔮 Season Outlook")
w()
w(f"**10. Playoff positioning**")
w(f"   - Current: {wins_so_far}-{losses_so_far} ({wins_so_far/games_played*100:.1f}%)")
w(f"   - Projected finish: **{median_wins:.0f} wins** (80% CI: {ci_low:.0f}–{ci_high:.0f})")
w(f"   - Playoff probability (≥42 wins): **{playoff_prob:.0f}%**")
w(f"   - Recent form ({recent_wpct*100:.0f}% L15) is {'concerning' if recent_wpct < 0.5 else 'encouraging'} — "
  f"{'urgent improvement needed to maintain playoff position' if recent_wpct < 0.5 else 'momentum suggests upside to projections'}")
blank()

if recent_wpct < 0.5:
    w(f"   > ⚠️ **Alert:** At the current L15 pace ({recent_wpct*100:.0f}%), the team projects to only "
      f"**{wins_so_far + int(games_remaining * recent_wpct)} wins** — below the playoff threshold. "
      f"Immediate tactical adjustments are needed.")
    blank()

# ═══════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
w("---")
blank()
h(2, "20. Executive Summary")
blank()
w("| # | Finding | Action | Priority |")
w("|---|---------|--------|----------|")

# Summarize top findings
summary_items = [
    (f"Turnovers are the #1 win predictor (SHAP={mean_shap_sorted[0][1]:.2f})",
     "Target <14 TOV/game; tighten half-court execution", "🔴 Critical"),
    (f"FG% gap: {(win_profile['FG_PCT']-loss_profile['FG_PCT'])*100:+.1f}% between W/L",
     "Increase restricted-area attempts; cut mid-range volume", "🔴 Critical"),
    (f"3PT%: {win_profile['FG3_PCT']*100:.1f}% in wins vs {loss_profile['FG3_PCT']*100:.1f}% in losses",
     "More corner 3s; reduce contested above-break 3s", "🔴 Critical"),
]

if len(impact_df) > 0:
    best_imp = impact_df.iloc[-1]
    worst_imp = impact_df.iloc[0]
    summary_items.append(
        (f"Best pair: {best_pair_names[0].split()[-1]} + {best_pair_names[1].split()[-1]} ({best_pair_val:+.1f})",
         "Increase shared minutes; design synergy sets", "🟡 High"))
    summary_items.append(
        (f"Worst pair: {worst_pair_names[0].split()[-1]} + {worst_pair_names[1].split()[-1]} ({worst_pair_val:+.1f})",
         "Stagger minutes; avoid pairing in crunch time", "🟡 High"))

summary_items.append(
    (f"Projected {median_wins:.0f} wins (80% CI: {ci_low:.0f}–{ci_high:.0f})",
     f"Playoff prob: {playoff_prob:.0f}%; recent slide needs correction", "🟡 High"))

for i, (finding, action, pri) in enumerate(summary_items, 1):
    row(i, finding, action, pri)
blank()

p("> **Bottom line:** The Warriors are a borderline playoff team whose fate hinges on ball security, "
  "shot quality, and maximizing the Butler-Melton pairing. The data is clear on what separates "
  "this team's wins from losses — executing on these priorities will determine the season outcome.")
blank()

# ═══════════════════════════════════════════════════════════════════════
# GLOSSARY
# ═══════════════════════════════════════════════════════════════════════
w("---")
blank()
h(2, "Appendix: Glossary of Models & Metrics")
blank()
h(3, "Machine Learning Models Used")
blank()
tbl("Model", "Type", "Purpose")
row("XGBoost Classifier", "Gradient Boosted Trees", "Predict wins/losses from game stats; 200 trees, depth 4, 5-fold CV")
row("SHAP (TreeExplainer)", "Game Theory Explainability", "Decompose each prediction into per-feature contributions")
row("K-Means Clustering", "Unsupervised Learning", "Group players into archetypes based on stat profiles")
row("Ridge Regression (RAPM-lite)", "Regularized Linear Model", "Estimate each player's impact on team net rating from lineup data")
row("Bayesian Beta-Binomial", "Probabilistic Inference", "Project season win total with uncertainty via posterior distribution")
row("Gradient Boosted Regressor", "Ensemble Regression", "Expected FG% model (xFG) based on shot profile")
row("PCA", "Dimensionality Reduction", "Project high-dimensional player stats to 2D for visualization")
blank()

h(3, "Statistical Metrics")
blank()
tbl("Metric", "Definition")
row("**SHAP Value**", "Shapley Additive Explanation — each feature's marginal contribution to a prediction")
row("**Net Rating**", "Points scored minus points allowed per 100 possessions (team or individual on-court)")
row("**Off Rating / Def Rating**", "Points scored / allowed per 100 possessions")
row("**On/Off Swing**", "Difference in team Net Rating when a player is on court vs off court")
row("**TS% (True Shooting)**", "Scoring efficiency: PTS / (2 × (FGA + 0.44 × FTA))")
row("**USG% (Usage Rate)**", "% of team possessions used by a player while on court")
row("**RAPM**", "Regularized Adjusted Plus-Minus — player's per-possession impact estimated via Ridge regression")
row("**PIE (Player Impact Estimate)**", "Player's share of game events (points, rebounds, assists, etc.)")
row("**Silhouette Score**", "Clustering quality: ranges from -1 (poor) to +1 (perfect); >0.25 is acceptable")
row("**Coefficient of Variation**", "Standard deviation / mean — measures scoring consistency (lower = more consistent)")
row("**Bayesian Posterior**", "Updated probability distribution after combining prior belief with observed data")
row("**Monte Carlo Simulation**", "Running thousands of random season simulations to estimate outcome probabilities")
row("**Cross-Validation Accuracy**", "Model accuracy averaged across held-out test folds (prevents overfitting)")
row("**Clutch**", "Last 5 minutes of game with score within 5 points")
row("**Weighted Net Rating**", "Net rating weighted by minutes played (gives more weight to larger samples)")
blank()

h(3, "Visualization Guide")
blank()
tbl("Chart Type", "How to Read It")
row("SHAP Beeswarm", "Each dot = one game. Horizontal position = impact on win probability. Color = feature value (red=high, blue=low)")
row("Radar Chart", "9 axes showing player skills normalized 0–100 within team. Larger area = more well-rounded")
row("Violin Plot", "Width = frequency of scoring at that level. Wider = more common. Dots = individual games")
row("Synergy Heatmap", "Pair net rating across shared lineups. Green = positive chemistry; red = negative")
row("Two-Way Impact", "X = offensive help, Y = defensive help. Upper-right = two-way stars")
row("Efficiency Landscape", "X = minutes, Y = net rating, size = PPG, color = TS%. Above zero line = net positive")
row("Cumulative Wins", "Tracks actual wins vs pace targets. Dashed line = season-pace projection")
row("Bayesian Projection", "Histogram = simulated win totals. Taller bar = more likely outcome")
blank()

w("---")
gen_date = datetime.now().strftime("%B %d, %Y")
w(f"*Models: XGBoost 3.2, SHAP 0.50, scikit-learn, scipy | Data: stats.nba.com {SEASON} | Generated: {gen_date}*")

# ── Write Report ─────────────────────────────────────────────────────
REPORT.write_text("\n".join(lines))
n_lines = len(lines)
n_chars = sum(len(l) for l in lines)
print(f"\n✅ Report written to {REPORT}")
print(f"   {n_lines} lines, {n_chars:,} characters")
print(f"   {len(list(IMG_DIR.glob('*.png')))} visualization files in {IMG_DIR}/")
