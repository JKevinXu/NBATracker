#!/usr/bin/env python3
"""
Machine Learning–Driven NBA Analytics for the Golden State Warriors.

Models:
  1. Player Archetype Clustering (K-Means)
  2. Win Probability Model (Random Forest)
  3. Expected Shooting Model (xFG% – Gradient Boosted)
  4. Player On/Off Impact (RAPM-Lite via Ridge Regression)
  5. Fatigue / Rest-Day Regression
  6. Lineup Synergy Analysis (Net Rating Prediction)
  7. Performance Momentum Detection (HMM-style)

Outputs a Markdown report to reports/warriors_ml_insights_2025_26.md
"""

import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "web" / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TEAM_ID = 1610612744


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_rs(name):
    data = json.loads((CACHE / f"{name}.json").read_text())
    rs = data.get("resultSets", data)
    if isinstance(rs, list):
        return {r["name"]: pd.DataFrame(r["rowSet"], columns=r["headers"]) for r in rs}
    elif isinstance(rs, dict):
        return {"main": pd.DataFrame(rs["rowSet"], columns=rs["headers"])}
    return {}


def load1(name):
    d = load_rs(name)
    return list(d.values())[0]


print("Loading data…")
base = load1("player_base")
adv = load1("player_adv")
gamelog = load1("gamelog")
standings = load1("standings")
clutch = load1("clutch")
gamelogs_player = load_rs("player_gamelogs")["PlayerGameLogs"]
lineups = load_rs("lineups")["Lineups"]
on_off = load_rs("on_off")

# On-court / off-court
on_court = on_off.get("PlayersOnCourtTeamPlayerOnOffDetails", pd.DataFrame())
off_court = on_off.get("PlayersOffCourtTeamPlayerOnOffDetails", pd.DataFrame())

# Hustle
hustle = load_rs("hustle")
hustle = list(hustle.values())[0]

# Shot locations
shot_raw = json.loads((CACHE / "shot_locations.json").read_text())
info_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "NICKNAME"]
zones = ["<5ft", "5-9ft", "10-14ft", "15-19ft", "20-24ft", "25-29ft", "30-34ft", "35-39ft", "40+ft"]
sl_cols = list(info_cols)
for z in zones:
    sl_cols.extend([f"{z}_FGM", f"{z}_FGA", f"{z}_FG_PCT"])
shot_loc = pd.DataFrame(shot_raw["resultSets"]["rowSet"], columns=sl_cols)

# Tracking
tracking = load_rs("tracking_speed")
tracking = list(tracking.values())[0]

# Splits
splits_raw = json.loads((CACHE / "splits.json").read_text())
splits = {rs["name"]: pd.DataFrame(rs["rowSet"], columns=rs["headers"]) for rs in splits_raw["resultSets"]}

print(f"  Loaded {len(gamelogs_player)} player-game records, {len(lineups)} lineups")


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


now = datetime.now().strftime("%B %d, %Y")

h(1, "Golden State Warriors — Machine Learning Insights Report")
p(f"*Generated: {now} | Models: scikit-learn, scipy, numpy*")
blank()
p("This report applies machine learning and statistical modeling techniques to the Warriors' 2025-26 season data to uncover patterns, quantify player impact, and identify actionable insights beyond traditional box-score analysis.")
blank()
p("---")
blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 1: PLAYER ARCHETYPE CLUSTERING
# ═══════════════════════════════════════════════════════════════
print("Model 1: Player Clustering…")

h(2, "1. Player Archetype Clustering (K-Means)")
p("Using K-Means clustering on per-game stats and advanced metrics to identify natural player archetypes within the Warriors roster.")
blank()

# Merge base + adv + shooting + hustle for feature matrix
players = base[base["GP"] >= 10].copy()
adv_f = adv[adv["GP"] >= 10][["PLAYER_ID", "OFF_RATING", "DEF_RATING", "NET_RATING",
                                "TS_PCT", "USG_PCT", "PIE", "AST_TO", "EFG_PCT"]].copy()
players = players.merge(adv_f, on="PLAYER_ID", how="inner")

# Add hustle features
gp_col = "G" if "G" in hustle.columns else "GP"
hustle_f = hustle[hustle[gp_col] >= 10][["PLAYER_ID", "CONTESTED_SHOTS", "DEFLECTIONS",
                                          "SCREEN_ASSISTS", "LOOSE_BALLS_RECOVERED"]].copy()
players = players.merge(hustle_f, on="PLAYER_ID", how="left")

# Add tracking
track_f = tracking[tracking["GP"] >= 10][["PLAYER_ID", "AVG_SPEED", "DIST_MILES"]].copy()
players = players.merge(track_f, on="PLAYER_ID", how="left")

# Add shot location features
shot_f = shot_loc[["PLAYER_ID"]].copy()
total_fga = sum(shot_loc[f"{z}_FGA"] for z in zones)
for z in ["<5ft", "25-29ft"]:
    shot_f[f"pct_fga_{z}"] = shot_loc[f"{z}_FGA"] / total_fga.replace(0, np.nan)
    shot_f[f"fg_pct_{z}"] = shot_loc[f"{z}_FG_PCT"]
players = players.merge(shot_f, on="PLAYER_ID", how="left")

feature_cols = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "TS_PCT", "USG_PCT", "PIE", "AST_TO",
    "CONTESTED_SHOTS", "DEFLECTIONS", "SCREEN_ASSISTS",
    "AVG_SPEED", "DIST_MILES",
    "pct_fga_<5ft", "pct_fga_25-29ft",
]
feature_cols = [c for c in feature_cols if c in players.columns]

X = players[feature_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal K using silhouette score
from sklearn.metrics import silhouette_score

sil_scores = {}
for k in range(2, min(7, len(X))):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

best_k = max(sil_scores, key=sil_scores.get)
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
players["cluster"] = km.fit_predict(X_scaled)

# PCA for interpretation
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
players["pca1"] = X_pca[:, 0]
players["pca2"] = X_pca[:, 1]

# Name the clusters based on characteristics
cluster_profiles = {}
for c in range(best_k):
    mask = players["cluster"] == c
    subset = players[mask]
    profile = {
        "pts": subset["PTS"].mean(),
        "ast": subset["AST"].mean(),
        "reb": subset["REB"].mean(),
        "usg": subset["USG_PCT"].mean() if "USG_PCT" in subset.columns else 0,
        "def_rtg": subset["DEF_RATING"].mean() if "DEF_RATING" in subset.columns else 0,
        "contested": subset["CONTESTED_SHOTS"].mean() if "CONTESTED_SHOTS" in subset.columns else 0,
        "speed": subset["AVG_SPEED"].mean() if "AVG_SPEED" in subset.columns else 0,
        "players": list(subset["PLAYER_NAME"]),
    }
    # Auto-name
    if profile["pts"] >= 18:
        name = "⭐ Primary Scorers"
    elif profile["contested"] >= 5:
        name = "🛡️ Defensive Anchors"
    elif profile["ast"] >= 3:
        name = "🎯 Playmakers / Connectors"
    elif profile["speed"] >= 4.6:
        name = "⚡ Energy / Hustle"
    else:
        name = "🔧 Role Players"
    profile["name"] = name
    cluster_profiles[c] = profile

h(3, "Cluster Configuration")
p(f"- **Algorithm:** K-Means (k={best_k}, selected by silhouette score)")
p(f"- **Features:** {len(feature_cols)} dimensions (stats, advanced metrics, tracking, shooting zones)")
p(f"- **Silhouette Score:** {sil_scores[best_k]:.3f} (1.0 = perfect separation)")
p(f"- **PCA Variance Explained:** {pca.explained_variance_ratio_.sum() * 100:.1f}% (2 components)")
blank()

# Silhouette comparison
h(3, "Optimal K Selection")
tbl("K", "Silhouette Score", "")
for k, s in sorted(sil_scores.items()):
    bar = "█" * int(s * 40) + "░" * (40 - int(s * 40))
    marker = " ← chosen" if k == best_k else ""
    row(k, f"{s:.3f}", f"{bar}{marker}")
blank()

h(3, "Player Archetypes")
for c, prof in sorted(cluster_profiles.items()):
    h(4, prof["name"])
    p(f"**Players:** {', '.join(prof['players'])}")
    blank()
    tbl("Metric", "Cluster Avg")
    row("PPG", f"{prof['pts']:.1f}")
    row("APG", f"{prof['ast']:.1f}")
    row("RPG", f"{prof['reb']:.1f}")
    row("USG%", f"{prof['usg'] * 100:.1f}%")
    row("DEF RTG", f"{prof['def_rtg']:.1f}")
    row("Contested Shots", f"{prof['contested']:.1f}")
    row("Avg Speed", f"{prof['speed']:.2f} mph")
    blank()

# Feature importance for clustering
h(3, "Most Discriminating Features (PCA Loadings)")
loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1", "PC2"])
loadings["importance"] = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)
loadings = loadings.sort_values("importance", ascending=False)
tbl("Feature", "PC1 Loading", "PC2 Loading", "Importance")
for feat, lr in loadings.head(10).iterrows():
    row(feat, f"{lr['PC1']:.3f}", f"{lr['PC2']:.3f}", f"{lr['importance']:.3f}")
blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 2: WIN PROBABILITY MODEL
# ═══════════════════════════════════════════════════════════════
print("Model 2: Win Probability…")

h(2, "2. Win Probability Model (Random Forest)")
p("A Random Forest classifier trained on per-game team stats to identify the statistical factors most predictive of Warriors wins and losses.")
blank()

# Prepare game-level features
gl = gamelog.copy()
gl["WIN"] = (gl["WL"] == "W").astype(int)
gl["GAME_NUM"] = range(len(gl), 0, -1)

# Feature engineering
game_features = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "STL", "BLK"]
available_feats = [c for c in game_features if c in gl.columns]

# Add rolling averages (last 5 games as context)
gl_sorted = gl.sort_values("GAME_NUM")
for col in available_feats:
    gl_sorted[f"{col}_roll5"] = gl_sorted[col].rolling(5, min_periods=1).mean().shift(1)

# Add rest days approximation
gl_sorted["GAME_DATE_DT"] = pd.to_datetime(gl_sorted["GAME_DATE"])
gl_sorted["REST_DAYS"] = gl_sorted["GAME_DATE_DT"].diff().dt.days.fillna(2)
gl_sorted["IS_HOME"] = gl_sorted["MATCHUP"].str.contains("vs.").astype(int)

# Features: current game stats + context
context_feats = [f"{c}_roll5" for c in available_feats] + ["REST_DAYS", "IS_HOME"]
context_feats = [c for c in context_feats if c in gl_sorted.columns]

# Combine current game stats + context
all_feats = available_feats + context_feats
X_win = gl_sorted[all_feats].fillna(0).values
y_win = gl_sorted["WIN"].values

# Drop first few games (no rolling data)
valid = ~np.isnan(X_win).any(axis=1) & (gl_sorted["GAME_NUM"] > 5).values
X_win = X_win[valid]
y_win = y_win[valid]

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, min_samples_leaf=3)
cv_scores = cross_val_score(rf, X_win, y_win, cv=5, scoring="accuracy")
rf.fit(X_win, y_win)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=all_feats).sort_values(ascending=False)

h(3, "Model Performance")
p(f"- **Algorithm:** Random Forest (200 trees, max_depth=6)")
p(f"- **Cross-Validation Accuracy:** {cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")
p(f"- **Training Samples:** {len(X_win)} games")
blank()

h(3, "Key Predictors of Wins")
p("Features ranked by their importance in predicting whether the Warriors win or lose.")
blank()
tbl("Rank", "Feature", "Importance", "")
for i, (feat, imp) in enumerate(importances.head(12).items()):
    bar = "█" * int(imp * 100) + "░" * max(0, 20 - int(imp * 100))
    row(i + 1, feat, f"{imp:.3f}", bar)
blank()

# Win threshold analysis
h(3, "Win Threshold Analysis")
p("Statistical thresholds that separate wins from losses:")
blank()
gl_wins = gl_sorted[gl_sorted["WIN"] == 1]
gl_losses = gl_sorted[gl_sorted["WIN"] == 0]
tbl("Stat", "Avg in Wins", "Avg in Losses", "Δ (Win-Loss)", "p-value")
for feat in available_feats:
    w_mean = gl_wins[feat].mean()
    l_mean = gl_losses[feat].mean()
    delta = w_mean - l_mean
    # t-test
    t_stat, p_val = sp_stats.ttest_ind(gl_wins[feat].dropna(), gl_losses[feat].dropna())
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    row(feat, f"{w_mean:.1f}", f"{l_mean:.1f}", f"{'+' if delta >= 0 else ''}{delta:.1f}",
        f"{p_val:.4f} {sig}")
blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 3: EXPECTED SHOOTING (xFG%)
# ═══════════════════════════════════════════════════════════════
print("Model 3: Expected Shooting…")

h(2, "3. Expected Shooting Model (xFG%)")
p("A Gradient Boosted regression model predicting expected FG% based on shot distance, shot type distribution, and volume, to identify players shooting above or below expectations.")
blank()

# Build shot profile features per game
pgl = gamelogs_player.copy()
pgl["MIN_FLOAT"] = pd.to_numeric(pgl["MIN"], errors="coerce")
pgl = pgl[pgl["MIN_FLOAT"] >= 10].copy()

# Get shot zone profiles per player
player_shot_features = {}
for _, sr in shot_loc.iterrows():
    pid = sr["PLAYER_ID"]
    total_fga = sum(sr[f"{z}_FGA"] for z in zones)
    if total_fga < 1:
        continue
    feats = {}
    for z in zones:
        feats[f"pct_{z}"] = sr[f"{z}_FGA"] / total_fga
    feats["avg_shot_dist"] = sum(
        sr[f"{z}_FGA"] * i * 5 for i, z in enumerate(zones)
    ) / total_fga
    player_shot_features[pid] = feats

# Build features for xFG model using player game logs
xfg_data = []
for _, r in pgl.iterrows():
    pid = r["PLAYER_ID"]
    if pid not in player_shot_features:
        continue
    if r["FGA"] < 3:
        continue
    feats = player_shot_features[pid].copy()
    feats["fga"] = r["FGA"]
    feats["fg3a_rate"] = r["FG3A"] / r["FGA"] if r["FGA"] > 0 else 0
    feats["fta_rate"] = r["FTA"] / r["FGA"] if r["FGA"] > 0 else 0
    feats["actual_fg_pct"] = r["FG_PCT"]
    feats["player_id"] = pid
    feats["player_name"] = r["PLAYER_NAME"]
    xfg_data.append(feats)

xfg_df = pd.DataFrame(xfg_data)

if len(xfg_df) > 20:
    xfg_features = [c for c in xfg_df.columns if c.startswith("pct_") or c in
                     ["fga", "fg3a_rate", "fta_rate", "avg_shot_dist"]]
    X_xfg = xfg_df[xfg_features].fillna(0).values
    y_xfg = xfg_df["actual_fg_pct"].values

    gb = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    xfg_cv = cross_val_score(gb, X_xfg, y_xfg, cv=5, scoring="r2")
    gb.fit(X_xfg, y_xfg)

    xfg_df["xFG_PCT"] = gb.predict(X_xfg)
    xfg_df["fg_diff"] = xfg_df["actual_fg_pct"] - xfg_df["xFG_PCT"]

    # Aggregate per player
    player_xfg = xfg_df.groupby("player_name").agg(
        games=("actual_fg_pct", "count"),
        actual_fg=("actual_fg_pct", "mean"),
        expected_fg=("xFG_PCT", "mean"),
        fg_over_expected=("fg_diff", "mean"),
    ).reset_index().sort_values("fg_over_expected", ascending=False)

    h(3, "Model Performance")
    p(f"- **Algorithm:** Gradient Boosted Regressor (150 trees)")
    p(f"- **Features:** Shot distance profile, 3PT rate, FTA rate, volume")
    p(f"- **Cross-Val R²:** {xfg_cv.mean():.3f} ± {xfg_cv.std():.3f}")
    p(f"- **Training Samples:** {len(X_xfg)} player-game records")
    blank()

    h(3, "FG% Over/Under Expected")
    p("Players shooting significantly above or below what the model expects given their shot profile.")
    blank()
    tbl("Player", "Games", "Actual FG%", "Expected FG% (xFG)", "Diff", "Verdict")
    for _, r in player_xfg.iterrows():
        if r["games"] >= 5:
            diff = r["fg_over_expected"]
            if diff > 0.03:
                verdict = "🔥 Hot"
            elif diff > 0.01:
                verdict = "✅ Above Exp"
            elif diff > -0.01:
                verdict = "➡️ On Track"
            elif diff > -0.03:
                verdict = "⚠️ Below Exp"
            else:
                verdict = "❄️ Cold"
            row(r["player_name"], int(r["games"]),
                f"{r['actual_fg'] * 100:.1f}%", f"{r['expected_fg'] * 100:.1f}%",
                f"{'+' if diff >= 0 else ''}{diff * 100:.1f}%", verdict)
    blank()

    # Feature importances
    xfg_imp = pd.Series(gb.feature_importances_, index=xfg_features).sort_values(ascending=False)
    h(3, "Shot Profile Factors (Feature Importance)")
    tbl("Feature", "Importance")
    for feat, imp in xfg_imp.head(8).items():
        row(feat, f"{imp:.3f}")
    blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 4: PLAYER ON/OFF IMPACT (RAPM-LITE)
# ═══════════════════════════════════════════════════════════════
print("Model 4: Player Impact…")

h(2, "4. Player Impact Quantification (RAPM-Lite)")
p("Using Ridge Regression on player on/off-court data to estimate each player's regularized impact on team performance, similar to Regularized Adjusted Plus-Minus (RAPM).")
blank()

# Use on/off court net ratings
if len(on_court) > 0 and len(off_court) > 0:
    on_df = on_court[["VS_PLAYER_NAME", "MIN", "NET_RATING", "OFF_RATING", "DEF_RATING", "W_PCT", "PACE"]].copy()
    on_df.columns = ["PLAYER", "ON_MIN", "ON_NET", "ON_OFF_RTG", "ON_DEF_RTG", "ON_WPCT", "ON_PACE"]

    off_df = off_court[["VS_PLAYER_NAME", "MIN", "NET_RATING", "OFF_RATING", "DEF_RATING", "W_PCT"]].copy()
    off_df.columns = ["PLAYER", "OFF_MIN", "OFF_NET", "OFF_OFF_RTG", "OFF_DEF_RTG", "OFF_WPCT"]

    impact = on_df.merge(off_df, on="PLAYER", how="inner")
    impact["NET_SWING"] = impact["ON_NET"] - impact["OFF_NET"]
    impact["OFF_SWING"] = impact["ON_OFF_RTG"] - impact["OFF_OFF_RTG"]
    impact["DEF_SWING"] = impact["ON_DEF_RTG"] - impact["OFF_DEF_RTG"]  # Lower is better
    impact["WIN_SWING"] = impact["ON_WPCT"] - impact["OFF_WPCT"]

    # RAPM-lite: Use Ridge on lineup data to decompose
    # Build binary player-in-lineup matrix
    all_players = sorted(players["PLAYER_NAME"].unique())
    player_to_idx = {p: i for i, p in enumerate(all_players)}

    lineup_data = lineups[lineups["MIN"] >= 5].copy()  # Minimum 5 min
    X_rapm = np.zeros((len(lineup_data), len(all_players)))
    y_rapm = lineup_data["NET_RATING"].values
    w_rapm = lineup_data["MIN"].values  # Weight by minutes

    # Build a lookup from last name / short name to full name
    def name_variants(full_name):
        """Generate match variants: 'Stephen Curry' -> ['S. Curry', 'Curry', 'Stephen Curry']"""
        parts = full_name.split()
        variants = [full_name]
        if len(parts) >= 2:
            variants.append(f"{parts[0][0]}. {' '.join(parts[1:])}")  # S. Curry
            variants.append(parts[-1])  # Curry (last resort)
            # Handle suffixes like III, Jr, Sr
            if parts[-1] in ("III", "II", "IV", "Jr", "Jr.", "Sr", "Sr."):
                variants.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
                variants.append(f"{parts[0][0]}. {parts[-2]} {parts[-1]}")
        return variants

    variant_map = {}  # short_name -> player_idx
    for pname, idx in player_to_idx.items():
        for v in name_variants(pname):
            variant_map[v] = idx

    for i, (_, lr) in enumerate(lineup_data.iterrows()):
        names = str(lr["GROUP_NAME"]).split(" - ")
        for name in names:
            name = name.strip()
            if name in variant_map:
                X_rapm[i, variant_map[name]] = 1
            else:
                # Fuzzy fallback: check if lineup name contains player last name
                for pname, idx in player_to_idx.items():
                    last = pname.split()[-1]
                    if last in ("III", "II", "IV", "Jr", "Jr.", "Sr", "Sr."):
                        last = pname.split()[-2] if len(pname.split()) >= 2 else last
                    if last in name:
                        X_rapm[i, idx] = 1
                        break

    # Check coverage
    players_found = (X_rapm.sum(axis=0) > 0).sum()
    print(f"  RAPM: {players_found}/{len(all_players)} players matched in lineups, {(X_rapm.sum(axis=1) >= 2).sum()}/{len(lineup_data)} lineups have 2+ players matched")

    # Ridge regression — lower alpha for more differentiation with this data size
    ridge = Ridge(alpha=50, fit_intercept=True)
    ridge.fit(X_rapm, y_rapm, sample_weight=w_rapm)

    rapm_results = pd.DataFrame({
        "Player": all_players,
        "RAPM": ridge.coef_,
    }).sort_values("RAPM", ascending=False)

    # Merge with on/off
    impact_sorted = impact.sort_values("NET_SWING", ascending=False)

    h(3, "On/Off Court Impact")
    p("Net Rating when each player is on vs. off the court — the most direct measure of individual impact.")
    p("*Filtered to players with ≥200 on-court minutes for statistical reliability.*")
    blank()
    impact_display = impact_sorted[impact_sorted["ON_MIN"] >= 200]
    tbl("Player", "On MIN", "On-Court Net", "Off-Court Net", "Net Swing", "Off Swing", "Def Swing", "Win% Swing")
    for _, r in impact_display.iterrows():
        ns = r["NET_SWING"]
        row(
            r["PLAYER"], f"{r['ON_MIN']:.0f}",
            f"{r['ON_NET']:+.1f}", f"{r['OFF_NET']:+.1f}",
            f"**{'+' if ns >= 0 else ''}{ns:.1f}**",
            f"{'+' if r['OFF_SWING'] >= 0 else ''}{r['OFF_SWING']:.1f}",
            f"{'+' if r['DEF_SWING'] >= 0 else ''}{r['DEF_SWING']:.1f}",
            f"{'+' if r['WIN_SWING'] >= 0 else ''}{r['WIN_SWING'] * 100:.1f}%",
        )
    blank()

    h(3, "RAPM-Lite (Ridge Regression)")
    p("Regularized Adjusted Plus-Minus isolates each player's contribution by decomposing lineup net ratings.")
    blank()
    tbl("Rank", "Player", "RAPM", "Interpretation")
    for i, (_, r) in enumerate(rapm_results.iterrows()):
        rapm_val = r["RAPM"]
        if rapm_val > 3:
            interp = "🟢 Strong Positive"
        elif rapm_val > 0:
            interp = "🔵 Positive"
        elif rapm_val > -3:
            interp = "🟡 Neutral"
        else:
            interp = "🔴 Negative"
        row(i + 1, r["Player"], f"{rapm_val:+.2f}", interp)
    blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 5: FATIGUE / REST-DAY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("Model 5: Fatigue Analysis…")

h(2, "5. Fatigue & Rest-Day Analysis")
p("Linear regression and statistical tests to quantify how rest days affect player and team performance.")
blank()

# Team-level rest analysis from splits
rest_split = splits.get("DaysRestTeamDashboard")
if rest_split is not None and len(rest_split) > 0:
    h(3, "5.1 Team Performance by Rest Days")
    tbl("Rest Days", "GP", "Win%", "PTS", "FG%", "+/-", "Trend")
    for _, r in rest_split.iterrows():
        wp = r["W_PCT"]
        pm = r.get("PLUS_MINUS", 0)
        if wp >= 0.6:
            trend = "📈 Strong"
        elif wp >= 0.45:
            trend = "➡️ Average"
        else:
            trend = "📉 Weak"
        row(r["GROUP_VALUE"], int(r["GP"]), f"{wp * 100:.1f}%",
            f"{r['PTS']:.1f}", f"{r['FG_PCT'] * 100:.1f}%",
            f"{'+' if pm >= 0 else ''}{pm:.1f}", trend)
    blank()

# Player-level fatigue analysis using game logs
h(3, "5.2 Player Fatigue Regression")
p("For each key player, we regress game performance on minutes played in prior 3 games to quantify fatigue effects.")
blank()

# Calculate prior workload for each player
pgl_sorted = gamelogs_player.sort_values(["PLAYER_ID", "GAME_DATE"])
pgl_sorted["MIN_FLOAT"] = pd.to_numeric(pgl_sorted["MIN"], errors="coerce")

fatigue_results = []
for pid in players["PLAYER_ID"].unique():
    player_games = pgl_sorted[pgl_sorted["PLAYER_ID"] == pid].copy()
    if len(player_games) < 15:
        continue
    player_games = player_games.sort_values("GAME_DATE")
    player_games["prior_3g_min"] = player_games["MIN_FLOAT"].rolling(3, min_periods=1).mean().shift(1)
    player_games["game_date_dt"] = pd.to_datetime(player_games["GAME_DATE"])
    player_games["rest_days"] = player_games["game_date_dt"].diff().dt.days.fillna(2)
    player_games = player_games.dropna(subset=["prior_3g_min", "PTS"])
    player_games = player_games[player_games["MIN_FLOAT"] >= 10]

    if len(player_games) < 10:
        continue

    name = player_games.iloc[0]["PLAYER_NAME"]

    # Regression: PTS ~ prior_3g_min + rest_days
    X_fat = player_games[["prior_3g_min", "rest_days"]].values
    y_fat = player_games["PTS"].values

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_fat, y_fat)

    # Correlation
    corr_workload, p_workload = sp_stats.pearsonr(player_games["prior_3g_min"], player_games["PTS"])
    corr_rest, p_rest = sp_stats.pearsonr(player_games["rest_days"], player_games["PTS"])

    fatigue_results.append({
        "name": name,
        "games": len(player_games),
        "coef_workload": lr.coef_[0],
        "coef_rest": lr.coef_[1],
        "r2": lr.score(X_fat, y_fat),
        "corr_workload": corr_workload,
        "p_workload": p_workload,
        "corr_rest": corr_rest,
        "p_rest": p_rest,
    })

if fatigue_results:
    fat_df = pd.DataFrame(fatigue_results).sort_values("coef_workload")
    tbl("Player", "Games", "Workload Effect", "Rest Effect", "R²", "Fatigue Signal")
    for _, r in fat_df.iterrows():
        if r["coef_workload"] < -0.15 and r["p_workload"] < 0.1:
            signal = "⚠️ Fatigue-Sensitive"
        elif r["coef_workload"] > 0.1:
            signal = "💪 Endurance"
        else:
            signal = "➡️ Neutral"
        row(
            r["name"], int(r["games"]),
            f"{r['coef_workload']:+.3f} pts/min",
            f"{r['coef_rest']:+.3f} pts/day",
            f"{r['r2']:.3f}", signal,
        )
    blank()

    # Interpretation
    p("**Key Findings:**")
    most_fatigued = fat_df.iloc[0]
    if most_fatigued["coef_workload"] < -0.1:
        p(f"- **{most_fatigued['name']}** shows the strongest fatigue effect: each additional minute of average workload in prior 3 games correlates with {most_fatigued['coef_workload']:.3f} fewer points")
    most_rested = fat_df.sort_values("coef_rest", ascending=False).iloc[0]
    if most_rested["coef_rest"] > 0.1:
        p(f"- **{most_rested['name']}** benefits most from rest: each extra rest day adds {most_rested['coef_rest']:.2f} points to scoring")
    blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 6: LINEUP SYNERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("Model 6: Lineup Synergy…")

h(2, "6. Lineup Synergy Analysis")
p("Analyzing 5-man lineup data to identify the most effective combinations and the synergies between player pairs.")
blank()

# Top and bottom lineups
lineup_f = lineups[lineups["MIN"] >= 10].copy()  # At least 10 minutes
lineup_f = lineup_f.sort_values("NET_RATING", ascending=False)

h(3, "6.1 Best & Worst Lineups (10+ min)")

h(4, "Top 10 Lineups")
tbl("Lineup", "MIN", "Net RTG", "Off RTG", "Def RTG", "TS%", "W-L")
for _, r in lineup_f.head(10).iterrows():
    row(r["GROUP_NAME"], f"{r['MIN']:.0f}", f"{r['NET_RATING']:+.1f}",
        f"{r['OFF_RATING']:.1f}", f"{r['DEF_RATING']:.1f}",
        f"{r['TS_PCT'] * 100:.1f}%", f"{int(r['W'])}-{int(r['L'])}")
blank()

h(4, "Bottom 5 Lineups")
tbl("Lineup", "MIN", "Net RTG", "Off RTG", "Def RTG", "TS%")
for _, r in lineup_f.tail(5).iterrows():
    row(r["GROUP_NAME"], f"{r['MIN']:.0f}", f"{r['NET_RATING']:+.1f}",
        f"{r['OFF_RATING']:.1f}", f"{r['DEF_RATING']:.1f}",
        f"{r['TS_PCT'] * 100:.1f}%")
blank()

# Pair synergy analysis
h(3, "6.2 Player Pair Synergy")
p("Weighted average net rating of all lineups containing each player pair (minimum 20 combined minutes).")
blank()

pair_stats = defaultdict(lambda: {"net_sum": 0, "min_sum": 0, "lineups": 0})
for _, r in lineup_f.iterrows():
    names = [n.strip() for n in str(r["GROUP_NAME"]).split(" - ")]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = tuple(sorted([names[i], names[j]]))
            pair_stats[pair]["net_sum"] += r["NET_RATING"] * r["MIN"]
            pair_stats[pair]["min_sum"] += r["MIN"]
            pair_stats[pair]["lineups"] += 1

pair_results = []
for (p1, p2), stats in pair_stats.items():
    if stats["min_sum"] >= 20:
        pair_results.append({
            "pair": f"{p1} + {p2}",
            "weighted_net": stats["net_sum"] / stats["min_sum"],
            "total_min": stats["min_sum"],
            "lineups": stats["lineups"],
        })

pair_df = pd.DataFrame(pair_results).sort_values("weighted_net", ascending=False)

h(4, "Best Pairings")
tbl("Pair", "Weighted Net RTG", "Total MIN", "Lineups")
for _, r in pair_df.head(10).iterrows():
    row(r["pair"], f"{r['weighted_net']:+.1f}", f"{r['total_min']:.0f}", int(r["lineups"]))
blank()

h(4, "Worst Pairings")
tbl("Pair", "Weighted Net RTG", "Total MIN", "Lineups")
for _, r in pair_df.tail(5).iterrows():
    row(r["pair"], f"{r['weighted_net']:+.1f}", f"{r['total_min']:.0f}", int(r["lineups"]))
blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 7: PERFORMANCE MOMENTUM / STREAKINESS
# ═══════════════════════════════════════════════════════════════
print("Model 7: Momentum Detection…")

h(2, "7. Performance Momentum & Consistency Analysis")
p("Statistical tests for streakiness and momentum detection — determining if hot/cold streaks are real or just noise.")
blank()

h(3, "7.1 Team Momentum")
# Runs test for team wins/losses
wl_seq = gamelog.sort_values("GAME_DATE")["WL"].values
runs = 1
for i in range(1, len(wl_seq)):
    if wl_seq[i] != wl_seq[i - 1]:
        runs += 1

n_w = sum(wl_seq == "W")
n_l = sum(wl_seq == "L")
n = n_w + n_l
expected_runs = 1 + (2 * n_w * n_l) / n
var_runs = (2 * n_w * n_l * (2 * n_w * n_l - n)) / (n ** 2 * (n - 1))
z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
p_runs = 2 * (1 - sp_stats.norm.cdf(abs(z_runs)))

p(f"**Wald-Wolfowitz Runs Test:**")
p(f"- Observed runs: {runs}")
p(f"- Expected runs (random): {expected_runs:.1f}")
p(f"- Z-score: {z_runs:.2f}")
p(f"- p-value: {p_runs:.4f}")
if p_runs < 0.05:
    if z_runs < 0:
        p(f"- **Conclusion:** Significant clustering detected (p < 0.05) — **streaky team** 🔥❄️")
    else:
        p(f"- **Conclusion:** Significant alternation detected (p < 0.05) — **bouncy team** ↕️")
else:
    p(f"- **Conclusion:** No significant evidence of streakiness (p = {p_runs:.3f}) — results appear **random**")
blank()

# 7.2 Player consistency
h(3, "7.2 Player Scoring Consistency")
p("Coefficient of Variation (CV) of per-game scoring — lower CV means more consistent.")
blank()

consistency = []
for pid in players["PLAYER_ID"].unique():
    pg = pgl_sorted[pgl_sorted["PLAYER_ID"] == pid]
    pg_valid = pg[pd.to_numeric(pg["MIN"], errors="coerce") >= 10]
    if len(pg_valid) < 10:
        continue
    pts = pg_valid["PTS"].astype(float)
    mean_pts = pts.mean()
    std_pts = pts.std()
    cv = std_pts / mean_pts if mean_pts > 0 else 0
    # Autocorrelation (lag 1) — positive = momentum, negative = mean-revert
    if len(pts) >= 5:
        autocorr = pts.autocorr(lag=1)
    else:
        autocorr = 0

    # Runs test on above/below median
    above = (pts > pts.median()).astype(int).values
    runs_p = 1
    r_count = 1
    for i in range(1, len(above)):
        if above[i] != above[i - 1]:
            r_count += 1
    n_a = above.sum()
    n_b = len(above) - n_a
    if n_a > 0 and n_b > 0:
        exp_r = 1 + (2 * n_a * n_b) / (n_a + n_b)
        var_r = (2 * n_a * n_b * (2 * n_a * n_b - n_a - n_b)) / ((n_a + n_b) ** 2 * (n_a + n_b - 1))
        if var_r > 0:
            z_r = (r_count - exp_r) / np.sqrt(var_r)
            runs_p = 2 * (1 - sp_stats.norm.cdf(abs(z_r)))

    consistency.append({
        "name": pg_valid.iloc[0]["PLAYER_NAME"],
        "games": len(pg_valid),
        "ppg": mean_pts,
        "std": std_pts,
        "cv": cv,
        "autocorr": autocorr if not np.isnan(autocorr) else 0,
        "runs_p": runs_p,
        "streaky": runs_p < 0.1 and autocorr > 0.1,
    })

cons_df = pd.DataFrame(consistency).sort_values("cv")
tbl("Player", "PPG", "Std Dev", "CV", "Autocorr", "Streaky?")
for _, r in cons_df.iterrows():
    streak = "🔥 Yes" if r["streaky"] else "➡️ No"
    ac = r["autocorr"]
    if ac > 0.2:
        ac_label = f"{ac:.2f} (momentum)"
    elif ac < -0.2:
        ac_label = f"{ac:.2f} (mean-revert)"
    else:
        ac_label = f"{ac:.2f} (neutral)"
    row(r["name"], f"{r['ppg']:.1f}", f"{r['std']:.1f}", f"{r['cv']:.3f}",
        ac_label, streak)
blank()

# ═══════════════════════════════════════════════════════════════
# MODEL 8: COMPOSITE PLAYER VALUE
# ═══════════════════════════════════════════════════════════════
print("Model 8: Composite Value…")

h(2, "8. Composite Player Value Index")
p("Combining all ML model outputs into a single composite score that represents each player's overall value to the Warriors.")
blank()

# Build composite
composite = players[["PLAYER_NAME", "PTS", "REB", "AST", "MIN"]].copy()

# Add cluster label
composite = composite.merge(
    players[["PLAYER_NAME", "cluster"]].rename(columns={"cluster": "archetype_id"}),
    on="PLAYER_NAME", how="left",
)

# Add RAPM if available
if "rapm_results" in dir():
    composite = composite.merge(
        rapm_results.rename(columns={"Player": "PLAYER_NAME"}),
        on="PLAYER_NAME", how="left",
    )
else:
    composite["RAPM"] = 0

# Add on/off impact
if len(impact_sorted) > 0:
    composite = composite.merge(
        impact_sorted[["PLAYER", "NET_SWING"]].rename(columns={"PLAYER": "PLAYER_NAME"}),
        on="PLAYER_NAME", how="left",
    )
else:
    composite["NET_SWING"] = 0

# Add xFG over expected
if "player_xfg" in dir():
    composite = composite.merge(
        player_xfg[["player_name", "fg_over_expected"]].rename(columns={"player_name": "PLAYER_NAME"}),
        on="PLAYER_NAME", how="left",
    )
else:
    composite["fg_over_expected"] = 0

# Add consistency
if len(cons_df) > 0:
    composite = composite.merge(
        cons_df[["name", "cv"]].rename(columns={"name": "PLAYER_NAME", "cv": "consistency_cv"}),
        on="PLAYER_NAME", how="left",
    )
else:
    composite["consistency_cv"] = 0.5

composite = composite.fillna(0)

# Normalize components to 0-100
def norm(s, higher_better=True):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    n = (s - mn) / (mx - mn) * 100
    return n if higher_better else 100 - n


composite["score_production"] = 0.40 * norm(composite["PTS"]) + 0.30 * norm(composite["AST"]) + 0.30 * norm(composite["REB"])
composite["score_impact"] = 0.50 * norm(composite["RAPM"]) + 0.50 * norm(composite["NET_SWING"])
composite["score_shooting"] = norm(composite["fg_over_expected"])
composite["score_consistency"] = norm(composite["consistency_cv"], higher_better=False)

composite["COMPOSITE"] = (
    0.30 * composite["score_production"]
    + 0.30 * composite["score_impact"]
    + 0.20 * composite["score_shooting"]
    + 0.20 * composite["score_consistency"]
)

composite = composite.sort_values("COMPOSITE", ascending=False)

tbl("Rank", "Player", "Production", "Impact", "Shooting", "Consistency", "**Composite**")
for i, (_, r) in enumerate(composite.iterrows()):
    row(
        i + 1, r["PLAYER_NAME"],
        f"{r['score_production']:.0f}", f"{r['score_impact']:.0f}",
        f"{r['score_shooting']:.0f}", f"{r['score_consistency']:.0f}",
        f"**{r['COMPOSITE']:.1f}**",
    )
blank()

# MVP interpretation
mvp = composite.iloc[0]
p(f"**ML-Based Team MVP:** {mvp['PLAYER_NAME']} (Composite: {mvp['COMPOSITE']:.1f})")
p(f"- Production Score: {mvp['score_production']:.0f}/100")
p(f"- Impact Score: {mvp['score_impact']:.0f}/100")
p(f"- Shooting Score: {mvp['score_shooting']:.0f}/100")
p(f"- Consistency Score: {mvp['score_consistency']:.0f}/100")
blank()

# ═══════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════
print("Writing summary…")

h(2, "9. Executive Summary — ML Insights")
blank()

h(3, "Key Findings")
p("1. **Player Archetypes:** The roster naturally separates into " +
  f"{best_k} distinct archetypes, with clear differentiation between primary scorers and defensive anchors.")
blank()

p(f"2. **Win Predictors:** {importances.index[0]} is the single most important predictor of wins " +
  f"(importance: {importances.iloc[0]:.3f}), followed by {importances.index[1]} " +
  f"and {importances.index[2]}.")
blank()

if "player_xfg" in dir() and len(player_xfg) > 0:
    hot_players = player_xfg[player_xfg["fg_over_expected"] > 0.01]
    cold_players = player_xfg[player_xfg["fg_over_expected"] < -0.01]
    p(f"3. **Shooting Luck vs Skill:** {len(hot_players)} player{'s' if len(hot_players)!=1 else ''} shooting above model expectations, " +
      f"{len(cold_players)} below — the tight clustering around expectations indicates sustainable shot profiles rather than luck-driven variance.")
    blank()

if len(impact_sorted) > 0:
    # Filter to players with meaningful minutes (>200 on-court min)
    impact_meaningful = impact_sorted[impact_sorted["ON_MIN"] >= 200]
    if len(impact_meaningful) == 0:
        impact_meaningful = impact_sorted.head(5)
    best_impact = impact_meaningful.iloc[0]
    p(f"4. **Biggest Impact Player:** {best_impact['PLAYER']} has the largest on/off swing " +
      f"({best_impact['NET_SWING']:+.1f} net rating), meaning the team is " +
      f"{abs(best_impact['NET_SWING']):.1f} points per 100 possessions better with them on court.")
    blank()

p(f"5. **Team Momentum:** {'The Warriors show statistically significant streaky behavior' if p_runs < 0.05 else 'Win/loss patterns appear statistically random (no true momentum)'} " +
  f"(runs test p={p_runs:.3f}).")
blank()

p(f"6. **Composite Value:** {composite.iloc[0]['PLAYER_NAME']} leads the ML-derived composite value index " +
  f"({composite.iloc[0]['COMPOSITE']:.1f}), integrating production, impact, shooting efficiency, and consistency.")
blank()

h(3, "Actionable Recommendations")
p("Based on the ML analysis:")
blank()

# Generate recommendations from data
recs = []

# Fatigue management
if fatigue_results:
    fatigued = sorted([r for r in fatigue_results if r["coef_workload"] < -0.15], key=lambda x: x["coef_workload"])
    if fatigued:
        names = ", ".join(r["name"] for r in fatigued[:3])
        coefs = ", ".join(f"{r['coef_workload']:+.2f}" for r in fatigued[:3])
        recs.append(f"- **Fatigue Management:** Monitor workload for {names} — their scoring drops significantly with high prior-game minutes (workload coefficients: {coefs} pts/min)")
    else:
        recs.append("- **Fatigue Management:** No players show statistically significant fatigue sensitivity — current minute distribution appears sustainable")

# Best lineup
if len(lineup_f) > 0:
    best_lineup = lineup_f.iloc[0]
    recs.append(f"- **Optimal Lineup:** Deploy '{best_lineup['GROUP_NAME']}' more — it has the best net rating ({best_lineup['NET_RATING']:+.1f}) among tested lineups")

# Pair synergy
if len(pair_df) > 0:
    best_pair = pair_df.iloc[0]
    worst_pair = pair_df.iloc[-1]
    recs.append(f"- **Best Pairing:** Maximize minutes for {best_pair['pair']} (weighted net rating: {best_pair['weighted_net']:+.1f})")
    recs.append(f"- **Avoid Pairing:** Minimize {worst_pair['pair']} together (weighted net rating: {worst_pair['weighted_net']:+.1f})")

# Shooting regression candidates
if "player_xfg" in dir():
    below_exp = player_xfg[player_xfg["fg_over_expected"] < -0.03].head(2)
    for _, r in below_exp.iterrows():
        recs.append(f"- **Shooting Bounce-Back:** {r['player_name']} is shooting {abs(r['fg_over_expected']) * 100:.1f}% below expected — likely to improve with the same shot profile")

for rec in recs:
    p(rec)
blank()

# Footer
p("---")
p(f"*Models: scikit-learn {__import__('sklearn').__version__} | Data: stats.nba.com 2025-26 | Generated: {now}*")

# ═══════════════════════════════════════════════════════════════
# WRITE REPORT
# ═══════════════════════════════════════════════════════════════
report = "\n".join(lines)
output = REPORTS / "warriors_ml_insights_2025_26.md"
output.write_text(report)
print(f"\n✅ Report written to {output}")
print(f"   {len(lines)} lines, {len(report):,} characters")
