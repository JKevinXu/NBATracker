"""
GSIS Model M2 — Dynamic Lineup Optimizer
Scores all valid 5-man lineup combinations from available players
against the upcoming opponent context and recommends top lineups.
"""

import json, os, warnings
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
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
# DATA
# ══════════════════════════════════════════════════════════════════

def load_lineups():
    """Load 5-man lineup data for the configured team."""
    team = get_team()
    lu = _load("lineups")
    df = _rs_to_df(lu)
    df = df[df["TEAM_ABBREVIATION"] == team].copy()
    for c in ["MIN", "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
              "EFG_PCT", "TS_PCT", "PIE", "GP", "W", "L", "W_PCT", "POSS"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    # Parse player names from GROUP_NAME
    df["PLAYERS"] = df["GROUP_NAME"].apply(
        lambda x: sorted([n.strip() for n in x.split(" - ")]) if isinstance(x, str) else []
    )
    return df


def get_roster():
    """Get current rotation players (≥15 GP) for the configured team."""
    team = get_team()
    try:
        lb = _load("league_base")
        base = _rs_to_df(lb)
        base = base[base["TEAM_ABBREVIATION"] == team]
        base["GP"] = pd.to_numeric(base["GP"], errors="coerce")
        roster = base[base["GP"] >= 15][["PLAYER_NAME", "GP"]].to_dict("records")
        return [r["PLAYER_NAME"] for r in roster]
    except FileNotFoundError:
        # Derive from player gamelogs
        pg = _load("player_gamelogs")
        df = _rs_to_df(pg)
        df = df[df["TEAM_ABBREVIATION"] == team]
        gp = df.groupby("PLAYER_NAME")["GAME_ID"].nunique()
        return gp[gp >= 15].index.tolist()


def shorten(name):
    """S. Curry format → Stephen Curry matching."""
    parts = name.strip().split()
    if len(parts) == 2 and len(parts[0]) <= 2:
        return parts[0].rstrip('.') + '. ' + parts[1]
    return name


# ══════════════════════════════════════════════════════════════════
# LINEUP RATING MODEL
# ══════════════════════════════════════════════════════════════════

def build_lineup_features(lineups_df, full_roster):
    """Create binary player-indicator features for each observed lineup."""
    # Map short names in lineups to full roster names
    name_map = {}
    for full in full_roster:
        parts = full.split()
        short = parts[0][0] + '. ' + parts[-1]
        name_map[short] = full
        # Handle suffixes
        if len(parts) > 2:
            short2 = parts[0][0] + '. ' + ' '.join(parts[1:])
            name_map[short2] = full
        name_map[full] = full

    all_names = sorted(full_roster)
    rows = []
    targets_net = []
    targets_off = []
    targets_def = []

    for _, lu in lineups_df.iterrows():
        if lu["MIN"] < 5 or lu["POSS"] < 10:
            continue
        players_short = lu["PLAYERS"]
        players_full = []
        for p in players_short:
            matched = name_map.get(p)
            if not matched:
                # Fuzzy last-name match
                last = p.split()[-1] if p else ""
                candidates = [r for r in full_roster if r.split()[-1] == last]
                matched = candidates[0] if candidates else None
            if matched:
                players_full.append(matched)

        if len(players_full) < 3:  # need at least 3 identified players
            continue

        # Binary indicators
        row = {name: 1 if name in players_full else 0 for name in all_names}
        row["LINEUP_SIZE"] = len(players_full)  # 3-5 matched
        rows.append(row)
        targets_net.append(lu["NET_RATING"])
        targets_off.append(lu["OFF_RATING"])
        targets_def.append(lu["DEF_RATING"])

    feat_df = pd.DataFrame(rows)
    return feat_df, np.array(targets_net), np.array(targets_off), np.array(targets_def), all_names


def train_lineup_model(feat_df, target):
    """Train XGBoost regressor to predict lineup rating."""
    X = feat_df.values.astype(float)
    y = target.astype(float)

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=5.0, random_state=42,
    )
    model.fit(X, y)

    # CV score
    if len(X) >= 10:
        tscv = TimeSeriesSplit(n_splits=3)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        cv_mae = -scores.mean()
    else:
        cv_mae = np.nan

    r2 = r2_score(y, model.predict(X))
    return model, cv_mae, r2


# ══════════════════════════════════════════════════════════════════
# COMBINATORIAL OPTIMIZER
# ══════════════════════════════════════════════════════════════════

def score_all_lineups(model_net, model_off, model_def, feat_cols, available_players, all_names, top_k=10):
    """Enumerate all C(n,5) lineups and score them."""
    combos = list(combinations(available_players, 5))
    print(f"    Scoring {len(combos)} possible 5-man lineups …")

    results = []
    for combo in combos:
        row = {name: 1 if name in combo else 0 for name in all_names}
        row["LINEUP_SIZE"] = 5
        X = pd.DataFrame([row])[feat_cols].values.astype(float)
        net = float(model_net.predict(X)[0])
        off = float(model_off.predict(X)[0])
        dfn = float(model_def.predict(X)[0])
        results.append({
            "players": combo,
            "net_rating": round(net, 1),
            "off_rating": round(off, 1),
            "def_rating": round(dfn, 1),
        })

    results.sort(key=lambda x: -x["net_rating"])
    return results


# ══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════

def plot_top_lineups(lineups_scored, img_dir):
    """Show top 10 lineups by predicted net rating."""
    top = lineups_scored[:10]
    fig, ax = plt.subplots(figsize=(14, 7))

    names = ["\n".join(lu["players"]) for lu in top]
    nets = [lu["net_rating"] for lu in top]
    offs = [lu["off_rating"] for lu in top]
    defs = [lu["def_rating"] for lu in top]

    x = np.arange(len(top))
    w = 0.25
    ax.barh(x - w, nets, height=w, color=GOLD, label="Net Rating", alpha=0.9)
    ax.barh(x, offs, height=w, color=GREEN, label="Off Rating", alpha=0.8)
    ax.barh(x + w, defs, height=w, color=RED, label="Def Rating", alpha=0.8)

    ax.set_yticks(x)
    # Short labels
    short_names = []
    for lu in top:
        n = ", ".join([p.split()[-1] for p in lu["players"]])
        short_names.append(n)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rating (pts per 100 possessions)")
    ax.set_title("Top 10 Predicted Lineups by Net Rating")
    ax.legend(fontsize=9)
    ax.axvline(0, color=WHITE, alpha=0.3)
    plt.tight_layout()
    fig.savefig(img_dir / "lineup_rankings.png", dpi=150)
    plt.close(fig)


def plot_player_impact(model_net, feat_cols, all_names, img_dir):
    """Show feature importance (which players most impact lineup Net Rating)."""
    importances = model_net.feature_importances_
    imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": importances})
    # Filter to actual player names (not LINEUP_SIZE)
    imp_df = imp_df[imp_df["Feature"] != "LINEUP_SIZE"]
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [GOLD if name in all_names else BLUE for name in imp_df["Feature"]]
    ax.barh(range(len(imp_df)), imp_df["Importance"].values, color=colors, alpha=0.9)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels([n.split()[-1] for n in imp_df["Feature"].values], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Player Impact on Lineup Net Rating")
    plt.tight_layout()
    fig.savefig(img_dir / "lineup_player_impact.png", dpi=150)
    plt.close(fig)


def plot_observed_vs_predicted(model_net, feat_df, target_net, img_dir):
    """Scatter plot of observed vs predicted net rating."""
    X = feat_df.values.astype(float)
    preds = model_net.predict(X)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(target_net, preds, color=GOLD, alpha=0.5, s=30)
    mn = min(target_net.min(), preds.min()) - 5
    mx = max(target_net.max(), preds.max()) + 5
    ax.plot([mn, mx], [mn, mx], color=WHITE, linestyle="--", alpha=0.5)
    ax.set_xlabel("Observed Net Rating")
    ax.set_ylabel("Predicted Net Rating")
    ax.set_title("Lineup Rating Model: Observed vs Predicted")
    r2 = r2_score(target_net, preds)
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
            fontsize=12, color=GOLD, fontweight="bold", va="top")
    plt.tight_layout()
    fig.savefig(img_dir / "lineup_obs_pred.png", dpi=150)
    plt.close(fig)


def plot_best_pairs(lineups_scored, img_dir):
    """Show best 2-man combinations based on how often they appear in top lineups."""
    top = lineups_scored[:20]  # top 20 lineups
    pair_scores = {}
    for lu in top:
        for pair in combinations(lu["players"], 2):
            key = tuple(sorted(pair))
            if key not in pair_scores:
                pair_scores[key] = []
            pair_scores[key].append(lu["net_rating"])

    pair_df = pd.DataFrame([
        {"Pair": f"{k[0].split()[-1]} + {k[1].split()[-1]}",
         "Avg Net": np.mean(v), "Count": len(v)}
        for k, v in pair_scores.items()
    ])
    pair_df = pair_df.sort_values("Avg Net", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(pair_df)), pair_df["Avg Net"].values, color=GOLD, alpha=0.9)
    ax.set_yticks(range(len(pair_df)))
    ax.set_yticklabels(pair_df["Pair"].values, fontsize=9)
    ax.set_xlabel("Avg Predicted Net Rating (in top-20 lineups)")
    ax.set_title("Best Player Pairings")
    for i, (bar, cnt) in enumerate(zip(bars, pair_df["Count"].values)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"({cnt}x)", va="center", fontsize=8, color="#a0a0a0")
    plt.tight_layout()
    fig.savefig(img_dir / "lineup_best_pairs.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════

def generate_report(lineups_scored, model_stats, img_dir, report_path):
    md = []
    p = md.append

    p("# Dynamic Lineup Optimizer")
    p("")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Model: XGBoost Regressors for Net/Off/Def Rating*")
    p("")
    p("This module trains XGBoost models on observed 5-man lineup data, then scores **every valid")
    p("combination** of available rotation players to find optimal lineups. Three separate models predict")
    p("net, offensive, and defensive ratings — enabling lineup recommendations optimized for different game situations.")
    p("")
    p("---")
    p("")

    # ── Top Lineups ──
    p("## 1. Top Recommended Lineups")
    p("")
    p("**How to read:** Lineups ranked by predicted net rating (offense − defense). Gold = net rating,")
    p("green = offensive rating, red = defensive rating. Higher net = more dominant.")
    p("")
    p("![Lineup Rankings](figures/lineup_rankings.png)")
    p("")

    p("### 🏀 Best Overall Lineup")
    p("")
    best = lineups_scored[0]
    p(f"**{' — '.join(best['players'])}**")
    p(f"- Predicted Net Rating: **{best['net_rating']:+.1f}**")
    p(f"- Off Rating: {best['off_rating']:.1f} | Def Rating: {best['def_rating']:.1f}")
    p("")

    # Best defensive
    by_def = sorted(lineups_scored[:50], key=lambda x: x["def_rating"])
    best_def = by_def[0]
    p("### 🛡️ Best Defensive Lineup")
    p("")
    p(f"**{' — '.join(best_def['players'])}**")
    p(f"- Def Rating: **{best_def['def_rating']:.1f}** | Net Rating: {best_def['net_rating']:+.1f}")
    p("")

    # Best offensive
    by_off = sorted(lineups_scored[:50], key=lambda x: -x["off_rating"])
    best_off = by_off[0]
    p("### 🔥 Best Offensive Lineup")
    p("")
    p(f"**{' — '.join(best_off['players'])}**")
    p(f"- Off Rating: **{best_off['off_rating']:.1f}** | Net Rating: {best_off['net_rating']:+.1f}")
    p("")

    # ── Top 5 Table ──
    p("### Complete Top-10 Rankings")
    p("")
    p("| Rank | Lineup | Net | Off | Def |")
    p("|---|---|---|---|---|")
    for i, lu in enumerate(lineups_scored[:10], 1):
        names = ", ".join([n.split()[-1] for n in lu["players"]])
        p(f"| {i} | {names} | {lu['net_rating']:+.1f} | {lu['off_rating']:.1f} | {lu['def_rating']:.1f} |")
    p("")

    # ── Player Impact ──
    p("## 2. Player Impact on Lineup Quality")
    p("")
    p("**How to read:** Feature importance from the XGBoost net-rating model shows which players")
    p("most influence lineup quality. Higher importance = this player's presence/absence has the")
    p("largest effect on whether a lineup is predicted to have a high or low net rating.")
    p("")
    p("![Player Impact](figures/lineup_player_impact.png)")
    p("")

    # ── Model Fit ──
    p("## 3. Model Calibration")
    p("")
    p("**How to read:** Each dot is an observed lineup. If the model were perfect, all dots would lie")
    p("on the dashed diagonal. The R² value tells us how much variance in actual lineup performance")
    p("the model captures.")
    p("")
    p("![Observed vs Predicted](figures/lineup_obs_pred.png)")
    p("")
    p(f"- **Training R²:** {model_stats['r2']:.3f}")
    p(f"- **CV MAE:** {model_stats['cv_mae']:.1f}")
    p(f"- **Lineups in training data:** {model_stats['n_lineups']}")
    p("")

    # ── Best Pairs ──
    p("## 4. Best Player Pairings")
    p("")
    p("**How to read:** Which 2-player combinations appear most frequently in the top-20 predicted lineups")
    p("and have the highest average net rating? These are your highest-synergy duos.")
    p("")
    p("![Best Pairs](figures/lineup_best_pairs.png)")
    p("")

    p("---")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Data: stats.nba.com 2025-26*")

    report_path.write_text("\n".join(md))
    print(f"  📄 Report: {report_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(img_dir=None, report_path=None):
    if img_dir is None:
        img_dir = ROOT / "reports" / "game_briefs" / "figures"
    if report_path is None:
        report_path = ROOT / "reports" / "game_briefs" / "lineup_recommendations.md"
    img_dir, report_path = Path(img_dir), Path(report_path)
    os.makedirs(img_dir, exist_ok=True)

    print("M2 — Dynamic Lineup Optimizer")
    print("  Loading lineup data …")
    lineups_df = load_lineups()
    roster = get_roster()
    print(f"  Roster: {len(roster)} rotation players | Observed lineups: {len(lineups_df)}")

    print("  Building features …")
    feat_df, target_net, target_off, target_def, all_names = build_lineup_features(lineups_df, roster)
    feat_cols = list(feat_df.columns)
    print(f"  Training data: {len(feat_df)} lineups × {len(feat_cols)} features")

    print("  Training models …")
    model_net, cv_mae_net, r2_net = train_lineup_model(feat_df, target_net)
    model_off, _, _ = train_lineup_model(feat_df, target_off)
    model_def, _, _ = train_lineup_model(feat_df, target_def)
    print(f"  Net Rating model: R²={r2_net:.3f}, CV MAE={cv_mae_net:.1f}")

    print("  Scoring all lineup combinations …")
    lineups_scored = score_all_lineups(
        model_net, model_off, model_def, feat_cols, roster, all_names
    )
    print(f"  Best lineup: {lineups_scored[0]['net_rating']:+.1f} net rating")

    print("  Generating visualizations …")
    plot_top_lineups(lineups_scored, img_dir)
    plot_player_impact(model_net, feat_cols, all_names, img_dir)
    plot_observed_vs_predicted(model_net, feat_df, target_net, img_dir)
    plot_best_pairs(lineups_scored, img_dir)

    print("  Generating report …")
    model_stats = {"r2": r2_net, "cv_mae": cv_mae_net, "n_lineups": len(feat_df)}
    generate_report(lineups_scored, model_stats, img_dir, report_path)
    print("  ✅ Lineup Optimizer complete.")
    return lineups_scored


if __name__ == "__main__":
    run()
