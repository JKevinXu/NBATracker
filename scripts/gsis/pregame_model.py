"""
GSIS Model M1 — Pre-Game Win Probability Predictor
Stacked ensemble: XGBoost + LightGBM + Logistic Regression → Meta-Learner
Includes SHAP explanations, calibration, and per-game predictions.
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Plotting Style ───────────────────────────────────────────────
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
})
GOLD = "#FFC72C"
BLUE = "#1D428A"
WHITE = "#FFFFFF"
GREEN = "#2ecc71"
RED = "#e74c3c"

# ── Friendly feature names ────────────────────────────────────────
FRIENDLY = {
    "L5_PTS": "Scoring (L5 avg)",
    "L10_PTS": "Scoring (L10 avg)",
    "L5_AST": "Assists (L5 avg)",
    "L10_AST": "Assists (L10 avg)",
    "L5_REB": "Rebounds (L5 avg)",
    "L10_REB": "Rebounds (L10 avg)",
    "L5_TOV": "Turnovers (L5 avg)",
    "L10_TOV": "Turnovers (L10 avg)",
    "L5_FG_PCT": "FG% (L5 avg)",
    "L10_FG_PCT": "FG% (L10 avg)",
    "L5_FG3_PCT": "3PT% (L5 avg)",
    "L5_FT_PCT": "FT% (L5 avg)",
    "L5_3PT_RATE": "3PT Rate (L5)",
    "L5_FTA_RATE": "FT Rate (L5)",
    "L5_TOV_RATE": "Turnover Rate (L5)",
    "L5_OREB_PCT": "Off. Reb% (L5)",
    "L5_AST_RATE": "Assist Rate (L5)",
    "L5_STL_RATE": "Steals (L5 avg)",
    "WIN_STREAK": "Win/Loss Streak",
    "L5_WIN_PCT": "Win% (L5)",
    "L10_WIN_PCT": "Win% (L10)",
    "SEASON_WIN_PCT": "Season Win%",
    "SEASON_PTS_AVG": "Season PPG",
    "SEASON_FG_PCT": "Season FG%",
    "HOME": "Home Game",
    "REST_DAYS": "Rest Days",
    "B2B": "Back-to-Back",
    "GAMES_IN_LAST_7": "Games in Last 7 Days",
    "MONTH": "Month",
    "DAY_OF_WEEK": "Day of Week",
    "GAME_NUM": "Game # in Season",
    "OPP_WIN_PCT": "Opponent Win%",
    "OPP_PPG": "Opponent PPG",
    "OPP_OPP_PPG": "Opp Def (PPG Allowed)",
    "OPP_NET_PPG": "Opponent Net PPG",
    "H2H_WIN_PCT": "H2H Win% This Season",
    "H2H_PTS_DIFF": "H2H Pts Diff",
    "H2H_GAMES": "H2H Games Played",
    "SEASON_MEETING_NUM": "Meeting # This Season",
    "CURRY_AVAILABLE": "Curry Available",
    "BUTLER_AVAILABLE": "Butler Available",
    "MELTON_AVAILABLE": "Melton Available",
    "GREEN_AVAILABLE": "Green Available",
    "KUMINGA_AVAILABLE": "Kuminga Available",
    "CURRY_FATIGUE": "Curry Fatigue (L3 min)",
    "BUTLER_FATIGUE": "Butler Fatigue (L3 min)",
    "MELTON_FATIGUE": "Melton Fatigue (L3 min)",
    "GREEN_FATIGUE": "Green Fatigue (L3 min)",
    "KUMINGA_FATIGUE": "Kuminga Fatigue (L3 min)",
}


def friendly(col):
    return FRIENDLY.get(col, col)


# ══════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ══════════════════════════════════════════════════════════════════

class PreGamePredictor:
    """Stacked ensemble for pre-game win probability."""

    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42,
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            verbose=-1, random_state=43,
        )
        self.lr_model = LogisticRegression(
            C=0.5, max_iter=1000, random_state=44,
        )
        self.meta_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=45,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_cols: list):
        """Train all base models and the meta-learner."""
        self.feature_cols = feature_cols
        X_arr = X[feature_cols].values.astype(float)
        y_arr = y.values.astype(int)
        n = len(y_arr)

    # Scale for logistic regression
        X_scaled = self.scaler.fit_transform(X_arr)

        # ── Manual OOF with TimeSeriesSplit ──
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_oof = np.full(n, 0.5)
        lgb_oof = np.full(n, 0.5)
        lr_oof = np.full(n, 0.5)

        for train_idx, val_idx in tscv.split(X_arr):
            # XGBoost
            self.xgb_model.fit(X_arr[train_idx], y_arr[train_idx])
            xgb_oof[val_idx] = self.xgb_model.predict_proba(X_arr[val_idx])[:, 1]
            # LightGBM
            self.lgb_model.fit(X_arr[train_idx], y_arr[train_idx])
            lgb_oof[val_idx] = self.lgb_model.predict_proba(X_arr[val_idx])[:, 1]
            # Logistic Regression
            self.lr_model.fit(X_scaled[train_idx], y_arr[train_idx])
            lr_oof[val_idx] = self.lr_model.predict_proba(X_scaled[val_idx])[:, 1]

        # ── Train meta-learner on OOF predictions ──
        meta_X = np.column_stack([xgb_oof, lgb_oof, lr_oof])
        self.meta_model.fit(meta_X, y_arr)

        # ── Refit base models on full data ──
        self.xgb_model.fit(X_arr, y_arr)
        self.lgb_model.fit(X_arr, y_arr)
        self.lr_model.fit(X_scaled, y_arr)

        self.is_fitted = True

        # ── Store OOF for evaluation ──
        self._oof_probs = self.meta_model.predict_proba(meta_X)[:, 1]
        self._oof_xgb = xgb_oof
        self._oof_lgb = lgb_oof
        self._oof_lr = lr_oof
        self._y_train = y_arr

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return win probability for each row."""
        X_arr = X[self.feature_cols].values.astype(float)
        X_scaled = self.scaler.transform(X_arr)

        p_xgb = self.xgb_model.predict_proba(X_arr)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X_arr)[:, 1]
        p_lr = self.lr_model.predict_proba(X_scaled)[:, 1]

        meta_X = np.column_stack([p_xgb, p_lgb, p_lr])
        return self.meta_model.predict_proba(meta_X)[:, 1]

    def evaluate(self) -> dict:
        """Evaluate using OOF predictions (no data leakage)."""
        probs = self._oof_probs
        preds = (probs >= 0.5).astype(int)
        y = self._y_train
        return {
            "accuracy": accuracy_score(y, preds),
            "log_loss": log_loss(y, probs),
            "auc_roc": roc_auc_score(y, probs),
            "brier_score": brier_score_loss(y, probs),
            "xgb_accuracy": accuracy_score(y, (self._oof_xgb >= 0.5).astype(int)),
            "lgb_accuracy": accuracy_score(y, (self._oof_lgb >= 0.5).astype(int)),
            "lr_accuracy": accuracy_score(y, (self._oof_lr >= 0.5).astype(int)),
        }

    def shap_explain(self, X: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP values using the XGBoost base model."""
        X_arr = X[self.feature_cols].values.astype(float)
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer(
            pd.DataFrame(X_arr, columns=self.feature_cols)
        )
        return shap_values

    def explain_game(self, X_row: pd.DataFrame, shap_vals: shap.Explanation,
                     game_idx: int) -> dict:
        """
        Return a dict of feature contributions for a single game.
        Positive = favors win, negative = favors loss.
        """
        vals = shap_vals.values[game_idx]
        feats = self.feature_cols
        data = X_row[self.feature_cols].values.flatten()
        contributions = []
        for fname, sval, fval in zip(feats, vals, data):
            contributions.append({
                "feature": fname,
                "friendly": friendly(fname),
                "shap_value": float(sval),
                "feature_value": float(fval),
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return contributions


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_model_comparison(metrics, img_dir):
    """Bar chart comparing base models vs ensemble."""
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = ["XGBoost", "LightGBM", "Logistic Reg", "Stacked\nEnsemble"]
    accs = [
        metrics["xgb_accuracy"] * 100, metrics["lgb_accuracy"] * 100,
        metrics["lr_accuracy"] * 100, metrics["accuracy"] * 100,
    ]
    colors = [BLUE, "#2ecc71", "#e67e22", GOLD]
    bars = ax.bar(model_names, accs, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold",
                color="white", fontsize=13)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pre-Game Model Accuracy Comparison (Time-Series CV)")
    ax.set_ylim(0, max(accs) + 12)
    ax.axhline(50, color="white", linestyle="--", alpha=0.3, label="Coin flip (50%)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(img_dir / "pregame_model_comparison.png", dpi=150)
    plt.close(fig)


def plot_pregame_shap_importance(shap_vals, feature_cols, img_dir):
    """SHAP feature importance bar chart."""
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [friendly(feature_cols[i]) for i in order]
    vals = [mean_abs[i] for i in order]
    colors = [GOLD if i < 5 else BLUE for i in range(len(order))]
    ax.barh(range(len(order)), vals[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Win Probability)")
    ax.set_title("Pre-Game Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(img_dir / "pregame_shap_importance.png", dpi=150)
    plt.close(fig)


def plot_pregame_timeline(df, probs, img_dir):
    """Timeline of pre-game predicted win probability vs actual outcomes."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(df)
    x = range(n)
    wins = df["WIN"].values
    for i in x:
        color = GREEN if wins[i] == 1 else RED
        ax.bar(i, probs[i], color=color, alpha=0.7, width=0.8)
    # Rolling average
    roll = pd.Series(probs).rolling(5, min_periods=1).mean()
    ax.plot(x, roll, color=GOLD, linewidth=2.5, label="5-game rolling avg")
    ax.axhline(0.5, color="white", linestyle="--", alpha=0.4, label="50% threshold")
    ax.set_xlabel("Game Number")
    ax.set_ylabel("Pre-Game Win Probability")
    ax.set_title("Pre-Game Win Probability Timeline — 2025-26 Season")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    for i in x:
        if i % 3 == 0:
            opp = df.iloc[i]["OPP_ABBREV"]
            ax.text(i, min(probs[i] + 0.04, 0.97), opp, ha="center",
                    fontsize=6, color="white", alpha=0.7, rotation=90)
    plt.tight_layout()
    fig.savefig(img_dir / "pregame_timeline.png", dpi=150)
    plt.close(fig)


def plot_calibration(y_true, probs, img_dir):
    """Calibration plot: predicted probability vs observed win rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_rates, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_rates.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
    ax.plot([0, 1], [0, 1], "--", color="white", alpha=0.4, label="Perfect calibration")
    ax.scatter(bin_centers, bin_rates, s=[c * 20 for c in bin_counts],
               color=GOLD, edgecolor="white", zorder=5)
    ax.plot(bin_centers, bin_rates, color=GOLD, linewidth=2)
    for bc, br, cnt in zip(bin_centers, bin_rates, bin_counts):
        ax.annotate(f"n={cnt}", (bc, br), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="white")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("Observed Win Rate")
    ax.set_title("Calibration Plot")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax2 = axes[1]
    ax2.hist(probs[y_true == 1], bins=15, alpha=0.7, color=GREEN, label="Actual Wins", edgecolor="white")
    ax2.hist(probs[y_true == 0], bins=15, alpha=0.7, color=RED, label="Actual Losses", edgecolor="white")
    ax2.set_xlabel("Predicted Win Probability")
    ax2.set_ylabel("Number of Games")
    ax2.set_title("Prediction Distribution by Outcome")
    ax2.legend(fontsize=9)
    ax2.axvline(0.5, color="white", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(img_dir / "pregame_calibration.png", dpi=150)
    plt.close(fig)


def plot_game_waterfall(contributions, game_label, img_dir, filename):
    """Waterfall chart for one game's SHAP contributions."""
    top_n = 12
    contribs = contributions[:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [c["friendly"] for c in contribs][::-1]
    vals = [c["shap_value"] for c in contribs][::-1]
    fvals = [c["feature_value"] for c in contribs][::-1]
    colors = [GREEN if v > 0 else RED for v in vals]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white",
                   linewidth=0.3, height=0.7)
    ax.set_yticks(range(len(names)))
    labels = [f"{n} = {fv:.2g}" for n, fv in zip(names, fvals)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="white", linewidth=0.5)
    ax.set_xlabel("SHAP Value (→ Win | ← Loss)")
    ax.set_title(f"Pre-Game Prediction Breakdown: {game_label}")
    plt.tight_layout()
    fig.savefig(img_dir / filename, dpi=150)
    plt.close(fig)


def plot_accuracy_by_confidence(y_true, probs, img_dir):
    """Show accuracy at different confidence levels."""
    fig, ax = plt.subplots(figsize=(10, 5))
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    accs = []
    counts = []
    for t in thresholds:
        mask = (probs >= t) | (probs <= 1 - t)
        if mask.sum() > 0:
            preds = (probs[mask] >= 0.5).astype(int)
            accs.append(accuracy_score(y_true[mask], preds) * 100)
            counts.append(mask.sum())
        else:
            accs.append(0)
            counts.append(0)
    bars = ax.bar([f">{t:.0%}" for t in thresholds], accs,
                  color=[BLUE, BLUE, GOLD, GOLD, GOLD], edgecolor="white",
                  linewidth=0.5, width=0.5)
    for bar, acc, cnt in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0f}%\n({cnt} games)", ha="center", fontsize=10,
                color="white", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Confidence Threshold")
    ax.set_title("Model Accuracy by Confidence Level")
    ax.set_ylim(0, max(accs) + 15)
    ax.axhline(50, color="white", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(img_dir / "pregame_accuracy_by_confidence.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(img_dir, report_path=None):
    """
    Train the pre-game predictor, evaluate, generate visualizations,
    and return results dict for the brief generator.
    """
    from scripts.gsis.features import build_feature_matrix

    print("  Building feature matrix …")
    df, feature_cols, target = build_feature_matrix()

    print(f"  {df.shape[0]} games × {len(feature_cols)} features")

    # ── Train ──
    print("  Training stacked ensemble …")
    model = PreGamePredictor()
    model.fit(df, df[target], feature_cols)

    # ── Evaluate ──
    metrics = model.evaluate()
    print(f"  Ensemble accuracy: {metrics['accuracy']:.1%}")
    print(f"  AUC-ROC:           {metrics['auc_roc']:.3f}")
    print(f"  Brier Score:       {metrics['brier_score']:.3f}")
    print(f"  Log-Loss:          {metrics['log_loss']:.3f}")

    # ── SHAP ──
    print("  Computing SHAP explanations …")
    shap_vals = model.shap_explain(df)

    # ── Per-game probabilities (refitted model on full data) ──
    all_probs = model.predict_proba(df)

    # ── Plots ──
    print("  Generating visualizations …")
    os.makedirs(img_dir, exist_ok=True)

    plot_model_comparison(metrics, img_dir)
    plot_pregame_shap_importance(shap_vals, feature_cols, img_dir)
    plot_pregame_timeline(df, model._oof_probs, img_dir)
    plot_calibration(model._y_train, model._oof_probs, img_dir)
    plot_accuracy_by_confidence(model._y_train, model._oof_probs, img_dir)

    # Best win & worst loss waterfalls
    wins_mask = df["WIN"] == 1
    losses_mask = df["WIN"] == 0
    if wins_mask.any():
        best_win_idx = model._oof_probs[wins_mask.values].argmax()
        actual_idx = df.index[wins_mask][best_win_idx]
        pos = df.index.get_loc(actual_idx)
        contribs = model.explain_game(df.iloc[[pos]], shap_vals, pos)
        label = f"{df.iloc[pos]['MATCHUP']} ({df.iloc[pos]['DATE'].strftime('%b %d')})"
        plot_game_waterfall(contribs, f"Best Win: {label}", img_dir,
                           "pregame_waterfall_best_win.png")

    if losses_mask.any():
        worst_loss_idx = model._oof_probs[losses_mask.values].argmin()
        actual_idx = df.index[losses_mask][worst_loss_idx]
        pos = df.index.get_loc(actual_idx)
        contribs = model.explain_game(df.iloc[[pos]], shap_vals, pos)
        label = f"{df.iloc[pos]['MATCHUP']} ({df.iloc[pos]['DATE'].strftime('%b %d')})"
        plot_game_waterfall(contribs, f"Worst Loss: {label}", img_dir,
                           "pregame_waterfall_worst_loss.png")

    # ── Last game analysis (most recent) ──
    last_idx = len(df) - 1
    last_game = df.iloc[last_idx]
    last_prob = model._oof_probs[last_idx]
    last_contribs = model.explain_game(df.iloc[[last_idx]], shap_vals, last_idx)

    # ── Feature importance ranking ──
    mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
    importance_order = np.argsort(mean_abs_shap)[::-1]
    top_features = [(feature_cols[i], mean_abs_shap[i]) for i in importance_order[:10]]

    results = {
        "metrics": metrics,
        "n_games": len(df),
        "n_features": len(feature_cols),
        "top_features": top_features,
        "last_game": {
            "matchup": last_game["MATCHUP"],
            "date": last_game["DATE"].strftime("%b %d, %Y"),
            "result": last_game["WL"],
            "prob": float(last_prob),
            "top_contribs": last_contribs[:8],
        },
        "oof_probs": model._oof_probs,
        "df": df,
        "model": model,
        "shap_vals": shap_vals,
        "feature_cols": feature_cols,
    }

    print("  ✅ Pre-Game Predictor complete.")
    return results


if __name__ == "__main__":
    from pathlib import Path

    IMG_DIR = Path("reports/game_briefs/figures")
    run(IMG_DIR)
