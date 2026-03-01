# Dynamic Lineup Optimizer

*Generated: March 01, 2026 | Model: XGBoost Regressors for Net/Off/Def Rating*

This module trains XGBoost models on observed 5-man lineup data, then scores **every valid
combination** of available rotation players to find optimal lineups. Three separate models predict
net, offensive, and defensive ratings — enabling lineup recommendations optimized for different game situations.

---

## 1. Top Recommended Lineups

**How to read:** Lineups ranked by predicted net rating (offense − defense). Gold = net rating,
green = offensive rating, red = defensive rating. Higher net = more dominant.

![Lineup Rankings](figures/lineup_rankings.png)

### 🏀 Best Overall Lineup

**Austin Reaves — Dalton Knecht — Jake LaRavia — Jaxson Hayes — Maxi Kleber**
- Predicted Net Rating: **+29.4**
- Off Rating: 123.9 | Def Rating: 96.1

### 🛡️ Best Defensive Lineup

**Austin Reaves — Dalton Knecht — Jaxson Hayes — Marcus Smart — Maxi Kleber**
- Def Rating: **96.0** | Net Rating: +29.0

### 🔥 Best Offensive Lineup

**Austin Reaves — Dalton Knecht — Jaxson Hayes — Marcus Smart — Maxi Kleber**
- Off Rating: **124.0** | Net Rating: +29.0

### Complete Top-10 Rankings

| Rank | Lineup | Net | Off | Def |
|---|---|---|---|---|
| 1 | Reaves, Knecht, LaRavia, Hayes, Kleber | +29.4 | 123.9 | 96.1 |
| 2 | Reaves, Knecht, Hayes, Smart, Kleber | +29.0 | 124.0 | 96.0 |
| 3 | Thiero, Reaves, Knecht, Hayes, Smart | +28.8 | 123.8 | 96.2 |
| 4 | Reaves, Knecht, Timme, Hayes, Smart | +28.8 | 123.8 | 96.2 |
| 5 | Reaves, Knecht, Hayes, Dončić, Smart | +28.8 | 123.8 | 96.2 |
| 6 | Reaves, Knecht, Hayes, Smart, Jr. | +28.8 | 123.8 | 96.2 |
| 7 | Thiero, Reaves, Knecht, LaRavia, Hayes | +28.7 | 123.8 | 96.2 |
| 8 | Reaves, Knecht, Timme, LaRavia, Hayes | +28.7 | 123.8 | 96.2 |
| 9 | Reaves, Knecht, LaRavia, Hayes, Dončić | +28.7 | 123.8 | 96.2 |
| 10 | Reaves, Knecht, LaRavia, Hayes, Jr. | +28.7 | 123.8 | 96.2 |

## 2. Player Impact on Lineup Quality

**How to read:** Feature importance from the XGBoost net-rating model shows which players
most influence lineup quality. Higher importance = this player's presence/absence has the
largest effect on whether a lineup is predicted to have a high or low net rating.

![Player Impact](figures/lineup_player_impact.png)

## 3. Model Calibration

**How to read:** Each dot is an observed lineup. If the model were perfect, all dots would lie
on the dashed diagonal. The R² value tells us how much variance in actual lineup performance
the model captures.

![Observed vs Predicted](figures/lineup_obs_pred.png)

- **Training R²:** 0.665
- **CV MAE:** 26.0
- **Lineups in training data:** 38

## 4. Best Player Pairings

**How to read:** Which 2-player combinations appear most frequently in the top-20 predicted lineups
and have the highest average net rating? These are your highest-synergy duos.

![Best Pairs](figures/lineup_best_pairs.png)

---
*Generated: March 01, 2026 | Data: stats.nba.com 2025-26*