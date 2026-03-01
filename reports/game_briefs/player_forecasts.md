# Player Performance Forecaster

*Generated: March 01, 2026 | Model: Per-Player XGBoost Regressors with Quantile Intervals*

Individual XGBoost models trained on each player's game-by-game history predict their next-game
stat line. Features include rolling averages (L3/L5/L10), trends, rest days, home/away, and
game context. Quantile regression provides 80% prediction intervals.

---

## 1. Next-Game Forecasts

**How to read this chart:** Each card shows one player's predicted stats for the next game.
Gold bars = point forecast. White error bars = 80% prediction interval (the true value will fall
in this range 80% of the time). Blue diamonds = season average for comparison. Arrows below each
stat show the trend (↑ = L5 avg > season avg, ↓ = below, → = stable).

![Player Forecasts](figures/player_forecasts.png)

| Player | PTS | REB | AST | FG% | PTS Trend |
|---|---|---|---|---|---|
| LeBron James | **16.1** [15–25] | 5.8 [3–7] | 4.9 [4–6] | 50.0% | → |
| Austin Reaves | **15.6** [9–20] | 2.9 [1–4] | 2.4 [2–4] | 40.0% | ↑ |
| Jake LaRavia | **11.4** [5–15] | 5.2 [2–5] | 1.2 [0–2] | 40.0% | ↓ |
| Marcus Smart | **11.4** [3–14] | 1.0 [0–3] | 2.9 [1–4] | 40.0% | ↓ |
| Luke Kennard | **8.3** [8–9] | 2.9 [1–3] | 0.4 [0–3] | 60.0% | ↑ |
| Jaxson Hayes | **5.5** [3–9] | 6.5 [4–8] | 1.2 [0–2] | 70.0% | ↑ |
| Deandre Ayton | **5.0** [2–17] | 4.3 [4–10] | 0.6 [0–2] | 40.0% | ↓ |
| Bronny James | **0.9** [0–2] | 0.2 [0–0] | 0.1 [0–1] | 10.0% | ↓ |
| Rui Hachimura | **0.9** [0–14] | 1.2 [0–2] | 0.2 [0–1] | 30.0% | ↓ |
| Jarred Vanderbilt | **0.6** [0–6] | 2.4 [0–4] | 0.2 [0–2] | 20.0% | ↓ |
| Dalton Knecht | **0.2** [0–5] | 0.1 [0–2] | 0.1 [0–1] | 10.0% | ↓ |
| Adou Thiero | **0.0** [0–0] | 0.0 [0–0] | 0.0 [0–0] | -0.0% | ↓ |
| Drew Timme | — | — | — | — | → |
| Luka Dončić | — | — | — | — | → |
| Maxi Kleber | **0.0** [0–0] | 0.2 [0–2] | 0.1 [0–1] | 10.0% | ↓ |
| Nick Smith Jr. | — | — | — | — | → |

## 2. Scoring Trends + Forecast

**How to read this chart:** Each panel tracks a player's game-by-game scoring (green = above
average, red = below). The gold line is the 5-game rolling average. The final gold bar with
error bar is the forecast for the *next* game. Compare the forecast to the season average
(dashed line) to see if the player is trending up or down.

![Forecast Trends](figures/forecast_trends.png)

## 3. Model Accuracy

**How to read this chart:** Cross-validated Mean Absolute Error (MAE) for each player-stat model.
Lower bars = more accurate predictions. For context, a PTS MAE of 6.0 means the model's
predictions are off by ±6 points on average. For FG%, divide by 100 (MAE of 0.08 = ±8% FG%).

![Forecast Accuracy](figures/forecast_accuracy.png)

## 4. Trend Alerts

### 🔥 Hot Streaks (L5 > Season Average)

- **Austin Reaves**: L5 avg 18.8 vs season 13.6 (**+5.2**) — forecast: 15.6
- **Luke Kennard**: L5 avg 8.8 vs season 7.2 (**+1.6**) — forecast: 8.3
- **Jaxson Hayes**: L5 avg 6.4 vs season 5.6 (**+0.8**) — forecast: 5.5

### ⚠️ Cold Streaks (L5 < Season Average)

- **Dalton Knecht**: L5 avg 0.0 vs season 3.4 (**-3.4**) — forecast: 0.2
- **Deandre Ayton**: L5 avg 8.0 vs season 11.2 (**-3.2**) — forecast: 5.0
- **Jake LaRavia**: L5 avg 6.4 vs season 9.2 (**-2.8**) — forecast: 11.4
- **Rui Hachimura**: L5 avg 7.8 vs season 9.5 (**-1.7**) — forecast: 0.9
- **Jarred Vanderbilt**: L5 avg 2.4 vs season 4.1 (**-1.7**) — forecast: 0.6
- **Marcus Smart**: L5 avg 6.4 vs season 7.9 (**-1.5**) — forecast: 11.4
- **Bronny James**: L5 avg 0.0 vs season 1.1 (**-1.1**) — forecast: 0.9
- **Maxi Kleber**: L5 avg 1.0 vs season 1.3 (**-0.3**) — forecast: 0.0
- **Adou Thiero**: L5 avg 0.4 vs season 0.6 (**-0.2**) — forecast: 0.0

---
*Models: 16 players × 4 stats = 64 individual XGBoost models*
*Generated: March 01, 2026 | Data: stats.nba.com 2025-26*