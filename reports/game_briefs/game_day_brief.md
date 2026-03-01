# 🏀 GSIS Game-Day Intelligence Brief
## Warriors vs. BOS — Feb 19, 2026

*Generated: March 01, 2026 09:55 | System: Game Strategy Intelligence System (GSIS)*
*Venue: Home | Models: 5 interconnected ML systems | Data: stats.nba.com 2025-26*

---

## 📋 Executive Summary

**Top Game-Day Priorities:**

1. ⚠️ **Monitor fatigue** for Curry, Curry, Spencer — consider minutes restrictions
2. 🔥 **Ride the hot hand**: Horford, Melton, Green trending above season averages
3. 📉 **Cold streak watch**: Post, Richard — adjust expectations / increase touches
4. 🏀 **Optimal lineup available**: projected +77.3 net rating

---

## 1. Pre-Game Win Probability (M1)

*Full analysis: [pregame_win_predictor.md](pregame_win_predictor.md)*

The stacked ensemble (XGBoost + LightGBM + Logistic Regression) provides a calibrated
win probability using only pre-game information (recent form, rest, opponent quality, etc.).

![Win Probability Factors](figures/pregame_shap_summary.png)

## 2. Opponent Scouting Report (M3)

*Full analysis: [opponent_scouting.md](opponent_scouting.md)*

**Opponent: BOS**

The K-Means archetype classifier has categorized all 30 teams by playing style.
See the full scouting report for this opponent's archetype, profile, and recommended
counter-strategies.

![Archetype Map](figures/opponent_pca_clusters.png)

## 3. Lineup Recommendations (M2)

*Full analysis: [lineup_recommendations.md](lineup_recommendations.md)*

### 🏀 Recommended Starting Lineup

**De'Anthony Melton — Gary Payton II — Gui Santos — Kristaps Porziņģis — Moses Moody**
- Predicted Net Rating: **+77.3**
- Off: 161.5 | Def: 79.8

### 🛡️ Best Defensive Lineup

**Brandin Podziemski — De'Anthony Melton — Gary Payton II — Gui Santos — Moses Moody**
- Def Rating: **77.3** | Net: +62.1

![Lineup Rankings](figures/lineup_rankings.png)

## 4. Player Performance Forecasts (M4)

*Full analysis: [player_forecasts.md](player_forecasts.md)*

| Player | PTS Forecast | REB | AST | Trend |
|---|---|---|---|---|
| Stephen Curry | **23.0** [22–36] | 1.5 | 2.3 | → |
| De'Anthony Melton | **17.1** [6–18] | 3.1 | 1.3 | ↑ |
| Jimmy Butler III | **16.9** [15–22] | 3.5 | 4.6 | ↑ |
| Will Richard | **16.8** [3–17] | 4.6 | 1.8 | ↓ |
| Gui Santos | **15.6** [9–17] | 5.4 | 2.2 | ↑ |
| Gary Payton II | **13.0** [6–14] | 2.8 | 2.1 | ↑ |
| Brandin Podziemski | **11.8** [7–13] | 6.6 | 5.5 | → |
| Moses Moody | **11.1** [8–14] | 3.1 | 1.9 | ↑ |

![Player Forecasts](figures/player_forecasts.png)

## 5. Fatigue & Load Management (M5)

*Full analysis: [fatigue_dashboard.md](fatigue_dashboard.md)*

| Player | Fatigue Index | Zone | Recommendation |
|---|---|---|---|
| Stephen Curry | 🟠 **69** | Orange | 🟠 Limit to 25 min |
| Seth Curry | 🟠 **64** | Orange | 🟠 Limit to 12 min |
| Pat Spencer | 🟠 **61** | Orange | 🟠 Limit to 12 min |
| Gui Santos | 🟠 **60** | Orange | 🟠 Limit to 12 min |
| Jimmy Butler III | 🟠 **56** | Orange | 🟠 Limit to 24 min |
| Al Horford | 🟡 **54** | Yellow | 🟡 Normal (20 min) |
| Gary Payton II | 🟡 **52** | Yellow | 🟡 Normal (13 min) |
| Jonathan Kuminga | 🟡 **49** | Yellow | 🟡 Normal (20 min) |
| Draymond Green | 🟡 **47** | Yellow | 🟡 Normal (26 min) |
| Moses Moody | 🟡 **46** | Yellow | 🟡 Normal (25 min) |
| Buddy Hield | 🟡 **45** | Yellow | 🟡 Normal (20 min) |
| Brandin Podziemski | 🟡 **43** | Yellow | 🟡 Normal (26 min) |
| Malevy Leons | 🟡 **42** | Yellow | 🟡 Normal (4 min) |
| Quinten Post | 🟡 **40** | Yellow | 🟡 Normal (17 min) |

![Fatigue Dashboard](figures/fatigue_dashboard.png)

## 6. Tactical Takeaways

Based on the integrated GSIS analysis:

1. **Feed Horford** — forecasted for 5 PTS, trending above season average
2. **Manage Curry's minutes** — fatigue index 69, consider <25 min
3. **Closing lineup**: Melton, II, Santos, Porziņģis, Moody (Net: +77.3)

---

## 7. Glossary

| Term | Definition |
|---|---|
| Net Rating | Points scored minus points allowed per 100 possessions |
| Off Rating | Points scored per 100 possessions |
| Def Rating | Points allowed per 100 possessions (lower = better) |
| Fatigue Index | 0–100 scale (0=fresh, 100=exhausted) based on minutes, rest, age |
| SHAP | Feature importance method showing each factor's contribution to prediction |
| Archetype | Team playing-style cluster from K-Means algorithm |
| Stacked Ensemble | Combining multiple ML models via a meta-learner for better predictions |
| Quantile Regression | Predicts ranges (80% CI) instead of single point estimates |
| 80% CI | 80% prediction interval — true value falls in this range 80% of the time |
| L5/L10 | Rolling average over last 5 or 10 games |

---

*GSIS v1.0 — March 01, 2026 | 5 Models | 09:55 | Golden State Warriors*