# ML Model Design — Warriors Game Strategy Intelligence System (GSIS)

*Author: NBATracker Analytics | Date: February 26, 2026*

---

## 1. Motivation & Gap Analysis

### 1.1 What We Already Have

| Model | Type | Limitation |
|---|---|---|
| XGBoost Win Predictor | Post-game explainer | Uses box-score stats *after* the game — can't predict before tip-off |
| K-Means Clustering | Static player archetypes | Doesn't change game-to-game; ignores opponent context |
| RAPM (Ridge) | Season-long impact | Single number per player; no situational granularity |
| Bayesian Projection | Season win total | Macro-level only; doesn't help with individual game decisions |
| Synergy Heatmap | Pairwise chemistry | Descriptive, not prescriptive; doesn't recommend lineups |
| xFG% Model | Expected shooting | Doesn't account for defender quality or fatigue |

### 1.2 The Gap

All existing models are **retrospective** — they explain what happened. None of them answer the questions a coaching staff asks *before* each game:

1. **"What's our win probability tonight?"** (Pre-game prediction)
2. **"What lineup should we start against this opponent?"** (Lineup optimization)
3. **"How should we adjust our strategy for this team?"** (Opponent-adaptive tactics)
4. **"Who's at fatigue risk and needs load management?"** (Player availability forecasting)
5. **"How will each player perform tonight?"** (Individual stat forecasting)

### 1.3 Proposed System

The **Game Strategy Intelligence System (GSIS)** is an integrated multi-model pipeline that provides **pre-game, actionable intelligence** for every upcoming Warriors game. It combines five interconnected models into a single decision-support system.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     GAME STRATEGY INTELLIGENCE SYSTEM                │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────────┐ │
│  │  DATA LAYER │   │ FEATURE ENG  │   │       MODEL LAYER         │ │
│  │             │──▶│              │──▶│                           │ │
│  │ • Game Logs │   │ • Rolling    │   │  M1: Pre-Game Predictor   │ │
│  │ • Player    │   │   Averages   │   │  M2: Lineup Optimizer     │ │
│  │   Stats     │   │ • Fatigue    │   │  M3: Opponent Classifier  │ │
│  │ • Lineups   │   │   Index      │   │  M4: Player Forecaster    │ │
│  │ • Opponent  │   │ • Matchup    │   │  M5: Load Manager         │ │
│  │   Scouting  │   │   Features   │   │                           │ │
│  │ • Schedule  │   │ • Contextual │   └─────────┬─────────────────┘ │
│  │ • Tracking  │   │   (H/A,rest) │             │                   │
│  └─────────────┘   └──────────────┘             ▼                   │
│                                        ┌─────────────────┐          │
│                                        │  INTEGRATION     │          │
│                                        │  LAYER           │          │
│                                        │                  │          │
│                                        │  Game-Day Brief  │          │
│                                        │  (unified report)│          │
│                                        └─────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Requirements

### 3.1 Existing Data (Already Cached)

| Source File | Contents | Used By |
|---|---|---|
| `gamelog.json` | Warriors game-by-game results (PTS, REB, AST, W/L, opponent, date) | M1, M3, M5 |
| `player_gamelogs.json` | Individual player game logs | M2, M4, M5 |
| `lineups.json` | 5-man lineup combinations with net rating | M2 |
| `on_off.json` | Player on/off court splits | M2, M4 |
| `player_adv.json` | Advanced player stats (TS%, USG%, PIE, NET_RTG) | M2, M4 |
| `player_base.json` | Base player stats (PTS, REB, AST, etc.) | M4 |
| `hustle.json` | Hustle stats (deflections, contested shots, charges) | M4 |
| `tracking_speed.json` | Speed/distance tracking | M5 |
| `clutch.json` | Clutch performance splits | M2 |
| `standings.json` | League standings (opponent strength) | M1, M3 |
| `splits.json` | Team splits by location, rest, opponent conference | M1, M3 |
| `shot_locations.json` | Team shot distribution by zone | M3 |
| `opp_shooting.json` | Opponent shooting against Warriors | M3 |

### 3.2 New Data Needed

| Data | Source | Purpose | API Endpoint |
|---|---|---|---|
| Opponent game logs (all 30 teams) | `stats.nba.com` | M1 needs opponent recent form | `leaguegamelog` |
| Full NBA schedule (remaining games) | `stats.nba.com` | M1, M5 need upcoming opponents | `leaguegamefinder` |
| Player injury reports | `stats.nba.com` or manual | M1, M2 need availability | Manual / RSS |
| Opponent defensive ratings by position | `stats.nba.com` | M3 needs matchup quality | `leaguedashteamstats` |
| Historical matchup data (Warriors vs each opponent) | `stats.nba.com` | M1 needs head-to-head trends | `teamgamelog` with filters |

### 3.3 Feature Engineering Pipeline

```
Raw Data ──▶ Clean & Join ──▶ Feature Engineering ──▶ Feature Store
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
           Rolling Features    Contextual Features   Interaction Features
           • L5/L10 PTS avg   • Home/Away           • Curry USG × OPP DEF RTG
           • L5 FG% trend     • Rest days           • Lineup RAPM × OPP pace
           • L5 TOV trend     • Back-to-back?       • Player fatigue × minutes
           • Win streak len   • Opponent rank        • Shot profile × OPP zone DEF
           • Form momentum    • Travel distance     
```

**Key Engineered Features (56 total):**

| # | Feature | Type | Description |
|---|---|---|---|
| 1–5 | `L5_PTS`, `L5_AST`, `L5_REB`, `L5_TOV`, `L5_FG_PCT` | Rolling | 5-game rolling averages |
| 6–10 | `L10_PTS`, `L10_AST`, `L10_REB`, `L10_TOV`, `L10_FG_PCT` | Rolling | 10-game rolling averages |
| 11 | `WIN_STREAK` | Momentum | Current consecutive W/L streak (positive=W, negative=L) |
| 12 | `L5_WIN_PCT` | Momentum | Win % over last 5 games |
| 13 | `HOME` | Context | 1 if home, 0 if away |
| 14 | `REST_DAYS` | Context | Days since last game (0=B2B, 1=normal, 2+=extra rest) |
| 15 | `B2B` | Context | Binary back-to-back flag |
| 16 | `OPP_WIN_PCT` | Opponent | Opponent's season win percentage |
| 17 | `OPP_NET_RTG` | Opponent | Opponent's net rating (pts/100 poss) |
| 18 | `OPP_DEF_RTG` | Opponent | Opponent's defensive rating |
| 19 | `OPP_PACE` | Opponent | Opponent's pace (possessions/game) |
| 20 | `OPP_L5_WIN_PCT` | Opponent | Opponent's recent 5-game form |
| 21–25 | `GSW_OFF_RTG`, `GSW_DEF_RTG`, `GSW_NET_RTG`, `GSW_PACE`, `GSW_TS_PCT` | Team | Warriors' current season rates |
| 26–30 | `CURRY_AVAILABLE`, `BUTLER_AVAILABLE`, `MELTON_AVAILABLE`, `GREEN_AVAILABLE`, `KUMINGA_AVAILABLE` | Availability | Binary flags for top 5 players |
| 31–35 | `CURRY_FATIGUE`, `BUTLER_FATIGUE`, `MELTON_FATIGUE`, etc. | Fatigue | Fatigue index (see M5) |
| 36–40 | `H2H_WIN_PCT`, `H2H_PTS_DIFF`, `H2H_GAMES`, `SEASON_MEETING_NUM` | Matchup | Head-to-head history this season |
| 41–45 | `OPP_CLUSTER` | Opponent | Opponent archetype cluster ID (see M3) |
| 46–50 | `MONTH`, `DAY_OF_WEEK`, `GAMES_IN_LAST_7`, `TRAVEL_CROSS_TZ` | Schedule | Calendar and travel features |
| 51–56 | `L5_3PT_RATE`, `L5_FTA_RATE`, `L5_TOV_RATE`, `L5_REB_PCT`, `L5_AST_RATE`, `L5_STL_RATE` | Rolling Rate | Possession-adjusted rolling rates |

---

## 4. Model M1 — Pre-Game Win Probability Predictor

### 4.1 Objective

Predict the probability that the Warriors win their next game **before tip-off**, using only information available at that point (no box-score stats).

### 4.2 Why This Matters

The current XGBoost model (71.4% accuracy) uses post-game stats and answers "what happened?" The pre-game model answers **"what will happen?"** — enabling the coaching staff to prepare strategy, set rotation plans, and manage expectations.

### 4.3 Model Selection: Stacked Ensemble

```
                    ┌──────────────────┐
                    │   Meta-Learner   │
                    │  (Logistic Reg)  │
                    └───────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │  XGBoost  │ │  LightGBM │ │  Logistic  │
        │           │ │           │ │ Regression │
        └───────────┘ └───────────┘ └───────────┘
              ▲             ▲             ▲
              └─────────────┼─────────────┘
                            │
                    ┌───────┴──────────┐
                    │  56 Features     │
                    │  (pre-game only) │
                    └──────────────────┘
```

**Why stacking?**
- XGBoost captures non-linear interactions (e.g., "rest matters more against elite opponents")
- LightGBM provides diversity via different tree-building strategy (leaf-wise vs level-wise)
- Logistic Regression provides a stable, interpretable baseline
- The meta-learner (logistic regression on the 3 base model outputs) combines their strengths

### 4.4 Training Strategy

| Aspect | Design Decision | Rationale |
|---|---|---|
| **Train set** | All Warriors games from 2023-24 + 2024-25 + 2025-26 (through today) | ~220 games; need multiple seasons for sufficient data |
| **Validation** | Time-series split (never train on future games) | Prevents data leakage from random CV |
| **Test set** | Last 10 games of 2025-26 (hold-out) | Realistic evaluation on most recent data |
| **Cross-validation** | Expanding-window CV with 5 splits | Respects temporal ordering |
| **Target** | Binary: WIN=1, LOSS=0 | Classification task |
| **Loss function** | Log-loss (binary cross-entropy) | Calibrated probabilities, not just accuracy |
| **Calibration** | Platt scaling on validation set | Ensures predicted probabilities are well-calibrated |

### 4.5 Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| **Log-Loss** | < 0.65 | Primary metric; measures probability calibration |
| **AUC-ROC** | > 0.70 | Discrimination ability |
| **Brier Score** | < 0.22 | Mean squared error of probability predictions |
| **Accuracy** | > 62% | Secondary; pre-game prediction is harder than post-game |
| **Calibration Plot** | Linear | Predicted probabilities should match observed win rates |

### 4.6 SHAP Integration

Apply SHAP to the pre-game XGBoost to generate a **pre-game scouting report** showing:
- Top 5 factors favoring a Warriors win
- Top 5 factors favoring a Warriors loss
- Overall win probability with confidence interval

### 4.7 Example Output

```
┌─────────────────────────────────────────────────────────┐
│  PRE-GAME PREDICTION: Warriors vs Lakers (Feb 28)       │
│  Win Probability: 58.3% [52.1% – 64.5%]                │
│                                                          │
│  ✅ Favoring Warriors:                                   │
│     • Warriors L5 form: 4-1 (+0.08 to win prob)         │
│     • Home game (+0.06)                                  │
│     • 2 rest days (+0.04)                                │
│     • Lakers L5: 2-3 (+0.03)                             │
│     • Curry available (+0.02)                            │
│                                                          │
│  ❌ Favoring Opponent:                                   │
│     • Lakers DEF RTG: 108.2 (top 10) (-0.05)            │
│     • H2H this season: 0-1 (-0.03)                      │
│     • Warriors 3-game road trip fatigue (-0.02)          │
│                                                          │
│  📋 Recommended Focus: Ball security (TOV < 14)         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Model M2 — Dynamic Lineup Optimizer

### 5.1 Objective

Given the available players and the upcoming opponent, recommend the **optimal starting lineup** and **top 3 closing lineups** by maximizing predicted net rating.

### 5.2 Why This Matters

The current synergy heatmap shows pairwise chemistry but can't recommend full 5-man lineups. With 12 available players, there are C(12,5) = **792 possible lineups** — too many for a coach to evaluate manually.

### 5.3 Model Design: Lineup Net Rating Predictor + Combinatorial Search

```
Step 1: Train a Lineup Rating Model
─────────────────────────────────────
Input:  10 binary features (player_in_lineup × available roster)
        + 5 opponent context features (OPP_DEF_RTG, OPP_PACE, etc.)
        + 3 situation features (game_time, score_diff, clutch_flag)
Output: Predicted Net Rating for this lineup

Step 2: Enumerate & Score All Valid Lineups
─────────────────────────────────────────────
For each valid 5-man combination:
  • Score with the lineup model
  • Apply position-compatibility constraints
  • Apply minutes-balance constraints

Step 3: Return Top-K Lineups
────────────────────────────
  • Best overall lineup (maximize Net Rating)
  • Best closing lineup (clutch-weighted)
  • Best defensive lineup (minimize OPP scoring)
  • Best offensive lineup (maximize own scoring)
```

### 5.4 Lineup Rating Model: Gradient Boosted Regressor

| Aspect | Design |
|---|---|
| **Algorithm** | XGBoost Regressor |
| **Features** | 10 player indicators + 5 opponent + 3 situational = 18 features |
| **Target** | Lineup NET_RTG (from `lineups.json`) |
| **Training data** | All lineups with ≥20 possessions (filter out noise) |
| **Regularization** | Weighted by possessions (more data = more trust) |

### 5.5 Position Compatibility Constraints

Not every 5-player combination makes basketball sense. Apply hard constraints:

```python
POSITION_SLOTS = {
    "PG": ["Curry", "Melton", "Podziemski", "Spencer"],
    "SG": ["Hield", "Melton", "Moody", "Richard"],
    "SF": ["Butler", "Santos", "Moody", "Kuminga"],
    "PF": ["Green", "Kuminga", "Butler", "Santos"],
    "C":  ["Horford", "Jackson-Davis", "Post", "Green"],
}
# Each lineup must have at least one player eligible for each slot
```

### 5.6 Optimization Strategy

With ~800 valid lineups per game, **brute-force enumeration** is feasible (no need for approximate optimization). For each valid lineup:

1. Predict net rating vs specific opponent
2. Apply fatigue penalties (reduce predicted rating for fatigued players)
3. Apply availability filter (remove injured/resting players)
4. Rank by predicted net rating
5. Return top 5 with situation labels

### 5.7 Example Output

```
┌──────────────────────────────────────────────────────────┐
│  LINEUP RECOMMENDATIONS vs Lakers                        │
│                                                           │
│  🏀 Best Starting 5 (predicted Net Rating: +6.2):        │
│     PG: Curry  SG: Melton  SF: Butler  PF: Green  C: Post│
│                                                           │
│  ⏱️ Best Closing 5 (clutch Net Rating: +8.1):            │
│     PG: Curry  SG: Melton  SF: Butler  PF: Green  C: TJD │
│                                                           │
│  🛡️ Best Defensive 5 (predicted OPP OFF RTG: 104.3):    │
│     PG: Payton  SG: Melton  SF: Santos  PF: Green  C: TJD│
│                                                           │
│  ⚡ Best Offensive 5 (predicted OFF RTG: 118.7):          │
│     PG: Curry  SG: Hield  SF: Butler  PF: Kuminga  C: Post│
└──────────────────────────────────────────────────────────┘
```

---

## 6. Model M3 — Opponent Archetype Classifier

### 6.1 Objective

Classify each NBA team into one of k archetypes based on their playing style, then map each archetype to a recommended Warriors counter-strategy.

### 6.2 Why This Matters

Preparing 29 different game plans is impractical. Clustering opponents into 4–5 archetypes lets the coaching staff prepare a **small set of flexible game plans** that cover every opponent.

### 6.3 Model Design: K-Means on Team Style Features

**Input features (per opponent, 15 dimensions):**

| Feature | What It Captures |
|---|---|
| PACE | Tempo / speed of play |
| OFF_RTG | Offensive quality |
| DEF_RTG | Defensive quality |
| 3PT_RATE | Perimeter orientation |
| FTA_RATE | Free throw / driving tendency |
| REB_PCT | Rebounding dominance |
| TOV_PCT | Ball security |
| AST_PCT | Ball movement |
| STL_PCT | Defensive aggression |
| BLK_PCT | Rim protection |
| PAINT_PTS_PCT | Interior scoring tendency |
| FB_PTS_PCT | Transition offense |
| 2ND_CHANCE_PCT | Offensive rebounding payoff |
| OPP_3PT_PCT | Perimeter defense quality |
| OPP_PAINT_PTS | Interior defense quality |

**Clustering:**
- K-Means with k = 4–6 (selected via silhouette score)
- StandardScaler normalization before clustering
- Each cluster gets a descriptive label based on centroid characteristics

### 6.4 Expected Archetypes (Hypothetical)

| Archetype | Description | Example Teams | Warriors Counter-Strategy |
|---|---|---|---|
| 🏃 **Pace Pushers** | High pace, high 3PT rate, transition-heavy | OKC, Indiana | Slow the pace; limit transition; crash defensive glass |
| 🏰 **Fortress Defense** | Low pace, elite DEF RTG, grind-it-out | Cleveland, Minnesota | Prioritize Curry PnR; attack switches; maximize FTA |
| 🎯 **Perimeter Snipers** | High 3PT rate + accuracy, spread floor | Boston, Dallas | Close out aggressively; switch-everything defense; attack inside |
| 💪 **Paint Beasts** | High paint scoring, low 3PT, dominant rebounding | NYK, Memphis | Pack the paint; wall off drives; run them off 3-point line |
| ⚖️ **Balanced** | Average across all metrics | LAL, Phoenix | Default game plan; exploit specific matchup edges |

### 6.5 Strategy Mapping

For each archetype, compute the Warriors' historical record and performance metrics. Identify:
- **Best stats in wins** vs this archetype
- **Worst stats in losses** vs this archetype
- **Recommended emphasis** (3 specific tactical focuses)

---

## 7. Model M4 — Player Performance Forecaster

### 7.1 Objective

Predict each player's stat line (PTS, REB, AST, FG%) for the upcoming game, accounting for opponent, fatigue, recent trend, and role.

### 7.2 Why This Matters

Enables the coaching staff to set realistic expectations, plan rotation minutes, and identify potential breakout or slump risks *before the game*.

### 7.3 Model Design: Gradient Boosted Regressor (Per Player, Per Stat)

**One model per (player, stat) pair.** For the top 8 rotation players × 4 key stats = **32 individual models**.

```
For each player p and stat s (PTS, REB, AST, FG%):

  Input Features (22):
  ├── Rolling stats: L3, L5, L10 averages of stat s
  ├── Season average of stat s
  ├── Opponent DEF RTG (or specific: OPP_PTS_ALLOWED_TO_PG for guards)
  ├── Opponent PACE
  ├── Home/Away
  ├── Rest days
  ├── Fatigue index (from M5)
  ├── Minutes L3 average
  ├── USG% L5 trend
  ├── Previous meeting stats vs this opponent
  ├── Back-to-back flag
  └── Month (captures season-long development arc)

  Output: Predicted value of stat s for player p
  
  Model: XGBoost Regressor (100 trees, depth 3, learning rate 0.1)
  Evaluation: MAE, R² via 5-fold time-series CV
```

### 7.4 Confidence Intervals

Use **quantile regression** (XGBoost with `quantile` objective at α=0.1 and α=0.9) to generate 80% prediction intervals alongside point estimates.

### 7.5 Example Output

```
┌──────────────────────────────────────────────────────────┐
│  PLAYER FORECASTS vs Lakers (Feb 28)                     │
│                                                           │
│  Stephen Curry                                            │
│    PTS: 26.3 [19.8 – 33.1]    FG%: 46.2% [38% – 55%]   │
│    AST: 5.8 [3.2 – 8.4]      REB: 4.1 [2.0 – 6.5]     │
│    ⬆️ Trend: PTS rising (L5 avg: 28.4 vs season: 25.1)  │
│                                                           │
│  Jimmy Butler III                                         │
│    PTS: 18.9 [12.4 – 25.7]    FG%: 50.1% [41% – 59%]   │
│    AST: 3.5 [1.8 – 5.2]      REB: 5.2 [2.8 – 7.6]     │
│    ➡️ Trend: Stable                                       │
│                                                           │
│  ⚠️ Slump Risk: Hield (L5 FG%: 34.2%, below L20: 41.8%) │
│  🔥 Breakout Signal: Santos (L3 PTS: 14.3, above avg 8.1)│
└──────────────────────────────────────────────────────────┘
```

---

## 8. Model M5 — Player Load & Fatigue Manager

### 8.1 Objective

Compute a **fatigue index** (0–100) for each player entering every game, combining minutes load, rest, travel, and age. Flag players at high fatigue risk and recommend minutes caps.

### 8.2 Why This Matters

The Warriors' core players (Curry age 37, Butler age 36, Green age 35) are among the oldest in the NBA. Fatigue management is critical for both injury prevention and performance optimization down the stretch. The existing rest-day analysis showed performance drops on back-to-backs — this model quantifies the risk *per player*.

### 8.3 Fatigue Index Formula

```
FATIGUE_INDEX(player, game) = w1 × MINUTES_LOAD
                             + w2 × REST_PENALTY
                             + w3 × SCHEDULE_DENSITY
                             + w4 × AGE_FACTOR
                             + w5 × TRAVEL_FACTOR

Where:
  MINUTES_LOAD     = (L5 avg minutes) / (season avg minutes)     [0–2 range]
  REST_PENALTY     = 1.0 if B2B, 0.5 if 1-day rest, 0.0 if 2+   [0–1 range]
  SCHEDULE_DENSITY = (games in last 7 days) / 4                   [0–1 range]
  AGE_FACTOR       = max(0, (age - 28) / 12)                     [0–1 range]
  TRAVEL_FACTOR    = (miles traveled in L3 days) / 3000           [0–1 range]
```

**Weight calibration:** Fit weights w1–w5 via Ridge Regression where the target is the **performance drop** (player's stat deviation below season average) in each game. This learns which fatigue factors most predict performance decline.

### 8.4 Minutes Recommendation Engine

```python
if fatigue_index > 80:
    recommendation = "REST (DNP) or hard cap at 20 minutes"
elif fatigue_index > 60:
    recommendation = f"Limit to {season_avg_min * 0.8:.0f} minutes"
elif fatigue_index > 40:
    recommendation = f"Normal minutes ({season_avg_min:.0f})"
else:
    recommendation = f"Full workload available ({season_avg_min + 3:.0f} max)"
```

### 8.5 Injury Risk Correlation

Track whether high fatigue index games correlate with:
- Performance drops (below season average)
- Increased turnovers
- Decreased defensive hustle
- Subsequent missed games

Over time, this builds an **early warning system** for injury risk.

### 8.6 Example Output

```
┌──────────────────────────────────────────────────────────┐
│  FATIGUE REPORT — Feb 28 vs Lakers                       │
│                                                           │
│  🔴 HIGH FATIGUE (60+):                                  │
│    Stephen Curry    — Fatigue: 72 — "Cap at 28 min"      │
│      (B2B, 36.2 avg L5 min, age 37, cross-country)       │
│    Draymond Green   — Fatigue: 65 — "Cap at 25 min"      │
│      (3 games in 5 days, age 35)                          │
│                                                           │
│  🟡 MODERATE (40–60):                                    │
│    Jimmy Butler III — Fatigue: 48 — "Normal (32 min)"    │
│    De'Anthony Melton— Fatigue: 44 — "Normal (28 min)"    │
│                                                           │
│  🟢 LOW (<40):                                           │
│    Podziemski, Kuminga, Moody, Santos, Post, Hield       │
│    → All available for full workload                      │
└──────────────────────────────────────────────────────────┘
```

---

## 9. Integration Layer — Game-Day Intelligence Brief

### 9.1 Unified Output

All five models feed into a single **Game-Day Brief** generated the morning of each game:

```
╔══════════════════════════════════════════════════════════════╗
║         WARRIORS GAME-DAY INTELLIGENCE BRIEF                ║
║         vs Los Angeles Lakers — Feb 28, 2026                ║
║         Chase Center (Home) | 7:30 PM PT                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 WIN PROBABILITY: 58.3% [52.1% – 64.5%]          [M1]  ║
║     Key factors: Home (+6%), L5 form 4-1 (+8%),             ║
║     Lakers DEF RTG top-10 (-5%)                              ║
║                                                              ║
║  🎯 OPPONENT TYPE: 🏰 Fortress Defense              [M3]  ║
║     Counter-strategy: Attack switches via Curry PnR;         ║
║     Maximize FTA; Limit turnovers to <13                     ║
║                                                              ║
║  🏀 RECOMMENDED STARTING 5:                          [M2]  ║
║     Curry / Melton / Butler / Green / Post                   ║
║     Predicted Net Rating: +6.2 vs LAL                        ║
║                                                              ║
║  ⏱️ CLOSING 5 (if within 5 pts):                     [M2]  ║
║     Curry / Melton / Butler / Green / TJD                    ║
║                                                              ║
║  👤 PLAYER FORECASTS:                                [M4]  ║
║     Curry: 26 pts / 6 ast / 4 reb (FG% ~46%)                ║
║     Butler: 19 pts / 4 ast / 5 reb (FG% ~50%)               ║
║     🔥 Breakout watch: Santos (trend ↑↑)                    ║
║     ⚠️ Slump risk: Hield (L5 FG%: 34%)                     ║
║                                                              ║
║  🏥 FATIGUE ALERTS:                                  [M5]  ║
║     🔴 Curry: Fatigue 72 → cap 28 min (B2B + age)          ║
║     🔴 Green: Fatigue 65 → cap 25 min                       ║
║     🟢 All others: full availability                         ║
║                                                              ║
║  📋 TOP 3 TACTICAL PRIORITIES:                              ║
║     1. Turnovers < 13 (vs LAL press defense)                 ║
║     2. FTA > 25 (attack Lakers' switching scheme)            ║
║     3. Pace < 98 (don't let Lakers run in transition)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 10. Implementation Plan

### 10.1 Phased Rollout

| Phase | Models | Timeline | Deliverable |
|---|---|---|---|
| **Phase 1** | M1 (Pre-Game Predictor) | Week 1–2 | Pre-game win probability for remaining 26 games |
| **Phase 2** | M5 (Fatigue Manager) | Week 2–3 | Daily fatigue dashboard |
| **Phase 3** | M3 (Opponent Classifier) | Week 3–4 | Opponent scouting cards for remaining schedule |
| **Phase 4** | M4 (Player Forecaster) | Week 4–5 | Per-player stat projections |
| **Phase 5** | M2 (Lineup Optimizer) | Week 5–6 | Lineup recommendations engine |
| **Phase 6** | Integration | Week 6–7 | Game-Day Brief generator |

### 10.2 Technical Stack

| Component | Technology | Rationale |
|---|---|---|
| Models | XGBoost, LightGBM, scikit-learn | Already in environment; proven on tabular data |
| Explainability | SHAP | Already integrated; enables trust and debugging |
| Data pipeline | Python + pandas | Consistent with existing codebase |
| Feature store | JSON files in `web/cache/` | Consistent with existing caching approach |
| Report generation | Python → Markdown + PNG | Consistent with existing report pipeline |
| API serving | FastAPI (existing `web/app.py`) | Add `/api/pregame/{opponent}` endpoint |
| Scheduling | Cron / manual trigger | Run daily for next-game brief |

### 10.3 New Files to Create

```
scripts/
├── gsis/
│   ├── __init__.py
│   ├── features.py          # Feature engineering pipeline (56 features)
│   ├── pregame_model.py     # M1: Pre-game win predictor
│   ├── lineup_optimizer.py  # M2: Lineup optimizer
│   ├── opponent_cluster.py  # M3: Opponent archetype classifier
│   ├── player_forecast.py   # M4: Player stat forecaster
│   ├── fatigue_manager.py   # M5: Load management
│   ├── game_day_brief.py    # Integration: generate unified brief
│   └── prefetch_opponent.py # Fetch opponent data to cache
│
web/
├── app.py                   # Add new API endpoints
│
reports/
├── game_briefs/             # Output directory for game-day briefs
│   ├── 2026-02-28_vs_LAL.md
│   └── ...
```

### 10.4 New API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/pregame/{team_abbrev}` | GET | Returns pre-game prediction JSON |
| `/api/lineup/{team_abbrev}` | GET | Returns recommended lineups |
| `/api/fatigue` | GET | Returns current fatigue index for all players |
| `/api/forecast/{player_name}` | GET | Returns player stat forecast |
| `/api/brief/{date}` | GET | Returns full game-day brief |

---

## 11. Evaluation & Monitoring

### 11.1 Model Quality Checks

| Model | Metric | Check Frequency | Action if Below Threshold |
|---|---|---|---|
| M1 Pre-Game | Brier Score < 0.22 | Every 10 games | Retrain with expanded window |
| M2 Lineup | MAE of Net Rating < 5.0 | Weekly | Add more lineup data; increase min possession threshold |
| M3 Opponent | Silhouette > 0.30 | Monthly | Re-cluster with updated team stats |
| M4 Forecaster | MAE within 20% of player's std dev | Every 10 games | Tune hyperparameters; add features |
| M5 Fatigue | Correlation with perf drop > 0.3 | Bi-weekly | Recalibrate weights |

### 11.2 Backtesting Protocol

Before deployment, backtest each model on the 2024-25 Warriors season:
1. Train on 2023-24 data
2. Predict each 2024-25 game in chronological order
3. Update model after each game (expanding window)
4. Report metrics across the full 82-game season
5. Compare to naive baselines (always predict home team wins, always predict season average)

### 11.3 Drift Detection

Monitor feature distributions weekly. If any feature's distribution shifts significantly (KL-divergence > 0.1 from training distribution), trigger a model refresh.

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Small sample size** (56 games/season) | High | Models overfit | Strong regularization; cross-season training; Bayesian priors |
| **Injury data unavailable programmatically** | Medium | M1/M2 accuracy drops | Manual injury flags; default to "available" if unknown |
| **API rate limiting** (opponent data) | Medium | Can't fetch fresh data | Expand prefetch cache; daily batch updates |
| **Lineup data sparse** (many lineups < 20 poss) | High | M2 unreliable for rare combos | Possession-weighted training; fall back to pairwise synergy for unseen lineups |
| **Opponent cluster instability** | Medium | Strategy recommendations flip | Ensemble of k=4,5,6; use soft cluster membership |
| **Model staleness mid-season** | Medium | Predictions degrade | Expanding-window retraining after every 5 games |

---

## 13. Success Criteria

The GSIS is successful if:

1. **M1 accuracy ≥ 60%** on held-out games (vs ~52% for "always predict home team" baseline)
2. **M2 recommended lineups** have higher actual net rating than non-recommended lineups (measured retroactively)
3. **M4 forecasts** have lower MAE than simple season-average baseline for ≥ 6 of 8 players
4. **M5 fatigue alerts** correlate (r > 0.3) with actual performance drops
5. **Coaching staff qualitative feedback** rates the Game-Day Brief as "useful" or "very useful"

---

*This design document serves as the blueprint for implementing the GSIS. Phase 1 (Pre-Game Win Predictor) can begin immediately using existing cached data. Shall we proceed with implementation?*
