# Golden State Warriors — Advanced ML Insights & Recommendations

*Generated: February 24, 2026 | Models: XGBoost, SHAP, Bayesian Inference, K-Means, Ridge Regression*

This report uses advanced machine learning models and statistical analysis to identify actionable areas for improvement, quantify player impact, project season outcomes, and provide data-driven recommendations for the Warriors' coaching staff and front office.

---

## 1. XGBoost Win Predictor + SHAP Explanations

**What is this model?** We trained an XGBoost gradient-boosted decision tree classifier on every Warriors game this season (56 games). The model takes 11 per-game statistics — points, rebounds, assists, steals, blocks, turnovers, FG%, 3PT%, plus 5-game rolling averages of PTS, AST, and FG% — and predicts whether the Warriors win or lose. Cross-validated accuracy: **71.4%**.

**Why XGBoost?** XGBoost is the gold standard for tabular data prediction. Unlike simpler models (logistic regression), it captures non-linear interactions — for example, high assists AND high FG% together may matter more than either alone. It also handles the small sample size (56 games) better than deep learning approaches.


### 1.1 SHAP Feature Importance

**What is SHAP?** SHAP (SHapley Additive exPlanations) is a game-theory-based method that assigns each feature a contribution value for every individual prediction. Unlike simple feature importance (which just says "FG% is important"), SHAP tells you *how much* FG% moved the win probability *for each specific game*.

**How to read this chart:** Each horizontal bar represents one feature. The length of the bar is the **mean absolute SHAP value** across all games — the longer the bar, the more that feature influences the win/loss outcome. Gold bars are the top 3 most important features; blue bars are less impactful. A feature with a SHAP value of 0.975 means that, on average, it shifts the model's win probability by ±97.5 percentage points across games.

![SHAP Feature Importance](figures/shap_importance.png)

### 1.2 SHAP Beeswarm — Every Game Explained

**How to read this chart:** Each dot represents one game. Features are listed vertically (most important at top). For each feature, dots are spread horizontally: dots pushed to the **right** increased the model's win probability for that game; dots pushed **left** decreased it. The color indicates the actual value of that stat in the game (red = high value, blue = low value).

**What to look for:** If a feature has red dots on the right and blue dots on the left (like FG_PCT), it means higher values of that stat drive wins — intuitive. If it's reversed (like TOV), higher values drive losses. Features with tightly clustered dots have consistent effects; spread-out dots indicate variable, game-dependent impact.

![SHAP Beeswarm](figures/shap_beeswarm.png)

### 1.3 Win Probability Timeline

**How to read this chart:** Each vertical bar represents one game. **Green bars** are wins; **red bars** are losses. The height of each bar is the model's estimated win probability *based on the box-score stats*. The **gold line** is a 5-game rolling average, which smooths out noise and reveals the team's momentum trend.

**What to look for:** Stretches where the gold line dips below 0.5 indicate sustained poor performance windows. Games where a green bar is short (low predicted win probability but still a win) represent "lucky" wins where the stats didn't support the outcome. Conversely, tall red bars are games the team "should have" won based on their statistical profile but didn't.

![Win Probability Timeline](figures/win_prob_timeline.png)

**Key Insight:** The top 3 win drivers are:

1. **TOV** (mean |SHAP| = 0.975)
2. **REB** (mean |SHAP| = 0.955)
3. **FG3_PCT** (mean |SHAP| = 0.885)

> 💡 **Recommendation:** The model confirms that **TOV** (turnovers) is the single most important factor separating wins from losses. Reducing turnovers should be the coaching staff's #1 tactical priority.


## 2. Win vs Loss Statistical DNA

**What is this analysis?** We split all Warriors games into two groups — wins and losses — and compute the average for every major stat in each group. The gap between these averages reveals the team's "win recipe": the statistical profile they must hit to come out on top.

**How to read this chart:** Each bar shows the difference between the Warriors' average stat in wins versus losses. **Green bars** extending right indicate stats that are higher in wins (good direction for scoring stats). **Red bars** indicate the opposite. The taller the bar, the larger the gap — and the more critical that stat is for winning. The **Priority** column in the table below uses gap size to classify each stat (🔴 Critical = must address, 🟡 Moderate = should improve, 🟢 Minor = marginal impact).

![Win vs Loss Gap](figures/win_loss_gap.png)

| Stat | In Wins | In Losses | Gap | Priority |
|---|---|---|---|---|
| PTS | 121.2 | 109.3 | +11.9 | 🔴 Critical |
| AST | 30.6 | 27.1 | +3.5 | 🟡 Moderate |
| REB | 43.2 | 41.7 | +1.5 | 🟢 Minor |
| STL | 10.0 | 9.9 | +0.1 | 🟢 Minor |
| BLK | 4.6 | 4.2 | +0.4 | 🟢 Minor |
| TOV | 14.0 | 15.9 | -1.9 | 🔴 Critical |
| FG_PCT | 48.3% | 43.8% | +4.6% | 🔴 Critical |
| FG3_PCT | 38.4% | 33.4% | +5.0% | 🔴 Critical |

## 3. Player Archetype Clustering

**What is this model?** K-Means clustering is an unsupervised machine learning algorithm that groups players based on similarity across 23 statistical dimensions (points, rebounds, assists, shooting percentages, advanced metrics, hustle stats, etc.). The algorithm automatically discovers natural groupings *without being told what the groups should be* — it purely looks at the data.

**How it works:** Each player is represented as a point in 23-dimensional space (one axis per stat, all scaled to the same range). K-Means finds cluster centers and assigns each player to the nearest center. We tested k=2 through k=6 and chose **k=2** because it had the highest silhouette score (0.281), which measures how well-separated the clusters are (1.0 = perfect separation, 0.0 = random, >0.25 is acceptable).

**How to read this chart:** We use **PCA (Principal Component Analysis)** to compress the 23 dimensions down to 2 for visualization. Each dot is a player; colors represent cluster membership. Players close together on the chart are statistically similar across *all* 23 metrics — not just scoring. The X and Y axes are the first two principal components, which capture the most variance in the data. Labels show player names for identification.

![Player Clustering](figures/cluster_scatter.png)

#### Cluster: 🔧 Core Rotation

**Players:** Al Horford, Brandin Podziemski, Buddy Hield, Draymond Green, Gary Payton II, Gui Santos, Jonathan Kuminga, Moses Moody, Pat Spencer, Quinten Post, Trayce Jackson-Davis, Will Richard

Avg: 8.0 PPG, 2.2 APG, 3.7 RPG, 19.4 MPG

#### Cluster: ⭐ Primary Scorers

**Players:** De'Anthony Melton, Jimmy Butler III, Stephen Curry

Avg: 19.7 PPG, 4.0 APG, 4.0 RPG, 28.1 MPG

> **Interpretation:** The algorithm clearly separates the three highest-usage players (Curry, Butler, Melton) from everyone else. This confirms the team's heavy reliance on a small core for scoring production.


## 4. Player Skill Profiles

**What is this chart?** A radar chart (also called a spider chart) displays multiple statistics simultaneously on a circular grid. Each spoke represents one skill dimension: **Points, Assists, Rebounds, Steals, Blocks, FG%, 3PT%, TS% (True Shooting), and Usage Rate**.

**How to read it:** All values are normalized within the team on a **0–100 scale**: 100 = team-best in that category, 0 = team-worst. A player with a large, round shape is a well-rounded contributor. A player with a lopsided shape has clear strengths (spikes outward) and weaknesses (dips inward). Compare shapes across players to see how their skill profiles complement or overlap each other.

**Example interpretation:** If Curry's radar spikes on Points, 3PT%, and TS% but dips on Rebounds and Blocks, he's a high-volume efficient scorer with limited rim impact. If Green's radar spikes on Assists, Steals, and Rebounds but dips on Points and FG%, he's a do-everything playmaker who doesn't score efficiently. A coaching staff can use these profiles to build complementary lineups where each player's strengths cover another's weaknesses.

![Player Radar Charts](figures/player_radars.png)

## 5. On/Off Court Impact Analysis

**What is this analysis?** On/Off analysis compares the team's **Net Rating** (points scored minus points allowed per 100 possessions) when a player is on the court versus when they sit on the bench. A player with a large positive swing is someone the team performs significantly better *with* on the floor. A negative swing means the team actually plays better without them. This is the most direct, lineup-based measure of individual impact.

**How to read the bar chart:** Each bar represents one player's net rating swing (on-court minus off-court). **Green bars** extending to the right indicate the team is better with this player on court. **Red bars** extending left mean the team is worse with this player. We filter to players with **200+ on-court minutes** to ensure statistical reliability. A swing of ±5 or more is considered highly significant in NBA analytics.

![On/Off Court Impact](figures/onoff_impact.png)

### 5.1 Two-Way Impact Map

**How to read this chart:** This scatter plot separates the on/off analysis into two dimensions — **offense** (X-axis) and **defense** (Y-axis). Each dot is a player; dot size reflects on-court minutes (larger dots = more reliable sample).

**Quadrant interpretation:**
- **Upper-right (Two-Way Star ★):** Helps both offense AND defense — the most valuable players
- **Upper-left (Defense Only):** Helps the defense but may hurt the offense
- **Lower-right (Offense Only):** Helps the offense but is a defensive liability
- **Lower-left (Liability):** Hurts the team on both ends

*Note: For defense, lower Defensive Rating is better. The Y-axis is constructed so that upward = better defense, making upper-right the ideal quadrant.*

![Two-Way Impact Map](figures/twoway_impact.png)

> 💡 **Most Impactful:** Melton, De'Anthony (+18.7 net swing) — the team is dramatically better in every way when Melton is on the floor

> ⚠️ **Least Impactful:** Hield, Buddy (-10.6 net swing) — the team's net rating drops significantly when Hield plays, suggesting a role or minutes adjustment is needed


## 6. Lineup Synergy Heatmap

**What is this chart?** A heatmap showing the **weighted Net Rating** for every possible pair of the top 10 players, aggregated across all 5-man lineups they've shared. This reveals which player combinations have natural chemistry and which pairs struggle together.

**How to read it:** Find a cell at the intersection of two player names. The number is their weighted Net Rating when both are on the floor simultaneously. **Green cells** indicate the team outscores opponents when this pair plays together (positive synergy); **red cells** mean the team is outscored (negative synergy). Darker colors = stronger effect. Only the lower-left triangle is filled (the upper-right would be a mirror image). Values are **weighted by shared minutes** to give more reliable estimates for pairs that have played more together.

**What to look for:** Bright green cells identify pairs that should share more minutes. Bright red cells identify pairs that should be staggered (play at different times). If a player has mostly green in their row, they're a positive influence on everyone; mostly red suggests fit issues.

![Lineup Synergy Heatmap](figures/synergy_heatmap.png)

> ✅ **Best Pairing:** Jimmy Butler III + De'Anthony Melton (+21.9 weighted net) — The team dominates when these two share the floor. Coaching staff should maximize their overlapping minutes and design plays that leverage their combined strengths.

> ❌ **Worst Pairing:** Jonathan Kuminga + Al Horford (-18.8 weighted net) — The team hemorrhages points with this pairing. These players should be staggered in the rotation whenever possible.


## 7. Season Trends & Momentum

**What is this chart?** A three-panel time series view tracking how the Warriors' performance has evolved over the entire season. Time series analysis reveals trends, hot streaks, cold spells, and whether the team is improving or declining as the season progresses.

**How to read the three panels:**
- **Top panel (Scoring Trend):** The faint gold line is actual points scored per game; the bold gold line is a **10-game rolling average**, which smooths out noise. The dashed white line is the season average. When the rolling average rises above the season average, the team is on a scoring hot streak.
- **Middle panel (Shooting Efficiency):** Gold line = FG% rolling average; blue line = 3PT% rolling average. The dotted line at ~45% is a general efficiency threshold. Watch for divergence between the two — if 3PT% drops while FG% holds, the team may be getting to the rim more but missing from distance.
- **Bottom panel (Win Probability):** Background bars (green = wins, red = losses) show actual outcomes. The gold line shows the model's 10-game rolling predicted win probability. Periods where this line dips below 0.5 indicate stretches where the team's underlying stats predict losing.

**What to look for:** Upward trends in the rolling averages suggest the team is peaking at the right time. Downward trends are warning signs. Sudden drops may correspond to injuries, schedule difficulty, or lineup changes.

![Season Trends](figures/scoring_trends.png)

### 7.1 Correlation Matrix

**What is this chart?** A correlation matrix shows the **Pearson correlation coefficient (r)** between every pair of game stats and winning. Correlation ranges from **-1** (perfect inverse relationship) to **+1** (perfect positive relationship). Values near 0 indicate no linear relationship.

**How to read it:** Each cell shows the correlation between two stats. **Dark red/warm colors** = strong positive correlation (they move together); **dark blue/cool colors** = strong negative correlation (one goes up when the other goes down); **white/neutral** = no meaningful relationship. The **bottom row ("WIN")** is the most important — it reveals which stats most strongly correlate with winning. Higher |r| = more important for winning.

![Correlation Matrix](figures/correlation_matrix.png)

**Strongest win correlates:**

- PTS: r = 0.456 — scoring more is (unsurprisingly) the strongest correlate
- FG_PCT: r = 0.454 — shooting efficiently is nearly as important as raw scoring
- AST: r = 0.374 — ball movement and team play strongly predict wins

## 8. Player Monthly Development

**What is this chart?** Each subplot tracks one player's monthly scoring (gold bars) and shooting efficiency (green dashed line, right axis) across the season. This reveals individual player **development arcs** — who is improving, who is slumping, and who has been consistent.

**How to read it:** The gold bars represent average PPG for each month. The green dashed line (right Y-axis) shows FG% for that month. Look for:
- **Rising bars + rising green line** = player is getting better at scoring AND efficiency (ideal growth)
- **Rising bars + falling green line** = scoring more but less efficiently (may be forcing shots or getting a bigger role than warranted)
- **Falling bars + rising green line** = scoring less but more efficiently (may deserve more shot attempts)
- **Falling bars + falling green line** = slumping on all fronts (needs coaching attention)

**Why it matters:** This chart helps identify whether a player's season averages mask important trends. A player averaging 15 PPG who scored 20 PPG in January and 10 PPG in February is very different from one who scored a steady 15 each month.

![Monthly Trends](figures/monthly_trends.png)

## 9. Bayesian Season Projection

**What is this model?** Bayesian inference treats the Warriors' "true" win probability as an unknown quantity and uses the observed win-loss record to estimate a **probability distribution** over it. Instead of saying "the Warriors will win 42 games," it says "there's a 51% chance they win 42+ and a 21% chance they win 44+." This honest accounting of uncertainty is more useful for decision-making than a single point estimate.

**How it works:** We combine two Bayesian posteriors:
1. **Full-season posterior** — Beta(30, 28) based on the complete 29-27 record, giving a narrow distribution centered at ~51.8%
2. **Recent-form posterior** — based on the last 15 games (46.7% win rate), wider because less data

These are blended 60/40 (favoring the full season but weighting recent form). We then run **50,000 Monte Carlo simulations** — randomly sampling a win probability and simulating the remaining 26 games each time — to get a full distribution of possible season outcomes.

**How to read the three panels:**
- **Left panel (Win Total Distribution):** Each bar is one possible season win total. Height = probability. Blue histogram uses the full-season prior; red uses recent form. The **gold dashed line** is the blended median projection. The spread of the histogram shows the range of plausible outcomes.
- **Center panel (Bayesian Win Rate Posterior):** The bell curves show the estimated "true" win rate. The **blue curve** (full season) is narrow (more data = more certainty). The **red curve** (last 15 games) is wider (less data = more uncertainty). Where the curves overlap is the most likely true win rate.
- **Right panel (Playoff Threshold Probability):** Bars show the probability of reaching various win totals. **Green** = >50% likely; **red** = <25% likely. This directly answers "how likely are we to make the playoffs?"

![Bayesian Projection](figures/bayesian_projection.png)

| Metric | Value |
|---|---|
| Current Record | 29-27 |
| Full Season Win% | 51.8% |
| Recent Form (L15) | 46.7% |
| **Projected Total Wins (Median)** | **42** |
| 80% Confidence Interval | 39 – 45 wins |
| Probability ≥42 Wins | 50.7% |
| Probability ≥44 Wins | 21.1% |
| Probability ≥46 Wins | 5.2% |

## 10. Game Anatomy — Best Win & Worst Loss

**What is this chart?** SHAP waterfall charts take one specific game and break down *exactly* which stats pushed the model's prediction toward a win or a loss. This is the most granular level of explainability — it answers "why did we win/lose *this specific game*?"

**How to read it:** Each horizontal bar represents one feature's SHAP contribution for that game. The label shows "Feature = Value" (e.g., "TOV = 12" means the team had 12 turnovers). **Green bars** pushed the prediction toward a win; **red bars** pushed toward a loss. Features are sorted by magnitude (most impactful at top). The left chart shows the team's **best win** (largest margin of victory); the right shows the **worst loss** (largest deficit).

**What to look for:** Compare the two charts side by side. Stats that appear green in the best win but red in the worst loss are the "swing factors" — the stats that most differentiate the team's ceiling from its floor. These are the highest-leverage areas for game-to-game improvement.

![SHAP Game Waterfall](figures/shap_waterfall.png)

## 11. Composite Player Value Index

**What is this chart?** A stacked horizontal bar chart showing each player's overall value, broken down into four components from different analytical methods. This is the most comprehensive single-number summary of player value on the roster — it synthesizes box-score stats, per-minute efficiency, lineup impact, and shooting efficiency into one score.

**The four components (and their weights):**
- **Production (30%, gold):** Raw counting stats — points, assists, rebounds, steals, blocks, minus turnovers. Higher = more total output. This rewards high-minute, high-volume players.
- **Per-Minute Production (20%, blue):** Same formula but divided by minutes played. This identifies efficient contributors who produce a lot relative to their playing time and may deserve more minutes.
- **Impact/RAPM (30%, green):** Regularized Adjusted Plus-Minus — estimated from lineup data via Ridge Regression. This measures how much the team's net rating changes when this player is on the court, controlling for teammates. A player can score zero points but have high RAPM if the team consistently outscores opponents when they play.
- **Efficiency/TS% (20%, light green):** True Shooting Percentage — how efficiently the player converts possessions into points, accounting for 2-pointers, 3-pointers, and free throws.

**How to read it:** Each player's total bar length is their composite score (0–100). Longer bars = more valuable. The color segments show *why* they're valuable — a player with a big gold segment is a high-volume producer; a player with a big green segment is a lineup-level impact player even if their box score is modest.

![Composite Value](figures/composite_value.png)

| Rank | Player | Production | Per-Min | RAPM | Efficiency | **Composite** |
|---|---|---|---|---|---|---|
| 1 | Stephen Curry | 100 | 100 | 51 | 92 | **83.7** |
| 2 | Jimmy Butler III | 91 | 85 | 36 | 100 | **75.2** |
| 3 | De'Anthony Melton | 36 | 45 | 100 | 12 | **52.3** |
| 4 | Moses Moody | 33 | 16 | 56 | 60 | **41.7** |
| 5 | Gary Payton II | 7 | 45 | 65 | 55 | **41.4** |
| 6 | Brandin Podziemski | 48 | 34 | 42 | 38 | **41.3** |
| 7 | Gui Santos | 8 | 28 | 45 | 84 | **38.4** |
| 8 | Jonathan Kuminga | 39 | 37 | 39 | 15 | **33.9** |
| 9 | Quinten Post | 18 | 39 | 37 | 35 | **31.2** |
| 10 | Will Richard | 11 | 0 | 40 | 70 | **29.4** |

## 12. Scoring Distribution

**What is this chart?** A violin plot combines a box plot with a kernel density estimate to show the **full distribution** of each player's game-by-game scoring. Unlike a simple "PPG average," this reveals the entire shape of a player's scoring output — their ceiling, floor, most common range, and outlier performances.

**How to read it:** Each violin's width at any given point represents how frequently the player scores that many points.
- A **fat section** = the player frequently scores in that range
- A **thin section** = that scoring level is rare
- The **white horizontal line** = mean (average)
- The **green horizontal line** = median (middle value)
- Small **blue dots** = individual game performances, jittered horizontally for visibility

**What to look for:**
- **Tall, narrow violins** = inconsistent scorer with a wide range (boom-or-bust player)
- **Short, fat violins** = consistent scorer who stays in a narrow range (reliable, predictable)
- **Dots far above/below the main body** = outlier performances (career nights or unusually bad games)
- **Mean much higher than median** = occasional big scoring games that inflate the average (right-skewed distribution)

![Scoring Distribution](figures/scoring_violin.png)

## 13. Usage Rate vs Shooting Efficiency

**What is this chart?** A bubble scatter plot that maps every player along two crucial dimensions: how much they use the ball (**Usage Rate**, X-axis) and how efficiently they score (**True Shooting %**, Y-axis). This is the fundamental efficiency-volume trade-off in basketball — can a player handle a heavy workload without becoming inefficient?

**How to read it:**
- **X-axis (Usage Rate %):** The percentage of team possessions a player uses while on court (via shot attempt, free throw attempt, or turnover). League average is ~20%. The vertical dashed line marks this threshold.
- **Y-axis (True Shooting %):** Scoring efficiency accounting for 2-pointers, 3-pointers, and free throws. League average is ~57%. The horizontal dashed line marks this threshold.
- **Bubble size:** Minutes per game — larger bubbles = more playing time (and more reliable data)
- **Bubble color:** Gold = positive net rating (team outscores opponents with this player); Red = negative net rating

**Quadrant interpretation:**
- **Upper-right ⭐ (High Usage + High Efficiency):** Franchise cornerstones — the best players in the NBA live here
- **Upper-left (Low Usage + High Efficiency):** Efficient role players who could potentially handle a bigger role
- **Lower-right ⚠️ (High Usage + Low Efficiency):** Volume shooters who are hurting the offense — improvement targets
- **Lower-left (Low Usage + Low Efficiency):** Limited contributors with small roles

![Usage vs Efficiency](figures/usage_vs_efficiency.png)

> ⭐ **Elite efficiency at high usage:** Jimmy Butler III, Stephen Curry — these two are the engine of the offense, and the data confirms they earn every shot they take


## 14. Shot Zone Efficiency

**What is this chart?** A bar chart showing the Warriors' **field goal percentage by shot zone** — different areas of the court where shots are taken. This reveals where the team generates high-quality looks and where they waste possessions on low-efficiency shots.

**How to read it:** Each bar represents a different shot zone (e.g., Restricted Area, In The Paint, Mid-Range, Left Corner 3, Right Corner 3, Above the Break 3). The bar height is the team's FG% from that zone. The number above each bar shows the percentage. **Taller bars = more efficient zones**. As a rule of thumb, NBA offenses should maximize shots from the Restricted Area (~65% FG%) and Corner 3s (~39% FG%), while minimizing mid-range shots (~40% FG%), since the mid-range is the least efficient area on the court per expected points.

**What to look for:** Compare the Warriors' zone efficiency to these benchmarks. If the mid-range bar is high, the team may have skilled mid-range shooters worth keeping. If the corner 3 bar is low, the team needs better looks or better shooters in the corners.

![Shot Zone Efficiency](figures/shot_zone_efficiency.png)


## 15. Clutch Performance Analysis

**What is this chart?** The NBA defines **"clutch"** as the last 5 minutes of any game where the score is within 5 points. These are the highest-pressure moments that often decide close games. This chart shows which Warriors players step up (or shrink) when it matters most.

**How to read it:** Each player is represented by grouped bars showing their clutch stats — typically PPG, FG%, and/or plus-minus in clutch situations. Players are sorted by clutch scoring output. Compare each player's clutch stats to their regular-season averages: players whose clutch numbers *exceed* their regular stats are "closers" who raise their game under pressure; players whose clutch numbers *drop* may not be reliable in crunch time.

**Why it matters:** In a season where the Warriors are 29-27 and fighting for playoff positioning, close games are make-or-break. Identifying who can be trusted in the final minutes — and who should cede touches — can directly flip 3-5 close losses into wins.

![Clutch Performance](figures/clutch_performance.png)


## 16. Rest Day Impact on Performance

**What is this chart?** This analysis groups all games by the number of rest days before the game (0 = back-to-back, 1 = one day off, 2+ = multiple days off) and compares the team's win rate and scoring output. This quantifies the impact of rest and schedule density on performance.

**How to read it:** Each group of bars represents a rest-day category. The bars show two metrics: **win percentage** (how often the team wins with that rest) and **average points scored** (offensive output). Higher bars = better performance. If the 0-rest (back-to-back) bars are significantly shorter than the 1-rest or 2-rest bars, the team is fatigue-sensitive and the coaching staff should aggressively manage minutes on back-to-backs.

**Why it matters:** NBA teams play 82 games in ~6 months, with frequent back-to-backs. Teams that manage rest effectively can gain 2-3 extra wins per season. This data helps the coaching staff decide when to rest key players (especially Curry at age 37) and how to structure rotation minutes on back-to-backs.

![Rest Day Impact](figures/rest_day_impact.png)


## 17. Cumulative Win Pace

**What is this chart?** A cumulative wins tracker that plots the Warriors' actual win accumulation against multiple pace targets throughout the season. This is the most intuitive way to see whether the team is on track for the playoffs.

**How to read it:**
- The **solid gold line** is the Warriors' actual cumulative wins over the season (each step up = one win)
- The **dashed lines** represent different pace targets: a .500 pace (41 wins), a 55% pace (45 wins), etc.
- The **shaded gray region** is the "playoff zone" — the win range (approximately 39-45) typically needed to make the playoffs in the Western Conference
- After the current game, the gold line extends as a **dashed projection** to the end of the 82-game season based on current win rate
- The **white dot** marks the current record with a label

**What to look for:** If the gold line is above the .500 pace line, the team is winning at a rate above 50%. If it enters the shaded playoff zone, they're on pace for the postseason. If the gold line flattens (goes horizontal for several games), that represents a losing streak. Steep upward sections are winning streaks.

![Cumulative Win Pace](figures/cumulative_wins.png)


## 18. Player Efficiency Landscape

**What is this chart?** A multi-dimensional bubble scatter that provides a holistic view of every player's role, impact, and efficiency in a single visualization. This is the "big picture" player chart — each player is represented as a bubble encoding four separate pieces of information.

**How to read it:**
- **X-axis (Minutes Per Game):** Higher = bigger role on the team. Players on the right play the most.
- **Y-axis (Net Rating):** Points scored minus points allowed per 100 possessions when this player is on court. **Above the zero line (dashed)** = the team outscores opponents with this player; **below** = the team is outscored. This is the most direct measure of on-court impact.
- **Bubble size (Points Per Game):** Larger bubbles = higher-volume scorers. Small bubbles may still have high net ratings if they contribute defensively or as facilitators.
- **Bubble color (True Shooting %):** Gold/warm = high efficiency; red/cool = low efficiency. The most valuable players combine large bubbles (high volume) with warm colors (high efficiency) above the zero line (positive impact).

**What to look for:** The ideal player is a **large, gold bubble in the upper-right** (high minutes, high scoring, high efficiency, positive net impact). Players in the lower-right quadrant with large red bubbles are concerns — they play a lot and score a lot but inefficiently and with negative net impact. Players in the upper-left with small gold bubbles may be underutilized gems.

![Player Efficiency Landscape](figures/efficiency_landscape.png)


## 19. Actionable Improvement Recommendations

Based on all models above, here are prioritized, specific recommendations:


### 🏀 Offensive Recommendations


**1. #1 win driver: ball security (turnovers)** (SHAP importance: 0.975)
   - Turnovers are the single most predictive stat separating Warriors wins from losses
   - Current: 14.9 TOV/game; in wins: 14.0, in losses: 15.9
   - Every turnover averted significantly increases win probability per the XGBoost model
   - Focus areas: tighten half-court passing, limit dribble hand-offs in traffic

**2. Reduce turnovers** (gap: -1.9 fewer in wins)
   - Current: 14.9 TOV/game; wins average 14.0, losses 15.9
   - Target: <14 turnovers per game
   - Key focus: Stephen Curry (2.8), Draymond Green (2.7), Jonathan Kuminga (2.3)

**3. Improve 3-point shot selection**
   - 3PT% gap between wins (38.4%) and losses (33.4%): +5.0%
   - Increase corner 3 attempts (highest efficiency) and reduce contested above-break 3s
   - Curry's off-ball screens create the best looks — run more Curry off-screen sets

### 🛡️ Defensive Recommendations


**4. Optimize defensive lineups**
   - Best defensive anchors (lower DEF RTG on court): Melton, De'Anthony, Santos, Gui, Podziemski, Brandin
   - Hide defensively in crunch time: Hield, Buddy, Curry, Stephen

**5. Increase steal-generating activity**
   - Steals gap in wins vs losses: +0.1/game
   - More aggressive trapping in half-court; the data shows minimal impact on fouls

### 📋 Rotation & Lineup Recommendations


**6. Maximize the best pair: Jimmy Butler III + De'Anthony Melton** (net: +21.9)
   - Increase their shared minutes; design sets that leverage their chemistry
   - Separate Jonathan Kuminga and Al Horford (-18.8 net) when possible

**7. Rest day optimization**
   - Best performance on 1 rest days (55% win rate)
   - Manage Curry's minutes on back-to-backs (fatigue coefficient from prior analysis)

### 📈 Player Development Recommendations


**8. Development priorities**

**9. Shooting improvement targets**
   - De'Anthony Melton: 29.7% on 4.8 3PA/game — needs shot selection improvement or volume reduction
   - Draymond Green: 32.0% on 4.7 3PA/game — needs shot selection improvement or volume reduction
   - Jonathan Kuminga: 32.1% on 2.7 3PA/game — needs shot selection improvement or volume reduction

### 🔮 Season Outlook


**10. Playoff positioning**
   - Current: 29-27 (51.8%)
   - Projected finish: **42 wins** (80% CI: 39–45)
   - Playoff probability (≥42 wins): **51%**
   - Recent form (47% L15) is concerning — urgent improvement needed to maintain playoff position

   > ⚠️ **Alert:** At the current L15 pace (47%), the team projects to only **41 wins** — below the playoff threshold. Immediate tactical adjustments are needed.

---

## 20. Executive Summary


| # | Finding | Action | Priority |
|---|---------|--------|----------|
| 1 | Turnovers are the #1 win predictor (SHAP=0.98) | Target <14 TOV/game; tighten half-court execution | 🔴 Critical |
| 2 | FG% gap: +4.6% between W/L | Increase restricted-area attempts; cut mid-range volume | 🔴 Critical |
| 3 | 3PT%: 38.4% in wins vs 33.4% in losses | More corner 3s; reduce contested above-break 3s | 🔴 Critical |
| 4 | Best pair: III + Melton (+21.9) | Increase shared minutes; design synergy sets | 🟡 High |
| 5 | Worst pair: Kuminga + Horford (-18.8) | Stagger minutes; avoid pairing in crunch time | 🟡 High |
| 6 | Projected 42 wins (80% CI: 39–45) | Playoff prob: 51%; recent slide needs correction | 🟡 High |

> **Bottom line:** The Warriors are a borderline playoff team whose fate hinges on ball security, shot quality, and maximizing the Butler-Melton pairing. The data is clear on what separates this team's wins from losses — executing on these priorities will determine the season outcome.


---

## Appendix: Glossary of Models & Metrics


### Machine Learning Models Used


| Model | Type | Purpose |
|---|---|---|
| XGBoost Classifier | Gradient Boosted Trees | Predict wins/losses from game stats; 200 trees, depth 4, 5-fold CV |
| SHAP (TreeExplainer) | Game Theory Explainability | Decompose each prediction into per-feature contributions |
| K-Means Clustering | Unsupervised Learning | Group players into archetypes based on stat profiles |
| Ridge Regression (RAPM-lite) | Regularized Linear Model | Estimate each player's impact on team net rating from lineup data |
| Bayesian Beta-Binomial | Probabilistic Inference | Project season win total with uncertainty via posterior distribution |
| Gradient Boosted Regressor | Ensemble Regression | Expected FG% model (xFG) based on shot profile |
| PCA | Dimensionality Reduction | Project high-dimensional player stats to 2D for visualization |

### Statistical Metrics


| Metric | Definition |
|---|---|
| **SHAP Value** | Shapley Additive Explanation — each feature's marginal contribution to a prediction |
| **Net Rating** | Points scored minus points allowed per 100 possessions (team or individual on-court) |
| **Off Rating / Def Rating** | Points scored / allowed per 100 possessions |
| **On/Off Swing** | Difference in team Net Rating when a player is on court vs off court |
| **TS% (True Shooting)** | Scoring efficiency: PTS / (2 × (FGA + 0.44 × FTA)) |
| **USG% (Usage Rate)** | % of team possessions used by a player while on court |
| **RAPM** | Regularized Adjusted Plus-Minus — player's per-possession impact estimated via Ridge regression |
| **PIE (Player Impact Estimate)** | Player's share of game events (points, rebounds, assists, etc.) |
| **Silhouette Score** | Clustering quality: ranges from -1 (poor) to +1 (perfect); >0.25 is acceptable |
| **Coefficient of Variation** | Standard deviation / mean — measures scoring consistency (lower = more consistent) |
| **Bayesian Posterior** | Updated probability distribution after combining prior belief with observed data |
| **Monte Carlo Simulation** | Running thousands of random season simulations to estimate outcome probabilities |
| **Cross-Validation Accuracy** | Model accuracy averaged across held-out test folds (prevents overfitting) |
| **Clutch** | Last 5 minutes of game with score within 5 points |
| **Weighted Net Rating** | Net rating weighted by minutes played (gives more weight to larger samples) |

### Visualization Guide


| Chart Type | How to Read It |
|---|---|
| SHAP Beeswarm | Each dot = one game. Horizontal position = impact on win probability. Color = feature value (red=high, blue=low) |
| Radar Chart | 9 axes showing player skills normalized 0–100 within team. Larger area = more well-rounded |
| Violin Plot | Width = frequency of scoring at that level. Wider = more common. Dots = individual games |
| Synergy Heatmap | Pair net rating across shared lineups. Green = positive chemistry; red = negative |
| Two-Way Impact | X = offensive help, Y = defensive help. Upper-right = two-way stars |
| Efficiency Landscape | X = minutes, Y = net rating, size = PPG, color = TS%. Above zero line = net positive |
| Cumulative Wins | Tracks actual wins vs pace targets. Dashed line = season-pace projection |
| Bayesian Projection | Histogram = simulated win totals. Taller bar = more likely outcome |

---
*Models: XGBoost 3.2, SHAP 0.50, scikit-learn, scipy | Data: stats.nba.com 2025-26 | Generated: February 24, 2026*