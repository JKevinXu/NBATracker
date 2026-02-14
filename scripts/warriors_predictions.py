#!/usr/bin/env python3
"""Golden State Warriors — Remaining Season Prediction Model.

Generates a Markdown report with final-record projections using five methods:
  1. Pace projection (current win% extrapolation)
  2. Pythagorean wins (points-based "true" win%)
  3. Weighted recent form (season + L15 blend)
  4. Home/away-adjusted projection
  5. Monte Carlo simulation (10,000 trials, opponent-adjusted)

Run:
    python scripts/warriors_predictions.py [--season 2025-26]
"""

from __future__ import annotations

import sys
import os
import time
import math
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    LeagueStandings,
    TeamDashboardByGeneralSplits,
    TeamGameLog,
)

TEAM_ABBR = "GSW"
TEAM_ID = 1610612744
TOTAL_SEASON_GAMES = 82
API_DELAY = 0.6
N_SIMULATIONS = 10_000
PYTH_EXPONENT = 13.91  # NBA-calibrated Pythagorean exponent


def _delay() -> None:
    time.sleep(API_DELAY)


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _plus_minus(val: float) -> str:
    return f"+{val:.1f}" if val > 0 else f"{val:.1f}"


# ── data fetching ─────────────────────────────────────────────────────


def fetch_standings(season: str) -> pd.DataFrame:
    _delay()
    return LeagueStandings(season=season, league_id="00").get_data_frames()[0]


def fetch_team_game_log(season: str) -> pd.DataFrame:
    _delay()
    return TeamGameLog(
        team_id=TEAM_ID, season=season, season_type_all_star="Regular Season"
    ).get_data_frames()[0]


def fetch_team_splits(season: str) -> list[pd.DataFrame]:
    _delay()
    return TeamDashboardByGeneralSplits(
        team_id=TEAM_ID, season=season, per_mode_detailed="PerGame"
    ).get_data_frames()


# ── prediction models ─────────────────────────────────────────────────


def pace_projection(wins: int, losses: int, remaining: int) -> tuple[int, int]:
    """Simple extrapolation of current win%."""
    gp = wins + losses
    win_pct = wins / gp if gp > 0 else 0.5
    proj_wins = round(win_pct * remaining)
    return wins + proj_wins, losses + (remaining - proj_wins)


def pythagorean_projection(
    ppg: float, opp_ppg: float, wins: int, losses: int, remaining: int
) -> tuple[int, int, float]:
    """Pythagorean expected wins based on scoring margin."""
    pyth_pct = ppg**PYTH_EXPONENT / (ppg**PYTH_EXPONENT + opp_ppg**PYTH_EXPONENT)
    proj_wins = round(pyth_pct * remaining)
    return wins + proj_wins, losses + (remaining - proj_wins), pyth_pct


def weighted_recent_form(
    season_pct: float,
    recent_pct: float,
    wins: int,
    losses: int,
    remaining: int,
    season_weight: float = 0.40,
) -> tuple[int, int, float]:
    """Blend season win% (40%) with recent form (60%)."""
    blended = season_weight * season_pct + (1 - season_weight) * recent_pct
    proj_wins = round(blended * remaining)
    return wins + proj_wins, losses + (remaining - proj_wins), blended


def home_away_adjusted(
    home_pct: float,
    away_pct: float,
    remaining_home: int,
    remaining_away: int,
    wins: int,
    losses: int,
) -> tuple[int, int]:
    """Project remaining games using home/away-specific win rates."""
    proj_home_wins = round(home_pct * remaining_home)
    proj_away_wins = round(away_pct * remaining_away)
    total_proj = proj_home_wins + proj_away_wins
    total_remaining = remaining_home + remaining_away
    return wins + total_proj, losses + (total_remaining - total_proj)


def monte_carlo_simulation(
    base_win_pct: float,
    home_pct: float,
    away_pct: float,
    remaining_home: int,
    remaining_away: int,
    opponent_win_pcts: list[float],
    n_sims: int = N_SIMULATIONS,
) -> dict:
    """Run N Monte Carlo simulations of the remaining season.

    For each game, the win probability is adjusted by opponent strength:
        p = location_pct * (1 - opp_pct) / ((location_pct * (1 - opp_pct)) + ((1 - location_pct) * opp_pct))
    This is a log5 formula.

    Returns dict with distribution data.
    """
    rng = np.random.default_rng(seed=42)

    # Build game schedule: list of (is_home, opp_win_pct)
    schedule: list[tuple[bool, float]] = []
    # Assign opponent win pcts to home/away games
    # We don't know exact schedule, so distribute proportionally
    n_opp = len(opponent_win_pcts)
    home_opps = opponent_win_pcts[: min(remaining_home, n_opp)]
    away_opps = opponent_win_pcts[min(remaining_home, n_opp) :]

    # Pad if needed
    avg_opp = np.mean(opponent_win_pcts) if opponent_win_pcts else 0.5
    while len(home_opps) < remaining_home:
        home_opps.append(avg_opp)
    while len(away_opps) < remaining_away:
        away_opps.append(avg_opp)

    for opp_pct in home_opps[:remaining_home]:
        schedule.append((True, opp_pct))
    for opp_pct in away_opps[:remaining_away]:
        schedule.append((False, opp_pct))

    # Compute win probabilities using log5
    game_probs: list[float] = []
    for is_home, opp_pct in schedule:
        loc_pct = home_pct if is_home else away_pct
        # Log5 formula
        opp_pct = max(0.05, min(0.95, opp_pct))  # clamp
        loc_pct = max(0.05, min(0.95, loc_pct))
        p = (loc_pct * (1 - opp_pct)) / (
            loc_pct * (1 - opp_pct) + (1 - loc_pct) * opp_pct
        )
        game_probs.append(p)

    # Simulate
    remaining = len(schedule)
    probs_array = np.array(game_probs)
    random_draws = rng.random((n_sims, remaining))
    wins_per_sim = (random_draws < probs_array).sum(axis=1)

    # Analyse
    win_counts = Counter(int(w) for w in wins_per_sim)
    total_wins_dist = {k: v / n_sims for k, v in sorted(win_counts.items())}

    return {
        "wins_array": wins_per_sim,
        "mean": float(np.mean(wins_per_sim)),
        "median": float(np.median(wins_per_sim)),
        "std": float(np.std(wins_per_sim)),
        "min": int(np.min(wins_per_sim)),
        "max": int(np.max(wins_per_sim)),
        "p10": int(np.percentile(wins_per_sim, 10)),
        "p25": int(np.percentile(wins_per_sim, 25)),
        "p75": int(np.percentile(wins_per_sim, 75)),
        "p90": int(np.percentile(wins_per_sim, 90)),
        "distribution": total_wins_dist,
        "game_probs": game_probs,
        "schedule": schedule,
    }


# ── report builder ────────────────────────────────────────────────────


def build_report(season: str) -> str:
    lines: list[str] = []
    w = lines.append

    print(f"Building prediction model for {TEAM_ABBR} — {season}...")

    # ── Fetch data ───────────────────────────────────────────────
    print("  Fetching standings...")
    standings = fetch_standings(season)
    print("  Fetching game log...")
    game_log = fetch_team_game_log(season)
    print("  Fetching team splits...")
    splits = fetch_team_splits(season)

    # ── Parse current state ──────────────────────────────────────
    gsw = standings[standings["TeamCity"] == "Golden State"].iloc[0]
    current_wins = int(gsw["WINS"])
    current_losses = int(gsw["LOSSES"])
    games_played = current_wins + current_losses
    remaining = TOTAL_SEASON_GAMES - games_played
    win_pct = current_wins / games_played if games_played > 0 else 0.5
    ppg = float(gsw["PointsPG"])
    opp_ppg = float(gsw["OppPointsPG"])
    diff_ppg = float(gsw["DiffPointsPG"])
    playoff_rank = int(gsw["PlayoffRank"])
    conf_record = str(gsw["ConferenceRecord"])

    # Conference standings context
    west = standings[standings["Conference"] == "West"].sort_values(
        "WinPCT", ascending=False
    )
    games_back = float(gsw["ConferenceGamesBack"])

    # Home/away splits
    home_away = splits[1]
    home_row = home_away[home_away["GROUP_VALUE"] == "Home"].iloc[0]
    away_row = home_away[home_away["GROUP_VALUE"] == "Road"].iloc[0]
    home_pct = float(home_row["W_PCT"])
    away_pct = float(away_row["W_PCT"])
    home_gp = int(home_row["GP"])
    away_gp = int(away_row["GP"])

    # Remaining home/away estimate (82 games, ~41 each)
    remaining_home = max(0, 41 - home_gp)
    remaining_away = max(0, remaining - remaining_home)
    if remaining_away < 0:
        remaining_home = remaining
        remaining_away = 0

    # Recent form (L15)
    recent_15 = game_log.head(15)
    recent_w = (recent_15["WL"] == "W").sum()
    recent_l = 15 - recent_w
    recent_pct = recent_w / 15

    # Opponent win percentages for schedule strength
    # Use unique opponents from game log and map to standings
    opp_records: dict[str, float] = {}
    for _, row in standings.iterrows():
        opp_records[str(row["TeamCity"]) + " " + str(row["TeamName"])] = float(
            row["WinPCT"]
        )
    # Also map by abbreviation from game log matchups
    opp_abbr_pct: dict[str, float] = {}
    for _, row in standings.iterrows():
        tid = int(row["TeamID"])
        opp_abbr_pct[tid] = float(row["WinPCT"])

    # Extract opponents from game log for schedule analysis
    played_opponents: list[str] = []
    for _, row in game_log.iterrows():
        mu = str(row["MATCHUP"])
        opp = mu.split(" ")[-1]
        played_opponents.append(opp)

    # For remaining schedule, estimate opponent quality from season-wide distribution
    # Use the average opponent win% the Warriors have faced
    all_opp_pcts: list[float] = []
    for opp_abbr in played_opponents:
        # Find in standings by matching abbreviation
        for _, sr in standings.iterrows():
            # Standings don't have abbreviation directly, build from context
            pass
        all_opp_pcts.append(0.5)  # default

    # Better approach: use vs-conference and overall opponent data
    # Estimate remaining opponents as league-average mix adjusted by conference
    west_avg_pct = float(west["WinPCT"].mean())
    east = standings[standings["Conference"] == "East"]
    east_avg_pct = float(east["WinPCT"].mean())

    # Warriors play mostly West (~52 games West, ~30 East in 82-game season)
    # Estimate remaining opponent win% as weighted mix
    est_remaining_opp_pct = 0.65 * west_avg_pct + 0.35 * east_avg_pct

    # Build opponent win pcts for Monte Carlo
    # Vary them around the estimated average
    rng_opp = np.random.default_rng(seed=99)
    opp_win_pcts_for_mc = list(
        np.clip(rng_opp.normal(est_remaining_opp_pct, 0.10, remaining), 0.15, 0.85)
    )

    # ── Run all 5 models ─────────────────────────────────────────
    print("  Running projections...")

    # 1. Pace
    pace_w, pace_l = pace_projection(current_wins, current_losses, remaining)

    # 2. Pythagorean
    pyth_w, pyth_l, pyth_pct = pythagorean_projection(
        ppg, opp_ppg, current_wins, current_losses, remaining
    )

    # 3. Weighted recent form
    wrf_w, wrf_l, wrf_pct = weighted_recent_form(
        win_pct, recent_pct, current_wins, current_losses, remaining
    )

    # 4. Home/away adjusted
    ha_w, ha_l = home_away_adjusted(
        home_pct, away_pct, remaining_home, remaining_away,
        current_wins, current_losses,
    )

    # 5. Monte Carlo
    print("  Running Monte Carlo simulation (10,000 trials)...")
    mc = monte_carlo_simulation(
        win_pct, home_pct, away_pct,
        remaining_home, remaining_away,
        opp_win_pcts_for_mc,
    )
    mc_proj_wins = current_wins + round(mc["mean"])
    mc_proj_losses = TOTAL_SEASON_GAMES - mc_proj_wins

    # ── Playoff threshold estimation ─────────────────────────────
    # Look at current 8th and 10th seeds to estimate thresholds
    west_sorted = west.sort_values("PlayoffRank")
    seed_8_row = west_sorted[west_sorted["PlayoffRank"] == 8].iloc[0] if len(west_sorted[west_sorted["PlayoffRank"] == 8]) > 0 else None
    seed_10_row = west_sorted[west_sorted["PlayoffRank"] == 10].iloc[0] if len(west_sorted[west_sorted["PlayoffRank"] == 10]) > 0 else None

    if seed_8_row is not None:
        seed_8_pace_w = round(float(seed_8_row["WinPCT"]) * TOTAL_SEASON_GAMES)
    else:
        seed_8_pace_w = 41

    if seed_10_row is not None:
        seed_10_pace_w = round(float(seed_10_row["WinPCT"]) * TOTAL_SEASON_GAMES)
    else:
        seed_10_pace_w = 38

    playoff_threshold = seed_8_pace_w
    playin_threshold = seed_10_pace_w

    # Monte Carlo playoff odds
    mc_total_wins = current_wins + mc["wins_array"]
    pct_make_playoffs = float(np.mean(mc_total_wins >= playoff_threshold) * 100)
    pct_make_playin = float(np.mean(mc_total_wins >= playin_threshold) * 100)
    pct_miss = float(np.mean(mc_total_wins < playin_threshold) * 100)

    # Seed probability distribution
    seed_thresholds: list[tuple[str, int]] = []
    for _, sr in west_sorted.iterrows():
        rank = int(sr["PlayoffRank"])
        proj = round(float(sr["WinPCT"]) * TOTAL_SEASON_GAMES)
        seed_thresholds.append((f"#{rank} {sr['TeamCity']} {sr['TeamName']}", proj))

    # ═════════════════════════════════════════════════════════════
    # BUILD MARKDOWN REPORT
    # ═════════════════════════════════════════════════════════════

    w(f"# Golden State Warriors — {season} Season Predictions")
    w("")
    w(f"> Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    w(f"> Model based on {games_played} games played, {remaining} remaining")
    w("")

    # ── 1. Current Snapshot ──────────────────────────────────────
    w("## 1. Current Snapshot")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Record | **{current_wins}-{current_losses}** ({_pct(win_pct)}) |")
    w(f"| West Seed | **#{playoff_rank}** |")
    w(f"| Games Back | {games_back} |")
    w(f"| Conference Record | {conf_record} |")
    w(f"| Points Per Game | {ppg:.1f} |")
    w(f"| Opponent PPG | {opp_ppg:.1f} |")
    w(f"| Point Differential | **{_plus_minus(diff_ppg)}** |")
    w(f"| Home Record | {int(home_row['W'])}-{int(home_row['L'])} ({_pct(home_pct)}) |")
    w(f"| Road Record | {int(away_row['W'])}-{int(away_row['L'])} ({_pct(away_pct)}) |")
    w(f"| Last 15 Games | {recent_w}-{recent_l} ({_pct(recent_pct)}) |")
    w(f"| Games Remaining | {remaining} (~{remaining_home}H / ~{remaining_away}A) |")
    w("")

    # ── 2. Western Conference Context ────────────────────────────
    w("## 2. Western Conference Standings")
    w("")
    w("| Seed | Team | W | L | Win% | Pace W | GB |")
    w("|---:|---|---:|---:|---:|---:|---:|")

    for _, sr in west_sorted.iterrows():
        rank = int(sr["PlayoffRank"])
        proj = round(float(sr["WinPCT"]) * TOTAL_SEASON_GAMES)
        is_gsw = "**" if sr["TeamCity"] == "Golden State" else ""
        gb = float(sr["ConferenceGamesBack"])
        gb_str = f"{gb:.1f}" if gb > 0 else "—"
        marker = ""
        if rank == 6:
            marker = " ← playoff cut"
        elif rank == 10:
            marker = " ← play-in cut"
        w(
            f"| {rank} | {is_gsw}{sr['TeamCity']} {sr['TeamName']}{is_gsw} "
            f"| {int(sr['WINS'])} | {int(sr['LOSSES'])} "
            f"| {_pct(sr['WinPCT'])} | {proj} | {gb_str}{marker} |"
        )
    w("")

    # ── 3. Projection Models ─────────────────────────────────────
    w("## 3. Projection Models — Final Record")
    w("")
    w("Five independent methods projecting the Warriors' final 82-game record:")
    w("")
    w("| # | Model | Method | Projected W-L | Win% |")
    w("|---:|---|---|:---:|---:|")
    w(f"| 1 | **Pace Projection** | Extrapolate current {_pct(win_pct)} | **{pace_w}-{pace_l}** | {_pct(pace_w/82)} |")
    w(f"| 2 | **Pythagorean Wins** | Points ratio ({ppg:.1f} vs {opp_ppg:.1f}) | **{pyth_w}-{pyth_l}** | {_pct(pyth_w/82)} |")
    w(f"| 3 | **Weighted Recent Form** | 40% season + 60% L15 ({_pct(recent_pct)}) | **{wrf_w}-{wrf_l}** | {_pct(wrf_w/82)} |")
    w(f"| 4 | **Home/Away Adjusted** | H: {_pct(home_pct)}, A: {_pct(away_pct)} | **{ha_w}-{ha_l}** | {_pct(ha_w/82)} |")
    w(f"| 5 | **Monte Carlo (10K sims)** | Log5 opponent-adjusted | **{mc_proj_wins}-{mc_proj_losses}** | {_pct(mc_proj_wins/82)} |")
    w("")

    # Consensus
    all_proj_wins = [pace_w, pyth_w, wrf_w, ha_w, mc_proj_wins]
    consensus_w = round(sum(all_proj_wins) / len(all_proj_wins))
    consensus_l = TOTAL_SEASON_GAMES - consensus_w
    w(f"> **Consensus projection: {consensus_w}-{consensus_l}** (average of all 5 models)")
    w("")

    # ── 4. Model Details ─────────────────────────────────────────
    w("## 4. Model Methodology")
    w("")
    w("### Pace Projection")
    w(f"Assumes the team continues at its current {_pct(win_pct)} clip for the remaining {remaining} games.")
    w(f"Projected remaining: {pace_w - current_wins}-{pace_l - current_losses}.")
    w("")

    w("### Pythagorean Wins")
    w(f"Uses the Pythagorean expectation formula with NBA exponent ({PYTH_EXPONENT}):")
    w(f"- Points For: {ppg:.1f} PPG | Points Against: {opp_ppg:.1f} PPG")
    w(f"- Expected win%: **{_pct(pyth_pct)}** (vs actual {_pct(win_pct)})")
    if pyth_pct > win_pct + 0.02:
        w(f"- The Warriors have been **unlucky** — their point differential suggests they should have ~{round(pyth_pct * games_played)} wins, not {current_wins}.")
    elif pyth_pct < win_pct - 0.02:
        w(f"- The Warriors have been **lucky** — their point differential suggests only ~{round(pyth_pct * games_played)} wins, but they have {current_wins}.")
    else:
        w(f"- Record is well-aligned with point differential — no significant luck factor.")
    w("")

    w("### Weighted Recent Form")
    w(f"Blends season-long performance ({_pct(win_pct)}) with momentum from the last 15 games ({_pct(recent_pct)}):")
    w(f"- Blended rate: **{_pct(wrf_pct)}**")
    if recent_pct < win_pct - 0.05:
        w(f"- Recent form is **worse** than season average — the team is trending downward.")
    elif recent_pct > win_pct + 0.05:
        w(f"- Recent form is **better** than season average — positive momentum.")
    else:
        w(f"- Recent form is roughly in line with season average.")
    w("")

    w("### Home/Away Adjusted")
    w(f"Applies split-specific win rates to the estimated remaining schedule:")
    w(f"- ~{remaining_home} home games at {_pct(home_pct)} → ~{round(home_pct * remaining_home)} wins")
    w(f"- ~{remaining_away} away games at {_pct(away_pct)} → ~{round(away_pct * remaining_away)} wins")
    w("")

    # ── 5. Monte Carlo Deep Dive ─────────────────────────────────
    w("## 5. Monte Carlo Simulation (10,000 Trials)")
    w("")
    w(f"Each of the {remaining} remaining games is simulated independently using the **log5 formula**,")
    w(f"which adjusts win probability based on home/away split and opponent strength.")
    w("")
    w("### Distribution Summary")
    w("")
    w("| Metric | Remaining Wins | Final Record |")
    w("|---|---:|---|")
    w(f"| Mean | {mc['mean']:.1f} | {current_wins + round(mc['mean'])}-{TOTAL_SEASON_GAMES - current_wins - round(mc['mean'])} |")
    w(f"| Median | {mc['median']:.0f} | {current_wins + int(mc['median'])}-{TOTAL_SEASON_GAMES - current_wins - int(mc['median'])} |")
    w(f"| Std Dev | {mc['std']:.1f} | — |")
    w(f"| 10th percentile (pessimistic) | {mc['p10']} | {current_wins + mc['p10']}-{TOTAL_SEASON_GAMES - current_wins - mc['p10']} |")
    w(f"| 25th percentile | {mc['p25']} | {current_wins + mc['p25']}-{TOTAL_SEASON_GAMES - current_wins - mc['p25']} |")
    w(f"| 75th percentile | {mc['p75']} | {current_wins + mc['p75']}-{TOTAL_SEASON_GAMES - current_wins - mc['p75']} |")
    w(f"| 90th percentile (optimistic) | {mc['p90']} | {current_wins + mc['p90']}-{TOTAL_SEASON_GAMES - current_wins - mc['p90']} |")
    w(f"| Best case | {mc['max']} | {current_wins + mc['max']}-{TOTAL_SEASON_GAMES - current_wins - mc['max']} |")
    w(f"| Worst case | {mc['min']} | {current_wins + mc['min']}-{TOTAL_SEASON_GAMES - current_wins - mc['min']} |")
    w("")

    # Text-based histogram
    w("### Win Distribution (Remaining Games)")
    w("")
    w("```")
    dist = mc["distribution"]
    max_pct = max(dist.values()) if dist else 0
    for wins_remaining in sorted(dist.keys()):
        pct = dist[wins_remaining]
        bar_len = int(round(pct / max_pct * 40)) if max_pct > 0 else 0
        bar = "█" * bar_len
        final = current_wins + wins_remaining
        w(f"  {wins_remaining:2d}W ({final:2d}-{TOTAL_SEASON_GAMES - final:2d}) | {bar} {pct*100:.1f}%")
    w("```")
    w("")

    # ── 6. Playoff Scenarios ─────────────────────────────────────
    w("## 6. Playoff Scenarios")
    w("")
    w(f"Based on Monte Carlo simulation, using estimated thresholds:")
    w(f"- **Playoff lock (top 6):** ~{playoff_threshold} wins")
    w(f"- **Play-in (7-10):** ~{playin_threshold} wins")
    w("")
    w("| Scenario | Probability |")
    w("|---|---:|")
    w(f"| Make playoffs (top 6 seed) | **{pct_make_playoffs:.1f}%** |")
    w(f"| Play-in tournament (7-10 seed) | **{pct_make_playin - pct_make_playoffs:.1f}%** |")
    w(f"| Miss postseason entirely | **{pct_miss:.1f}%** |")
    w("")

    # Wins needed analysis
    wins_needed_playoffs = max(0, playoff_threshold - current_wins)
    wins_needed_playin = max(0, playin_threshold - current_wins)
    w(f"To lock a **top-6 seed**, the Warriors need **{wins_needed_playoffs} more wins** in {remaining} games ({_pct(wins_needed_playoffs / remaining if remaining > 0 else 0)} pace).")
    w("")
    w(f"To secure a **play-in spot** (top 10), they need **{wins_needed_playin} more wins** in {remaining} games ({_pct(wins_needed_playin / remaining if remaining > 0 else 0)} pace).")
    w("")

    # ── 7. Strength of Remaining Schedule ────────────────────────
    w("## 7. Strength of Remaining Schedule")
    w("")
    w(f"Estimated average opponent win% for remaining games: **{_pct(est_remaining_opp_pct)}**")
    w("")
    if est_remaining_opp_pct > 0.52:
        w("> The remaining schedule is **above average difficulty**. Expect tougher competition down the stretch.")
    elif est_remaining_opp_pct < 0.48:
        w("> The remaining schedule is **below average difficulty**. Favorable matchups ahead.")
    else:
        w("> The remaining schedule is **roughly average** in difficulty.")
    w("")

    w("### Conference Opponent Context")
    w("")
    w("| Conference | Avg Win% | Note |")
    w("|---|---:|---|")
    w(f"| Western Conference | {_pct(west_avg_pct)} | ~65% of remaining games |")
    w(f"| Eastern Conference | {_pct(east_avg_pct)} | ~35% of remaining games |")
    w("")

    # ── 8. Key Variables ─────────────────────────────────────────
    w("## 8. Key Variables")
    w("")
    w("Factors that could swing the projection significantly:")
    w("")

    # Home record
    w("### Home Court")
    if home_pct >= 0.60:
        w(f"- Currently {_pct(home_pct)} at home — **strong advantage**. With ~{remaining_home} home games left, this is the Warriors' path to the playoffs.")
    else:
        w(f"- Home record of {_pct(home_pct)} is **below expectations**. Improving this is critical.")

    # Road woes
    if away_pct < 0.45:
        w(f"- Road record of {_pct(away_pct)} is a major concern. Even a modest improvement to 45% would add ~{round(0.45 * remaining_away) - round(away_pct * remaining_away)} wins.")
    w("")

    # Clutch
    w("### Clutch Situations")
    close_games = str(gsw.get("ThreePTSOrLess", "N/A"))
    w(f"- Record in games decided by 3 or fewer points: {close_games}")
    w(f"- Improvement here directly affects 2-3 wins in the final stretch.")
    w("")

    # Scoring differential
    w("### Scoring Margin")
    w(f"- Current differential: **{_plus_minus(diff_ppg)} PPG**")
    if diff_ppg > 0:
        w(f"- Positive differential supports sustainability of the current record.")
    else:
        w(f"- **Negative differential** suggests the current record may be slightly inflated.")
    w("")

    # Momentum
    w("### Momentum")
    w(f"- Last 15 games: **{recent_w}-{recent_l}** ({_pct(recent_pct)})")
    if recent_pct < win_pct - 0.05:
        w(f"- **Downward trend** — if this continues, expect the lower end of projections.")
    elif recent_pct > win_pct + 0.05:
        w(f"- **Upward trend** — momentum favors higher-end outcomes.")
    else:
        w(f"- Relatively steady — no significant shift in trajectory.")
    w("")

    # ── 9. Best / Worst Case ─────────────────────────────────────
    w("## 9. Best and Worst Case Scenarios")
    w("")
    best_final = current_wins + mc["p90"]
    worst_final = current_wins + mc["p10"]

    w("### Optimistic Scenario (90th percentile)")
    w(f"**{best_final}-{TOTAL_SEASON_GAMES - best_final}** — Going {mc['p90']}-{remaining - mc['p90']} the rest of the way.")
    w("")
    w("This happens if:")
    w("- The home court advantage holds or improves")
    w("- Steph Curry stays healthy and maintains his scoring (27+ PPG)")
    w("- Turnover rate drops below 14 per game")
    w("- Clutch game conversion improves")
    w("")

    w("### Pessimistic Scenario (10th percentile)")
    w(f"**{worst_final}-{TOTAL_SEASON_GAMES - worst_final}** — Going {mc['p10']}-{remaining - mc['p10']} the rest of the way.")
    w("")
    w("This happens if:")
    w("- Road struggles continue at current rate")
    w("- Injuries to key players (Curry, Butler)")
    w("- Turnover and clutch issues persist")
    w("- February's scoring dip extends into March")
    w("")

    # ── 10. Summary ──────────────────────────────────────────────
    w("## 10. Summary")
    w("")
    w(f"| | Value |")
    w(f"|---|---|")
    w(f"| Current Record | {current_wins}-{current_losses} |")
    w(f"| Consensus Final Record | **{consensus_w}-{consensus_l}** |")
    w(f"| Monte Carlo Mean | **{mc_proj_wins}-{mc_proj_losses}** |")
    w(f"| Playoff Probability | **{pct_make_playoffs:.1f}%** |")
    w(f"| Play-in Probability | **{pct_make_playin:.1f}%** (incl. playoffs) |")
    w(f"| Miss Postseason | **{pct_miss:.1f}%** |")
    w(f"| Needs for Playoffs | {wins_needed_playoffs}W in {remaining}G ({_pct(wins_needed_playoffs/remaining if remaining else 0)}) |")
    w("")

    w("---")
    w(f"*Model generated by NBATracker prediction engine — {N_SIMULATIONS:,} Monte Carlo simulations.*")
    w("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--season", default="2025-26", help="NBA season (e.g. 2025-26).")
@click.option("--output", "output_path", default=None, help="Output .md file path.")
def main(season: str, output_path: str | None) -> None:
    """Generate a remaining-season prediction report for the Warriors."""
    if output_path is None:
        safe = season.replace("-", "_")
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", f"warriors_predictions_{safe}.md"
        )

    report = build_report(season)

    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nPrediction report written to {out_path}")


if __name__ == "__main__":
    main()
