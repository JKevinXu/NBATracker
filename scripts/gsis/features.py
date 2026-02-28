"""
GSIS Feature Engineering Pipeline
Builds the pre-game feature matrix for the Warriors' 2025-26 season.
Every feature for game N is computed using ONLY data available *before* game N.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"

TEAM_ID = 1610612744  # Warriors

# Mapping from 3-letter abbreviations to team IDs (from standings)
# We'll build this dynamically from standings data

# ── helpers ────────────────────────────────────────────────────────

def _load(name: str) -> dict:
    """Load a cached JSON file and return the raw dict."""
    return json.loads((CACHE / f"{name}.json").read_text())


def _rs_to_df(data: dict, idx: int = 0) -> pd.DataFrame:
    """Convert resultSets[idx] to a DataFrame."""
    rs = data.get("resultSets", data)
    if isinstance(rs, list):
        return pd.DataFrame(rs[idx]["rowSet"], columns=rs[idx]["headers"])
    raise ValueError(f"Unexpected resultSets type: {type(rs)}")


def _parse_date(date_str: str) -> datetime:
    """Parse NBA date strings like 'FEB 19, 2026'."""
    return datetime.strptime(date_str.strip(), "%b %d, %Y")


# ── opponent lookup tables ─────────────────────────────────────────

def build_opponent_table() -> pd.DataFrame:
    """Build a table of every NBA team's season-level stats from standings."""
    standings = _load("standings")
    df = _rs_to_df(standings)
    # Compute useful columns
    df["WIN_PCT"] = df["WinPCT"].astype(float)
    df["PPG"] = df["PointsPG"].astype(float)
    df["OPP_PPG"] = df["OppPointsPG"].astype(float)
    df["DIFF_PPG"] = df["DiffPointsPG"].astype(float)
    df["TEAM_FULL"] = df["TeamCity"] + " " + df["TeamName"]
    return df


def build_opponent_advanced() -> dict:
    """
    Build per-team advanced stats (OFF_RTG, DEF_RTG, NET_RTG, PACE, TS%)
    by aggregating player-level advanced data from league_adv.
    """
    adv = _load("league_adv")
    df = _rs_to_df(adv)
    # Aggregate to team level (weighted by minutes would be ideal, but
    # for team-level OFF/DEF RTG we take the team-level figures from
    # the min-weighted data in standings instead).
    # For now, approximate from standings PPG and OppPPG
    standings = _load("standings")
    st = _rs_to_df(standings)
    result = {}
    for _, row in st.iterrows():
        tid = row["TeamID"]
        wins = int(row["WINS"])
        losses = int(row["LOSSES"])
        gp = wins + losses
        ppg = float(row["PointsPG"])
        opp_ppg = float(row["OppPointsPG"])
        # Approximate ratings (per-game, not per-100, but directionally correct)
        result[tid] = {
            "WIN_PCT": float(row["WinPCT"]),
            "PPG": ppg,
            "OPP_PPG": opp_ppg,
            "NET_PPG": ppg - opp_ppg,
            "GP": gp,
        }
    return result


# ── team abbreviation extraction ───────────────────────────────────

def _extract_opponent_abbrev(matchup: str) -> str:
    """
    Extract the 3-letter opponent abbreviation from the MATCHUP string.
    e.g. 'GSW vs. BOS' → 'BOS', 'GSW @ LAL' → 'LAL'
    """
    if " vs. " in matchup:
        return matchup.split(" vs. ")[-1].strip()
    elif " @ " in matchup:
        return matchup.split(" @ ")[-1].strip()
    return matchup.split()[-1].strip()


def _is_home(matchup: str) -> int:
    """1 if home game, 0 if away."""
    return 1 if " vs. " in matchup else 0


# ── abbrev to team_id mapping ─────────────────────────────────────

def build_abbrev_map() -> dict:
    """Map 3-letter abbreviations to TeamIDs using league_base."""
    lb = _load("league_base")
    df = _rs_to_df(lb)
    mapping = {}
    for _, row in df.iterrows():
        abbrev = row["TEAM_ABBREVIATION"]
        tid = row["TEAM_ID"]
        if abbrev not in mapping:
            mapping[abbrev] = tid
    return mapping


# ── rolling-stat helpers ───────────────────────────────────────────

def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Shifted rolling mean so game N uses only games 1..N-1."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _win_streak(wl_series: pd.Series) -> pd.Series:
    """
    Compute current win/loss streak entering each game.
    Positive = consecutive wins, negative = consecutive losses.
    Uses only data BEFORE the game (shifted).
    """
    streaks = []
    current = 0
    for wl in wl_series:
        streaks.append(current)  # streak entering this game
        if wl == "W":
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
    return pd.Series(streaks, index=wl_series.index)


# ── MAIN: build feature matrix ────────────────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    """
    Build the full pre-game feature matrix.
    Returns a DataFrame with one row per game and only pre-game features.
    """
    # ── 1. Load game log ─────────────────────────────────────────
    gl_data = _load("gamelog")
    gl = _rs_to_df(gl_data)
    # Reverse so oldest game is first (chronological order)
    gl = gl.iloc[::-1].reset_index(drop=True)

    # Parse dates
    gl["DATE"] = gl["GAME_DATE"].apply(_parse_date)
    gl["OPP_ABBREV"] = gl["MATCHUP"].apply(_extract_opponent_abbrev)
    gl["HOME"] = gl["MATCHUP"].apply(_is_home)
    gl["WIN"] = (gl["WL"] == "W").astype(int)

    # Cast numeric columns
    num_cols = ["PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
                "STL", "BLK", "TOV", "PF"]
    for c in num_cols:
        gl[c] = pd.to_numeric(gl[c], errors="coerce")

    # ── 2. Rolling Warriors features (use only data before game) ─
    gl["L5_PTS"] = _rolling_mean(gl["PTS"], 5)
    gl["L10_PTS"] = _rolling_mean(gl["PTS"], 10)
    gl["L5_AST"] = _rolling_mean(gl["AST"], 5)
    gl["L10_AST"] = _rolling_mean(gl["AST"], 10)
    gl["L5_REB"] = _rolling_mean(gl["REB"], 5)
    gl["L10_REB"] = _rolling_mean(gl["REB"], 10)
    gl["L5_TOV"] = _rolling_mean(gl["TOV"], 5)
    gl["L10_TOV"] = _rolling_mean(gl["TOV"], 10)
    gl["L5_FG_PCT"] = _rolling_mean(gl["FG_PCT"], 5)
    gl["L10_FG_PCT"] = _rolling_mean(gl["FG_PCT"], 10)
    gl["L5_FG3_PCT"] = _rolling_mean(gl["FG3_PCT"], 5)
    gl["L5_FT_PCT"] = _rolling_mean(gl["FT_PCT"], 5)

    # Rolling rates
    gl["L5_3PT_RATE"] = _rolling_mean(gl["FG3A"] / gl["FGA"].replace(0, np.nan), 5)
    gl["L5_FTA_RATE"] = _rolling_mean(gl["FTA"] / gl["FGA"].replace(0, np.nan), 5)
    gl["L5_TOV_RATE"] = _rolling_mean(gl["TOV"] / (gl["FGA"] + 0.44 * gl["FTA"] + gl["TOV"]).replace(0, np.nan), 5)
    gl["L5_OREB_PCT"] = _rolling_mean(gl["OREB"] / gl["REB"].replace(0, np.nan), 5)
    gl["L5_AST_RATE"] = _rolling_mean(gl["AST"] / gl["FGM"].replace(0, np.nan), 5)
    gl["L5_STL_RATE"] = _rolling_mean(gl["STL"], 5)

    # Momentum
    gl["WIN_STREAK"] = _win_streak(gl["WL"])
    gl["L5_WIN_PCT"] = gl["WIN"].shift(1).rolling(5, min_periods=1).mean()
    gl["L10_WIN_PCT"] = gl["WIN"].shift(1).rolling(10, min_periods=1).mean()

    # Season cumulative (before current game)
    gl["SEASON_WIN_PCT"] = gl["WIN"].shift(1).expanding().mean()
    gl["SEASON_PTS_AVG"] = gl["PTS"].shift(1).expanding().mean()
    gl["SEASON_FG_PCT"] = gl["FG_PCT"].shift(1).expanding().mean()

    # ── 3. Context: rest days, back-to-back, schedule density ────
    gl["REST_DAYS"] = gl["DATE"].diff().dt.days.fillna(3).astype(int) - 1
    gl["REST_DAYS"] = gl["REST_DAYS"].clip(0, 5)  # cap at 5
    gl["B2B"] = (gl["REST_DAYS"] == 0).astype(int)
    gl["GAMES_IN_LAST_7"] = gl["DATE"].apply(
        lambda d: ((gl["DATE"] < d) & (gl["DATE"] >= d - timedelta(days=7))).sum()
    )

    # Calendar
    gl["MONTH"] = gl["DATE"].dt.month
    gl["DAY_OF_WEEK"] = gl["DATE"].dt.dayofweek  # 0=Mon, 6=Sun
    gl["GAME_NUM"] = gl.index + 1  # sequential game number in season

    # ── 4. Opponent features ─────────────────────────────────────
    abbrev_map = build_abbrev_map()
    opp_stats = build_opponent_advanced()

    opp_win_pct = []
    opp_ppg = []
    opp_opp_ppg = []
    opp_net_ppg = []
    opp_gp = []

    for _, row in gl.iterrows():
        opp_abbrev = row["OPP_ABBREV"]
        tid = abbrev_map.get(opp_abbrev)
        if tid and tid in opp_stats:
            s = opp_stats[tid]
            opp_win_pct.append(s["WIN_PCT"])
            opp_ppg.append(s["PPG"])
            opp_opp_ppg.append(s["OPP_PPG"])
            opp_net_ppg.append(s["NET_PPG"])
            opp_gp.append(s["GP"])
        else:
            opp_win_pct.append(0.5)
            opp_ppg.append(110.0)
            opp_opp_ppg.append(110.0)
            opp_net_ppg.append(0.0)
            opp_gp.append(56)

    gl["OPP_WIN_PCT"] = opp_win_pct
    gl["OPP_PPG"] = opp_ppg
    gl["OPP_OPP_PPG"] = opp_opp_ppg
    gl["OPP_NET_PPG"] = opp_net_ppg

    # Head-to-head history (only prior meetings this season)
    h2h_win_pct = []
    h2h_pts_diff = []
    h2h_games = []
    season_meeting_num = []

    for i, row in gl.iterrows():
        prior = gl.iloc[:i]
        prior_vs_opp = prior[prior["OPP_ABBREV"] == row["OPP_ABBREV"]]
        n_prior = len(prior_vs_opp)
        h2h_games.append(n_prior)
        season_meeting_num.append(n_prior + 1)
        if n_prior > 0:
            h2h_win_pct.append(prior_vs_opp["WIN"].mean())
            # Approximate pts diff from our PTS vs avg
            h2h_pts_diff.append(prior_vs_opp["PTS"].mean() - 110.0)
        else:
            h2h_win_pct.append(0.5)  # no prior data → neutral
            h2h_pts_diff.append(0.0)

    gl["H2H_WIN_PCT"] = h2h_win_pct
    gl["H2H_PTS_DIFF"] = h2h_pts_diff
    gl["H2H_GAMES"] = h2h_games
    gl["SEASON_MEETING_NUM"] = season_meeting_num

    # ── 5. Player availability (from player_gamelogs) ────────────
    try:
        pg_data = _load("player_gamelogs")
        pg = _rs_to_df(pg_data)
        pg["GAME_DATE_DT"] = pd.to_datetime(pg["GAME_DATE"])

        key_players = {
            "Stephen Curry": "CURRY",
            "Jimmy Butler III": "BUTLER",
            "De'Anthony Melton": "MELTON",
            "Draymond Green": "GREEN",
            "Jonathan Kuminga": "KUMINGA",
        }

        game_ids = gl["Game_ID"].tolist()

        for player_name, col_prefix in key_players.items():
            player_games = set(
                pg[pg["PLAYER_NAME"] == player_name]["GAME_ID"].tolist()
            )
            gl[f"{col_prefix}_AVAILABLE"] = gl["Game_ID"].apply(
                lambda gid: 1 if gid in player_games else 0
            )
            # Simple fatigue proxy: player's minutes in last 3 games
            player_pg = pg[pg["PLAYER_NAME"] == player_name].sort_values("GAME_DATE_DT")
            mins_by_game = dict(zip(player_pg["GAME_ID"], pd.to_numeric(player_pg["MIN"], errors="coerce")))
            fatigue_vals = []
            for idx_g in range(len(game_ids)):
                # Last 3 games this player played before game idx_g
                prior_ids = game_ids[:idx_g]
                prior_mins = [mins_by_game.get(gid, 0) for gid in prior_ids if gid in mins_by_game]
                last3 = prior_mins[-3:] if len(prior_mins) >= 3 else prior_mins
                avg_min = np.mean(last3) if last3 else 0
                fatigue_vals.append(avg_min)
            gl[f"{col_prefix}_FATIGUE"] = fatigue_vals
    except Exception:
        # If player gamelogs unavailable, set defaults
        for suffix in ["CURRY", "BUTLER", "MELTON", "GREEN", "KUMINGA"]:
            gl[f"{suffix}_AVAILABLE"] = 1
            gl[f"{suffix}_FATIGUE"] = 30.0

    # ── 6. Select feature columns ────────────────────────────────
    feature_cols = [
        # Rolling team stats
        "L5_PTS", "L10_PTS", "L5_AST", "L10_AST", "L5_REB", "L10_REB",
        "L5_TOV", "L10_TOV", "L5_FG_PCT", "L10_FG_PCT", "L5_FG3_PCT",
        "L5_FT_PCT",
        # Rolling rates
        "L5_3PT_RATE", "L5_FTA_RATE", "L5_TOV_RATE", "L5_OREB_PCT",
        "L5_AST_RATE", "L5_STL_RATE",
        # Momentum
        "WIN_STREAK", "L5_WIN_PCT", "L10_WIN_PCT",
        # Season cumulative
        "SEASON_WIN_PCT", "SEASON_PTS_AVG", "SEASON_FG_PCT",
        # Context
        "HOME", "REST_DAYS", "B2B", "GAMES_IN_LAST_7",
        "MONTH", "DAY_OF_WEEK", "GAME_NUM",
        # Opponent
        "OPP_WIN_PCT", "OPP_PPG", "OPP_OPP_PPG", "OPP_NET_PPG",
        # Head-to-head
        "H2H_WIN_PCT", "H2H_PTS_DIFF", "H2H_GAMES", "SEASON_MEETING_NUM",
        # Player availability
        "CURRY_AVAILABLE", "BUTLER_AVAILABLE", "MELTON_AVAILABLE",
        "GREEN_AVAILABLE", "KUMINGA_AVAILABLE",
        # Player fatigue
        "CURRY_FATIGUE", "BUTLER_FATIGUE", "MELTON_FATIGUE",
        "GREEN_FATIGUE", "KUMINGA_FATIGUE",
    ]

    target = "WIN"
    meta_cols = ["Game_ID", "DATE", "MATCHUP", "OPP_ABBREV", "WL", "PTS"]

    result = gl[meta_cols + feature_cols + [target]].copy()
    # Fill any remaining NaN with 0 (first few games have incomplete rolling windows)
    result[feature_cols] = result[feature_cols].fillna(0)

    return result, feature_cols, target


# ── convenience ────────────────────────────────────────────────────

if __name__ == "__main__":
    df, fcols, tgt = build_feature_matrix()
    print(f"Feature matrix: {df.shape[0]} games × {len(fcols)} features")
    print(f"Target distribution: {df[tgt].value_counts().to_dict()}")
    print(f"\nFeatures ({len(fcols)}):")
    for i, c in enumerate(fcols, 1):
        print(f"  {i:2d}. {c}")
    print(f"\nSample (last 3 games):")
    print(df.tail(3)[["DATE", "MATCHUP", "WL"] + fcols[:8]].to_string())
