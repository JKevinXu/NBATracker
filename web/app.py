"""NBATracker — Interactive Dashboard (FastAPI)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

log = logging.getLogger("nbatracker")

# ── constants ─────────────────────────────────────────────────────

TEAM_ID = 1610612744
TEAM_ABBR = "GSW"
DEFAULT_SEASON = "2025-26"
PYTH_EXPONENT = 13.91
N_SIMULATIONS = 10_000

# ── app setup ─────────────────────────────────────────────────────

app = FastAPI(title="NBATracker Dashboard")

WEB_DIR = Path(__file__).parent
CACHE_DIR = WEB_DIR / "cache"
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


# ── JSON cache reader ─────────────────────────────────────────────

_mem: dict[str, list[pd.DataFrame]] = {}


def _load_cached(name: str) -> list[pd.DataFrame]:
    """Load pre-fetched JSON from web/cache/ and parse into DataFrames."""
    if name in _mem:
        return _mem[name]
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache file {path} not found. Run: python -m web.prefetch"
        )
    data = json.loads(path.read_text())
    frames = []
    for rs in data["resultSets"]:
        frames.append(pd.DataFrame(rs["rowSet"], columns=rs["headers"]))
    _mem[name] = frames
    return frames


# ── data fetching helpers (from cache) ────────────────────────────


def get_standings(season: str) -> pd.DataFrame:
    return _load_cached("standings")[0]

def get_team_game_log(season: str) -> pd.DataFrame:
    return _load_cached("gamelog")[0]

def get_team_splits(season: str) -> list[pd.DataFrame]:
    return _load_cached("splits")

def get_player_stats_base(season: str) -> pd.DataFrame:
    return _load_cached("player_base")[0]

def get_player_stats_adv(season: str) -> pd.DataFrame:
    return _load_cached("player_adv")[0]

def get_clutch_stats(season: str) -> pd.DataFrame:
    return _load_cached("clutch")[0]

def get_league_adv(season: str) -> pd.DataFrame:
    return _load_cached("league_adv")[0]

def get_league_base(season: str) -> pd.DataFrame:
    return _load_cached("league_base")[0]

def get_league_clutch(season: str) -> pd.DataFrame:
    return _load_cached("league_clutch")[0]


# ── helpers ───────────────────────────────────────────────────────

def tier(net: float) -> str:
    if net >= 8: return "Elite"
    if net >= 3: return "Positive"
    if net >= -3: return "Neutral"
    if net >= -8: return "Negative"
    return "Struggling"


def norm(s: pd.Series, hb=True) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    n = (s - mn) / (mx - mn) * 100
    return n if hb else 100 - n


# ══════════════════════════════════════════════════════════════════
# API ENDPOINTS (JSON)
# ══════════════════════════════════════════════════════════════════


@app.get("/api/team-stats")
def api_team_stats(season: str = DEFAULT_SEASON):
    try:
        base = get_player_stats_base(season)
        cols = ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "PTS", "REB", "AST",
                "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
        available = [c for c in cols if c in base.columns]
        df = base[available].copy()
        df = df.sort_values("PTS", ascending=False)
        return df.to_dict(orient="records")
    except Exception as exc:
        log.exception("api_team_stats failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/advanced")
def api_advanced(season: str = DEFAULT_SEASON):
    try:
        adv = get_player_stats_adv(season)
        adv = adv[adv["GP"] >= 10].copy()
        adv["TIER"] = adv["NET_RATING"].apply(tier)
        cols = ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "NET_RATING",
                "OFF_RATING", "DEF_RATING", "TS_PCT", "USG_PCT", "PIE",
                "AST_TO", "EFG_PCT", "TIER"]
        available = [c for c in cols if c in adv.columns]
        return adv[available].sort_values("NET_RATING", ascending=False).to_dict(orient="records")
    except Exception as exc:
        log.exception("api_advanced failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/game-log")
def api_game_log(season: str = DEFAULT_SEASON):
    try:
        gl = get_team_game_log(season)
        cols = ["GAME_DATE", "MATCHUP", "WL", "PTS", "FG_PCT", "FG3_PCT",
                "REB", "AST", "TOV", "STL", "BLK", "W", "L", "W_PCT", "PLUS_MINUS"]
        available = [c for c in cols if c in gl.columns]
        return gl[available].to_dict(orient="records")
    except Exception as exc:
        log.exception("api_game_log failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/standings")
def api_standings(season: str = DEFAULT_SEASON):
    try:
        st = get_standings(season)
        west = st[st["Conference"] == "West"].sort_values("WinPCT", ascending=False)
        cols = ["TeamCity", "TeamName", "TeamID", "WINS", "LOSSES", "WinPCT",
                "PlayoffRank", "ConferenceGamesBack", "PointsPG", "OppPointsPG", "DiffPointsPG"]
        available = [c for c in cols if c in west.columns]
        return west[available].to_dict(orient="records")
    except Exception as exc:
        log.exception("api_standings failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/splits")
def api_splits(season: str = DEFAULT_SEASON):
    try:
        splits = get_team_splits(season)
        result = {}
        # 0=overall, 1=home/away, 2=wins/losses, 3=monthly, 5=rest days
        for i, name in [(0, "overall"), (1, "location"), (2, "outcome"), (3, "month")]:
            if i < len(splits):
                result[name] = splits[i].to_dict(orient="records")
        if len(splits) > 5:
            result["rest"] = splits[5].to_dict(orient="records")
        return result
    except Exception as exc:
        log.exception("api_splits failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/predictions")
def api_predictions(season: str = DEFAULT_SEASON):
    try:
        standings = get_standings(season)
        game_log = get_team_game_log(season)
        splits = get_team_splits(season)

        gsw = standings[standings["TeamCity"] == "Golden State"].iloc[0]
        w = int(gsw["WINS"])
        lo = int(gsw["LOSSES"])
        gp = w + lo
        remaining = 82 - gp
        win_pct = w / gp if gp > 0 else 0.5
        ppg = float(gsw["PointsPG"])
        opp_ppg = float(gsw["OppPointsPG"])

        # Home/away
        ha = splits[1]
        home_row = ha[ha["GROUP_VALUE"] == "Home"].iloc[0]
        away_row = ha[ha["GROUP_VALUE"] == "Road"].iloc[0]
        home_pct = float(home_row["W_PCT"])
        away_pct = float(away_row["W_PCT"])
        home_gp = int(home_row["GP"])
        away_gp = int(away_row["GP"])
        rem_home = max(0, 41 - home_gp)
        rem_away = max(0, remaining - rem_home)

        # Recent 15
        recent = game_log.head(15)
        recent_w = int((recent["WL"] == "W").sum())
        recent_pct = recent_w / 15

        pace_w = w + round(win_pct * remaining)
        pyth_pct = ppg ** PYTH_EXPONENT / (ppg ** PYTH_EXPONENT + opp_ppg ** PYTH_EXPONENT)
        pyth_w = w + round(pyth_pct * remaining)
        blend = 0.4 * win_pct + 0.6 * recent_pct
        blend_w = w + round(blend * remaining)
        ha_w = w + round(home_pct * rem_home) + round(away_pct * rem_away)

        west = standings[standings["Conference"] == "West"]
        est_opp = float(west["WinPCT"].mean()) * 0.65 + float(standings[standings["Conference"] == "East"]["WinPCT"].mean()) * 0.35
        rng = np.random.default_rng(42)
        opp_pcts = list(np.clip(rng.normal(est_opp, 0.10, remaining), 0.15, 0.85))

        schedule = []
        for idx, opc in enumerate(opp_pcts[:rem_home]):
            schedule.append((True, opc))
        for opc in opp_pcts[rem_home:rem_home + rem_away]:
            schedule.append((False, opc))
        while len(schedule) < remaining:
            schedule.append((len(schedule) % 2 == 0, est_opp))

        probs = []
        for is_home, opc in schedule:
            lp = home_pct if is_home else away_pct
            lp = max(0.05, min(0.95, lp))
            opc = max(0.05, min(0.95, opc))
            p = (lp * (1 - opc)) / (lp * (1 - opc) + (1 - lp) * opc)
            probs.append(p)

        probs_arr = np.array(probs)
        draws = rng.random((N_SIMULATIONS, len(schedule)))
        sim_wins = (draws < probs_arr).sum(axis=1)

        counts = Counter(int(x) for x in sim_wins)
        dist = {k: v / N_SIMULATIONS for k, v in sorted(counts.items())}
        mc_mean = float(np.mean(sim_wins))
        mc_w = w + round(mc_mean)

        west_sorted = west.sort_values("PlayoffRank")
        s8 = west_sorted[west_sorted["PlayoffRank"] == 8]
        s10 = west_sorted[west_sorted["PlayoffRank"] == 10]
        threshold_playoff = round(float(s8.iloc[0]["WinPCT"]) * 82) if len(s8) > 0 else 41
        threshold_playin = round(float(s10.iloc[0]["WinPCT"]) * 82) if len(s10) > 0 else 38

        total_wins = w + sim_wins
        pct_playoff = float(np.mean(total_wins >= threshold_playoff) * 100)
        pct_playin = float(np.mean(total_wins >= threshold_playin) * 100)

        return {
            "current": {"wins": w, "losses": lo, "remaining": remaining, "win_pct": win_pct,
                         "ppg": ppg, "opp_ppg": opp_ppg, "seed": int(gsw["PlayoffRank"])},
            "models": [
                {"name": "Pace", "wins": pace_w, "losses": 82 - pace_w},
                {"name": "Pythagorean", "wins": pyth_w, "losses": 82 - pyth_w},
                {"name": "Weighted Form", "wins": blend_w, "losses": 82 - blend_w},
                {"name": "Home/Away", "wins": ha_w, "losses": 82 - ha_w},
                {"name": "Monte Carlo", "wins": mc_w, "losses": 82 - mc_w},
            ],
            "monte_carlo": {
                "mean": mc_mean, "median": float(np.median(sim_wins)),
                "std": float(np.std(sim_wins)),
                "p10": int(np.percentile(sim_wins, 10)),
                "p90": int(np.percentile(sim_wins, 90)),
                "min": int(np.min(sim_wins)), "max": int(np.max(sim_wins)),
                "distribution": {str(k): v for k, v in dist.items()},
            },
            "playoffs": {
                "threshold_playoff": threshold_playoff,
                "threshold_playin": threshold_playin,
                "pct_playoff": pct_playoff,
                "pct_playin": pct_playin,
                "pct_miss": 100 - pct_playin,
            },
        }
    except Exception as exc:
        log.exception("api_predictions failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/gap-fill")
def api_gap_fill(season: str = DEFAULT_SEASON):
    try:
        league_adv = get_league_adv(season)
        league_base = get_league_base(season)
        league_clutch = get_league_clutch(season)

        adv_f = league_adv[(league_adv["TEAM_ID"] != TEAM_ID) & (league_adv["GP"] >= 30) & (league_adv["MIN"] >= 20)].copy()
        base_f = league_base[(league_base["TEAM_ID"] != TEAM_ID) & (league_base["GP"] >= 30) & (league_base["MIN"] >= 20)].copy()

        merged = base_f.merge(
            adv_f[["PLAYER_ID", "NET_RATING", "OFF_RATING", "DEF_RATING", "TS_PCT", "USG_PCT", "PIE", "AST_TO"]],
            on="PLAYER_ID", how="inner", suffixes=("", "_adv"),
        )
        cl = league_clutch[["PLAYER_ID", "PTS", "FG_PCT"]].rename(columns={"PTS": "CLUTCH_PTS", "FG_PCT": "CLUTCH_FG_PCT"})
        merged = merged.merge(cl, on="PLAYER_ID", how="left")
        merged["CLUTCH_PTS"] = merged["CLUTCH_PTS"].fillna(0)
        merged["CLUTCH_FG_PCT"] = merged["CLUTCH_FG_PCT"].fillna(0)

        if len(merged) == 0:
            return []

        merged["dim_defense"] = 0.50 * norm(merged["DEF_RATING"], False) + 0.25 * norm(merged["STL"]) + 0.25 * norm(merged["BLK"])
        merged["dim_scoring"] = norm(merged["PTS"] * merged["TS_PCT"])
        merged["dim_rebounding"] = norm(0.4 * merged["REB"] + 0.6 * merged["DREB"])
        merged["dim_playmaking"] = 0.60 * norm(merged["AST_TO"]) + 0.40 * norm(merged["TOV"], False)
        merged["dim_versatility"] = 0.50 * norm(merged["NET_RATING"]) + 0.50 * norm(merged["PIE"])
        merged["dim_clutch"] = 0.60 * norm(merged["CLUTCH_PTS"]) + 0.40 * norm(merged["CLUTCH_FG_PCT"])
        merged["gap_fill"] = (
            0.25 * merged["dim_defense"] + 0.20 * merged["dim_scoring"]
            + 0.15 * merged["dim_rebounding"] + 0.15 * merged["dim_playmaking"]
            + 0.15 * merged["dim_versatility"] + 0.10 * merged["dim_clutch"]
        )

        merged = merged.sort_values("gap_fill", ascending=False).head(20)
        cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "STL", "BLK",
                "DEF_RATING", "TS_PCT", "NET_RATING", "PIE", "USG_PCT", "AST_TO",
                "gap_fill", "dim_defense", "dim_scoring", "dim_rebounding",
                "dim_playmaking", "dim_versatility", "dim_clutch", "CLUTCH_PTS"]
        available = [c for c in cols if c in merged.columns]
        return merged[available].to_dict(orient="records")
    except Exception as exc:
        log.exception("api_gap_fill failed")
        return JSONResponse({"error": str(exc)}, status_code=502)


# ══════════════════════════════════════════════════════════════════
# PAGE ROUTES (HTML)
# ══════════════════════════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
def page_dashboard(request: Request, season: str = DEFAULT_SEASON):
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "season": season, "active_page": "dashboard",
    })


@app.get("/players", response_class=HTMLResponse)
def page_players(request: Request, season: str = DEFAULT_SEASON):
    return templates.TemplateResponse("players.html", {
        "request": request, "season": season, "active_page": "players",
    })


@app.get("/advanced", response_class=HTMLResponse)
def page_advanced(request: Request, season: str = DEFAULT_SEASON):
    return templates.TemplateResponse("advanced.html", {
        "request": request, "season": season, "active_page": "advanced",
    })


@app.get("/predictions", response_class=HTMLResponse)
def page_predictions(request: Request, season: str = DEFAULT_SEASON):
    return templates.TemplateResponse("predictions.html", {
        "request": request, "season": season, "active_page": "predictions",
    })


@app.get("/trades", response_class=HTMLResponse)
def page_trades(request: Request, season: str = DEFAULT_SEASON):
    return templates.TemplateResponse("trades.html", {
        "request": request, "season": season, "active_page": "trades",
    })


@app.get("/gaps", response_class=HTMLResponse)
def page_gaps(request: Request, season: str = DEFAULT_SEASON):
    return templates.TemplateResponse("gaps.html", {
        "request": request, "season": season, "active_page": "gaps",
    })
