"""Pre-fetch NBA data and save to JSON cache files for the web app."""

import json
import time
from pathlib import Path

import requests

TEAM_ID = 1610612744
SEASON = "2025-26"
API_DELAY = 0.8
TIMEOUT = 45

NBA_BASE = "https://stats.nba.com/stats"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

CACHE_DIR = Path(__file__).parent / "cache"


def fetch(name: str, endpoint: str, params: dict):
    """Fetch from stats.nba.com and save to cache."""
    print(f"  Fetching {name}...", end=" ", flush=True)
    url = f"{NBA_BASE}/{endpoint}"
    t0 = time.time()
    resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0
    path = CACHE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=None))
    rows = sum(len(rs["rowSet"]) for rs in data["resultSets"])
    print(f"OK ({rows} rows, {elapsed:.1f}s)")
    time.sleep(API_DELAY)
    return data


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pre-fetching NBA data for {SEASON}...")
    print()

    fetch("standings", "leaguestandings", {
        "LeagueID": "00", "Season": SEASON,
        "SeasonType": "Regular Season", "SeasonYear": "",
    })

    fetch("gamelog", "teamgamelog", {
        "TeamID": TEAM_ID, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "DateFrom": "", "DateTo": "",
    })

    fetch("splits", "teamdashboardbygeneralsplits", {
        "TeamID": TEAM_ID, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Base",
        "DateFrom": "", "DateTo": "", "GameSegment": "",
        "LastNGames": 0, "Location": "", "Month": 0,
        "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
        "PaceAdjust": "N", "Period": 0, "PlusMinus": "N",
        "Rank": "N", "SeasonSegment": "", "ShotClockRange": "",
        "VsConference": "", "VsDivision": "",
    })

    fetch("player_base", "leaguedashplayerstats", {
        "TeamID": TEAM_ID, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Base",
        "DateFrom": "", "DateTo": "", "GameScope": "",
        "GameSegment": "", "LastNGames": 0, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": 0, "PaceAdjust": "N", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "",
        "VsConference": "", "VsDivision": "",
    })

    fetch("player_adv", "leaguedashplayerstats", {
        "TeamID": TEAM_ID, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Advanced",
        "DateFrom": "", "DateTo": "", "GameScope": "",
        "GameSegment": "", "LastNGames": 0, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": 0, "PaceAdjust": "N", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "",
        "VsConference": "", "VsDivision": "",
    })

    fetch("clutch", "leaguedashplayerclutch", {
        "TeamID": TEAM_ID, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Base",
        "AheadBehind": "Ahead or Behind",
        "ClutchTime": "Last 5 Minutes", "PointDiff": 5,
        "DateFrom": "", "DateTo": "", "GameScope": "",
        "GameSegment": "", "LastNGames": 0, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": 0, "PaceAdjust": "N", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "",
        "VsConference": "", "VsDivision": "",
    })

    fetch("league_adv", "leaguedashplayerstats", {
        "TeamID": 0, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Advanced",
        "DateFrom": "", "DateTo": "", "GameScope": "",
        "GameSegment": "", "LastNGames": 0, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": 0, "PaceAdjust": "N", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "",
        "VsConference": "", "VsDivision": "",
    })

    fetch("league_base", "leaguedashplayerstats", {
        "TeamID": 0, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Base",
        "DateFrom": "", "DateTo": "", "GameScope": "",
        "GameSegment": "", "LastNGames": 0, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": 0, "PaceAdjust": "N", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "",
        "VsConference": "", "VsDivision": "",
    })

    fetch("league_clutch", "leaguedashplayerclutch", {
        "TeamID": 0, "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00",
        "PerMode": "PerGame", "MeasureType": "Base",
        "AheadBehind": "Ahead or Behind",
        "ClutchTime": "Last 5 Minutes", "PointDiff": 5,
        "DateFrom": "", "DateTo": "", "GameScope": "",
        "GameSegment": "", "LastNGames": 0, "Location": "",
        "Month": 0, "OpponentTeamID": 0, "Outcome": "",
        "PORound": 0, "PaceAdjust": "N", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "",
        "VsConference": "", "VsDivision": "",
    })

    print()
    print("All data cached successfully!")
    print(f"Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    main()
