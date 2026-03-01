"""
Basketball Reference Scraper → NBA-API-compatible cache
Fetches team game logs + player game logs for any team and
saves them in the same JSON format the GSIS modules expect.

Usage:
    python scripts/gsis/fetch_bref.py LAL    # Lakers
    python scripts/gsis/fetch_bref.py GSW    # Warriors
    python scripts/gsis/fetch_bref.py BOS    # Celtics
"""

import json, time, re, sys
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}
DELAY = 3.5  # seconds between requests (respect b-ref rate limit)
SEASON_END_YEAR = 2026  # 2025-26 season → b-ref uses the END year

# ── Team metadata ─────────────────────────────────────────────────
TEAM_META = {
    "ATL": {"bref": "ATL", "id": 1610612737, "name": "Atlanta Hawks"},
    "BOS": {"bref": "BOS", "id": 1610612738, "name": "Boston Celtics"},
    "BKN": {"bref": "BKN", "id": 1610612751, "name": "Brooklyn Nets"},
    "CHA": {"bref": "CHA", "id": 1610612766, "name": "Charlotte Hornets"},
    "CHI": {"bref": "CHI", "id": 1610612741, "name": "Chicago Bulls"},
    "CLE": {"bref": "CLE", "id": 1610612739, "name": "Cleveland Cavaliers"},
    "DAL": {"bref": "DAL", "id": 1610612742, "name": "Dallas Mavericks"},
    "DEN": {"bref": "DEN", "id": 1610612743, "name": "Denver Nuggets"},
    "DET": {"bref": "DET", "id": 1610612765, "name": "Detroit Pistons"},
    "GSW": {"bref": "GSW", "id": 1610612744, "name": "Golden State Warriors"},
    "HOU": {"bref": "HOU", "id": 1610612745, "name": "Houston Rockets"},
    "IND": {"bref": "IND", "id": 1610612754, "name": "Indiana Pacers"},
    "LAC": {"bref": "LAC", "id": 1610612746, "name": "LA Clippers"},
    "LAL": {"bref": "LAL", "id": 1610612747, "name": "Los Angeles Lakers"},
    "MEM": {"bref": "MEM", "id": 1610612763, "name": "Memphis Grizzlies"},
    "MIA": {"bref": "MIA", "id": 1610612748, "name": "Miami Heat"},
    "MIL": {"bref": "MIL", "id": 1610612749, "name": "Milwaukee Bucks"},
    "MIN": {"bref": "MIN", "id": 1610612750, "name": "Minnesota Timberwolves"},
    "NOP": {"bref": "NOP", "id": 1610612740, "name": "New Orleans Pelicans"},
    "NYK": {"bref": "NYK", "id": 1610612752, "name": "New York Knicks"},
    "OKC": {"bref": "OKC", "id": 1610612760, "name": "Oklahoma City Thunder"},
    "ORL": {"bref": "ORL", "id": 1610612753, "name": "Orlando Magic"},
    "PHI": {"bref": "PHI", "id": 1610612755, "name": "Philadelphia 76ers"},
    "PHX": {"bref": "PHO", "id": 1610612756, "name": "Phoenix Suns"},
    "POR": {"bref": "POR", "id": 1610612757, "name": "Portland Trail Blazers"},
    "SAC": {"bref": "SAC", "id": 1610612758, "name": "Sacramento Kings"},
    "SAS": {"bref": "SAS", "id": 1610612759, "name": "San Antonio Spurs"},
    "TOR": {"bref": "TOR", "id": 1610612761, "name": "Toronto Raptors"},
    "UTA": {"bref": "UTA", "id": 1610612762, "name": "Utah Jazz"},
    "WAS": {"bref": "WAS", "id": 1610612764, "name": "Washington Wizards"},
}

# B-ref uses PHO for Phoenix but our cache uses PHX
BREF_TO_NBA_ABBREV = {v["bref"]: k for k, v in TEAM_META.items()}
BREF_TO_NBA_ABBREV.update({k: k for k in TEAM_META})  # also map self


def _get(url: str) -> BeautifulSoup:
    """Fetch a page with retry logic."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                raise
    raise RuntimeError(f"Failed to fetch {url}")


def _parse_table(soup: BeautifulSoup, table_id: str) -> list[dict]:
    """Parse a b-ref table into list of row dicts."""
    table = soup.find("table", {"id": table_id})
    if not table:
        return []
    thead = table.find("thead")
    ths = thead.find_all("tr")[-1].find_all("th")
    cols = [th.get("data-stat", th.text) for th in ths]

    tbody = table.find("tbody")
    rows = []
    for tr in tbody.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = tr.find_all(["td", "th"])
        vals = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            if stat:
                vals[stat] = cell.text.strip()
        if vals.get("ranker", "").isdigit():
            rows.append(vals)
    return rows


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _parse_minutes(mp_str):
    """Parse '35:42' → 35.7 (float minutes)."""
    if not mp_str or mp_str == "":
        return 0.0
    if ":" in mp_str:
        parts = mp_str.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    return _safe_float(mp_str)


# ══════════════════════════════════════════════════════════════════
# 1. TEAM GAME LOG
# ══════════════════════════════════════════════════════════════════

def fetch_team_gamelog(abbrev: str) -> dict:
    """Fetch team game log from Basketball Reference.
    Returns data in NBA API gamelog format."""
    meta = TEAM_META[abbrev]
    bref_code = meta["bref"]
    team_id = meta["id"]

    url = f"https://www.basketball-reference.com/teams/{bref_code}/{SEASON_END_YEAR}/gamelog/"
    print(f"  Fetching team game log: {url}")
    soup = _get(url)

    games = _parse_table(soup, "team_game_log_reg")
    if not games:
        raise ValueError(f"No team game log found for {abbrev}")

    print(f"    → {len(games)} games found")

    # Convert to NBA API format
    nba_headers = [
        "Team_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL",
        "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    ]

    row_set = []
    wins, losses = 0, 0
    for g in games:
        wl = g.get("team_game_result", "L")[0]  # "W" or "L"
        if wl == "W":
            wins += 1
        else:
            losses += 1
        gp = wins + losses
        wpct = wins / max(gp, 1)

        date_str = g.get("date", "")
        # Convert "2025-10-21" to "OCT 21, 2025" (NBA format)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            nba_date = dt.strftime("%b %d, %Y").upper()
        except ValueError:
            nba_date = date_str

        opp = g.get("opp_name_abbr", "UNK")
        opp = BREF_TO_NBA_ABBREV.get(opp, opp)
        is_away = g.get("game_location", "") == "@"
        if is_away:
            matchup = f"{abbrev} @ {opp}"
        else:
            matchup = f"{abbrev} vs. {opp}"

        # Synthetic game ID
        game_id = f"002{SEASON_END_YEAR - 1}{gp:05d}"

        row = [
            team_id,
            game_id,
            nba_date,
            matchup,
            wl,
            wins,
            losses,
            round(wpct, 3),
            240,  # team minutes (48 * 5)
            _safe_int(g.get("fg", 0)),
            _safe_int(g.get("fga", 0)),
            _safe_float(g.get("fg_pct", 0)),
            _safe_int(g.get("fg3", 0)),
            _safe_int(g.get("fg3a", 0)),
            _safe_float(g.get("fg3_pct", 0)),
            _safe_int(g.get("ft", 0)),
            _safe_int(g.get("fta", 0)),
            _safe_float(g.get("ft_pct", 0)),
            _safe_int(g.get("orb", 0)),
            _safe_int(g.get("drb", 0)),
            _safe_int(g.get("trb", 0)),
            _safe_int(g.get("ast", 0)),
            _safe_int(g.get("stl", 0)),
            _safe_int(g.get("blk", 0)),
            _safe_int(g.get("tov", 0)),
            _safe_int(g.get("pf", 0)),
            _safe_int(g.get("team_game_score", g.get("fg", 0))),  # PTS
        ]
        # Fix PTS — it's the team_game_score field from b-ref
        row[-1] = _safe_int(g.get("team_game_score", 0))

        row_set.append(row)

    # B-ref returns newest first; NBA API also returns newest first
    # But we already have chronological order from b-ref, so reverse
    row_set.reverse()

    return {
        "resultSets": [{
            "name": "TeamGameLog",
            "headers": nba_headers,
            "rowSet": row_set,
        }]
    }


# ══════════════════════════════════════════════════════════════════
# 2. PLAYER GAME LOGS
# ══════════════════════════════════════════════════════════════════

def fetch_roster(abbrev: str) -> list[dict]:
    """Fetch team roster from Basketball Reference."""
    meta = TEAM_META[abbrev]
    bref_code = meta["bref"]
    url = f"https://www.basketball-reference.com/teams/{bref_code}/{SEASON_END_YEAR}.html"
    print(f"  Fetching roster: {url}")
    soup = _get(url)

    table = soup.find("table", {"id": "roster"})
    if not table:
        print("    ⚠️ No roster table found")
        return []

    tbody = table.find("tbody")
    players = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        vals = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            if stat:
                vals[stat] = cell.text.strip()
                if stat == "player" and cell.find("a"):
                    vals["player_url"] = cell.find("a")["href"]
        if vals.get("player") and vals.get("player_url"):
            players.append(vals)

    print(f"    → {len(players)} players on roster")
    return players


def fetch_player_gamelog(player_url: str, player_name: str, team_abbrev: str) -> list[dict]:
    """Fetch one player's game log from Basketball Reference."""
    # Extract player ID from URL: /players/j/jamesle01.html → jamesle01
    pid = player_url.split("/")[-1].replace(".html", "")
    url = f"https://www.basketball-reference.com/players/{pid[0]}/{pid}/gamelog/{SEASON_END_YEAR}/"

    soup = _get(url)
    games = _parse_table(soup, "player_game_log_reg")
    return games


def build_player_gamelogs(abbrev: str, roster: list[dict]) -> dict:
    """Build NBA API compatible player_gamelogs data for all roster players."""
    meta = TEAM_META[abbrev]
    team_id = meta["id"]
    team_name = meta["name"]

    nba_headers = [
        "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "NICKNAME",
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
        "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD",
        "PTS", "PLUS_MINUS", "NBA_FANTASY_PTS", "DD2", "TD3",
        "WNBA_FANTASY_PTS",
    ]

    all_rows = []
    fetched = 0
    for player in roster:
        name = player["player"]
        url = player["player_url"]

        if fetched > 0:
            time.sleep(DELAY)

        try:
            games = fetch_player_gamelog(url, name, abbrev)
            fetched += 1
        except Exception as e:
            print(f"    ⚠️ Failed for {name}: {e}")
            continue

        if not games:
            print(f"    ⚠️ No games for {name}")
            continue

        print(f"    {name}: {len(games)} games")

        for i, g in enumerate(games):
            date_str = g.get("date", "")
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                nba_date = dt.strftime("%Y-%m-%dT00:00:00")
            except ValueError:
                nba_date = date_str

            # Parse game result to get WL
            result = g.get("game_result", "")
            wl = result[0] if result else "L"

            opp = g.get("opp_name_abbr", "UNK")
            opp = BREF_TO_NBA_ABBREV.get(opp, opp)
            is_away = g.get("game_location", "") == "@"
            if is_away:
                matchup = f"{abbrev} @ {opp}"
            else:
                matchup = f"{abbrev} vs. {opp}"

            game_num = g.get("team_game_num_season", str(i + 1))
            game_id = f"002{SEASON_END_YEAR - 1}{_safe_int(game_num):05d}"

            minutes = _parse_minutes(g.get("mp", "0"))
            pts = _safe_int(g.get("pts", 0))
            reb = _safe_int(g.get("trb", 0))
            ast = _safe_int(g.get("ast", 0))

            row = [
                f"{SEASON_END_YEAR - 1}-{str(SEASON_END_YEAR)[2:]}",  # "2025-26"
                hash(name) % 100000000,  # synthetic player ID
                name,
                name.split()[-1],  # nickname
                team_id,
                abbrev,
                team_name,
                game_id,
                nba_date,
                matchup,
                wl,
                round(minutes, 1),
                _safe_int(g.get("fg", 0)),
                _safe_int(g.get("fga", 0)),
                _safe_float(g.get("fg_pct", 0)),
                _safe_int(g.get("fg3", 0)),
                _safe_int(g.get("fg3a", 0)),
                _safe_float(g.get("fg3_pct", 0)),
                _safe_int(g.get("ft", 0)),
                _safe_int(g.get("fta", 0)),
                _safe_float(g.get("ft_pct", 0)),
                _safe_int(g.get("orb", 0)),
                _safe_int(g.get("drb", 0)),
                reb,
                ast,
                _safe_int(g.get("tov", 0)),
                _safe_int(g.get("stl", 0)),
                _safe_int(g.get("blk", 0)),
                0,  # BLKA
                _safe_int(g.get("pf", 0)),
                0,  # PFD
                pts,
                _safe_float(g.get("plus_minus", 0)),
                0.0,  # NBA_FANTASY_PTS
                1 if sum(1 for x in [pts, reb, ast] if x >= 10) >= 2 else 0,  # DD2
                1 if sum(1 for x in [pts, reb, ast] if x >= 10) >= 3 else 0,  # TD3
                0.0,  # WNBA_FANTASY_PTS
            ]
            all_rows.append(row)

    print(f"    → Total: {len(all_rows)} player-game rows")

    return {
        "resultSets": [{
            "name": "PlayerGameLogs",
            "headers": nba_headers,
            "rowSet": all_rows,
        }]
    }


# ══════════════════════════════════════════════════════════════════
# 3. SYNTHETIC LINEUPS (from player game logs)
# ══════════════════════════════════════════════════════════════════

def build_synthetic_lineups(player_gamelogs_data: dict, abbrev: str) -> dict:
    """
    Build synthetic lineup data from player game logs.
    Since b-ref doesn't provide lineup combos easily, we create
    'pseudo-lineups' from the 5 starters of each game.
    """
    import pandas as pd

    df = pd.DataFrame(
        player_gamelogs_data["resultSets"][0]["rowSet"],
        columns=player_gamelogs_data["resultSets"][0]["headers"],
    )
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)

    # For each game, pick the top 5 players by minutes as the "lineup"
    nba_headers = [
        "GROUP_ID", "GROUP_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
        "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
        "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE",
        "EFG_PCT", "TS_PCT", "PIE", "POSS",
    ]

    lineup_dict = {}
    for game_id, grp in df.groupby("GAME_ID"):
        top5 = grp.nlargest(5, "MIN")
        if len(top5) < 5:
            continue
        names = sorted(top5["PLAYER_NAME"].tolist())
        key = " - ".join(names)
        wl = top5["WL"].iloc[0]

        if key not in lineup_dict:
            lineup_dict[key] = {
                "names": key, "gp": 0, "w": 0, "l": 0,
                "pts_for": [], "pts_against": [], "plus_minus": [],
            }
        lineup_dict[key]["gp"] += 1
        if wl == "W":
            lineup_dict[key]["w"] += 1
        else:
            lineup_dict[key]["l"] += 1
        # Approximate team pts from top5 pts
        top5_pts = pd.to_numeric(top5["PTS"], errors="coerce").sum()
        lineup_dict[key]["pts_for"].append(top5_pts)
        pm = pd.to_numeric(top5["PLUS_MINUS"], errors="coerce").mean()
        lineup_dict[key]["plus_minus"].append(pm)

    team_id = TEAM_META[abbrev]["id"]
    rows = []
    for key, lu in lineup_dict.items():
        gp = lu["gp"]
        w = lu["w"]
        l_ = lu["l"]
        avg_pts = np.mean(lu["pts_for"]) if lu["pts_for"] else 0
        avg_pm = np.mean(lu["plus_minus"]) if lu["plus_minus"] else 0
        # Estimate ratings
        net_rtg = avg_pm * 2.5  # rough approximation
        off_rtg = 110 + net_rtg / 2
        def_rtg = 110 - net_rtg / 2

        row = [
            hash(key) % 100000000,
            lu["names"],
            team_id,
            abbrev,
            gp, w, l_,
            round(w / max(gp, 1), 3),
            25.0,  # MIN approximation
            0, 0, 0.0,  # FG stats
            0, 0, 0.0,  # FG3 stats
            0, 0, 0.0,  # FT stats
            0, 0, 0,    # rebounds
            0, 0, 0, 0, 0, 0, 0,  # other box score
            round(avg_pts, 1),
            round(avg_pm, 1),
            round(net_rtg, 1),
            round(off_rtg, 1),
            round(def_rtg, 1),
            100.0,  # PACE
            0.0, 0.0, 0.0,  # EFG, TS, PIE
            50.0,  # POSS
        ]
        rows.append(row)

    return {
        "resultSets": [{
            "name": "Lineups",
            "headers": nba_headers,
            "rowSet": rows,
        }]
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def fetch_all(abbrev: str):
    """Fetch all data for a team and save to cache."""
    abbrev = abbrev.upper()
    if abbrev not in TEAM_META:
        print(f"❌ Unknown team: {abbrev}")
        print(f"   Available: {', '.join(sorted(TEAM_META.keys()))}")
        return

    prefix = abbrev.lower()
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Fetching {TEAM_META[abbrev]['name']} data from Basketball Reference")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    # 1. Team game log
    print("[1/3] Team game log")
    gamelog = fetch_team_gamelog(abbrev)
    path1 = CACHE / f"{prefix}_gamelog.json"
    path1.write_text(json.dumps(gamelog))
    n_games = len(gamelog["resultSets"][0]["rowSet"])
    print(f"  ✅ Saved {path1.name} ({n_games} games)")
    print()

    time.sleep(DELAY)

    # 2. Roster + player game logs
    print("[2/3] Player game logs")
    roster = fetch_roster(abbrev)
    time.sleep(DELAY)

    player_gamelogs = build_player_gamelogs(abbrev, roster)
    path2 = CACHE / f"{prefix}_player_gamelogs.json"
    path2.write_text(json.dumps(player_gamelogs))
    n_rows = len(player_gamelogs["resultSets"][0]["rowSet"])
    print(f"  ✅ Saved {path2.name} ({n_rows} player-game rows)")
    print()

    # 3. Synthetic lineups
    print("[3/3] Building synthetic lineups")
    lineups = build_synthetic_lineups(player_gamelogs, abbrev)
    path3 = CACHE / f"{prefix}_lineups.json"
    path3.write_text(json.dumps(lineups))
    n_lu = len(lineups["resultSets"][0]["rowSet"])
    print(f"  ✅ Saved {path3.name} ({n_lu} lineup combos)")
    print()

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  ✅ All {abbrev} data cached successfully!")
    print(f"║  Files: {prefix}_gamelog.json, {prefix}_player_gamelogs.json, {prefix}_lineups.json")
    print(f"╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/gsis/fetch_bref.py <TEAM_ABBREV>")
        print("Example: python scripts/gsis/fetch_bref.py LAL")
        sys.exit(1)
    fetch_all(sys.argv[1])
