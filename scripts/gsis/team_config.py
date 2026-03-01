"""
Shared team configuration for GSIS modules.
Provides a single source of truth for which team is being analyzed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"

# Default team
_current_team = "GSW"


def set_team(abbrev: str):
    """Set the current team for all GSIS modules."""
    global _current_team
    _current_team = abbrev.upper()


def get_team() -> str:
    """Get the current team abbreviation."""
    return _current_team


def cache_prefix() -> str:
    """Return the cache file prefix for the current team.
    GSW uses no prefix (legacy); other teams use lowercase abbreviation."""
    if _current_team == "GSW":
        return ""
    return _current_team.lower() + "_"


def load_cache(name: str) -> dict:
    """Load a cached JSON file for the current team.
    For team-specific files (gamelog, player_gamelogs, lineups),
    uses the prefix. For league-wide files (standings, league_base, league_adv),
    uses no prefix.
    """
    # League-wide files that have no team prefix
    league_wide = {"standings", "league_base", "league_adv", "league_clutch"}

    if name in league_wide:
        path = CACHE / f"{name}.json"
    else:
        prefix = cache_prefix()
        path = CACHE / f"{prefix}{name}.json"

    if not path.exists():
        # Fallback: try without prefix
        fallback = CACHE / f"{name}.json"
        if fallback.exists():
            return json.loads(fallback.read_text())
        raise FileNotFoundError(f"Cache file not found: {path}")

    return json.loads(path.read_text())
