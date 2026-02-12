# NBATracker

A Python command-line tool that fetches, processes, and analyzes NBA player statistics for each of the 30 NBA teams.

## Features

- **Team Roster** — List all players on any NBA team for a given season.
- **Per-Team Player Stats** — Season averages (PTS, REB, AST, STL, BLK, FG%, 3P%, FT%, MIN, TOV) sortable by any column with minimum-games filtering.
- **Team Leaders** — Identify the statistical leader in every major category.
- **Team Summary** — Aggregate team-level statistics.
- **Player Comparison** — Compare 2+ players side-by-side with highlighted leaders.
- **Export** — Save stats to CSV or JSON for further analysis.
- **Caching** — Automatic local SQLite cache to avoid redundant API calls.

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd NBATracker

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Show help
python -m nba_tracker.cli --help

# List the Lakers roster for 2024-25
python -m nba_tracker.cli team roster LAL --season 2024-25

# Per-player stats sorted by points, minimum 10 games
python -m nba_tracker.cli team stats BOS --sort PTS --min-games 10

# Statistical leaders
python -m nba_tracker.cli team leaders GSW

# Team aggregate summary
python -m nba_tracker.cli team summary MIL

# Compare two players
python -m nba_tracker.cli compare "LeBron James" "Stephen Curry"

# Export to CSV
python -m nba_tracker.cli export LAL --format csv --output lakers.csv

# Export to JSON
python -m nba_tracker.cli export LAL --format json

# Clear cache
python -m nba_tracker.cli cache clear
```

## Team Abbreviations

Use standard 3-letter NBA abbreviations: `LAL`, `BOS`, `GSW`, `MIL`, `PHX`, `DEN`, `MIA`, `NYK`, `DAL`, `LAC`, etc.

## Running Tests

```bash
pytest tests/ -v
```

## Technology Stack

| Component | Choice |
|---|---|
| Language | Python 3.10+ |
| Data source | `nba_api` (free, no API key required) |
| CLI | `click` |
| Data processing | `pandas` |
| Terminal output | `rich` |
| Caching | SQLite (`sqlite3` stdlib) |
| Models | `pydantic` |
| Testing | `pytest` |

## Project Structure

```
NBATracker/
  nba_tracker/
    __init__.py       # Package init + version
    cli.py            # CLI entry point (click)
    fetcher.py        # Data fetching + caching from nba_api
    analyzer.py       # Statistical analysis logic
    models.py         # Pydantic data models
    exporter.py       # Output formatting (table, CSV, JSON)
    cache.py          # Local SQLite cache
  tests/
    test_fetcher.py
    test_analyzer.py
    test_exporter.py
  requirements.txt
  README.md
```
