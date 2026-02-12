"""Local SQLite cache for NBA API responses."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

_CACHE_DIR = Path.home() / ".nba_tracker" / "cache"
_DB_PATH = _CACHE_DIR / "nba_cache.db"
_DEFAULT_TTL = 60 * 60 * 6  # 6 hours


def _ensure_db() -> sqlite3.Connection:
    """Create the cache directory and table if they don't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            ts    REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get(key: str, ttl: int = _DEFAULT_TTL) -> Optional[Any]:
    """Return cached JSON value for *key* if it exists and is fresh, else None."""
    conn = _ensure_db()
    try:
        row = conn.execute(
            "SELECT value, ts FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value, ts = row
        if time.time() - ts > ttl:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None
        return json.loads(value)
    finally:
        conn.close()


def put(key: str, value: Any) -> None:
    """Store a JSON-serialisable *value* under *key*."""
    conn = _ensure_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def clear_cache() -> None:
    """Delete the entire cache database file."""
    if _DB_PATH.exists():
        os.remove(_DB_PATH)
