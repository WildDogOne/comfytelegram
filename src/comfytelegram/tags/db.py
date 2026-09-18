"""Local sqlite tag database for `/tags`/`/tagcheck` search — bulk
reference data imported wholesale by `scripts/update_tag_db.py`, kept in
its own file (`Settings.tags_db_path`) rather than `state.sqlite3`'s
per-chat mutable state, the same separation `model_profiles/*.json` already
draws against `storage.py`.

Deliberately as primitive as `storage.py`: stdlib sqlite3, no ORM, no
migrations framework. `replace_source` is the only write path, so there's
never a partially-migrated schema to worry about — a re-import just
replaces one source's rows wholesale in a single transaction.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from comfytelegram.tags.schema import TagResult, TagRow, TagSource

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tag (
    source TEXT NOT NULL,
    name TEXT NOT NULL,
    category INTEGER NOT NULL,
    post_count INTEGER NOT NULL,
    PRIMARY KEY (source, name)
);

CREATE TABLE IF NOT EXISTS tag_alias (
    source TEXT NOT NULL,
    alias TEXT NOT NULL,
    tag_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tag_alias ON tag_alias(source, alias);

-- One row per source, written by replace_source — lets a caller (the
-- startup auto-refresh check in tags/update.py) tell "never imported"
-- apart from "imported N days ago" without scanning the tag table itself.
CREATE TABLE IF NOT EXISTS tag_source_meta (
    source TEXT PRIMARY KEY,
    imported_at REAL NOT NULL
);
"""

#: Bounds how many rows a single LIKE clause can hand back to Python for
#: ranking — a one-character query would otherwise pull a large fraction of
#: the whole table. Ordered by post_count DESC first, so truncation drops
#: the least relevant rows, not arbitrary ones.
_CANDIDATE_FETCH_LIMIT = 500


def _normalize(text: str) -> str:
    """Booru tags use underscores, not spaces — accept either from a query
    typed by hand."""
    return text.strip().lower().replace(" ", "_")


def _escape_like(text: str) -> str:
    """Escape a normalized query for use inside a `LIKE ... ESCAPE '\\'`
    pattern, so a literal '_' in a tag name (extremely common, e.g.
    "hi_res") isn't treated as SQL's single-character wildcard."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class TagDatabase:
    """One connection per process — same `check_same_thread=False`
    reasoning as `storage.Storage` (python-telegram-bot runs all handlers
    on a single event loop thread; concurrency there is cooperative, not
    parallel)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)  # already commits internally

    def close(self) -> None:
        self._conn.close()

    def replace_source(self, source: TagSource, rows: Iterable[TagRow]) -> tuple[int, int]:
        """Atomically swap out everything stored for `source` with `rows`;
        the other source's rows are untouched. Returns the (tag_count,
        alias_count) actually written."""
        rows = list(rows)
        tag_values = [(source.value, r.name, r.category, r.post_count) for r in rows]
        alias_values = [(source.value, alias, r.name) for r in rows for alias in r.aliases if alias]
        with self._conn:
            self._conn.execute("DELETE FROM tag WHERE source = ?", (source.value,))
            self._conn.execute("DELETE FROM tag_alias WHERE source = ?", (source.value,))
            self._conn.executemany(
                "INSERT INTO tag (source, name, category, post_count) VALUES (?, ?, ?, ?)",
                tag_values,
            )
            self._conn.executemany(
                "INSERT INTO tag_alias (source, alias, tag_name) VALUES (?, ?, ?)",
                alias_values,
            )
            self._conn.execute(
                "INSERT INTO tag_source_meta (source, imported_at) VALUES (?, ?) "
                "ON CONFLICT(source) DO UPDATE SET imported_at = excluded.imported_at",
                (source.value, time.time()),
            )
        return len(tag_values), len(alias_values)

    def last_imported(self) -> dict[TagSource, float | None]:
        """Unix timestamp of the last successful `replace_source` per
        source, or None if that source has never been imported — what
        `tags/update.py`'s startup auto-refresh check uses to decide what's
        stale."""
        rows = dict(self._conn.execute("SELECT source, imported_at FROM tag_source_meta"))
        return {source: rows.get(source.value) for source in TagSource}

    def stats(self) -> dict[TagSource, int]:
        """Tag count per source — lets a caller tell "nothing imported yet"
        apart from "no matches for that query"."""
        return {
            source: self._conn.execute(
                "SELECT COUNT(*) FROM tag WHERE source = ?", (source.value,)
            ).fetchone()[0]
            for source in TagSource
        }

    def search(self, query: str, sources: Iterable[TagSource], limit: int = 15) -> list[TagResult]:
        """Tag name/alias search: prefix matches rank above substring
        matches, then by post_count descending. A tag matched directly by
        its own name always wins over the same tag surfaced through one of
        its aliases, even if the alias match technically ranks higher."""
        needle = _normalize(query)
        sources = list(sources)
        if not needle or not sources:
            return []
        placeholders = ",".join("?" for _ in sources)
        source_values = [s.value for s in sources]
        pattern = f"%{_escape_like(needle)}%"

        name_rows = self._conn.execute(
            f"SELECT source, name, category, post_count FROM tag "
            f"WHERE source IN ({placeholders}) AND name LIKE ? ESCAPE '\\' "
            f"ORDER BY post_count DESC LIMIT {_CANDIDATE_FETCH_LIMIT}",
            [*source_values, pattern],
        ).fetchall()
        alias_rows = self._conn.execute(
            f"SELECT t.source, t.name, t.category, t.post_count, a.alias FROM tag_alias a "
            f"JOIN tag t ON t.source = a.source AND t.name = a.tag_name "
            f"WHERE a.source IN ({placeholders}) AND a.alias LIKE ? ESCAPE '\\' "
            f"ORDER BY t.post_count DESC LIMIT {_CANDIDATE_FETCH_LIMIT}",
            [*source_values, pattern],
        ).fetchall()

        candidates: dict[tuple[str, str], TagResult] = {}
        for source, name, category, post_count in name_rows:
            candidates[(source, name)] = TagResult(TagSource(source), name, category, post_count)
        for source, name, category, post_count, alias in alias_rows:
            key = (source, name)
            if key not in candidates:
                candidates[key] = TagResult(TagSource(source), name, category, post_count, alias)

        def rank(result: TagResult) -> tuple[int, int]:
            text = result.matched_alias or result.name
            return (0 if text.startswith(needle) else 1, -result.post_count)

        return sorted(candidates.values(), key=rank)[:limit]

    def lookup_exact(self, token: str, sources: Iterable[TagSource]) -> TagResult | None:
        """Exact (case/space-insensitive) match against a tag's own name,
        falling back to its aliases — for `/tagcheck`, where a token either
        is a known tag or it isn't."""
        needle = _normalize(token)
        sources = list(sources)
        if not needle or not sources:
            return None
        placeholders = ",".join("?" for _ in sources)
        source_values = [s.value for s in sources]

        row = self._conn.execute(
            f"SELECT source, name, category, post_count FROM tag "
            f"WHERE source IN ({placeholders}) AND name = ? "
            f"ORDER BY post_count DESC LIMIT 1",
            [*source_values, needle],
        ).fetchone()
        if row is not None:
            source, name, category, post_count = row
            return TagResult(TagSource(source), name, category, post_count)

        row = self._conn.execute(
            f"SELECT t.source, t.name, t.category, t.post_count, a.alias FROM tag_alias a "
            f"JOIN tag t ON t.source = a.source AND t.name = a.tag_name "
            f"WHERE a.source IN ({placeholders}) AND a.alias = ? "
            f"ORDER BY t.post_count DESC LIMIT 1",
            [*source_values, needle],
        ).fetchone()
        if row is not None:
            source, name, category, post_count, alias = row
            return TagResult(TagSource(source), name, category, post_count, alias)
        return None
