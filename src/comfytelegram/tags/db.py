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
from difflib import SequenceMatcher
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

#: Fuzzy fallback (see `_fuzzy_matches`) only scores tags whose length is
#: within this many characters of the query — a typo rarely adds/drops
#: more than a couple, and this keeps the fallback's own candidate query
#: from having to consider the whole table.
_FUZZY_LENGTH_SLOP = 3

#: Same purpose as `_CANDIDATE_FETCH_LIMIT`, for the fuzzy fallback's own
#: (length-filtered, not substring-filtered) candidate query.
_FUZZY_CANDIDATE_LIMIT = 5000

#: Minimum `difflib.SequenceMatcher` ratio for a fuzzy fallback candidate
#: to be considered a match at all, vs. just two unrelated tags that happen
#: to be a similar length — e.g. a "dimple" search fuzzy-matching "temple"
#: or "male" at 0.6. Real single-edit typos of a tag of any reasonable
#: length still clear this comfortably (0.8+ in practice).
_FUZZY_MIN_RATIO = 0.72


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

    def search(
        self,
        query: str,
        sources: Iterable[TagSource],
        limit: int = 15,
        *,
        by_frequency: bool = False,
    ) -> list[TagResult]:
        """Tag name/alias search. Default ranking puts prefix matches above
        substring matches, then sorts by post_count descending within each
        group — good for a "did you mean" hint, where the closest textual
        match matters more than raw popularity. `by_frequency=True` instead
        sorts the whole candidate pool by post_count alone (`/tags`'s own
        listing wants the most-used matching tags first, not a rare prefix
        match outranking a far more common substring one). Either way, a
        tag matched directly by its own name always wins over the same tag
        surfaced through one of its aliases, even if the alias match
        technically ranks higher.

        Only falls back to `_fuzzy_matches`' edit-distance-tolerant
        candidates when substring matching finds *nothing at all* — e.g. a
        typo like "1gril" shares no substring with "1girl". A query with
        even one real hit never reaches for a fuzzy guess just to pad the
        result list out to `limit`: real, if few, beats padding a "dimple"
        search out to 15 rows with "temple"/"nipples"/"male" once its 5
        genuine matches run out. Fuzzy results are ranked by closeness of
        match first and post_count second — a fuzzy guess is only useful
        if it's actually the tag meant, no matter how popular a
        less-similar guess is."""
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

        if by_frequency:
            ranked = sorted(candidates.values(), key=lambda r: -r.post_count)
        else:

            def rank(result: TagResult) -> tuple[int, int]:
                text = result.matched_alias or result.name
                return (0 if text.startswith(needle) else 1, -result.post_count)

            ranked = sorted(candidates.values(), key=rank)

        if ranked:
            return ranked[:limit]

        fuzzy = self._fuzzy_matches(needle, sources, exclude=set(candidates))
        fuzzy.sort(key=lambda item: (-item[0], -item[1].post_count))
        return [result for _, result in fuzzy][:limit]

    def _fuzzy_matches(
        self, needle: str, sources: list[TagSource], exclude: set[tuple[str, str]]
    ) -> list[tuple[float, TagResult]]:
        """`search()`'s edit-distance-tolerant fallback, for typos that
        share no substring with the tag they meant. Scored with stdlib
        `difflib.SequenceMatcher` against a candidate pool filtered by
        name length (within `_FUZZY_LENGTH_SLOP` of `needle`) rather than
        substring containment — cheap enough to run without a dedicated
        fuzzy index, and only reached when `search()`'s real matches don't
        already fill `limit`. Alias names aren't scored — a fuzzy match
        against a tag's own name is what "did you mean" callers actually
        want, and skipping aliases here keeps the fallback to one query."""
        source_values = [s.value for s in sources]
        placeholders = ",".join("?" for _ in sources)
        min_len = max(1, len(needle) - _FUZZY_LENGTH_SLOP)
        max_len = len(needle) + _FUZZY_LENGTH_SLOP

        rows = self._conn.execute(
            f"SELECT source, name, category, post_count FROM tag "
            f"WHERE source IN ({placeholders}) AND LENGTH(name) BETWEEN ? AND ? "
            f"ORDER BY post_count DESC LIMIT {_FUZZY_CANDIDATE_LIMIT}",
            [*source_values, min_len, max_len],
        ).fetchall()

        matches = []
        for source, name, category, post_count in rows:
            if (source, name) in exclude:
                continue
            ratio = SequenceMatcher(None, needle, name).ratio()
            if ratio >= _FUZZY_MIN_RATIO:
                matches.append((ratio, TagResult(TagSource(source), name, category, post_count)))
        return matches

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
