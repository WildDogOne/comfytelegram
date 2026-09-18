"""Parses the DraconicDragon `danbooru-e621-tag-list-processor` archive's
CSV format (github.com/DraconicDragon/dbr-e621-lists-archive) — no header
row, columns `tag,category,post_count,"alias1,alias2,..."`, e.g.:

    anthro,0,4156082,"anthromorph,anthropomorph,anthropomorphic,antro"

— into `TagRow`s, and imports them into a `TagDatabase`.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.schema import TagRow, TagSource


def parse_csv(path: Path) -> Iterator[TagRow]:
    """Yield one `TagRow` per non-empty CSV row. `csv.reader` already
    handles the quoted, comma-separated aliases field; a row with no
    aliases at all just has fewer than 4 columns."""
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            name, category, post_count, *rest = row
            aliases_field = rest[0] if rest else ""
            aliases = tuple(
                alias for alias in (a.strip() for a in aliases_field.split(",")) if alias
            )
            yield TagRow(
                name=name.strip(),
                category=int(category),
                post_count=int(post_count),
                aliases=aliases,
            )


def import_csv(db: TagDatabase, source: TagSource, path: Path) -> tuple[int, int]:
    """Parse `path` and swap it in as `source`'s entire tag set. Returns
    the (tag_count, alias_count) `TagDatabase.replace_source` reports."""
    return db.replace_source(source, parse_csv(path))
