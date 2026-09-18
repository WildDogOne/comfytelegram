"""Shared types for the local tag database.

Danbooru and e621 each define their own tag category numbers, and are kept
as two separate sources (see `db.py`) rather than merged into one id space
— so there's no need for the "+7" collision-avoidance offset the upstream
archive (github.com/DraconicDragon/dbr-e621-lists-archive) uses for its own
merged danbooru+e621 list.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class TagSource(str, Enum):
    DANBOORU = "danbooru"
    E621 = "e621"


#: https://danbooru.donmai.us/wiki_pages/help:tags#category (2 is unused)
_DANBOORU_CATEGORIES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

#: https://e621.net/help/tags#category
_E621_CATEGORIES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "species",
    6: "invalid",
    7: "meta",
    8: "lore",
}

CATEGORY_LABELS: dict[TagSource, dict[int, str]] = {
    TagSource.DANBOORU: _DANBOORU_CATEGORIES,
    TagSource.E621: _E621_CATEGORIES,
}


def category_label(source: TagSource, category: int) -> str:
    """Human-readable category name, or the raw number if unrecognized
    (e.g. a category the upstream site added after this table was written)."""
    return CATEGORY_LABELS[source].get(category, str(category))


class TagRow(NamedTuple):
    """One parsed CSV row, before `TagDatabase.replace_source` splits it
    across the `tag`/`tag_alias` tables."""

    name: str
    category: int
    post_count: int
    aliases: tuple[str, ...]


class TagResult(NamedTuple):
    """One `TagDatabase.search`/`lookup_exact` hit. `matched_alias` is set
    only when the query matched an alias rather than the tag's own name."""

    source: TagSource
    name: str
    category: int
    post_count: int
    matched_alias: str | None = None
