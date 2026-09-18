"""Fetches/imports tag data from the DraconicDragon tag-list archive
(https://github.com/DraconicDragon/dbr-e621-lists-archive). Two callers
share this: `scripts/update_tag_db.py` (manual/offline CLI) and
`refresh_if_stale` below, which `main.py` fires as a background task at
startup so a fresh checkout — or one whose data has aged past
`Settings.tag_db_max_age_days` — self-populates without anyone having to
remember to run the script by hand.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.importer import import_csv
from comfytelegram.tags.schema import TagSource

logger = logging.getLogger(__name__)

ARCHIVE_REPO = "DraconicDragon/dbr-e621-lists-archive"
ARCHIVE_BRANCH = "main"
_REQUEST_HEADERS = {"User-Agent": "comfytelegram-update-tag-db"}

#: e.g. "e621_2026-04-01_pt20-ia-ed.csv" / "danbooru_2026-04-01_pt20-ia-dd.csv".
#: Excludes the "_cleaner" variant (a whitespace/format cleanup pass over
#: the same data, not a different tag list) via the caller checking for
#: "_cleaner" in the matched filename.
_FILENAME_RE = re.compile(r"^(danbooru|e621)_(\d{4}-\d{2}-\d{2})_.+\.csv$")

RefreshSource = Callable[[TagDatabase, TagSource], tuple[int, int]]


def latest_csv_filename(source: TagSource) -> str:
    """Newest dated CSV in the archive's `tag-lists/<source>/` directory,
    via the public (unauthenticated, rate-limited) GitHub Contents API —
    fine for something run occasionally, not in a tight loop."""
    url = f"https://api.github.com/repos/{ARCHIVE_REPO}/contents/tag-lists/{source.value}"
    request = urllib.request.Request(url, headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.load(response)

    candidates = [
        (match.group(2), entry["name"])
        for entry in entries
        if (match := _FILENAME_RE.match(entry["name"])) and "_cleaner" not in entry["name"]
    ]
    if not candidates:
        raise RuntimeError(f"No dated CSV found for {source.value} at {url}")
    candidates.sort()
    return candidates[-1][1]


def download_csv(source: TagSource, filename: str, dest: Path) -> None:
    url = (
        f"https://raw.githubusercontent.com/{ARCHIVE_REPO}/{ARCHIVE_BRANCH}/"
        f"tag-lists/{source.value}/{filename}"
    )
    request = urllib.request.Request(url, headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())


def refresh_source(db: TagDatabase, source: TagSource) -> tuple[int, int]:
    """Blocking: download the newest CSV for `source` and import it.
    Synchronous end to end (stdlib `urllib`, no event loop involved) so it
    can run either directly (the CLI script) or off-thread via
    `asyncio.to_thread` (the startup auto-refresh below) without pulling in
    an async HTTP client just for an occasional multi-MB download."""
    filename = latest_csv_filename(source)
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / filename
        download_csv(source, filename, csv_path)
        return import_csv(db, source, csv_path)


async def refresh_if_stale(
    db: TagDatabase,
    max_age_days: float,
    *,
    refresh: RefreshSource = refresh_source,
) -> None:
    """Startup check: refresh any source that's never been imported or is
    older than `max_age_days`, one at a time. Each refresh runs via
    `asyncio.to_thread` so a slow or failed GitHub fetch can't block
    Telegram polling on the event loop thread it shares with every other
    handler. A source that fails (offline, GitHub unreachable/rate-limited)
    just logs a warning and keeps whatever was already imported — this must
    never raise, since it's fired as an unawaited background task at
    startup and an uncaught exception there would just vanish into
    asyncio's default exception logging instead of being handled here.
    `refresh` is swappable so tests can exercise the staleness logic
    without hitting the network."""
    last_imported = db.last_imported()
    cutoff = time.time() - max_age_days * 86400
    for source in TagSource:
        imported_at = last_imported.get(source)
        if imported_at is not None and imported_at >= cutoff:
            continue
        age = (
            "never imported"
            if imported_at is None
            else f"{(time.time() - imported_at) / 86400:.0f}d old"
        )
        logger.info("Tag database: refreshing %s (%s)", source.value, age)
        try:
            tag_count, alias_count = await asyncio.to_thread(refresh, db, source)
        except Exception:
            logger.warning("Tag database: failed to refresh %s", source.value, exc_info=True)
        else:
            logger.info(
                "Tag database: %s refreshed (%d tags, %d aliases)",
                source.value,
                tag_count,
                alias_count,
            )
