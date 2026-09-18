import time
from pathlib import Path

import pytest

from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.schema import TagRow, TagSource
from comfytelegram.tags.update import refresh_if_stale


@pytest.fixture
def db(tmp_path: Path) -> TagDatabase:
    database = TagDatabase(tmp_path / "tags.sqlite3")
    yield database
    database.close()


def _fake_refresh():
    """A `RefreshSource` stub that writes one dummy row for whichever
    source it's called with (so `last_imported` advances the same way the
    real `refresh_source` would, via `replace_source`) and records every
    call it received, without touching the network."""
    calls: list[TagSource] = []

    def refresh(database: TagDatabase, source: TagSource) -> tuple[int, int]:
        calls.append(source)
        return database.replace_source(source, [TagRow("dummy", 0, 1, ())])

    refresh.calls = calls
    return refresh


@pytest.mark.asyncio
async def test_refresh_if_stale_refreshes_never_imported_sources(db: TagDatabase):
    refresh = _fake_refresh()
    await refresh_if_stale(db, max_age_days=30, refresh=refresh)

    assert set(refresh.calls) == {TagSource.DANBOORU, TagSource.E621}
    imported = db.last_imported()
    assert imported[TagSource.DANBOORU] is not None
    assert imported[TagSource.E621] is not None


@pytest.mark.asyncio
async def test_refresh_if_stale_skips_recently_imported_sources(db: TagDatabase):
    db.replace_source(TagSource.DANBOORU, [TagRow("1girl", 0, 100, ())])
    db.replace_source(TagSource.E621, [TagRow("fox", 5, 100, ())])

    refresh = _fake_refresh()
    await refresh_if_stale(db, max_age_days=30, refresh=refresh)

    assert refresh.calls == []


@pytest.mark.asyncio
async def test_refresh_if_stale_refreshes_a_source_older_than_max_age(db: TagDatabase):
    db.replace_source(TagSource.DANBOORU, [TagRow("1girl", 0, 100, ())])
    stale_time = time.time() - 40 * 86400
    with db._conn:
        db._conn.execute(
            "UPDATE tag_source_meta SET imported_at = ? WHERE source = ?",
            (stale_time, TagSource.DANBOORU.value),
        )

    refresh = _fake_refresh()
    await refresh_if_stale(db, max_age_days=30, refresh=refresh)

    # e621 was never imported at all, so both are expected to refresh.
    assert set(refresh.calls) == {TagSource.DANBOORU, TagSource.E621}


@pytest.mark.asyncio
async def test_refresh_if_stale_continues_past_a_failing_source(db: TagDatabase):
    def refresh(database: TagDatabase, source: TagSource) -> tuple[int, int]:
        if source is TagSource.DANBOORU:
            raise RuntimeError("network down")
        return database.replace_source(source, [TagRow("fox", 5, 100, ())])

    await refresh_if_stale(db, max_age_days=30, refresh=refresh)

    imported = db.last_imported()
    assert imported[TagSource.DANBOORU] is None
    assert imported[TagSource.E621] is not None
