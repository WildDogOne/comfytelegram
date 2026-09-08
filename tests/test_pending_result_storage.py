import time
from pathlib import Path

import pytest

from comfytelegram.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def test_pending_result_roundtrip(storage: Storage):
    assert storage.get_pending_result("abc123") is None

    base_params = {
        "checkpoint": "ckpt.safetensors",
        "positive_prompt": "a fox",
        "negative_prompt": "low quality",
        "clip_skip": -2,
        "loras": [{"name": "a.safetensors", "strength_model": 0.8, "strength_clip": 1.0}],
    }
    storage.store_pending_result("abc123", 42, "TG_FILE_ID", "out.png", base_params)

    result = storage.get_pending_result("abc123")
    assert result is not None
    assert result["chat_id"] == 42
    assert result["file_id"] == "TG_FILE_ID"
    assert result["filename"] == "out.png"
    assert result["base_params"] == base_params


def test_pending_result_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.store_pending_result("abc123", 1, "FILE_ID", "out.png", {"checkpoint": "x"})
    s1.close()

    # this is the exact scenario reported: a bot restart must not lose an
    # "Upscale" button's ability to find its source image
    s2 = Storage(db_path)
    result = s2.get_pending_result("abc123")
    assert result is not None
    assert result["file_id"] == "FILE_ID"
    s2.close()


def test_pending_result_pruned_after_ttl(storage: Storage):
    storage.store_pending_result("old", 1, "FILE_ID", "out.png", {})
    # simulate an old row by writing directly with a stale created_at
    storage._conn.execute(
        "UPDATE pending_result SET created_at = ? WHERE result_id = ?",
        (time.time() - 999999999, "old"),
    )
    storage._conn.commit()
    # force the next store to actually run a prune sweep, bypassing
    # PRUNE_INTERVAL_SECONDS' cadence gate (see storage.py's `_prune`)
    storage._last_prune.clear()

    # storing a new result triggers pruning of expired rows
    storage.store_pending_result("new", 1, "FILE_ID2", "out2.png", {})

    assert storage.get_pending_result("old") is None
    assert storage.get_pending_result("new") is not None


def test_pending_result_prune_is_rate_limited(storage: Storage):
    """A prune sweep shouldn't re-run on every single store — see
    PRUNE_INTERVAL_SECONDS. A stale row should survive a store that
    immediately follows a just-ran sweep."""
    storage.store_pending_result("old", 1, "FILE_ID", "out.png", {})  # first sweep runs (cache was empty)
    storage._conn.execute(
        "UPDATE pending_result SET created_at = ? WHERE result_id = ?",
        (time.time() - 999999999, "old"),
    )
    storage._conn.commit()

    # this store happens well within PRUNE_INTERVAL_SECONDS of the first —
    # no sweep should run, so "old" survives for now
    storage.store_pending_result("new", 1, "FILE_ID2", "out2.png", {})

    assert storage.get_pending_result("old") is not None
    assert storage.get_pending_result("new") is not None
