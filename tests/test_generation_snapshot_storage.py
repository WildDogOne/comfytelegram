import time
from pathlib import Path

import pytest

from comfytelegram.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def test_generation_snapshot_roundtrip(storage: Storage):
    assert storage.get_generation_snapshot("abc123") is None

    params = {
        "checkpoint": "ckpt.safetensors",
        "positive_prompt": "a fox",
        "negative_prompt": "low quality",
        "steps": 25,
        "cfg": 6.5,
    }
    storage.store_generation_snapshot("abc123", 42, params)

    result = storage.get_generation_snapshot("abc123")
    assert result is not None
    assert result["chat_id"] == 42
    assert result["params"] == params


def test_generation_snapshot_is_scoped_per_message_not_chat(storage: Storage):
    """Two snapshots for the same chat (e.g. a generation followed by a
    "Generate Again" tap) must stay independently addressable — the bug this
    replaced collapsed them into one chat-wide record, so an older message's
    button silently repeated whatever the chat generated most recently."""
    storage.store_generation_snapshot("first", 42, {"checkpoint": "a.safetensors"})
    storage.store_generation_snapshot("second", 42, {"checkpoint": "b.safetensors"})

    assert storage.get_generation_snapshot("first")["params"] == {"checkpoint": "a.safetensors"}
    assert storage.get_generation_snapshot("second")["params"] == {"checkpoint": "b.safetensors"}


def test_generation_snapshot_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.store_generation_snapshot("abc123", 42, {"checkpoint": "ckpt.safetensors"})
    s1.close()

    s2 = Storage(db_path)
    result = s2.get_generation_snapshot("abc123")
    assert result is not None
    assert result["params"] == {"checkpoint": "ckpt.safetensors"}
    s2.close()


def test_generation_snapshot_pruned_after_ttl(storage: Storage):
    storage.store_generation_snapshot("old", 1, {})
    # simulate an old row by writing directly with a stale created_at
    storage._conn.execute(
        "UPDATE generation_snapshot SET created_at = ? WHERE snapshot_id = ?",
        (time.time() - 999999999, "old"),
    )
    storage._conn.commit()
    # force the next store to actually run a prune sweep, bypassing
    # PRUNE_INTERVAL_SECONDS' cadence gate (see storage.py's `_prune`)
    storage._last_prune.clear()

    # storing a new snapshot triggers pruning of expired rows
    storage.store_generation_snapshot("new", 1, {})

    assert storage.get_generation_snapshot("old") is None
    assert storage.get_generation_snapshot("new") is not None
