import time
from pathlib import Path

import pytest

from comfytelegram.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def test_derived_prompt_roundtrip(storage: Storage):
    assert storage.get_derived_prompt("abc123") is None

    storage.store_derived_prompt("abc123", 42, "ckpt.safetensors", "fox, forest, solo")

    result = storage.get_derived_prompt("abc123")
    assert result is not None
    assert result["chat_id"] == 42
    assert result["checkpoint"] == "ckpt.safetensors"
    assert result["prompt"] == "fox, forest, solo"


def test_derived_prompt_scoped_per_message_not_chat(storage: Storage):
    """The "🏷️ Analyze" button writes two rows per tap (WD14 tags and a
    Qwen-VL caption) — each must stay independently addressable by its own
    "🎨 Generate" button, not collapse into one chat-wide record."""
    storage.store_derived_prompt("tags", 42, "ckpt.safetensors", "fox, forest, solo")
    storage.store_derived_prompt("caption", 42, "ckpt.safetensors", "a fox standing in a forest")

    assert storage.get_derived_prompt("tags")["prompt"] == "fox, forest, solo"
    assert storage.get_derived_prompt("caption")["prompt"] == "a fox standing in a forest"


def test_derived_prompt_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.store_derived_prompt("abc123", 42, "ckpt.safetensors", "fox, forest, solo")
    s1.close()

    s2 = Storage(db_path)
    result = s2.get_derived_prompt("abc123")
    assert result is not None
    assert result["prompt"] == "fox, forest, solo"
    s2.close()


def test_derived_prompt_pruned_after_ttl(storage: Storage):
    storage.store_derived_prompt("old", 1, "ckpt.safetensors", "fox")
    storage._conn.execute(
        "UPDATE derived_prompt SET created_at = ? WHERE prompt_id = ?",
        (time.time() - 999999999, "old"),
    )
    storage._conn.commit()
    storage._last_prune.clear()

    storage.store_derived_prompt("new", 1, "ckpt.safetensors", "wolf")

    assert storage.get_derived_prompt("old") is None
    assert storage.get_derived_prompt("new") is not None
