import sqlite3
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


def test_derived_prompt_stores_negative_prompt(storage: Storage):
    storage.store_derived_prompt(
        "abc123", 42, "ckpt.safetensors", "fox, forest, solo", "blurry, watermark"
    )

    result = storage.get_derived_prompt("abc123")
    assert result is not None
    assert result["negative_prompt"] == "blurry, watermark"


def test_derived_prompt_defaults_to_empty_negative_prompt(storage: Storage):
    storage.store_derived_prompt("abc123", 42, "ckpt.safetensors", "fox, forest, solo")

    result = storage.get_derived_prompt("abc123")
    assert result is not None
    assert result["negative_prompt"] == ""


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


def test_storage_adds_negative_prompt_column_to_a_pre_existing_table(tmp_path: Path):
    """A real deployed state.sqlite3 predating this column would otherwise
    hit "table derived_prompt has no column named negative_prompt" on the
    very first store — see `Storage._add_column_if_missing`."""
    db_path = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE derived_prompt (
            prompt_id TEXT PRIMARY KEY,
            chat_id INTEGER NOT NULL,
            checkpoint TEXT NOT NULL,
            prompt TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO derived_prompt (prompt_id, chat_id, checkpoint, prompt, created_at) "
        "VALUES ('old', 1, 'ckpt.safetensors', 'fox', ?)",
        (time.time(),),
    )
    conn.commit()
    conn.close()

    storage = Storage(db_path)
    try:
        assert storage.get_derived_prompt("old")["negative_prompt"] == ""
        storage.store_derived_prompt("new", 1, "ckpt.safetensors", "wolf", "blurry")
        assert storage.get_derived_prompt("new")["negative_prompt"] == "blurry"
    finally:
        storage.close()


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
