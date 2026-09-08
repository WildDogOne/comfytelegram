"""SQLite-backed persistent state: per-chat model selection, per-chat/
per-checkpoint profile-default overrides, the post-processing result
registry (which image a "🔍 Upscale" button refers to), saved
character designs (reusable prompt snippets the user can activate per
chat instead of retyping a subject description every time), and the
last-resolved-generation record that backs the chat-wide
"🔁 Generate Again" button.

That last one used to live only in an in-memory dict (`state.py`, now
removed) with the reasoning "there's no point persisting raw image bytes
across a restart, the bot has no memory of the prompt_id that made them
anyway" — which missed the obvious fix: we don't need to persist the bytes
at all. Telegram already stores the file once it's been sent; the message
we get back carries a `file_id` that's valid indefinitely and can be
re-downloaded on demand via `bot.get_file()`. So what actually needs to
survive a restart is tiny — a `file_id` plus the generation params needed
to build the next stage's graph — and it fits the same sqlite file as
everything else here.

Deliberately "primitive": stdlib sqlite3, small tables, no migrations
framework, no ORM.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

#: How long a post-processing button stays valid before its row is pruned.
#: Generous on purpose — the whole point is surviving restarts and letting
#: people come back to an old result later, not a tight session window.
PENDING_RESULT_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_checkpoint (
    chat_id INTEGER PRIMARY KEY,
    checkpoint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_override (
    chat_id INTEGER NOT NULL,
    checkpoint TEXT NOT NULL,
    overrides_json TEXT NOT NULL,
    PRIMARY KEY (chat_id, checkpoint)
);

CREATE TABLE IF NOT EXISTS pending_result (
    result_id TEXT PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    file_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    base_params_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS character (
    chat_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    positive_prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    PRIMARY KEY (chat_id, name)
);

CREATE TABLE IF NOT EXISTS active_character (
    chat_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS last_generation (
    chat_id INTEGER PRIMARY KEY,
    params_json TEXT NOT NULL
);
"""


class Storage:
    """Not async — sqlite3 is fast local disk I/O and every call here is a
    single small query, so it's called directly from async handlers the same
    way you'd call any other quick synchronous helper. One connection per
    process; `check_same_thread=False` is safe because python-telegram-bot
    runs all handlers on a single event loop thread."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_checkpoint(self, chat_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT checkpoint FROM chat_checkpoint WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row[0] if row else None

    def set_checkpoint(self, chat_id: int, checkpoint: str) -> None:
        self._conn.execute(
            "INSERT INTO chat_checkpoint (chat_id, checkpoint) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET checkpoint = excluded.checkpoint",
            (chat_id, checkpoint),
        )
        self._conn.commit()

    def get_override(self, chat_id: int, checkpoint: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT overrides_json FROM profile_override WHERE chat_id = ? AND checkpoint = ?",
            (chat_id, checkpoint),
        ).fetchone()
        return json.loads(row[0]) if row else {}

    def set_override_fields(self, chat_id: int, checkpoint: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Merge `fields` (already-validated ProfileDefaults-shaped values) into
        the existing override for (chat_id, checkpoint) and persist it. Returns
        the merged result."""
        current = self.get_override(chat_id, checkpoint)
        current.update(fields)
        self._conn.execute(
            "INSERT INTO profile_override (chat_id, checkpoint, overrides_json) VALUES (?, ?, ?) "
            "ON CONFLICT(chat_id, checkpoint) DO UPDATE SET overrides_json = excluded.overrides_json",
            (chat_id, checkpoint, json.dumps(current)),
        )
        self._conn.commit()
        return current

    def clear_override(self, chat_id: int, checkpoint: str) -> None:
        """Remove every overridden field for (chat_id, checkpoint) — "reset all"."""
        self._conn.execute(
            "DELETE FROM profile_override WHERE chat_id = ? AND checkpoint = ?", (chat_id, checkpoint)
        )
        self._conn.commit()

    def clear_override_field(self, chat_id: int, checkpoint: str, field: str) -> dict[str, Any]:
        """Remove a single field from the override, keeping the rest. Returns
        what's left (deletes the row entirely once it's empty)."""
        current = self.get_override(chat_id, checkpoint)
        if field not in current:
            return current
        del current[field]
        if current:
            self._conn.execute(
                "INSERT INTO profile_override (chat_id, checkpoint, overrides_json) VALUES (?, ?, ?) "
                "ON CONFLICT(chat_id, checkpoint) DO UPDATE SET overrides_json = excluded.overrides_json",
                (chat_id, checkpoint, json.dumps(current)),
            )
        else:
            self._conn.execute(
                "DELETE FROM profile_override WHERE chat_id = ? AND checkpoint = ?", (chat_id, checkpoint)
            )
        self._conn.commit()
        return current

    def store_pending_result(
        self, result_id: str, chat_id: int, file_id: str, filename: str, base_params: dict[str, Any]
    ) -> None:
        """Record what a post-processing/regenerate button (result_id) refers
        to: the Telegram file_id to re-download the source image from, and
        the full resolved generation params (already a plain dict — see
        handlers.py's (de)serialization helpers) needed to build the next
        graph, whether that's an upscale/face-detail pass or a fresh
        "🔁 Regenerate" run."""
        self._prune_pending_results()
        self._conn.execute(
            "INSERT OR REPLACE INTO pending_result "
            "(result_id, chat_id, file_id, filename, base_params_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (result_id, chat_id, file_id, filename, json.dumps(base_params), time.time()),
        )
        self._conn.commit()

    def get_pending_result(self, result_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT chat_id, file_id, filename, base_params_json FROM pending_result WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        chat_id, file_id, filename, base_params_json = row
        return {
            "chat_id": chat_id,
            "file_id": file_id,
            "filename": filename,
            "base_params": json.loads(base_params_json),
        }

    def _prune_pending_results(self, ttl_seconds: float = PENDING_RESULT_TTL_SECONDS) -> None:
        cutoff = time.time() - ttl_seconds
        self._conn.execute("DELETE FROM pending_result WHERE created_at < ?", (cutoff,))
        self._conn.commit()

    def save_character(
        self, chat_id: int, name: str, positive_prompt: str, negative_prompt: str = ""
    ) -> None:
        """Create or overwrite a saved character design for this chat."""
        self._conn.execute(
            "INSERT INTO character (chat_id, name, positive_prompt, negative_prompt, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(chat_id, name) DO UPDATE SET "
            "positive_prompt = excluded.positive_prompt, negative_prompt = excluded.negative_prompt",
            (chat_id, name, positive_prompt, negative_prompt, time.time()),
        )
        self._conn.commit()

    def get_character(self, chat_id: int, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT positive_prompt, negative_prompt FROM character WHERE chat_id = ? AND name = ?",
            (chat_id, name),
        ).fetchone()
        if row is None:
            return None
        return {"name": name, "positive_prompt": row[0], "negative_prompt": row[1]}

    def list_characters(self, chat_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, positive_prompt, negative_prompt FROM character "
            "WHERE chat_id = ? ORDER BY name COLLATE NOCASE",
            (chat_id,),
        ).fetchall()
        return [
            {"name": name, "positive_prompt": positive, "negative_prompt": negative}
            for name, positive, negative in rows
        ]

    def delete_character(self, chat_id: int, name: str) -> None:
        self._conn.execute("DELETE FROM character WHERE chat_id = ? AND name = ?", (chat_id, name))
        active = self.get_active_character_name(chat_id)
        if active == name:
            self.clear_active_character(chat_id)
        self._conn.commit()

    def get_active_character_name(self, chat_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT name FROM active_character WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row[0] if row else None

    def set_active_character(self, chat_id: int, name: str) -> None:
        self._conn.execute(
            "INSERT INTO active_character (chat_id, name) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET name = excluded.name",
            (chat_id, name),
        )
        self._conn.commit()

    def clear_active_character(self, chat_id: int) -> None:
        self._conn.execute("DELETE FROM active_character WHERE chat_id = ?", (chat_id,))
        self._conn.commit()

    def set_last_generation(self, chat_id: int, params: dict[str, Any]) -> None:
        """Remember the resolved generation params (same shape as
        `pending_result.base_params_json` — see handlers.py's serialize
        helper) for the "🔁 Generate Again" button: a chat-wide "run that
        prompt once more" that doesn't require picking any specific image."""
        self._conn.execute(
            "INSERT INTO last_generation (chat_id, params_json) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET params_json = excluded.params_json",
            (chat_id, json.dumps(params)),
        )
        self._conn.commit()

    def get_last_generation(self, chat_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT params_json FROM last_generation WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()
