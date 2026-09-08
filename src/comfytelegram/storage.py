"""SQLite-backed persistent state: per-chat model selection and per-chat,
per-checkpoint profile-default overrides.

This replaces what used to live only in the in-memory `BotState` (see
`state.py`) — that dict was lost on every restart, which is exactly the gap
reported after the first live test. `BotState` still exists for genuinely
ephemeral things (the post-processing result registry — there's no point
persisting raw image bytes across a restart when the bot has no memory of
the ComfyUI prompt_id that made them anyway).

Deliberately "primitive" per the request that prompted this: stdlib sqlite3,
two small tables, no migrations framework, no ORM.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

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

    def close(self) -> None:
        self._conn.close()
