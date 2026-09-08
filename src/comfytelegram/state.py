"""Minimal in-memory state for the running bot process.

Everything here resets on restart — acceptable for v1 per TODO.md section 7
("lightweight solution first, avoid over-engineering"). Swap for sqlite if
surviving restarts turns out to matter.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from comfytelegram.workflows import PostProcessBaseParams


@dataclass
class PendingResult:
    """A generated image kept around just long enough for its post-processing
    inline-keyboard buttons to still resolve to something."""

    image_bytes: bytes
    filename: str
    base_params: PostProcessBaseParams
    created_at: float = field(default_factory=time.monotonic)


class BotState:
    def __init__(self, *, result_ttl_seconds: float = 3600) -> None:
        self._chat_checkpoint: dict[int, str] = {}
        self._results: dict[str, PendingResult] = {}
        self._result_ttl = result_ttl_seconds

    def get_checkpoint(self, chat_id: int) -> str | None:
        return self._chat_checkpoint.get(chat_id)

    def set_checkpoint(self, chat_id: int, checkpoint: str) -> None:
        self._chat_checkpoint[chat_id] = checkpoint

    def store_result(self, result: PendingResult) -> str:
        self._gc()
        result_id = uuid.uuid4().hex[:12]
        self._results[result_id] = result
        return result_id

    def get_result(self, result_id: str) -> PendingResult | None:
        return self._results.get(result_id)

    def _gc(self) -> None:
        cutoff = time.monotonic() - self._result_ttl
        expired = [k for k, v in self._results.items() if v.created_at < cutoff]
        for k in expired:
            del self._results[k]
