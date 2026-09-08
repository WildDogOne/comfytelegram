"""In-memory, per-process-lifetime state: the post-processing result registry.

Durable state (per-chat model selection, profile overrides) lives in
`storage.py` (sqlite) instead — it needs to survive a bot restart, this
doesn't. There would be no point persisting raw generated-image bytes
across a restart anyway: the bot has no memory of the ComfyUI prompt_id
that made them, so a stale "Upscale" button after a restart can only ever
be told to generate a fresh image instead.
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
        self._results: dict[str, PendingResult] = {}
        self._result_ttl = result_ttl_seconds

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
