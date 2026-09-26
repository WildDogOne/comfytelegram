"""Telegram-independent CivitAI lookup for a locally-installed LoRA file.

Backs `/lora`'s "ℹ️ Info" button (see `lora_menu.py`) — the two things a
`model_profiles/*.json` entry's curated `loras` list can't tell a user by
itself: what trigger words actually activate a given LoRA, and what base
model it was trained against. Confirmed against ComfyUI-Custom-Scripts'
own source (`web/js/common/modelInfoDialog.js`, `py/model_info.py`) that
this is exactly the two-step lookup its "Info" dialog does: hash the model
file (SHA256) and query CivitAI's public hash-lookup API with it. That
project splits the work across a custom ComfyUI-server route (for the
hash) and a client-side browser fetch (for the CivitAI call); this bot has
no custom ComfyUI node code of its own to hang a route on, so both steps
happen here instead, server-side, driven by `Settings.comfyui_loras_dir` —
a filesystem path to ComfyUI's models/loras directory this process can
read directly (bind-mounted read-only in Docker). Stock ComfyUI's own
`/object_info` gives `LoraLoader`'s dropdown as bare filenames with no
hash or metadata at all, confirmed against ComfyUI core — there is no way
to do this lookup without filesystem access to the actual model file.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


class CivitaiLookupError(RuntimeError):
    """Raised when a CivitAI request itself fails — network error, timeout,
    or an unexpected non-404 HTTP status. Deliberately distinct from
    `fetch_civitai_info` returning `None`, which means CivitAI was actually
    reached and genuinely has no model version for that hash. A caller that
    caches results (`lora_menu.py`'s "ℹ️ Info" button, `lora_discovery.py`'s
    boot scan) must not treat the two the same way — caching a transient
    network hiccup as "not on CivitAI" would stick forever, since nothing
    re-checks a cached miss on its own."""


#: CivitAI's public model-version-by-hash endpoint — no API key required
#: (confirmed: ComfyUI-Custom-Scripts' own client-side call sends none
#: either). Keyed on a SHA256 hex digest of the raw file bytes.
CIVITAI_HASH_LOOKUP_URL = "https://civitai.com/api/v1/model-versions/by-hash/{sha256}"

_CIVITAI_TIMEOUT = aiohttp.ClientTimeout(total=15)

#: Streamed in fixed-size chunks rather than `Path.read_bytes()` — a LoRA
#: file commonly runs several hundred MB to a couple GB, and loading one
#: wholesale into memory just to hash it would be wasteful, and a real
#: spike if a few "ℹ️ Info" taps land concurrently.
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass
class CivitaiLoraInfo:
    """The fields `lora_menu.py`'s "ℹ️ Info" button actually shows."""

    model_name: str
    base_model: str
    trained_words: list[str]
    civitai_url: str


def resolve_lora_path(loras_dir: Path, lora_name: str) -> Path | None:
    """Resolve `lora_name` (as `LoraLoader`'s enum / a model profile's
    `loras` list names it — a path relative to ComfyUI's loras folder,
    possibly with subdirectories) against `loras_dir`. Returns None if the
    resolved path would land outside `loras_dir` (a profile's JSON is
    trusted config already, but this still refuses to let a `..`-laced
    name escape the mounted directory) or the resulting file doesn't
    actually exist (mount not populated yet, filename drifted from what's
    actually on disk, etc.)."""
    resolved_root = loras_dir.resolve()
    candidate = (loras_dir / lora_name).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _sha256_file(path: Path) -> str:
    """Blocking — always run this through `hash_lora_file`, never called
    directly from an async handler."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


async def hash_lora_file(path: Path) -> str:
    """SHA256 hex digest of `path`'s full contents, off the event loop
    thread (`asyncio.to_thread`) — hashing a large file is CPU/disk-bound
    blocking work that would otherwise stall every other chat's handler on
    this bot's single event loop for its duration."""
    return await asyncio.to_thread(_sha256_file, path)


async def fetch_civitai_info(sha256: str) -> CivitaiLoraInfo | None:
    """Query CivitAI for the model version matching `sha256`. Returns None
    when CivitAI was reached and genuinely has no match (an ordinary 404 —
    the LoRA may be privately trained, hosted elsewhere, or simply not on
    CivitAI). Raises `CivitaiLookupError` when the request fails outright
    (network error, timeout, unexpected non-404 status) — a caller that
    caches results must let that propagate rather than caching it as a
    clean miss (see `CivitaiLookupError`'s docstring)."""
    url = CIVITAI_HASH_LOOKUP_URL.format(sha256=sha256)
    try:
        async with (
            aiohttp.ClientSession(timeout=_CIVITAI_TIMEOUT) as session,
            session.get(url) as resp,
        ):
            if resp.status == 404:
                return None
            resp.raise_for_status()
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise CivitaiLookupError(f"CivitAI lookup failed for hash {sha256}: {exc}") from exc

    model = data.get("model") or {}
    # The parent model's id is a top-level `modelId` on the model-version
    # response, NOT nested under `model` — confirmed against a live
    # response: `model` only carries `name`/`type`/`nsfw`/`poi`, no `id` at
    # all. Reading `model.get("id")` here used to silently return None on
    # every single lookup, so the link always fell back to the bare
    # civitai.com URL below instead of a real model page.
    model_id = data.get("modelId")
    version_id = data.get("id")
    civitai_url = (
        f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
        if model_id is not None
        else "https://civitai.com"
    )
    return CivitaiLoraInfo(
        model_name=model.get("name") or "Unknown",
        base_model=data.get("baseModel") or "Unknown",
        trained_words=list(data.get("trainedWords") or []),
        civitai_url=civitai_url,
    )
