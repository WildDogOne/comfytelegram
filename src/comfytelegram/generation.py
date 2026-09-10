"""Glue between a bot-level request, the model-profile system, and ComfyUI.

Turns (checkpoint, user prompt, profile) into a submitted-and-collected
result, independent of Telegram. `handlers.py` calls into this;
`scripts/smoke_test*.py` do the equivalent inline for quick manual checks
against a live server.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.profiles import ModelProfile, resolve_generation_params
from comfytelegram.workflows import (
    FaceDetailerParams,
    GenerationParams,
    HandDetailerParams,
    PostProcessBaseParams,
    UpscaleParams,
    build_face_detailer,
    build_hand_detailer,
    build_txt2img,
    build_upscale,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[JobProgress], Awaitable[None]]


@dataclass
class GeneratedImage:
    """One image out of ComfyUI, plus the resolved settings that made it."""

    data: bytes
    filename: str
    #: The fully-resolved settings this image (or the base generation it was
    #: post-processed from) was made with. Carried forward through
    #: post-processing so the checkpoint/filename it came from is known to
    #: "🔬 Analyze & Regenerate" and "🔁 Generate Again" (see handlers.py).
    full_params: GenerationParams


def _to_post_process_base(params: GenerationParams) -> PostProcessBaseParams:
    """Narrow a full `GenerationParams` down to the checkpoint/prompt/LoRA/
    clip-skip subset `build_upscale`/`build_face_detailer` need."""
    return PostProcessBaseParams(
        checkpoint=params.checkpoint,
        positive_prompt=params.positive_prompt,
        negative_prompt=params.negative_prompt,
        loras=params.loras,
        clip_skip=params.clip_skip,
    )


async def _run_graph(
    client: ComfyClient,
    prompt_graph: dict,
    save_node_id: str,
    *,
    on_progress: ProgressCallback | None,
) -> list[tuple[bytes, str]]:
    """Submit a graph, wait for it to finish, and download every image the
    given save node produced. Returns (bytes, filename) pairs."""
    client_id = uuid.uuid4().hex
    prompt_id = await client.queue_prompt(prompt_graph, client_id=client_id)

    async for progress in client.watch(prompt_id, client_id=client_id):
        if on_progress is not None:
            await on_progress(progress)
        if progress.done:
            break

    history = await client.get_history(prompt_id)
    if history is None:
        raise ComfyUIError(f"No history entry for prompt {prompt_id} after completion")

    status = history.get("status", {})
    if status.get("status_str") == "error":
        raise ComfyUIError(f"Job {prompt_id} failed: {status}")

    images = history.get("outputs", {}).get(save_node_id, {}).get("images", [])
    if not images:
        raise ComfyUIError(f"Job {prompt_id} produced no images at save node {save_node_id}")

    # Independent /view downloads — run them concurrently instead of one
    # round-trip at a time (matters most for a multi-image batch_size).
    data_list = await asyncio.gather(
        *(client.get_image_bytes(img["filename"], img["subfolder"], img["type"]) for img in images)
    )
    return list(zip(data_list, (img["filename"] for img in images)))


async def generate(
    client: ComfyClient,
    checkpoint: str,
    user_prompt: str,
    profile: ModelProfile | None,
    *,
    extra_negative_prompt: str = "",
    overrides: dict[str, Any] | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[GeneratedImage]:
    """Resolve profile defaults, build the base txt2img graph, run it.

    `user_prompt` should already have any active character's positive
    prompt folded in by the caller (see `handlers.py`'s `generate_message`)
    since it's just free text; `extra_negative_prompt` carries that same
    character's negative prompt through separately, since profile
    resolution owns the negative prompt entirely otherwise. `overrides`
    wins over both the profile and its own defaults (see
    `resolve_generation_params`) — used by "/stream" to force
    `batch_size=1` per request without touching the chat's persisted
    `/settings` override.
    """
    params = resolve_generation_params(
        checkpoint,
        user_prompt,
        profile,
        overrides=overrides,
        extra_negative_prompt=extra_negative_prompt,
    )
    prompt_graph, save_node_id = build_txt2img(params)
    logger.info(
        "Submitting txt2img: checkpoint=%s cfg=%s steps=%s", checkpoint, params.cfg, params.steps
    )

    raw = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    return [GeneratedImage(data=data, filename=name, full_params=params) for data, name in raw]


async def post_process(
    client: ComfyClient,
    kind: Literal["upscale", "face", "hand"],
    source_image: bytes,
    source_filename: str,
    full_params: GenerationParams,
    *,
    on_progress: ProgressCallback | None = None,
) -> GeneratedImage:
    """Upload a previously-generated image and run one post-processing stage on it."""
    base_params = _to_post_process_base(full_params)
    upload = await client.upload_image(source_image, filename=source_filename)
    uploaded_name = upload["name"]

    if kind == "upscale":
        prompt_graph, save_node_id = build_upscale(uploaded_name, base_params, UpscaleParams())
    elif kind == "face":
        prompt_graph, save_node_id = build_face_detailer(
            uploaded_name, base_params, FaceDetailerParams()
        )
    elif kind == "hand":
        prompt_graph, save_node_id = build_hand_detailer(
            uploaded_name, base_params, HandDetailerParams()
        )
    else:
        raise ValueError(f"Unknown post-processing kind: {kind}")

    logger.info("Submitting post-process (%s) on %s", kind, uploaded_name)
    raw = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    data, filename = raw[0]
    return GeneratedImage(data=data, filename=filename, full_params=full_params)


async def repeat(
    client: ComfyClient,
    full_params: GenerationParams,
    *,
    on_progress: ProgressCallback | None = None,
) -> list[GeneratedImage]:
    """Re-run an entire base txt2img request — all `batch_size` images — with
    the same resolved settings but a forced-fresh seed. Backs the chat-level
    "🔁 Generate Again" button, for quickly cranking out more variations of
    the same prompt without retyping it or picking a specific image."""
    fresh_params = replace(full_params, seed=None)
    prompt_graph, save_node_id = build_txt2img(fresh_params)
    logger.info(
        "Repeating: checkpoint=%s cfg=%s steps=%s",
        fresh_params.checkpoint,
        fresh_params.cfg,
        fresh_params.steps,
    )

    raw = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    return [
        GeneratedImage(data=data, filename=name, full_params=fresh_params) for data, name in raw
    ]
