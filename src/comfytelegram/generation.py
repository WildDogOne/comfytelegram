"""Glue between a bot-level request, the model-profile system, and ComfyUI.

This is the piece TODO.md section 4 flagged as missing: turning
(checkpoint, user prompt, profile) into a submitted-and-collected result,
independent of Telegram. `handlers.py` calls into this; `scripts/smoke_test*.py`
do the equivalent inline for quick manual checks against a live server.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.profiles import ModelProfile, resolve_generation_params
from comfytelegram.workflows import (
    FaceDetailerParams,
    PostProcessBaseParams,
    UpscaleParams,
    build_face_detailer,
    build_txt2img,
    build_upscale,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[JobProgress], Awaitable[None]]


@dataclass
class GeneratedImage:
    data: bytes
    filename: str
    base_params: PostProcessBaseParams


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

    results = []
    for img in images:
        data = await client.get_image_bytes(img["filename"], img["subfolder"], img["type"])
        results.append((data, img["filename"]))
    return results


async def generate(
    client: ComfyClient,
    checkpoint: str,
    user_prompt: str,
    profile: ModelProfile | None,
    *,
    on_progress: ProgressCallback | None = None,
) -> list[GeneratedImage]:
    """Resolve profile defaults, build the base txt2img graph, run it."""
    params = resolve_generation_params(checkpoint, user_prompt, profile)
    prompt_graph, save_node_id = build_txt2img(params)
    logger.info("Submitting txt2img: checkpoint=%s cfg=%s steps=%s", checkpoint, params.cfg, params.steps)

    base_params = PostProcessBaseParams(
        checkpoint=params.checkpoint,
        positive_prompt=params.positive_prompt,
        negative_prompt=params.negative_prompt,
        loras=params.loras,
        clip_skip=params.clip_skip,
    )

    raw = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    return [GeneratedImage(data=data, filename=name, base_params=base_params) for data, name in raw]


async def post_process(
    client: ComfyClient,
    kind: Literal["upscale", "face"],
    source_image: bytes,
    source_filename: str,
    base_params: PostProcessBaseParams,
    *,
    on_progress: ProgressCallback | None = None,
) -> GeneratedImage:
    """Upload a previously-generated image and run one post-processing stage on it."""
    upload = await client.upload_image(source_image, filename=source_filename)
    uploaded_name = upload["name"]

    if kind == "upscale":
        prompt_graph, save_node_id = build_upscale(uploaded_name, base_params, UpscaleParams())
    elif kind == "face":
        prompt_graph, save_node_id = build_face_detailer(uploaded_name, base_params, FaceDetailerParams())
    else:
        raise ValueError(f"Unknown post-processing kind: {kind}")

    logger.info("Submitting post-process (%s) on %s", kind, uploaded_name)
    raw = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    data, filename = raw[0]
    return GeneratedImage(data=data, filename=filename, base_params=base_params)
