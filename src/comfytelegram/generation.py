"""Glue between a bot-level request, the model-profile system, and ComfyUI.

Turns (checkpoint, user prompt, profile) into a submitted-and-collected
result, independent of Telegram. `handlers.py` calls into this;
`scripts/smoke_test*.py` do the equivalent inline for quick manual checks
against a live server.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from PIL import Image

from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.profiles import ModelProfile, resolve_generation_params
from comfytelegram.workflows import (
    FaceDetailerParams,
    GenerationParams,
    HandDetailerParams,
    ManualHandDetailerParams,
    PostProcessBaseParams,
    UpscaleParams,
    build_face_detailer,
    build_hand_detailer,
    build_hand_detailer_manual,
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
    #: "🔁 Generate Again" (see handlers.py).
    full_params: GenerationParams
    #: True if this came from `post_process(kind="face"/"hand")` and Impact
    #: Pack's bbox detector found nothing to refine, so the node silently
    #: passed the source image through unchanged instead of erroring — see
    #: `_detailer_found_nothing`. Always False for a fresh txt2img/
    #: `repeat()` image or an upscale (which always changes pixel
    #: dimensions), since there's either no source to compare against or a
    #: no-op isn't possible. `postprocess_callback` surfaces this as a
    #: warning instead of silently handing back what looks like a
    #: successful refinement.
    unchanged: bool = False


#: Max grayscale pixel value `_detailer_found_nothing` still treats as "no
#: mask at all" — an all-zero mask stays exactly 0 through PNG encoding (no
#: floating-point rounding is possible at that boundary), so this is just a
#: hair of slack, not a real tolerance band.
_ZERO_MASK_TOLERANCE = 1


async def _detailer_found_nothing(
    client: ComfyClient, history: dict[str, Any], detection_node_id: str
) -> bool:
    """True only if we can positively confirm the detailer's detection mask
    (see `_build_detailer`'s docstring — `detection_node_id` is the
    `PreviewImage` fed by its `mask` output) came back completely black,
    meaning Impact Pack's bbox detector found nothing and the detailer's
    main `image` output is an unmodified pass-through of the source. This
    is what `post_process()` actually wants to know — diffing the final
    output image's pixels against the source doesn't work reliably, since
    ComfyUI's own float32 round-trip (uint8 -> tensor -> uint8) can shift
    pixel values by a level or two even on a true no-op pass-through.
    Returns False (i.e. "assume it did something") whenever that can't be
    confirmed — no image at that node, or a download/decode failure —
    since wrongly suppressing a real result is worse than an occasional
    missed "nothing detected" notice."""
    images = history.get("outputs", {}).get(detection_node_id, {}).get("images", [])
    if not images:
        return False
    img = images[0]
    try:
        data = await client.get_image_bytes(img["filename"], img["subfolder"], img["type"])
        pixels = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
    except Exception:
        logger.warning("Couldn't inspect detailer detection mask", exc_info=True)
        return False
    return bool(pixels.max() <= _ZERO_MASK_TOLERANCE)


def _to_post_process_base(params: GenerationParams) -> PostProcessBaseParams:
    """Narrow a full `GenerationParams` down to the checkpoint/prompt/LoRA/
    clip-skip/loader subset `build_upscale`/`build_face_detailer`/
    `build_hand_detailer` need. Carrying the loader fields through matters
    for split-architecture checkpoints (e.g. Anima) — without them a
    post-processing pass would default back to `loader="checkpoint"` and
    try to load the UNET filename through `CheckpointLoaderSimple`."""
    return PostProcessBaseParams(
        checkpoint=params.checkpoint,
        positive_prompt=params.positive_prompt,
        negative_prompt=params.negative_prompt,
        loras=params.loras,
        clip_skip=params.clip_skip,
        loader=params.loader,
        clip_name=params.clip_name,
        clip_type=params.clip_type,
        vae_name=params.vae_name,
        model_sampling_shift=params.model_sampling_shift,
        tile_controlnet=params.tile_controlnet,
        tile_controlnet_strength=params.tile_controlnet_strength,
        upscale_denoise=params.upscale_denoise,
    )


async def _run_graph(
    client: ComfyClient,
    prompt_graph: dict,
    save_node_id: str,
    *,
    on_progress: ProgressCallback | None,
) -> tuple[list[tuple[bytes, str]], dict[str, Any]]:
    """Submit a graph, wait for it to finish, and download every image the
    given save node produced. Returns ((bytes, filename) pairs, the raw
    history dict) — the history is handed back too so a caller that needs
    to inspect another node's output (`post_process()`'s detection-check
    node, see `_detailer_found_nothing`) doesn't need a second /history
    round-trip."""
    client_id = uuid.uuid4().hex
    # Connect before queueing, never after: ComfyUI drops execution events
    # aimed at a client that isn't connected yet, so submitting first can
    # lose the terminal event of a fast job and hang here forever. See
    # `ComfyClient.connect_events`.
    async with client.connect_events(client_id=client_id) as events:
        prompt_id = await client.queue_prompt(prompt_graph, client_id=client_id)

        async for progress in events.watch(prompt_id):
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
    return list(zip(data_list, (img["filename"] for img in images))), history


async def generate(
    client: ComfyClient,
    checkpoint: str,
    user_prompt: str,
    profile: ModelProfile | None,
    *,
    extra_negative_prompt: str = "",
    raw_positive_prompt: str = "",
    raw_negative_prompt: str = "",
    overrides: dict[str, Any] | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[GeneratedImage]:
    """Resolve profile defaults, build the base txt2img graph, run it.

    `user_prompt` should already have any active character's positive
    prompt folded in by the caller (see `handlers.py`'s `generate_message`)
    since it's just free text; `extra_negative_prompt` carries that same
    character's negative prompt through separately, since profile
    resolution owns the negative prompt entirely otherwise.
    `raw_positive_prompt`/`raw_negative_prompt` are the caller's record of
    what the user actually typed, before any profile/character prompt got
    folded in — see `GenerationParams.raw_positive_prompt` — and ride along
    unmodified onto the returned images' `full_params` for "🐛 Show Prompt"
    to display. `overrides` wins over both the profile and its own
    defaults (see `resolve_generation_params`) — used by "/stream" to force
    `batch_size=1` per request without touching the chat's persisted
    `/settings` override.
    """
    params = resolve_generation_params(
        checkpoint,
        user_prompt,
        profile,
        overrides=overrides,
        extra_negative_prompt=extra_negative_prompt,
        raw_positive_prompt=raw_positive_prompt,
        raw_negative_prompt=raw_negative_prompt,
    )
    prompt_graph, save_node_id = build_txt2img(params)
    logger.info(
        "Submitting txt2img: checkpoint=%s cfg=%s steps=%s", checkpoint, params.cfg, params.steps
    )

    raw, _history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    return [GeneratedImage(data=data, filename=name, full_params=params) for data, name in raw]


async def post_process(
    client: ComfyClient,
    kind: Literal["upscale", "face", "hand", "hand_manual"],
    source_image: bytes,
    source_filename: str,
    full_params: GenerationParams,
    *,
    point_frac: tuple[float, float] | None = None,
    box_size_frac: float | None = None,
    on_progress: ProgressCallback | None = None,
) -> GeneratedImage:
    """Upload a previously-generated image and run one post-processing stage
    on it. For `kind="face"`/`"hand"`, the returned `GeneratedImage.unchanged`
    is True if Impact Pack's bbox detector found nothing to refine — the
    node just passes the source through as-is in that case rather than
    erroring, which otherwise looks like a successful (but pointless)
    refinement (see `_detailer_found_nothing`). `kind="hand_manual"` skips
    detection entirely and inpaints a small mask centered on `point_frac`
    (required for this kind — see `build_hand_detailer_manual`, backing
    handlers.py's "✋ Tap to mark" flow); `unchanged` is always False for it,
    since a manually placed mask can't come back empty. `box_size_frac`
    overrides `ManualHandDetailerParams`' default mask size for `"hand_manual"`
    — handlers.py's `hand_point_callback` shrinks it for a denser tap grid
    (see `HAND_POINT_BOX_SIZE_FRAC_BASE`) so the marked region stays roughly
    cell-sized instead of a fixed fraction of the image regardless of
    density; ignored for every other `kind`."""
    base_params = _to_post_process_base(full_params)
    upload = await client.upload_image(source_image, filename=source_filename)
    uploaded_name = upload["name"]

    detection_node_id: str | None = None
    if kind == "upscale":
        upscale_params = UpscaleParams()
        if base_params.upscale_denoise is not None:
            upscale_params = replace(upscale_params, denoise=base_params.upscale_denoise)
        prompt_graph, save_node_id = build_upscale(uploaded_name, base_params, upscale_params)
    elif kind == "face":
        prompt_graph, save_node_id, detection_node_id = build_face_detailer(
            uploaded_name, base_params, FaceDetailerParams()
        )
    elif kind == "hand":
        prompt_graph, save_node_id, detection_node_id = build_hand_detailer(
            uploaded_name, base_params, HandDetailerParams()
        )
    elif kind == "hand_manual":
        assert point_frac is not None, "hand_manual requires point_frac"
        image_size = Image.open(io.BytesIO(source_image)).size
        manual_params = (
            ManualHandDetailerParams()
            if box_size_frac is None
            else ManualHandDetailerParams(box_size_frac=box_size_frac)
        )
        prompt_graph, save_node_id = build_hand_detailer_manual(
            uploaded_name, base_params, manual_params, point_frac, image_size
        )
    else:
        raise ValueError(f"Unknown post-processing kind: {kind}")

    logger.info("Submitting post-process (%s) on %s", kind, uploaded_name)
    raw, history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    data, filename = raw[0]
    unchanged = detection_node_id is not None and await _detailer_found_nothing(
        client, history, detection_node_id
    )
    return GeneratedImage(
        data=data, filename=filename, full_params=full_params, unchanged=unchanged
    )


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

    raw, _history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    return [
        GeneratedImage(data=data, filename=name, full_params=fresh_params) for data, name in raw
    ]
