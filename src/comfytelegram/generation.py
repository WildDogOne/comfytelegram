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
from comfytelegram.profiles import ModelProfile, join_nonempty, resolve_generation_params
from comfytelegram.workflows import (
    DrawnMaskFixParams,
    DrawnMaskHandDetailerParams,
    FaceDetailerParams,
    GenerationParams,
    HandDetailerParams,
    ManualHandDetailerParams,
    PostProcessBaseParams,
    TiledRefineParams,
    UpscaleParams,
    build_face_detailer,
    build_fix_drawn_mask,
    build_hand_detailer,
    build_hand_detailer_drawn_mask,
    build_hand_detailer_manual,
    build_tiled_refine,
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
        anima_lllite_inpaint_patch=params.anima_lllite_inpaint_patch,
        anima_lllite_inpaint_patch_strength=params.anima_lllite_inpaint_patch_strength,
        upscale_denoise=params.upscale_denoise,
    )


def _fix_artifact_override_base(profiles: list[ModelProfile]) -> PostProcessBaseParams | None:
    """`kind="fix_drawn"` support: if a profile sets `fix_artifact_checkpoint`
    (currently only Anima Aesthetic), "🩹 Fix Artifact" always runs against
    *that* profile's checkpoint/loader/clip/vae/LoRAs/anima_lllite_inpaint_patch
    instead of the image's own original checkpoint — see
    `ModelProfile.fix_artifact_checkpoint`'s docstring for why (most
    checkpoints have no inpainting-aware path wired at all yet, so their
    removal results are unreliable regardless of tuning). `None` if no
    profile has it set, in which case the caller falls back to the image's
    own checkpoint as before. First match wins if more than one profile sets
    it, same "first match wins" convention `profiles.loader.resolve_profile`
    uses for `match` globs.

    The positive prompt is the profile's own `positive_prompt_prefix` plus
    "background scenery" — not blank. Matched against a real
    krita-ai-diffusion "remove object" job pulled from this server's own
    `/history`: its positive prompt was `"<style's quality prefix>,
    background scenery"`, exactly `prepare_prompts()`'s documented fallback
    (`if cond.positive == "" and inpaint is InpaintMode.remove_object:
    cond.positive = "background scenery"`) merged with the style prompt — a
    genuinely blank prompt was this bot's own choice, not krita's."""
    profile = next((p for p in profiles if p.fix_artifact_checkpoint), None)
    if profile is None:
        return None
    return PostProcessBaseParams(
        checkpoint=profile.fix_artifact_checkpoint,
        positive_prompt=join_nonempty([profile.positive_prompt_prefix, "background scenery"]),
        negative_prompt=profile.negative_prompt_prefix,
        loras=[lora.to_spec() for lora in profile.loras if lora.default_enabled],
        clip_skip=profile.defaults.clip_skip if profile.defaults.clip_skip is not None else -1,
        loader=profile.loader,
        clip_name=profile.clip_name,
        clip_type=profile.clip_type,
        vae_name=profile.vae_name,
        model_sampling_shift=profile.model_sampling_shift,
        tile_controlnet=profile.tile_controlnet,
        tile_controlnet_strength=profile.tile_controlnet_strength,
        anima_lllite_inpaint_patch=profile.anima_lllite_inpaint_patch,
        anima_lllite_inpaint_patch_strength=profile.anima_lllite_inpaint_patch_strength,
        upscale_denoise=None,
    )


def _fix_drawn_work_size(
    image_size: tuple[int, int], params: DrawnMaskFixParams
) -> tuple[int, int]:
    """`build_fix_drawn_mask`'s `work_size` — `image_size` downscaled
    (preserving aspect ratio) so its longest side is at most
    `params.work_max_size`, or `image_size` unchanged if it's already
    smaller. Only `_build_anima_fix_drawn_mask` actually resizes with this,
    but it's computed unconditionally here since it's cheap and
    `build_fix_drawn_mask` needs the value regardless of which path it ends
    up taking.

    A *crop*-based predecessor of this (bounding a region around the drawn
    mask's own bbox instead of downscaling the whole image) was tried
    first and rejected: confirmed on a real removal case pulled from this
    server's own `/history` that cropping away the surrounding scene
    starves the diffusion pass of context it needs to correctly continue
    "this is fabric" into the masked hole — it painted a literal window
    showing background through the masked garment instead, regardless of
    seed/checkpoint/prompt tried. A resize keeps the whole scene, just
    smaller — matching krita-ai-diffusion's own `scale_to_initial`
    approach, and the actual root problem this exists for is a runaway
    30-step diffusion pass over a raw, un-cropped 4096x4096 photo with
    nothing bounding the resolution — solved just as well by a resize."""
    img_w, img_h = image_size
    longest = max(img_w, img_h)
    if longest <= params.work_max_size:
        return (img_w, img_h)
    scale = params.work_max_size / longest
    return (round(img_w * scale), round(img_h * scale))


#: How long `_run_graph` waits for a websocket event before checking
#: `/history` directly instead. Connecting before queueing (see
#: `ComfyClient.connect_events`) closes the *usual* window for ComfyUI to
#: drop a job's terminal event into the void, but not every one — verified
#: against a live server: a fast job (steps small enough, or the checkpoint/
#: prompt nodes execution-cached from a prior run, per its own
#: `execution_cached` history message) can still finish and fire its
#: terminal `executing{node: null}` before this process's websocket
#: handshake has been fully registered server-side, with nothing left to
#: ever arrive after that — no buffering, no replay. `/history` is
#: authoritative regardless of the socket's state, so silence past this
#: many seconds falls back to polling it directly rather than waiting on an
#: event that may already have been lost. Short enough to catch a lost-event
#: hang quickly, long enough that a real in-progress step (checked every
#: tick this fires during) doesn't trigger more than the occasional spare
#: /history call.
_WS_EVENT_FALLBACK_SECONDS = 5.0


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
    history: dict[str, Any] | None = None
    async with client.connect_events(client_id=client_id) as events:
        prompt_id = await client.queue_prompt(prompt_graph, client_id=client_id)

        watcher = events.watch(prompt_id).__aiter__()
        watcher_active = True
        completed = False
        while not completed:
            progress = None
            if watcher_active:
                try:
                    progress = await asyncio.wait_for(
                        watcher.__anext__(), timeout=_WS_EVENT_FALLBACK_SECONDS
                    )
                except StopAsyncIteration:
                    watcher_active = False
                except TimeoutError:
                    pass
                else:
                    if on_progress is not None:
                        await on_progress(progress)
                    if not progress.done:
                        continue
            else:
                # Once the watcher stops being trustworthy (below), keep
                # checking on the same cadence a lost event would use
                # rather than hammering /history in a tight loop.
                await asyncio.sleep(_WS_EVENT_FALLBACK_SECONDS)

            # /history is the sole source of truth for completion — a lost
            # terminal event (watcher ran dry or timed out with nothing,
            # the original reason for this fallback) and a *premature* one
            # both need corroborating, not trusting outright. Verified
            # against a live server: a job whose leading nodes were all
            # `execution_cached` fired `executing{node: null}` on the
            # websocket within milliseconds of being queued, roughly 90
            # seconds before its actual (non-cached) work finished and
            # `/history` agreed the job was complete — trusting that event
            # alone reported a job that went on to succeed as failed.
            history = await client.get_history(prompt_id)
            completed = history is not None and bool(history.get("status", {}).get("completed"))
            if progress is not None and progress.done and not completed:
                # Confirmed premature: `JobEvents.watch` already returned
                # after yielding this, so there's nothing left to await on
                # it — fall back to polling /history like a lost event.
                watcher_active = False

    # The loop above only ever exits with `completed` True, which requires
    # `history` to be a real dict (see its assignment above) — there's no
    # "give up" path left; a job ComfyUI hasn't finished is worth waiting
    # for indefinitely rather than reporting a false failure (see the
    # comment above on the premature-terminal-event case this replaced).
    assert history is not None
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
    kind: Literal[
        "upscale", "homogenize", "face", "hand", "hand_manual", "hand_drawn", "fix_drawn"
    ],
    source_image: bytes,
    source_filename: str,
    full_params: GenerationParams,
    *,
    point_frac: tuple[float, float] | None = None,
    box_size_frac: float | None = None,
    mask_bytes: bytes | None = None,
    on_progress: ProgressCallback | None = None,
    profiles: list[ModelProfile] | None = None,
) -> GeneratedImage:
    """Upload a previously-generated image and run one post-processing stage
    on it. `kind="homogenize"` is the same UltimateSDUpscale tiled img2img
    pass as `"upscale"`, but at `TiledRefineParams.upscale_by=1.0` — no
    resolution change, just a lower-denoise pass over the whole image to
    blend seams left behind by independent Face/Hand Detail patches (each
    of those only ever sees its own crop, at its own denoise/seed).
    `unchanged` is always False for it too, the same as `"upscale"` — there's
    no detection step to have found nothing. For `kind="face"`/`"hand"`, the
    returned `GeneratedImage.unchanged`
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
    density; ignored for every other `kind`. `kind="hand_drawn"` also skips
    detection, inpainting a freehand mask instead (required for this kind —
    see `build_hand_detailer_drawn_mask`, backing the Telegram WebApp mask
    editor); `unchanged` is always False for it too, for the same reason.
    `kind="fix_drawn"` is the general-purpose "🩹 Fix Artifact" counterpart to
    `"hand_drawn"` — same freehand-drawn-mask requirement (`mask_bytes`) and
    the same `unchanged=False` reasoning, just built via
    `build_fix_drawn_mask`/`DrawnMaskFixParams` (higher denoise/crop_factor,
    since removing an arbitrary artifact needs more creative latitude than
    a hand touch-up) instead of the hand-tuned graph — for painting over any
    unwanted region (a stray object, a background glitch, a watermark)
    rather than just a hand. `profiles`, if given, is only consulted for
    `kind="fix_drawn"` — see `_fix_artifact_override_base` — to run that
    pass against a fixed, known-inpainting-aware checkpoint regardless of
    which one the image was originally generated with; every other `kind`
    ignores it entirely and keeps using the image's own checkpoint."""
    base_params = _to_post_process_base(full_params)
    upload = await client.upload_image(source_image, filename=source_filename)
    uploaded_name = upload["name"]

    detection_node_id: str | None = None
    if kind == "upscale":
        upscale_params = UpscaleParams()
        if base_params.upscale_denoise is not None:
            upscale_params = replace(upscale_params, denoise=base_params.upscale_denoise)
        prompt_graph, save_node_id = build_upscale(uploaded_name, base_params, upscale_params)
    elif kind == "homogenize":
        prompt_graph, save_node_id = build_tiled_refine(
            uploaded_name, base_params, TiledRefineParams()
        )
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
    elif kind == "hand_drawn":
        assert mask_bytes is not None, "hand_drawn requires mask_bytes"
        mask_upload = await client.upload_image(mask_bytes, filename=f"mask_{source_filename}")
        prompt_graph, save_node_id = build_hand_detailer_drawn_mask(
            uploaded_name, mask_upload["name"], base_params, DrawnMaskHandDetailerParams()
        )
    elif kind == "fix_drawn":
        assert mask_bytes is not None, "fix_drawn requires mask_bytes"
        mask_upload = await client.upload_image(mask_bytes, filename=f"mask_{source_filename}")
        # Removal, not refinement: the original positive prompt describes the
        # whole scene, including whatever the user just marked for deletion,
        # so conditioning the inpaint on it steers the model toward a nicer
        # version of the exact thing being removed instead of erasing it.
        # When there's no fix_artifact_checkpoint override, drop it entirely
        # and let the fill be driven by `build_fix_drawn_mask`'s
        # content-aware fill pass and the surrounding image context instead.
        # The override path composes its own prompt in
        # `_fix_artifact_override_base` instead — "<profile prefix>,
        # background scenery", matched against a real krita-ai-diffusion job
        # (a genuinely blank prompt was this bot's own choice, not krita's).
        override_base = _fix_artifact_override_base(profiles) if profiles else None
        if override_base is not None:
            fix_base_params = override_base
            checkpoint_source = "fix_artifact_checkpoint override"
        else:
            fix_base_params = replace(base_params, positive_prompt="")
            checkpoint_source = "image's own checkpoint"
        fix_params = DrawnMaskFixParams()
        if fix_base_params.loader == "split" and fix_base_params.anima_lllite_inpaint_patch:
            engaged_reason = (
                f"patch={fix_base_params.anima_lllite_inpaint_patch!r} "
                f"strength={fix_base_params.anima_lllite_inpaint_patch_strength}"
            )
        elif fix_base_params.loader != "split":
            engaged_reason = f"skipped (loader={fix_base_params.loader!r}, not 'split')"
        else:
            engaged_reason = "skipped (anima_lllite_inpaint_patch not set on this profile)"
        image_size = Image.open(io.BytesIO(source_image)).size
        work_size = _fix_drawn_work_size(image_size, fix_params)
        logger.info(
            "Fix Artifact: checkpoint=%s (%s) loader=%s fill_model=%s anima_lllite_patch: %s "
            "positive_prompt=%r work_size=%s (image_size=%s)",
            fix_base_params.checkpoint,
            checkpoint_source,
            fix_base_params.loader,
            fix_params.fill_model,
            engaged_reason,
            fix_base_params.positive_prompt,
            work_size,
            image_size,
        )
        prompt_graph, save_node_id = build_fix_drawn_mask(
            uploaded_name,
            mask_upload["name"],
            fix_base_params,
            fix_params,
            image_size=image_size,
            work_size=work_size,
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
