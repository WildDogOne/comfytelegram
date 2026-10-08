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
from comfytelegram.params_serde import serialize_generation_params
from comfytelegram.png_metadata import build_metadata, embed_metadata, extract_seed
from comfytelegram.profiles import (
    ModelProfile,
    join_nonempty,
    resolve_generation_params,
    resolve_profile,
)
from comfytelegram.workflows import (
    DrawnMaskFixParams,
    DrawnMaskHandDetailerParams,
    FaceDetailerParams,
    GenerationParams,
    HandDetailerParams,
    KontextModelFiles,
    KontextParams,
    ManualHandDetailerParams,
    PostProcessBaseParams,
    RegionSpec,
    TiledRefineParams,
    UpscaleParams,
    build_face_detailer,
    build_fix_drawn_mask,
    build_hand_detailer,
    build_hand_detailer_drawn_mask,
    build_hand_detailer_manual,
    build_kontext_edit,
    build_tiled_refine,
    build_txt2img,
    build_upscale,
    kontext_resolution,
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
        detail_prompt_tile_controlnet=params.detail_prompt_tile_controlnet,
        anima_lllite_inpaint_patch=params.anima_lllite_inpaint_patch,
        anima_lllite_inpaint_patch_strength=params.anima_lllite_inpaint_patch_strength,
        upscale_denoise=params.upscale_denoise,
        detailer_denoise=params.detailer_denoise,
        detailer_cfg=params.detailer_cfg,
        detailer_steps=params.detailer_steps,
        detailer_disable_lora=params.detailer_disable_lora,
    )


#: `post_process` kinds whose whole-image tiled pass reads
#: `PostProcessBaseParams.tile_controlnet`/`tile_controlnet_strength`/
#: `upscale_denoise` — see `_refresh_tunable_defaults`.
_TILED_UPSCALE_KINDS = ("upscale", "homogenize")

#: `post_process` kinds whose detailer pass reads
#: `PostProcessBaseParams.detailer_denoise`/`detailer_cfg`/`detailer_steps`/
#: `detailer_disable_lora` — see `_refresh_detailer_tunables`. `"fix_drawn"` is deliberately
#: excluded: it always runs against `DrawnMaskFixParams`' own fixed
#: removal-oriented denoise/cfg/steps (and, when `fix_artifact_checkpoint`
#: is set, a different checkpoint's profile entirely — see
#: `_fix_artifact_override_base`), so this checkpoint's own detailer
#: tunables have no business reaching it.
_DETAILER_TUNABLE_KINDS = ("face", "hand", "hand_manual", "hand_drawn")


def _refresh_tunable_defaults(
    base: PostProcessBaseParams, profiles: list[ModelProfile] | None
) -> PostProcessBaseParams:
    """`tile_controlnet(_strength)`/`upscale_denoise` are tuning knobs meant
    to be edited in a profile's JSON file over time — unlike checkpoint/
    LoRAs/prompt, nothing requires them to match what a given image was
    originally generated with for post-processing to make sense. But `base`
    was narrowed off `full_params` (`_to_post_process_base`), which froze
    whatever the profile said *at generation time* — so editing the profile
    afterward had no effect on that image's own post-processing passes
    until it was regenerated from scratch (confirmed live: a same-day
    profile edit to `upscale_denoise`/`tile_controlnet_strength` didn't
    change the very next "🔍 Upscale 4x" tap on an image generated earlier
    that day). Re-resolving against the checkpoint's *current* profile here
    instead means an edited profile takes effect on the next tap, no
    regeneration needed. Falls back to `base` unchanged if the checkpoint no
    longer matches any shipped profile (a foreign/imported image, or one
    whose profile was since removed) or `profiles` wasn't supplied at all."""
    live_profile = resolve_profile(base.checkpoint, profiles or [])
    if live_profile is None:
        return base
    return replace(
        base,
        tile_controlnet=live_profile.tile_controlnet,
        tile_controlnet_strength=live_profile.tile_controlnet_strength,
        upscale_denoise=live_profile.defaults.upscale_denoise,
    )


def _refresh_detailer_tunables(
    base: PostProcessBaseParams, profiles: list[ModelProfile] | None
) -> PostProcessBaseParams:
    """Same "edit the profile, see it on the very next tap" reasoning as
    `_refresh_tunable_defaults`, scoped to `ModelProfile.defaults.
    detailer_denoise`/`detailer_cfg`/`detailer_steps`/`detailer_disable_lora`
    — applied for every kind in `_DETAILER_TUNABLE_KINDS` regardless of
    `is_detail_prompt`, unlike `_refresh_detail_prompt_tile_controlnet`,
    since a checkpoint that needs a different detailer denoise/cfg/steps (or
    needs its LoRAs left out of the detailer pass entirely — surfaced by
    Banana Splitz XXL, whose detailer pass got visibly worse with any LoRA
    attached) needs it whether the tap was "🎯 Face Detail", "🖐️ Hand
    Detail", "✋ Tap to mark", "🖌️ Draw Mask", or "✏️ Detail Prompt" —
    there's no reason one of those should get the tuned values and another
    shouldn't. Falls back to `base` unchanged if the checkpoint no longer
    matches any shipped profile or `profiles` wasn't supplied."""
    live_profile = resolve_profile(base.checkpoint, profiles or [])
    if live_profile is None:
        return base
    return replace(
        base,
        detailer_denoise=live_profile.defaults.detailer_denoise,
        detailer_cfg=live_profile.defaults.detailer_cfg,
        detailer_steps=live_profile.defaults.detailer_steps,
        detailer_disable_lora=live_profile.defaults.detailer_disable_lora,
    )


def _detailer_cfg_steps_overrides(base: PostProcessBaseParams) -> dict[str, Any]:
    """The `replace()` kwargs a detailer-family dataclass (`FaceDetailerParams`/
    `HandDetailerParams`/`ManualHandDetailerParams`/`DrawnMaskHandDetailerParams`,
    all sharing the same `cfg`/`steps` field names) should apply from
    `base`'s live-refreshed `detailer_cfg`/`detailer_steps` — empty (a no-op
    `replace()`) for any left `None`. `denoise` isn't included here since
    `"hand_drawn"` has its own one-shot override to fold in first (see
    `post_process`'s `denoise` parameter); every other kind applies it the
    same simple way right alongside these two."""
    overrides: dict[str, Any] = {}
    if base.detailer_cfg is not None:
        overrides["cfg"] = base.detailer_cfg
    if base.detailer_steps is not None:
        overrides["steps"] = base.detailer_steps
    return overrides


def _refresh_detail_prompt_tile_controlnet(
    base: PostProcessBaseParams, profiles: list[ModelProfile] | None
) -> PostProcessBaseParams:
    """Same "edit the profile, see it on the very next tap" reasoning as
    `_refresh_tunable_defaults`, scoped to "✏️ Detail Prompt"'s own
    tile-ControlNet experiment (`ModelProfile.detail_prompt_tile_controlnet`)
    — plus `tile_controlnet`/`tile_controlnet_strength` themselves, since
    that's what the experiment actually conditions on. Only ever called for
    `kind == "hand_drawn"` when `is_detail_prompt` is also True (see
    `post_process`) — a plain "🖌️ Draw Mask" call never re-resolves any of
    this, so a profile edit meant only to be tried via "✏️ Detail Prompt"
    can't accidentally change what "🖌️ Draw Mask" does too. Falls back to
    `base` unchanged if the checkpoint no longer matches any shipped profile
    or `profiles` wasn't supplied."""
    live_profile = resolve_profile(base.checkpoint, profiles or [])
    if live_profile is None:
        return base
    return replace(
        base,
        tile_controlnet=live_profile.tile_controlnet,
        tile_controlnet_strength=live_profile.tile_controlnet_strength,
        detail_prompt_tile_controlnet=live_profile.detail_prompt_tile_controlnet,
    )


def resolve_live_upscale_defaults(
    checkpoint: str, profiles: list[ModelProfile]
) -> tuple[float, float]:
    """The `(tile_controlnet_strength, upscale_denoise)` pair a "🔍 Upscale
    4x" tap on `checkpoint` would use right now with no override — the same
    live-profile resolution `_refresh_tunable_defaults` applies, with
    `UpscaleParams`'s own 0.2 denoise default filled in when the matched
    profile (or lack of one) leaves `defaults.upscale_denoise` unset. Used by
    handlers.py to show the actual numbers in the "use defaults or
    customize?" prompt that precedes an upscale, instead of a blind "use
    defaults" button telling the user nothing about what they'd get."""
    profile = resolve_profile(checkpoint, profiles)
    # `0.4` mirrors `ModelProfile.tile_controlnet_strength`'s own field
    # default — read straight off `profile` when one matched rather than
    # constructing a placeholder `ModelProfile` just to get it back.
    tile_strength = profile.tile_controlnet_strength if profile is not None else 0.4
    denoise = (
        profile.defaults.upscale_denoise
        if profile is not None and profile.defaults.upscale_denoise is not None
        else UpscaleParams().denoise
    )
    return tile_strength, denoise


#: The `post_process` kinds a caller-supplied detail prompt/denoise
#: override applies to — just `"hand_drawn"`, which is what
#: `DETAIL_PROMPT_CALLBACK_KIND` ("✏️ Detail Prompt") actually submits: a
#: freehand-drawn mask plus a one-shot prompt/denoise describing that one
#: region, run immediately rather than saved anywhere. Every other kind
#: never receives an override at all — `"face"`/`"hand"`/`"hand_manual"`
#: (auto-detect/tap) always condition on the image's own whole-scene
#: prompt, and `"upscale"`/`"homogenize"` condition the *whole* image (each
#: already has its own whole-image denoise —
#: `PostProcessBaseParams.upscale_denoise` for the former; the latter is a
#: fixed low-denoise seam-blend pass by design).
DETAIL_PROMPT_KINDS = ("hand_drawn",)


def _apply_detail_prompt(
    base: PostProcessBaseParams, positive: str | None, negative: str | None
) -> PostProcessBaseParams:
    """`base` with the user's detailer prompt override applied. Each side is
    independent — `None` means "keep what the image was generated with", so
    overriding only the positive leaves the original negative in place."""
    if positive is None and negative is None:
        return base
    return replace(
        base,
        positive_prompt=base.positive_prompt if positive is None else positive,
        negative_prompt=base.negative_prompt if negative is None else negative,
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
        positive_prompt=join_nonempty(
            [
                profile.fix_artifact_positive_prefix
                if profile.fix_artifact_positive_prefix is not None
                else profile.positive_prompt_prefix,
                "background scenery",
            ]
        ),
        negative_prompt=(
            profile.fix_artifact_negative_prefix
            if profile.fix_artifact_negative_prefix is not None
            else profile.negative_prompt_prefix
        ),
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
        detailer_denoise=None,
        detailer_cfg=None,
        detailer_steps=None,
    )


#: krita-ai-diffusion's own defaults for the settings `calc_selection_pre_process`
#: (`ai_diffusion/model/model.py`) reads — confirmed against this server's own
#: `/history` for a real krita job: a drawn mask with bounding-box diagonal
#: ~750px produced feather=75/grow=41/blend=25 there, which is exactly what
#: this formula reproduces for these constants at `strength=1.0` (a "remove
#: object" job is always full-strength, hence no `strength` parameter here).
_SELECTION_FEATHER_PCT = 10
_SELECTION_MIN_TRANSITION_PX = 32
_SELECTION_GROW_OFFSET_PX = 4
_SELECTION_BLEND_CAP_PX = 25


def _mask_bbox(mask_bytes: bytes) -> tuple[int, int, int, int] | None:
    """The drawn mask's own marked-pixel bounding box, or None if nothing is
    marked at all."""
    return Image.open(io.BytesIO(mask_bytes)).convert("L").getbbox()


def _bbox_diagonal(bbox: tuple[int, int, int, int] | None) -> float:
    if bbox is None:
        return 0.0
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5


def _native_fix_mask_params(diagonal: float) -> tuple[int, int, int]:
    """krita's `calc_selection_pre_process` formula, in the source image's own
    (un-downscaled) pixel units."""
    feather = max(int(_SELECTION_FEATHER_PCT / 100 * diagonal), _SELECTION_MIN_TRANSITION_PX)
    grow = _SELECTION_GROW_OFFSET_PX + feather // 2
    blend = min(_SELECTION_BLEND_CAP_PX, grow + feather // 2)
    return grow, feather, blend


#: Diffusion resolutions should be a multiple of this — same constant, and
#: the same reason ("required latent compression factor, or to avoid border
#: artifacts with UNET models"), as krita's own `resolution.diffusion_multiple`.
_DIFFUSION_MULTIPLE = 16
#: Extra padding around the drawn mask's bounding box for the hi-res refine
#: crop, on top of the mask's own grow+feather: krita's `selection_padding`
#: default (6%) of the bbox's longest side, never less than 16px.
_REFINE_PAD_FRAC = 0.06
_REFINE_MIN_PAD = 16
#: Smallest refine crop worth cropping to — krita's own checkpoint-resolution
#: floor for a diffusion pass is 256 too.
_REFINE_MIN_SIZE = 256
#: How much more resolution the second pass has to actually gain over what
#: the first pass already had for that region before it's worth running.
_REFINE_MIN_GAIN = 1.25


def _snap_down(value: int, multiple: int = _DIFFUSION_MULTIPLE) -> int:
    return max(multiple, (value // multiple) * multiple)


def _expand_axis(lo: int, hi: int, minimum: int, limit: int) -> tuple[int, int]:
    """Grow `(lo, hi)` to at least `minimum` long, staying inside `[0, limit]`."""
    minimum = min(minimum, limit)
    if hi - lo >= minimum:
        return lo, hi
    lo = max(0, lo - (minimum - (hi - lo)) // 2)
    hi = min(limit, lo + minimum)
    return max(0, hi - minimum), hi


def _snap_span(
    lo: int, hi: int, limit: int, multiple: int = _DIFFUSION_MULTIPLE
) -> tuple[int, int]:
    """`(lo, hi)` shrunk so its length is a multiple of `multiple`, anchored
    to whichever side already sits flush against the true image boundary
    (`lo == 0` or `hi == limit`) instead of always keeping `lo` fixed.

    `target`'s span used to be built as `(x0, _snap_down(x1 - x0))` —
    `_snap_down` floors the *length*, but `x0` never moved to compensate,
    so a crop that `_expand_axis` had deliberately pushed flush against the
    canvas edge (`x1 == full_w`, the drawn mask itself touching it) lost up
    to `multiple - 1` (15) pixels off that same edge the moment its length
    wasn't already a multiple of 16 — pulling the crop's own far edge back
    *inside* the true boundary. Confirmed live on a real "🩹 Fix Artifact"
    job (Anima pipeline — the only caller of this): the compositing mask
    built from `target`/`refine_region` (see `_build_anima_fix_drawn_mask`'s
    `full_crop`) came out a few pixels short of the source's own edge, so
    `ImageCompositeMasked` left that outermost strip as the *original,
    unfixed* source pixels instead of the refined patch — a mask painted
    solid to the true edge still showed old content there. Anchoring to
    `lo` (as before) is still correct whenever `lo == 0` and `hi < limit`
    (flush against the left/top edge); this only changes behavior when
    `hi == limit` and `lo` isn't already 0, which is exactly the
    right/bottom-edge case that lost pixels."""
    size = _snap_down(hi - lo, multiple)
    if lo <= 0:
        return 0, size
    if hi >= limit:
        return limit - size, limit
    return lo, lo + size


def _expand_about_centre(
    bounds: tuple[int, int, int, int], scale: float, limits: tuple[int, int]
) -> tuple[int, int, int, int]:
    """`bounds` grown by `scale` around its own centre, clamped inside
    `limits` and kept a multiple of `_DIFFUSION_MULTIPLE`."""
    x, y, w, h = bounds
    lim_w, lim_h = limits
    new_w = min(_snap_down(lim_w), _snap_down(round(w * scale)))
    new_h = min(_snap_down(lim_h), _snap_down(round(h * scale)))
    new_x = max(0, min(lim_w - new_w, x + w // 2 - new_w // 2))
    new_y = max(0, min(lim_h - new_h, y + h // 2 - new_h // 2))
    return (new_x, new_y, new_w, new_h)


def _scale_to_max(size: tuple[int, int], max_size: int) -> tuple[int, int]:
    """`size` fitted inside `max_size` on its longest side, preserving aspect
    ratio and snapped to `_DIFFUSION_MULTIPLE`. Never upscales."""
    w, h = size
    scale = min(1.0, max_size / max(w, h))
    return (_snap_down(round(w * scale)), _snap_down(round(h * scale)))


@dataclass
class FixDrawnGeometry:
    """Every resolution/region decision `build_fix_drawn_mask` needs for the
    Anima path, derived from the drawn mask itself.

    The load-bearing one is `context_crop`. Pass 1 used to downscale the
    *whole* source to `work_max_size`, which on a 4096x4096 image left the
    region actually being repaired occupying only ~200px of the pass — far
    too few to resolve what it was supposed to continue, so it produced
    something vague and no amount of second-pass refinement could recover
    detail that had never been decided. Measured against a real krita
    "remove object" job for the same edit: krita's first pass gave the
    region 504x536 of its 1024 pass (49% of the frame), this bot's gave it
    205x215 of 1280 (16%) — a 2.5x difference in the linear resolution of
    the only part that matters. krita gets that ratio because its canvas is
    only about twice the size of the region; this reproduces it directly by
    cropping a `DrawnMaskFixParams.context_scale` multiple of the region out
    of the source and running pass 1 on *that* instead of the whole frame.

    This is deliberately a partial reversal of an earlier finding that
    cropping "starves the model of scene context". That still holds for a
    *tight* crop around the mask — the version that painted a window through
    a garment cropped to barely more than the mask itself. A crop at twice
    the region's size is a different thing, and is the proportion krita
    itself works at."""

    #: Region of the source image pass 1 actually sees, as (x, y, w, h).
    context_crop: tuple[int, int, int, int]
    #: Resolution pass 1 samples `context_crop` at.
    work_size: tuple[int, int]
    #: Mask grow/blur/blend, in `work_size`'s pixel space.
    mask_grow: int
    mask_blur: int
    blend: int
    #: The hi-res second pass, as ((x, y, w, h) in the *source image's* own
    #: space, (width, height) to sample it at), or None to skip it — which
    #: is what happens when pass 1 already resolved the region well enough
    #: that a second pass couldn't beat it, the same condition krita's
    #: `ScaledExtent.refinement_scaling` reports as `none`.
    refine_region: tuple[tuple[int, int, int, int], tuple[int, int]] | None


def _fix_drawn_geometry(
    mask_bytes: bytes, image_size: tuple[int, int], params: DrawnMaskFixParams
) -> FixDrawnGeometry:
    """Work out `FixDrawnGeometry` for one "🩹 Fix Artifact" job."""
    full_w, full_h = image_size
    bbox = _mask_bbox(mask_bytes)
    grow_n, feather_n, blend_n = _native_fix_mask_params(_bbox_diagonal(bbox))

    if bbox is None:
        # Nothing marked — nothing to centre a crop on, so fall back to the
        # whole frame and skip the second pass.
        work_size = _scale_to_max(image_size, params.work_max_size)
        scale = work_size[0] / full_w if full_w else 1.0
        return FixDrawnGeometry(
            context_crop=(0, 0, full_w, full_h),
            work_size=work_size,
            mask_grow=max(0, round(grow_n * scale)),
            mask_blur=max(1, round(feather_n * scale)),
            blend=max(2, round(blend_n * scale)),
            refine_region=None,
        )

    # The region to repair: the mark plus its own grow/feather falloff plus
    # krita's selection padding. This is krita's `target_bounds`.
    x0, y0, x1, y1 = bbox
    pad = grow_n + feather_n + max(_REFINE_MIN_PAD, round(_REFINE_PAD_FRAC * max(x1 - x0, y1 - y0)))
    x0, x1 = _expand_axis(max(0, x0 - pad), min(full_w, x1 + pad), _REFINE_MIN_SIZE, full_w)
    y0, y1 = _expand_axis(max(0, y0 - pad), min(full_h, y1 + pad), _REFINE_MIN_SIZE, full_h)
    x0, x1 = _snap_span(x0, x1, full_w)
    y0, y1 = _snap_span(y0, y1, full_h)
    target = (x0, y0, x1 - x0, y1 - y0)

    context_crop = _expand_about_centre(target, params.context_scale, image_size)
    work_size = _scale_to_max((context_crop[2], context_crop[3]), params.work_max_size)
    scale = work_size[0] / context_crop[2]

    # What pass 1 gives the region, in its own pixel space. The second pass
    # is only worth running if it can beat that by a real margin.
    target_longest = max(target[2], target[3])
    pass1_longest = target_longest * scale
    refine_scale = min(
        1.0,
        params.refine_max_size / target_longest,
        pass1_longest * params.refine_max_upscale / target_longest,
    )
    refine_w = _snap_down(round(target[2] * refine_scale))
    refine_h = _snap_down(round(target[3] * refine_scale))
    refine_region = None
    if max(refine_w, refine_h) > pass1_longest * _REFINE_MIN_GAIN:
        refine_region = (target, (refine_w, refine_h))

    return FixDrawnGeometry(
        context_crop=context_crop,
        work_size=work_size,
        mask_grow=max(0, round(grow_n * scale)),
        mask_blur=max(1, round(feather_n * scale)),
        blend=max(2, round(blend_n * scale)),
        refine_region=refine_region,
    )


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


def _tag_images(
    raw: list[tuple[bytes, str]],
    params: GenerationParams,
    *,
    kind: str,
    graph: dict[str, Any],
) -> list[tuple[bytes, str]]:
    """Stamp each just-collected image with its own generation metadata.

    Done here, at the single point every image passes through on its way
    out of ComfyUI, rather than at send time in `handlers.py` — that way
    the >10MB document fallback, the inpaint poller's background sends and
    `scripts/smoke_test*.py` all get it without each remembering to. It has
    to happen per run rather than once per image file, too: a
    post-processing pass loads a previous image back into ComfyUI and
    `SaveImage` writes an entirely fresh PNG, so whatever chunk the source
    carried is already gone by the time the bytes come back here.

    `graph` is the API-format graph that was actually submitted, read only
    for the seed — `params.seed` is normally None, since the builders roll
    the real value at build time (see `png_metadata.extract_seed`).
    Embedding never raises; see `embed_metadata`.
    """
    seed = extract_seed(graph)
    serialized = serialize_generation_params(params)
    return [
        (
            embed_metadata(
                data,
                build_metadata(params=serialized, kind=kind, filename=name, seed=seed),
            ),
            name,
        )
        for data, name in raw
    ]


async def generate(
    client: ComfyClient,
    checkpoint: str,
    user_prompt: str,
    profile: ModelProfile | None,
    *,
    extra_negative_prompt: str = "",
    raw_positive_prompt: str = "",
    raw_negative_prompt: str = "",
    regions: list[RegionSpec] | None = None,
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
    `/settings` override. `regions` turns this into an `/rp` regional
    generation, with `user_prompt` as the global prompt (see
    `GenerationParams.regions`).
    """
    params = resolve_generation_params(
        checkpoint,
        user_prompt,
        profile,
        overrides=overrides,
        extra_negative_prompt=extra_negative_prompt,
        raw_positive_prompt=raw_positive_prompt,
        raw_negative_prompt=raw_negative_prompt,
        regions=regions,
    )
    prompt_graph, save_node_id = build_txt2img(params)
    logger.info(
        "Submitting txt2img: checkpoint=%s cfg=%s steps=%s regions=%s",
        checkpoint,
        params.cfg,
        params.steps,
        len(params.regions),
    )

    raw, _history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    raw = _tag_images(raw, params, kind="txt2img", graph=prompt_graph)
    return [GeneratedImage(data=data, filename=name, full_params=params) for data, name in raw]


def _fix_base_params(
    base_params: PostProcessBaseParams, profiles: list[ModelProfile] | None
) -> PostProcessBaseParams:
    """The checkpoint/loader half of `_build_fix_drawn_post_process`'s base
    choice — `_fix_artifact_override_base`'s profile if one matches, else
    the image's own — without the prompt rewrite, for callers that only
    need to know which pipeline "🩹 Fix Artifact" will route to."""
    override_base = _fix_artifact_override_base(profiles) if profiles else None
    return override_base if override_base is not None else base_params


def _fix_drawn_denoise_field(fix_base_params: PostProcessBaseParams) -> str:
    """Which `DrawnMaskFixParams` field a "🩹 Fix Artifact" denoise override
    lands on. The generic detailer path has one plain `denoise`; the Anima
    pipeline (`_build_anima_fix_drawn_mask`) runs its first pass at a fixed
    1.0 and has no such knob at all, so the closest equivalent there is the
    second pass's own `refine_denoise` strength."""
    return "refine_denoise" if fix_base_params.uses_anima_inpaint_pipeline else "denoise"


def resolve_drawn_mask_denoise(
    kind: Literal["hand_drawn", "fix_drawn"],
    full_params: GenerationParams,
    profiles: list[ModelProfile] | None,
    override: float | None = None,
) -> float:
    """The denoise a drawn-mask `post_process(kind=...)` call will actually
    run at given `override` (a "🎚️ Redo…" pick, or a "✏️ Detail Prompt"
    submission's own value) — mirrors `post_process`'s own fallback chain
    so handlers.py's picker can show the real current value rather than
    "default". `"hand_drawn"`: override, else the live profile's
    `detailer_denoise`, else `DrawnMaskHandDetailerParams`' own default.
    `"fix_drawn"`: override, else `DrawnMaskFixParams`' default for
    whichever field `_fix_drawn_denoise_field` picks."""
    if override is not None:
        return override
    base_params = _to_post_process_base(full_params)
    if kind == "hand_drawn":
        detailer_denoise = _refresh_detailer_tunables(base_params, profiles).detailer_denoise
        if detailer_denoise is not None:
            return detailer_denoise
        return DrawnMaskHandDetailerParams().denoise
    field = _fix_drawn_denoise_field(_fix_base_params(base_params, profiles))
    return getattr(DrawnMaskFixParams(), field)


def _build_fix_drawn_post_process(
    base_params: PostProcessBaseParams,
    uploaded_name: str,
    mask_upload_name: str,
    source_image: bytes,
    mask_bytes: bytes,
    profiles: list[ModelProfile] | None,
    denoise_override: float | None = None,
) -> tuple[dict, str]:
    """`post_process`'s `kind="fix_drawn"` graph-building branch, pulled out
    on its own since — unlike every other `kind`, which just delegates
    straight to a `build_*` function — this one has real work to do first:
    pick an override checkpoint (or fall back to the image's own), work out
    the positive-prompt fallback, and compute the Anima mask-grow/blur/blend
    geometry. Keeps `post_process` itself down to the same one-branch-per-kind
    shape every other `kind` uses. `denoise_override` is the "🎚️ Redo…"
    picker's value (see `_fix_drawn_denoise_field`). Returns
    `(prompt_graph, save_node_id)`."""
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
    elif base_params.uses_anima_inpaint_pipeline:
        # No fix_artifact_checkpoint override matched, but this image's
        # own checkpoint already has the Anima lllite patch configured
        # — build_fix_drawn_mask's own routing check (loader=="split"
        # and anima_lllite_inpaint_patch set) will still send this to
        # _build_anima_fix_drawn_mask, which documents that it expects a
        # non-blank positive prompt. A blank one here would silently
        # reproduce the "masked region came back almost untouched"
        # regression that whole prompt exists to avoid.
        fix_base_params = replace(base_params, positive_prompt="background scenery")
        checkpoint_source = "image's own checkpoint"
    else:
        fix_base_params = replace(base_params, positive_prompt="")
        checkpoint_source = "image's own checkpoint"
    fix_params = DrawnMaskFixParams()
    if denoise_override is not None:
        fix_params = replace(
            fix_params, **{_fix_drawn_denoise_field(fix_base_params): denoise_override}
        )
    if fix_base_params.uses_anima_inpaint_pipeline:
        engaged_reason = (
            f"patch={fix_base_params.anima_lllite_inpaint_patch!r} "
            f"strength={fix_base_params.anima_lllite_inpaint_patch_strength}"
        )
    elif fix_base_params.loader != "split":
        engaged_reason = f"skipped (loader={fix_base_params.loader!r}, not 'split')"
    else:
        engaged_reason = "skipped (anima_lllite_inpaint_patch not set on this profile)"
    image_size = Image.open(io.BytesIO(source_image)).size
    geometry = _fix_drawn_geometry(mask_bytes, image_size, fix_params)
    if fix_base_params.uses_anima_inpaint_pipeline:
        fix_params = replace(
            fix_params,
            mask_grow=geometry.mask_grow,
            mask_blur=geometry.mask_blur,
            blend=geometry.blend,
        )
    region_longest = max(geometry.context_crop[2], geometry.context_crop[3])
    logger.info(
        "Fix Artifact: checkpoint=%s (%s) loader=%s fill_model=%s anima_lllite_patch: %s "
        "positive_prompt=%r context=%s -> work_size=%s (image_size=%s) "
        "mask_grow=%s mask_blur=%s blend=%s refine=%s",
        fix_base_params.checkpoint,
        checkpoint_source,
        fix_base_params.loader,
        fix_params.fill_model,
        engaged_reason,
        fix_base_params.positive_prompt,
        geometry.context_crop,
        geometry.work_size,
        image_size,
        fix_params.mask_grow,
        fix_params.mask_blur,
        fix_params.blend,
        (
            f"crop={geometry.refine_region[0]} at {geometry.refine_region[1]} "
            f"strength={fix_params.refine_denoise}"
            if geometry.refine_region
            else f"skipped (pass 1 already resolves it at ~{region_longest}px)"
        ),
    )
    return build_fix_drawn_mask(
        uploaded_name,
        mask_upload_name,
        fix_base_params,
        fix_params,
        image_size=image_size,
        work_size=geometry.work_size,
        context_crop=geometry.context_crop,
        refine_region=geometry.refine_region,
    )


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
    detail_prompt: str | None = None,
    detail_negative_prompt: str | None = None,
    denoise: float | None = None,
    is_detail_prompt: bool = False,
    tile_controlnet_override: bool | None = None,
    detailer_disable_lora_override: bool | None = None,
    upscale_denoise_override: float | None = None,
    tile_controlnet_strength_override: float | None = None,
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
    rather than just a hand. `profiles`, if given, is consulted for four
    unrelated things depending on `kind`: `"fix_drawn"` — see
    `_fix_artifact_override_base` — runs that pass against a fixed,
    known-inpainting-aware checkpoint regardless of which one the image was
    originally generated with; `"upscale"`/`"homogenize"` instead re-resolve
    `tile_controlnet`/`tile_controlnet_strength`/`upscale_denoise` off the
    checkpoint's *current* profile (see `_refresh_tunable_defaults`) rather
    than the values frozen into the image at generation time, so editing a
    profile's JSON file is reflected on the very next tap instead of only
    on the next fresh generation; `"face"`/`"hand"`/`"hand_manual"`/
    `"hand_drawn"` (every `kind` in `_DETAILER_TUNABLE_KINDS`, regardless of
    `is_detail_prompt`) similarly re-resolve `ModelProfile.defaults.
    detailer_denoise`/`detailer_cfg`/`detailer_steps`/`detailer_disable_lora`
    (see `_refresh_detailer_tunables`) — a checkpoint whose detailer pass
    needs a much lower denoise, a cfg/steps that actually matches its own
    tuned generation values, or its LoRAs left out of the detailer pass
    entirely (`detailer_disable_lora`, since attaching any LoRA at all
    visibly degraded the detailer pass for it), than `FaceDetailerParams`/
    `HandDetailerParams`/`ManualHandDetailerParams`/
    `DrawnMaskHandDetailerParams`' own hard-coded
    defaults (surfaced by Banana Splitz XXL); `"hand_drawn"`
    *only when `is_detail_prompt` is also True* additionally re-resolves
    `tile_controlnet`/`tile_controlnet_strength`/`detail_prompt_tile_controlnet`
    (see `_refresh_detail_prompt_tile_controlnet`) for the same "edit the
    profile, see it on the next tap" reason. Every refresh above is a no-op
    whenever `profiles` is `None`/empty or the checkpoint no longer matches
    any shipped profile, falling back to the image's own checkpoint/values
    as recorded in `full_params`. `upscale_denoise_override`/`tile_controlnet_strength_override`
    are `kind="upscale"`'s own one-shot override, applied on top of whatever
    `profiles` already resolved — handlers.py's "⚙️ Customize" step on the
    "🔍 Upscale 4x" prompt (see `resolve_live_upscale_defaults`), for a
    single image where the live profile defaults aren't what's wanted
    without editing the profile's JSON file for every future image too.
    `None` (the default for each, independently) keeps whichever value
    `profiles` resolved for that one. Ignored for every other `kind`,
    including `"homogenize"` — its `TiledRefineParams` denoise is a fixed
    low-denoise seam-blend pass by design, with no customize step of its
    own, though `tile_controlnet_strength_override` would need extending to
    reach it if that ever changes.
    `detail_prompt`/`detail_negative_prompt`/`denoise` are a one-shot
    override for `kind="hand_drawn"` only (`DETAIL_PROMPT_KINDS` —
    handlers.py's "✏️ Detail Prompt" flow: a freehand mask plus a prompt
    describing that region, collected together in one webapp visit and run
    immediately, never saved for a later tap) — each independent, `None`
    keeping the image's own prompt / falling through to
    `base_params.detailer_denoise` (the checkpoint's own tuned default, if
    any — see `_refresh_detailer_tunables` above) and only then
    `DrawnMaskHandDetailerParams`' own hard-coded default denoise (0.5).
    `detailer_cfg`/`detailer_steps` have no equivalent one-shot field on
    "✏️ Detail Prompt" at all — they apply straight from the checkpoint's
    profile whenever set, for every `_DETAILER_TUNABLE_KINDS` kind alike.
    Every other kind ignores all three (save `denoise` for `"fix_drawn"`, below); in
    particular `kind="fix_drawn"` always uses its own fixed prompt choice
    (blank, or the `fix_artifact_checkpoint` profile's "background
    scenery") — "🩹 Fix Artifact" has no prompt field of its own to pass
    one from. `denoise` alone *does* reach `"fix_drawn"` too, for the
    "🎚️ Redo…" picker (see `_fix_drawn_denoise_field` for which field it
    lands on); `None` keeps `DrawnMaskFixParams`' own defaults.
    `is_detail_prompt` is a separate signal from all three of those — it's
    perfectly possible for "✏️ Detail Prompt" to be submitted with both
    fields left blank (falling back to the image's own prompt, same
    outcome as a plain "🖌️ Draw Mask" tap that never had prompt fields to
    begin with), so `detail_prompt`/`detail_negative_prompt` being None
    doesn't mean "this is a Draw Mask call". `handlers.py`'s
    `_DRAWN_MASK_KINDS` is what actually knows which of the three
    `storage.py` `inpaint_job.kind`s ("hand"/"fix"/"detail") a given call
    came from, and passes `is_detail_prompt=True` only for "detail" — the
    one and only thing it gates is whether `kind="hand_drawn"` even
    considers conditioning on `tile_controlnet` at all (see `ModelProfile.
    detail_prompt_tile_controlnet`, `build_hand_detailer_drawn_mask`'s
    `enable_tile_controlnet`). Ignored entirely for every other `kind`.
    `tile_controlnet_override` is the WebApp's own checkbox on "✏️ Detail
    Prompt" (unlike `detail_prompt`/`detail_negative_prompt`/`denoise`,
    which are free text/blank-means-unset, a checkbox is always
    definitely checked or unchecked, so this is `True`/`False` for every
    real submission and `None` only when there was never a checkbox to
    submit in the first place — a plain "🖌️ Draw Mask"/"🩹 Fix Artifact"
    job, or a `"detail"` job's stored `inpaint_redo` row from before this
    existed). When not `None` it wins outright over
    `base_params.detail_prompt_tile_controlnet`, whatever the checkpoint's
    *current* profile says — the whole point of a per-submission checkbox
    is letting one tap try it without needing to flip the profile's JSON
    field first. `None` falls back to `is_detail_prompt and
    base_params.detail_prompt_tile_controlnet` exactly as before this
    parameter existed. Ignored entirely for every `kind` other than
    `"hand_drawn"`.
    `detailer_disable_lora_override` is the WebApp's own "Disable LoRAs for
    this detailer pass" checkbox — same always-`True`/`False`-or-`None`
    shape as `tile_controlnet_override`, but *not* scoped to
    `is_detail_prompt`: the editor shows it for a plain "🖌️ Draw Mask" job
    too (see `handlers.py`'s `postprocess_callback`), since
    `detailer_disable_lora` matters for any detailer pass, not just the
    tile-ControlNet experiment "✏️ Detail Prompt" owns. When not `None` it
    wins outright over `base_params.detailer_disable_lora` — whatever the
    checkpoint's current profile says (`_refresh_detailer_tunables`) or
    whatever was frozen into the image at generation time — the same
    "one tap tries it without touching the profile's JSON file" reasoning
    as the tile-ControlNet checkbox. Reaches `_build_drawn_mask_detailer`/
    `_build_anima_fix_drawn_mask` via `_detailer_base` regardless of `kind`,
    but today only `"hand_drawn"` callers (`handlers.py`) ever supply a
    non-`None` value — "🩹 Fix Artifact" (`"fix_drawn"`) has no WebApp
    checkbox for it (see `handlers.py`'s `postprocess_callback`, which never
    builds one for that job kind)."""
    base_params = _to_post_process_base(full_params)
    if kind in DETAIL_PROMPT_KINDS:
        base_params = _apply_detail_prompt(base_params, detail_prompt, detail_negative_prompt)
    if kind in _TILED_UPSCALE_KINDS:
        base_params = _refresh_tunable_defaults(base_params, profiles)
    if kind in _DETAILER_TUNABLE_KINDS:
        base_params = _refresh_detailer_tunables(base_params, profiles)
    if kind == "hand_drawn" and is_detail_prompt:
        base_params = _refresh_detail_prompt_tile_controlnet(base_params, profiles)
    if detailer_disable_lora_override is not None:
        base_params = replace(base_params, detailer_disable_lora=detailer_disable_lora_override)
    if kind == "upscale" and (
        upscale_denoise_override is not None or tile_controlnet_strength_override is not None
    ):
        base_params = replace(
            base_params,
            upscale_denoise=(
                upscale_denoise_override
                if upscale_denoise_override is not None
                else base_params.upscale_denoise
            ),
            tile_controlnet_strength=(
                tile_controlnet_strength_override
                if tile_controlnet_strength_override is not None
                else base_params.tile_controlnet_strength
            ),
        )
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
        face_params = FaceDetailerParams()
        overrides = _detailer_cfg_steps_overrides(base_params)
        if base_params.detailer_denoise is not None:
            overrides["denoise"] = base_params.detailer_denoise
        face_params = replace(face_params, **overrides)
        prompt_graph, save_node_id, detection_node_id = build_face_detailer(
            uploaded_name, base_params, face_params
        )
    elif kind == "hand":
        hand_params = HandDetailerParams()
        overrides = _detailer_cfg_steps_overrides(base_params)
        if base_params.detailer_denoise is not None:
            overrides["denoise"] = base_params.detailer_denoise
        hand_params = replace(hand_params, **overrides)
        prompt_graph, save_node_id, detection_node_id = build_hand_detailer(
            uploaded_name, base_params, hand_params
        )
    elif kind == "hand_manual":
        assert point_frac is not None, "hand_manual requires point_frac"
        image_size = Image.open(io.BytesIO(source_image)).size
        manual_params = (
            ManualHandDetailerParams()
            if box_size_frac is None
            else ManualHandDetailerParams(box_size_frac=box_size_frac)
        )
        overrides = _detailer_cfg_steps_overrides(base_params)
        if base_params.detailer_denoise is not None:
            overrides["denoise"] = base_params.detailer_denoise
        manual_params = replace(manual_params, **overrides)
        prompt_graph, save_node_id = build_hand_detailer_manual(
            uploaded_name, base_params, manual_params, point_frac, image_size
        )
    elif kind == "hand_drawn":
        assert mask_bytes is not None, "hand_drawn requires mask_bytes"
        mask_upload = await client.upload_image(mask_bytes, filename=f"mask_{source_filename}")
        drawn_hand_params = DrawnMaskHandDetailerParams()
        overrides = _detailer_cfg_steps_overrides(base_params)
        effective_denoise = denoise if denoise is not None else base_params.detailer_denoise
        if effective_denoise is not None:
            overrides["denoise"] = effective_denoise
        drawn_hand_params = replace(drawn_hand_params, **overrides)
        if not is_detail_prompt:
            enable_tile_controlnet = False
        elif tile_controlnet_override is not None:
            enable_tile_controlnet = tile_controlnet_override
        else:
            enable_tile_controlnet = base_params.detail_prompt_tile_controlnet
        prompt_graph, save_node_id = build_hand_detailer_drawn_mask(
            uploaded_name,
            mask_upload["name"],
            base_params,
            drawn_hand_params,
            enable_tile_controlnet=enable_tile_controlnet,
        )
    elif kind == "fix_drawn":
        assert mask_bytes is not None, "fix_drawn requires mask_bytes"
        mask_upload = await client.upload_image(mask_bytes, filename=f"mask_{source_filename}")
        # Off the event loop: this decodes the source image and the mask
        # (both up to the original's full resolution) to work out the
        # inpaint geometry, real CPU work that would otherwise block every
        # other concurrently-scheduled update (see `concurrent_updates(True)`
        # in main.py) for its duration.
        prompt_graph, save_node_id = await asyncio.to_thread(
            _build_fix_drawn_post_process,
            base_params,
            uploaded_name,
            mask_upload["name"],
            source_image,
            mask_bytes,
            profiles,
            denoise,
        )
    else:
        raise ValueError(f"Unknown post-processing kind: {kind}")

    logger.info("Submitting post-process (%s) on %s", kind, uploaded_name)
    raw, history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    data, filename = _tag_images(raw, full_params, kind=kind, graph=prompt_graph)[0]
    unchanged = detection_node_id is not None and await _detailer_found_nothing(
        client, history, detection_node_id
    )
    return GeneratedImage(
        data=data, filename=filename, full_params=full_params, unchanged=unchanged
    )


def _scale_for_kontext(source_image: bytes) -> tuple[bytes, tuple[int, int]]:
    """Resize `source_image` to its nearest `KONTEXT_RESOLUTIONS` entry
    (Lanczos, then a centre crop of whatever sliver the aspect-ratio snap
    leaves over — the same thing `FluxKontextImageScale` does), returned as
    PNG bytes plus the resulting size.

    Done here rather than with that node in-graph for two reasons: the
    graph's empty latent needs the dimensions up front, and a 4x-upscaled
    source is ~16MP — shipping all of it to ComfyUI just to throw 15/16ths
    away on arrival is pointless upload time."""
    image = Image.open(io.BytesIO(source_image)).convert("RGB")
    target_w, target_h = kontext_resolution(image.size)
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue(), (target_w, target_h)


async def kontext_edit(
    client: ComfyClient,
    source_image: bytes,
    source_filename: str,
    full_params: GenerationParams,
    instruction: str,
    *,
    files: KontextModelFiles,
    params: KontextParams,
    on_progress: ProgressCallback | None = None,
) -> GeneratedImage:
    """Edit a previously generated image with FLUX.1 Kontext, following a
    natural-language `instruction` ("change the jacket to red leather, keep
    everything else the same").

    Deliberately not a `post_process` kind: every one of those rebuilds the
    image's *own* checkpoint/LoRA/prompt graph, and none of that applies
    here — Kontext is a separate model (`files`) conditioned only on the
    instruction and the source pixels. The result still carries the
    source's `full_params` though, so its own post-processing buttons
    (upscale, detailers) keep working against the original checkpoint.

    The output is always ~1MP (see `KONTEXT_RESOLUTIONS`), however large
    the source was — a 4x upscale comes back at roughly a quarter of its
    side length, to be upscaled again afterwards if wanted."""
    scaled, size = await asyncio.to_thread(_scale_for_kontext, source_image)
    upload = await client.upload_image(scaled, filename=f"kontext_{source_filename}")
    prompt_graph, save_node_id = build_kontext_edit(
        upload["name"], instruction, size, files, params
    )
    logger.info("Submitting Kontext edit on %s at %dx%d", upload["name"], *size)
    raw, _history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    data, filename = _tag_images(raw, full_params, kind="kontext", graph=prompt_graph)[0]
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

    raw, _history = await _run_graph(client, prompt_graph, save_node_id, on_progress=on_progress)
    raw = _tag_images(raw, fresh_params, kind="repeat", graph=prompt_graph)
    return [
        GeneratedImage(data=data, filename=name, full_params=fresh_params) for data, name in raw
    ]
