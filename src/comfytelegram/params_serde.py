"""Plain-dict (de)serialization for `GenerationParams`.

Lives in its own module rather than in `handlers.py` (where it used to)
because it now has two consumers on opposite sides of the import graph:
`handlers.py` writes these dicts into `storage.py`'s `pending_result`/
`generation_snapshot` rows, and `generation.py` writes the same dict into
every generated PNG's metadata chunk (`png_metadata.py`) — and
`handlers` imports `generation`, never the reverse.

One dict shape serves both on purpose: an image imported from its own
embedded metadata can hand `params` straight to `store_pending_result`
with no translation step, so every post-processing button works on an
image the database has never seen. See `png_metadata` for why the
metadata is written by us rather than read back out of ComfyUI's graph.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Any

from comfytelegram.workflows import GenerationParams, LoraSpec, RegionSpec


def serialize_generation_params(params: GenerationParams) -> dict[str, Any]:
    """Flatten a `GenerationParams` into a plain JSON-able dict for
    `Storage` (deliberately drops `seed` and `filename_prefix` — see
    `test_serialization_omits_seed_so_regenerate_gets_a_fresh_roll`).
    Inverse of `deserialize_generation_params`."""
    return {
        "checkpoint": params.checkpoint,
        "positive_prompt": params.positive_prompt,
        "negative_prompt": params.negative_prompt,
        "steps": params.steps,
        "cfg": params.cfg,
        "sampler_name": params.sampler_name,
        "scheduler": params.scheduler,
        "width": params.width,
        "height": params.height,
        "batch_size": params.batch_size,
        "clip_skip": params.clip_skip,
        "loras": [
            {
                "name": lora.name,
                "strength_model": lora.strength_model,
                "strength_clip": lora.strength_clip,
            }
            for lora in params.loras
        ],
        "loader": params.loader,
        "clip_name": params.clip_name,
        "clip_type": params.clip_type,
        "vae_name": params.vae_name,
        "model_sampling_shift": params.model_sampling_shift,
        "tile_controlnet": params.tile_controlnet,
        "tile_controlnet_strength": params.tile_controlnet_strength,
        "detail_prompt_tile_controlnet": params.detail_prompt_tile_controlnet,
        "anima_lllite_inpaint_patch": params.anima_lllite_inpaint_patch,
        "anima_lllite_inpaint_patch_strength": params.anima_lllite_inpaint_patch_strength,
        "upscale_denoise": params.upscale_denoise,
        "detailer_denoise": params.detailer_denoise,
        "detailer_cfg": params.detailer_cfg,
        "detailer_steps": params.detailer_steps,
        "detailer_disable_lora": params.detailer_disable_lora,
        "raw_positive_prompt": params.raw_positive_prompt,
        "raw_negative_prompt": params.raw_negative_prompt,
        "regions": [_serialize_region(region) for region in params.regions],
        "regional_base_weight": params.regional_base_weight,
        "regional_region_weight": params.regional_region_weight,
        "regional_lora_denoise": params.regional_lora_denoise,
        "pose_png": (
            base64.b64encode(params.pose_png).decode("ascii")
            if params.pose_png is not None
            else None
        ),
        "pose_controlnet": params.pose_controlnet,
        "pose_controlnet_strength": params.pose_controlnet_strength,
        "pose_controlnet_end": params.pose_controlnet_end,
    }


def deserialize_generation_params(data: dict[str, Any]) -> GenerationParams:
    """`.get(..., <field default>)` on everything but checkpoint/prompts lets
    this still read pending_result rows written before the "🔁 Regenerate"
    button existed (when only the post-processing subset of fields was
    stored) — those rows just fall back to GenerationParams' own generic
    defaults for steps/cfg/etc. instead of the exact original values."""
    return GenerationParams(
        checkpoint=data["checkpoint"],
        positive_prompt=data["positive_prompt"],
        negative_prompt=data["negative_prompt"],
        steps=data.get("steps", 30),
        cfg=data.get("cfg", 7.0),
        sampler_name=data.get("sampler_name", "euler"),
        scheduler=data.get("scheduler", "normal"),
        width=data.get("width", 1024),
        height=data.get("height", 1024),
        batch_size=data.get("batch_size", 1),
        clip_skip=data.get("clip_skip", -1),
        loras=[LoraSpec(**lora) for lora in data.get("loras", [])],
        loader=data.get("loader", "checkpoint"),
        clip_name=data.get("clip_name", ""),
        clip_type=data.get("clip_type", "stable_diffusion"),
        vae_name=data.get("vae_name", ""),
        model_sampling_shift=data.get("model_sampling_shift"),
        tile_controlnet=data.get("tile_controlnet"),
        tile_controlnet_strength=data.get("tile_controlnet_strength", 0.4),
        detail_prompt_tile_controlnet=data.get("detail_prompt_tile_controlnet", False),
        anima_lllite_inpaint_patch=data.get("anima_lllite_inpaint_patch"),
        anima_lllite_inpaint_patch_strength=data.get("anima_lllite_inpaint_patch_strength", 1.0),
        upscale_denoise=data.get("upscale_denoise"),
        detailer_denoise=data.get("detailer_denoise"),
        detailer_cfg=data.get("detailer_cfg"),
        detailer_steps=data.get("detailer_steps"),
        detailer_disable_lora=data.get("detailer_disable_lora"),
        raw_positive_prompt=data.get("raw_positive_prompt", ""),
        raw_negative_prompt=data.get("raw_negative_prompt", ""),
        regions=[_deserialize_region(region) for region in data.get("regions", [])],
        regional_base_weight=data.get("regional_base_weight", 0.1),
        regional_region_weight=data.get("regional_region_weight", 0.9),
        regional_lora_denoise=data.get("regional_lora_denoise"),
        pose_png=base64.b64decode(data["pose_png"]) if data.get("pose_png") else None,
        pose_controlnet=data.get("pose_controlnet"),
        pose_controlnet_strength=data.get("pose_controlnet_strength", 0.8),
        pose_controlnet_end=data.get("pose_controlnet_end", 0.8),
    )


def _serialize_region(region: RegionSpec) -> dict[str, Any]:
    """`asdict`, with a painted mask's PNG bytes as base64 — the dict has to
    survive `json.dumps` on its way into SQLite and the PNG metadata."""
    data = asdict(region)
    if region.mask_png is not None:
        data["mask_png"] = base64.b64encode(region.mask_png).decode("ascii")
    return data


def _deserialize_region(data: dict[str, Any]) -> RegionSpec:
    """Inverse of `_serialize_region`; a box region from before painted
    masks existed has neither mask field and gets the defaults."""
    mask_png = data.get("mask_png")
    return RegionSpec(
        **{
            **data,
            "loras": [LoraSpec(**lora) for lora in data.get("loras", [])],
            "mask_png": base64.b64decode(mask_png) if mask_png is not None else None,
        }
    )
