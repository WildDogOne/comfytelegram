"""Builds ComfyUI API-format prompt graphs in Python.

We don't use the `comfy` CLI's fragment/blueprint system at runtime — the
bot assembles a graph per user request (variable number of LoRAs, optional
post-processing stage) which is a natural fit for a small Python graph
builder rather than static template JSON with placeholders.

Node wiring and default parameter values here mirror the hand-built
`sample.json` workflow at the project root (see the main README's
"Background: the reference workflow" section for the node-by-node summary),
so a request built here should behave like a manual run of that workflow
once a checkpoint/LoRA/model-file name is substituted in.
One deliberate deviation: `sample.json` saves via WAS Suite's "Image Save"
node; these graphs use the core `SaveImage` node instead, so the bot only
depends on custom node packs that actually add capability (LoRA/upscale/
face-detail packs), not ones that just add convenience widgets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

NodeRef = tuple[str, int]


def _resolve_seed(seed: int | None) -> int:
    """`seed` unchanged if given, otherwise a fresh random one — shared by
    `GenerationParams.resolved_seed()` and the upscale/face-detail builders'
    own seed handling below."""
    return seed if seed is not None else random.randint(0, 2**32 - 1)


class PromptGraph:
    """A mutable ComfyUI API-format prompt being assembled node by node."""

    def __init__(self) -> None:
        """Start empty; node ids are assigned sequentially from `add()`."""
        self._nodes: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def add(self, class_type: str, inputs: dict[str, Any], *, title: str | None = None) -> str:
        """Append one node and return its freshly-assigned id, for wiring
        into later nodes' `inputs` as `[node_id, output_index]`."""
        node_id = str(self._next_id)
        self._next_id += 1
        node: dict[str, Any] = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self._nodes[node_id] = node
        return node_id

    def as_prompt(self) -> dict[str, Any]:
        """The assembled graph in ComfyUI's API format, ready for
        `ComfyClient.queue_prompt()`."""
        return self._nodes


@dataclass
class LoraSpec:
    """One LoRA to chain into the model/clip pipeline, and at what
    strength."""

    name: str
    strength_model: float = 1.0
    strength_clip: float = 1.0


@dataclass
class GenerationParams:
    """Everything needed to build the base txt2img graph for one request.

    Defaults (steps/cfg/sampler/clip_skip/...) are meant to be overridden by
    a resolved ModelProfile (see profiles.py) before building — these are
    just generic fallbacks for when no profile matches.
    """

    checkpoint: str
    positive_prompt: str
    negative_prompt: str
    seed: int | None = None
    steps: int = 30
    cfg: float = 7.0
    sampler_name: str = "euler"
    scheduler: str = "normal"
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    clip_skip: int = -1
    loras: list[LoraSpec] = field(default_factory=list)
    filename_prefix: str = "comfytelegram"

    def resolved_seed(self) -> int:
        """This request's seed, or a freshly-rolled random one if unset."""
        return _resolve_seed(self.seed)


def _apply_loras(
    g: PromptGraph, model_ref: NodeRef, clip_ref: NodeRef, loras: list[LoraSpec]
) -> tuple[NodeRef, NodeRef]:
    """Chain zero or more `LoraLoader` nodes onto `model_ref`/`clip_ref`, in
    list order. Returns the refs unchanged if `loras` is empty."""
    for lora in loras:
        node_id = g.add(
            "LoraLoader",
            {
                "model": list(model_ref),
                "clip": list(clip_ref),
                "lora_name": lora.name,
                "strength_model": lora.strength_model,
                "strength_clip": lora.strength_clip,
            },
            title=f"LoRA: {lora.name}",
        )
        model_ref = (node_id, 0)
        clip_ref = (node_id, 1)
    return model_ref, clip_ref


def build_txt2img(params: GenerationParams) -> tuple[dict[str, Any], str]:
    """Build the base generation graph. Returns (prompt_dict, save_image_node_id)."""
    g = PromptGraph()

    model_ref, _clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, params)

    latent = g.add(
        "EmptyLatentImage",
        {"width": params.width, "height": params.height, "batch_size": params.batch_size},
        title="Empty Latent",
    )

    sampler = g.add(
        "KSampler",
        {
            "model": list(model_ref),
            "positive": [positive, 0],
            "negative": [negative, 0],
            "latent_image": [latent, 0],
            "seed": params.resolved_seed(),
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            "denoise": 1.0,
        },
        title="KSampler",
    )

    decode = g.add("VAEDecode", {"samples": [sampler, 0], "vae": list(vae_ref)}, title="VAE Decode")

    save = g.add(
        "SaveImage",
        {"images": [decode, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )

    return g.as_prompt(), save


@dataclass
class PostProcessBaseParams:
    """Model/prompt context carried over from the original generation.

    Post-processing stages (upscale, face-detail) run as their own,
    freshly-submitted graph seeded from the previously saved image (see
    `LoadImage` usage below) rather than being chained onto the original
    KSampler run — that keeps them decoupled from batch/seed reproduction
    details and lets the user pick any single image out of a batch.
    They still need the same checkpoint/LoRAs/prompt/clip-skip as the
    original request to refine consistently, so the caller should retain
    the originating `GenerationParams` and pass its relevant fields here.
    """

    checkpoint: str
    positive_prompt: str
    negative_prompt: str
    loras: list[LoraSpec] = field(default_factory=list)
    clip_skip: int = -1


def _build_model_clip_vae(
    g: PromptGraph, base: GenerationParams | PostProcessBaseParams
) -> tuple[NodeRef, NodeRef, NodeRef, str, str]:
    """Shared checkpoint+LoRA+clip-skip+prompt wiring, used by `build_txt2img`
    and both post-processing graph builders below — `GenerationParams` and
    `PostProcessBaseParams` both carry the five fields this needs.

    Returns (model_ref, clip_ref, vae_ref, positive_node_id, negative_node_id).
    """
    ckpt = g.add("CheckpointLoaderSimple", {"ckpt_name": base.checkpoint}, title="Checkpoint")
    model_ref: NodeRef = (ckpt, 0)
    clip_ref: NodeRef = (ckpt, 1)
    vae_ref: NodeRef = (ckpt, 2)

    model_ref, clip_ref = _apply_loras(g, model_ref, clip_ref, base.loras)

    if base.clip_skip != -1:
        clip_skip_node = g.add(
            "CLIPSetLastLayer",
            {"clip": list(clip_ref), "stop_at_clip_layer": base.clip_skip},
            title="Clip Skip",
        )
        clip_ref = (clip_skip_node, 0)

    positive = g.add(
        "CLIPTextEncode",
        {"clip": list(clip_ref), "text": base.positive_prompt},
        title="Prompt Positive",
    )
    negative = g.add(
        "CLIPTextEncode",
        {"clip": list(clip_ref), "text": base.negative_prompt},
        title="Prompt Negative",
    )
    return model_ref, clip_ref, vae_ref, positive, negative


@dataclass
class UpscaleParams:
    """Defaults mirror sample.json node 585 (UltimateSDUpscale, enabled branch)."""

    upscale_model: str = "RealESRGAN_x4.pth"
    upscale_by: float = 4.0
    steps: int = 30
    cfg: float = 5.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.2
    seed: int | None = None
    tile_width: int = 2048
    tile_height: int = 2048
    mask_blur: int = 16
    tile_padding: int = 32
    tiled_decode: bool = True
    filename_prefix: str = "comfytelegram_upscaled"


def build_upscale(
    source_filename: str,
    base: PostProcessBaseParams,
    params: UpscaleParams,
) -> tuple[dict[str, Any], str]:
    """Build the 4x UltimateSDUpscale post-processing graph. Returns
    (prompt_dict, save_image_node_id). `source_filename` must already exist
    in ComfyUI's `input` directory (upload it first via
    `ComfyClient.upload_image`)."""
    g = PromptGraph()

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    model_ref, _clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)
    upscale_model = g.add(
        "UpscaleModelLoader", {"model_name": params.upscale_model}, title="Upscale Model"
    )

    upscale = g.add(
        "UltimateSDUpscale",
        {
            "image": [load, 0],
            "model": list(model_ref),
            "positive": [positive, 0],
            "negative": [negative, 0],
            "vae": list(vae_ref),
            "upscale_model": [upscale_model, 0],
            "upscale_by": params.upscale_by,
            "seed": _resolve_seed(params.seed),
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            "denoise": params.denoise,
            "mode_type": "Linear",
            "tile_width": params.tile_width,
            "tile_height": params.tile_height,
            "mask_blur": params.mask_blur,
            "tile_padding": params.tile_padding,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 1.0,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": True,
            "tiled_decode": params.tiled_decode,
            "batch_size": 1,
        },
        title="Ultimate SD Upscale",
    )

    save = g.add(
        "SaveImage",
        {"images": [upscale, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )
    return g.as_prompt(), save


@dataclass
class FaceDetailerParams:
    """Defaults mirror sample.json node 590 (FaceDetailer, enabled post-upscale pass)."""

    bbox_model: str = "bbox/FacesV1.pt"
    sam_model: str = "sam_vit_b_01ec64.pth"
    guide_size: int = 512
    max_size: int = 1024
    steps: int = 25
    cfg: float = 5.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.6
    feather: int = 10
    bbox_threshold: float = 0.5
    bbox_dilation: int = 20
    bbox_crop_factor: float = 3.0
    seed: int | None = None
    filename_prefix: str = "comfytelegram_face"


def build_face_detailer(
    source_filename: str,
    base: PostProcessBaseParams,
    params: FaceDetailerParams,
) -> tuple[dict[str, Any], str]:
    """Build the Impact Pack FaceDetailer post-processing graph. Returns
    (prompt_dict, save_image_node_id). `source_filename` must already exist
    in ComfyUI's `input` directory (upload it first via
    `ComfyClient.upload_image`)."""
    g = PromptGraph()

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    model_ref, clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)
    bbox_detector = g.add(
        "UltralyticsDetectorProvider", {"model_name": params.bbox_model}, title="Face Detector"
    )
    sam_model = g.add(
        "SAMLoader", {"model_name": params.sam_model, "device_mode": "AUTO"}, title="SAM Model"
    )

    detailer = g.add(
        "FaceDetailer",
        {
            "image": [load, 0],
            "model": list(model_ref),
            "clip": list(clip_ref),
            "vae": list(vae_ref),
            "positive": [positive, 0],
            "negative": [negative, 0],
            "bbox_detector": [bbox_detector, 0],
            "sam_model_opt": [sam_model, 0],
            "guide_size": params.guide_size,
            "guide_size_for": True,
            "max_size": params.max_size,
            "seed": _resolve_seed(params.seed),
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            "denoise": params.denoise,
            "feather": params.feather,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": params.bbox_threshold,
            "bbox_dilation": params.bbox_dilation,
            "bbox_crop_factor": params.bbox_crop_factor,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "wildcard": "",
            "cycle": 1,
            "inpaint_model": False,
            "noise_mask_feather": 20,
            "tiled_encode": False,
            "tiled_decode": False,
        },
        title="Face Detailer",
    )

    save = g.add(
        "SaveImage",
        {"images": [detailer, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )
    return g.as_prompt(), save
