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
from typing import Any, Literal

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
    #: "checkpoint" (default) loads `checkpoint` through a single
    #: `CheckpointLoaderSimple` node. "split" instead loads `checkpoint` as
    #: a `UNETLoader` filename plus `clip_name`/`vae_name` through their own
    #: loader nodes — the architecture models like Anima ship as (see
    #: `ModelProfile.loader` for where this gets set from a profile).
    loader: Literal["checkpoint", "split"] = "checkpoint"
    #: split-loader only: `CLIPLoader`'s filename and `type` value.
    clip_name: str = ""
    clip_type: str = "stable_diffusion"
    #: split-loader only: `VAELoader`'s filename.
    vae_name: str = ""
    #: split-loader only: shift for a `ModelSamplingAuraFlow` node inserted
    #: after the UNET load (Anima's AuraFlow-style sampling); `None` skips
    #: that node entirely for split architectures that don't need it.
    model_sampling_shift: float | None = None
    #: `build_upscale` only: a ControlNet Tile model filename (loaded via
    #: `ControlNetLoader`) to condition the UltimateSDUpscale pass on the
    #: pre-upscale source image. `UltimateSDUpscale` itself crops this hint
    #: per tile, so it adds detail across the whole image instead of just
    #: the face/hand-detailer regions, while staying structurally faithful
    #: to the source. `None` (the default) skips the branch entirely — an
    #: architecture fact like `loader`/`clip_name` above (a tile ControlNet
    #: trained for one base model won't work on another), so it rides the
    #: same profile-to-`GenerationParams`-to-`PostProcessBaseParams` path.
    tile_controlnet: str | None = None
    #: `ControlNetApplyAdvanced`'s `strength` for `tile_controlnet` above.
    tile_controlnet_strength: float = 0.4
    #: `build_fix_drawn_mask` only: split-loader (Anima) architecture fact,
    #: same reasoning as `tile_controlnet` above — a `weights` filename for
    #: `ETN_control_load`/`ETN_control_apply` (`comfyui-tooling-nodes`,
    #: Acly's own node pack — the same one `krita-ai-diffusion` itself uses
    #: for this exact weight format; ComfyUI core's superficially similar
    #: `ModelPatchLoader`+`AnimaLLLiteApply` is a *different*, unrelated
    #: mechanism that happens to also list this file in its dropdown
    #: without actually being able to interpret it — do not use those for
    #: this field), applied to the model before "🩹 Fix Artifact" samples
    #: so it actually knows how to reason about a masked hole instead of
    #: running plain noise-masked img2img on a text2img UNET. `None` (the
    #: default) skips the branch entirely.
    anima_lllite_inpaint_patch: str | None = None
    #: `ETN_control_apply`'s `strength` for `anima_lllite_inpaint_patch` above.
    anima_lllite_inpaint_patch_strength: float = 1.0
    #: `build_upscale` only: overrides `UpscaleParams.denoise` for this
    #: checkpoint's "🔍 Upscale 4x" pass. Unlike `tile_controlnet` above this
    #: is a tunable generation setting rather than an architecture fact, so
    #: it comes from a profile's `defaults` block (`ProfileDefaults.
    #: upscale_denoise`) instead of riding straight off the profile. `None`
    #: (the default) leaves `UpscaleParams.denoise`'s own default in place.
    upscale_denoise: float | None = None
    #: The exact text the user typed, before `resolve_generation_params`
    #: folded in the profile's `positive_prompt_prefix`/
    #: `negative_prompt_prefix` or an active character's saved prompt.
    #: Not used to build the graph (`positive_prompt`/`negative_prompt` are)
    #: — it just rides along for `handlers.py`'s "🐛 Show Prompt" to show a
    #: second, unprefixed view. Empty for pending_result rows serialized
    #: before this field existed.
    raw_positive_prompt: str = ""
    raw_negative_prompt: str = ""

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
    loader: Literal["checkpoint", "split"] = "checkpoint"
    clip_name: str = ""
    clip_type: str = "stable_diffusion"
    vae_name: str = ""
    model_sampling_shift: float | None = None
    tile_controlnet: str | None = None
    tile_controlnet_strength: float = 0.4
    anima_lllite_inpaint_patch: str | None = None
    anima_lllite_inpaint_patch_strength: float = 1.0
    upscale_denoise: float | None = None


def _build_model_clip_vae(
    g: PromptGraph, base: GenerationParams | PostProcessBaseParams
) -> tuple[NodeRef, NodeRef, NodeRef, str, str]:
    """Shared checkpoint+LoRA+clip-skip+prompt wiring, used by `build_txt2img`
    and both post-processing graph builders below — `GenerationParams` and
    `PostProcessBaseParams` both carry the fields this needs.

    `base.loader == "checkpoint"` (the default) loads everything through one
    `CheckpointLoaderSimple` node. `"split"` instead loads `base.checkpoint`
    as a `UNETLoader` filename plus `base.clip_name`/`base.vae_name` through
    their own loader nodes, then wraps the model in a `ModelSamplingAuraFlow`
    node if `base.model_sampling_shift` is set — the shape Anima-family
    models need (see `ModelProfile.loader`'s docstring for why).

    Returns (model_ref, clip_ref, vae_ref, positive_node_id, negative_node_id).
    """
    if base.loader == "split":
        unet = g.add(
            "UNETLoader",
            {"unet_name": base.checkpoint, "weight_dtype": "default"},
            title="UNET",
        )
        model_ref: NodeRef = (unet, 0)
        clip = g.add(
            "CLIPLoader",
            {"clip_name": base.clip_name, "type": base.clip_type, "device": "default"},
            title="Text Encoder",
        )
        clip_ref: NodeRef = (clip, 0)
        vae = g.add("VAELoader", {"vae_name": base.vae_name}, title="VAE")
        vae_ref: NodeRef = (vae, 0)
    else:
        ckpt = g.add("CheckpointLoaderSimple", {"ckpt_name": base.checkpoint}, title="Checkpoint")
        model_ref = (ckpt, 0)
        clip_ref = (ckpt, 1)
        vae_ref = (ckpt, 2)

    model_ref, clip_ref = _apply_loras(g, model_ref, clip_ref, base.loras)

    if base.clip_skip != -1:
        clip_skip_node = g.add(
            "CLIPSetLastLayer",
            {"clip": list(clip_ref), "stop_at_clip_layer": base.clip_skip},
            title="Clip Skip",
        )
        clip_ref = (clip_skip_node, 0)

    if base.loader == "split" and base.model_sampling_shift is not None:
        sampling_node = g.add(
            "ModelSamplingAuraFlow",
            {"model": list(model_ref), "shift": base.model_sampling_shift},
            title="Model Sampling (AuraFlow)",
        )
        model_ref = (sampling_node, 0)

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


@dataclass
class TiledRefineParams:
    """Same UltimateSDUpscale tiled-img2img shape as `UpscaleParams` (see
    `_build_tiled_upscale`), but `upscale_by` is fixed at 1.0: resolution
    stays unchanged. Backs "🧵 Homogenize" — for smoothing over the visibly
    different regions Face/Hand Detail passes leave behind (each patch has
    its own denoise/seed and only sees its own crop), by re-running the
    *whole* image through one consistent tiled pass instead of patching
    regions independently. Smaller default tile size than `UpscaleParams`
    too — at 1x, a 2048px tile would cover most or all of a typical
    1024-2048px generation, leaving nothing to actually tile across — and a
    lower default `denoise`, since the goal is blending existing content,
    not adding new detail the way the 4x upscale pass does."""

    upscale_model: str = "RealESRGAN_x4.pth"
    upscale_by: float = 1.0
    steps: int = 20
    cfg: float = 5.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.15
    seed: int | None = None
    tile_width: int = 768
    tile_height: int = 768
    mask_blur: int = 16
    tile_padding: int = 32
    tiled_decode: bool = True
    filename_prefix: str = "comfytelegram_homogenized"


def build_upscale(
    source_filename: str,
    base: PostProcessBaseParams,
    params: UpscaleParams,
) -> tuple[dict[str, Any], str]:
    """Build the 4x UltimateSDUpscale post-processing graph. Returns
    (prompt_dict, save_image_node_id). `source_filename` must already exist
    in ComfyUI's `input` directory (upload it first via
    `ComfyClient.upload_image`). See `_build_tiled_upscale` for the shared
    graph shape behind this and `build_tiled_refine`."""
    return _build_tiled_upscale(source_filename, base, params)


def build_tiled_refine(
    source_filename: str,
    base: PostProcessBaseParams,
    params: TiledRefineParams,
) -> tuple[dict[str, Any], str]:
    """Build the "🧵 Homogenize" post-processing graph — the same
    UltimateSDUpscale tiled-img2img pass as `build_upscale`, just run at
    `params.upscale_by=1.0` (see `TiledRefineParams`) so the output stays
    the same resolution as the source. See `_build_tiled_upscale`."""
    return _build_tiled_upscale(source_filename, base, params)


def _build_tiled_upscale(
    source_filename: str,
    base: PostProcessBaseParams,
    params: UpscaleParams | TiledRefineParams,
) -> tuple[dict[str, Any], str]:
    """Shared graph shape behind `build_upscale` and `build_tiled_refine` —
    an UltimateSDUpscale tiled img2img pass over the whole image, differing
    only in `params.upscale_by` (4x vs. 1x) and the rest of `params`'s
    tunables (see `TiledRefineParams`). Returns (prompt_dict,
    save_image_node_id). `source_filename` must already exist in ComfyUI's
    `input` directory (upload it first via `ComfyClient.upload_image`)."""
    g = PromptGraph()

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    model_ref, _clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)
    upscale_model = g.add(
        "UpscaleModelLoader", {"model_name": params.upscale_model}, title="Upscale Model"
    )

    positive_ref: list[Any] = [positive, 0]
    negative_ref: list[Any] = [negative, 0]
    if base.tile_controlnet:
        cn_loader = g.add(
            "ControlNetLoader", {"control_net_name": base.tile_controlnet}, title="ControlNet Tile"
        )
        cn_apply = g.add(
            "ControlNetApplyAdvanced",
            {
                "positive": positive_ref,
                "negative": negative_ref,
                "control_net": [cn_loader, 0],
                "image": [load, 0],
                "vae": list(vae_ref),
                "strength": base.tile_controlnet_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            title="Apply ControlNet Tile",
        )
        positive_ref = [cn_apply, 0]
        negative_ref = [cn_apply, 1]

    upscale = g.add(
        "UltimateSDUpscale",
        {
            "image": [load, 0],
            "model": list(model_ref),
            "positive": positive_ref,
            "negative": negative_ref,
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


@dataclass
class HandDetailerParams:
    """Same shape as `FaceDetailerParams`; defaults tuned for hands instead
    of faces — a hand-trained bbox model, and a higher denoise since extra-
    or missing-finger deformities need more correction latitude than face
    blemishes do, plus a wider crop factor since a hand's bounding box is
    usually smaller relative to the frame than a face's."""

    bbox_model: str = "bbox/hand_yolov8s.pt"
    sam_model: str = "sam_vit_b_01ec64.pth"
    guide_size: int = 512
    max_size: int = 1024
    steps: int = 25
    cfg: float = 5.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.5
    feather: int = 10
    bbox_threshold: float = 0.5
    bbox_dilation: int = 20
    bbox_crop_factor: float = 3.5
    seed: int | None = None
    filename_prefix: str = "comfytelegram_hand"


def _build_detailer(
    source_filename: str,
    base: PostProcessBaseParams,
    params: FaceDetailerParams | HandDetailerParams,
    *,
    label: str,
) -> tuple[dict[str, Any], str, str]:
    """Shared graph shape behind `build_face_detailer` and
    `build_hand_detailer` — Impact Pack's `FaceDetailer` node is really a
    generic detect-crop-inpaint-composite node despite the name, so hand-
    detailing is the same graph with a hand-trained `bbox_detector` model
    swapped in (see `HandDetailerParams`). `label` ("Face"/"Hand") only
    affects node titles shown in the ComfyUI UI, not graph behavior.
    Returns (prompt_dict, save_image_node_id, detection_preview_node_id).
    `source_filename` must already exist in ComfyUI's `input` directory
    (upload it first via `ComfyClient.upload_image`).

    The detailer's `mask` output (index 3) is the accumulated detection
    mask over the whole image — separate from the main composited `image`
    output (index 0), which the node always emits regardless of whether it
    actually detected anything (a silent no-op pass-through of the source
    when it didn't). Converting that mask to an image (`MaskToImage`) and
    routing it to its own `PreviewImage` gives `post_process()` a direct,
    ComfyUI-native "did this detect anything at all" signal — completely
    black means nothing was detected — instead of trying to infer it by
    diffing the final output's pixels against the source, which doesn't
    work reliably since ComfyUI's own float32 image round-trip (uint8 ->
    tensor -> uint8) can shift pixel values by a level or two even on a
    true pass-through. `PreviewImage` rather than `SaveImage` here on
    purpose — `/history`+`/view` only expose a node's output at all for an
    "output node" like either of these, but `PreviewImage` writes to
    ComfyUI's ephemeral `temp` folder instead of the permanent `output`
    one, so this internal detection check doesn't leave a stray extra file
    behind per face/hand post-process the way a second `SaveImage` would."""
    g = PromptGraph()

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    model_ref, clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)
    bbox_detector = g.add(
        "UltralyticsDetectorProvider",
        {"model_name": params.bbox_model},
        title=f"{label} Detector",
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
        title=f"{label} Detailer",
    )

    save = g.add(
        "SaveImage",
        {"images": [detailer, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )
    mask_image = g.add("MaskToImage", {"mask": [detailer, 3]}, title=f"{label} Detection Mask")
    detection_preview = g.add(
        "PreviewImage",
        {"images": [mask_image, 0]},
        title=f"{label} Detection Check (internal — not a result image)",
    )
    return g.as_prompt(), save, detection_preview


def build_face_detailer(
    source_filename: str,
    base: PostProcessBaseParams,
    params: FaceDetailerParams,
) -> tuple[dict[str, Any], str, str]:
    """Build the Impact Pack FaceDetailer post-processing graph, targeting
    faces. See `_build_detailer` for the shared graph shape."""
    return _build_detailer(source_filename, base, params, label="Face")


def build_hand_detailer(
    source_filename: str,
    base: PostProcessBaseParams,
    params: HandDetailerParams,
) -> tuple[dict[str, Any], str, str]:
    """Build the same Impact Pack detailer graph as `build_face_detailer`,
    but with a hand-trained bbox model — see `_build_detailer` and
    `HandDetailerParams`."""
    return _build_detailer(source_filename, base, params, label="Hand")


@dataclass
class ManualHandDetailerParams:
    """Same tunables as `HandDetailerParams` minus everything specific to
    YOLO/SAM auto-detection — used when the user taps a point instead of
    relying on the bbox detector (see `build_hand_detailer_manual`, backing
    "✋ Tap to mark")."""

    guide_size: int = 512
    max_size: int = 1024
    steps: int = 25
    cfg: float = 5.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.5
    feather: int = 10
    seed: int | None = None
    filename_prefix: str = "comfytelegram_hand"
    #: Side length of the manually-marked mask, as a fraction of the
    #: image's shorter edge — centered on the tapped point.
    box_size_frac: float = 0.35
    #: How far around the marked box `MaskToSEGS` widens the crop for
    #: guide-size resizing — same role as `HandDetailerParams.bbox_crop_factor`.
    crop_factor: float = 3.0


def build_hand_detailer_manual(
    source_filename: str,
    base: PostProcessBaseParams,
    params: ManualHandDetailerParams,
    point_frac: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    """Build a hand-detail graph that skips YOLO/SAM detection entirely and
    instead inpaints a small rectangular mask centered on a user-tapped
    point — for when the auto-detector in `build_hand_detailer` can't find
    the hand at all (see `handlers.py`'s "✋ Tap to mark" flow).

    `point_frac` is the tap location as (x_fraction, y_fraction) of the
    image; `image_size` is that image's actual (width, height) in pixels —
    the caller resolves both since ComfyUI's `SolidMask` needs concrete
    pixel dimensions, not a dynamic size derived from the loaded image.

    The mask itself is built from core nodes only (`SolidMask` +
    `MaskComposite`, "add" a small filled square onto an all-zero canvas at
    the clamped tap offset), then handed to Impact Pack's `MaskToSEGS` to
    become a `SEGS` region and `DetailerForEach` — the modular sibling of
    `FaceDetailer`/`_build_detailer`'s node that inpaints a given `SEGS`
    directly instead of running its own bbox detection. No detection-mask
    preview node is needed here (unlike `_build_detailer`) since a manually
    placed mask is never empty, so this returns just
    (prompt_dict, save_image_node_id), matching `build_upscale`'s shape.
    `source_filename` must already exist in ComfyUI's `input` directory
    (upload it first via `ComfyClient.upload_image`)."""
    g = PromptGraph()

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    model_ref, clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)

    width, height = image_size
    box_size = max(1, int(min(width, height) * params.box_size_frac))
    x_center = point_frac[0] * width
    y_center = point_frac[1] * height
    box_x = int(min(max(x_center - box_size / 2, 0), width - box_size))
    box_y = int(min(max(y_center - box_size / 2, 0), height - box_size))

    base_mask = g.add(
        "SolidMask", {"value": 0.0, "width": width, "height": height}, title="Blank Mask"
    )
    patch_mask = g.add(
        "SolidMask", {"value": 1.0, "width": box_size, "height": box_size}, title="Marked Patch"
    )
    marked_mask = g.add(
        "MaskComposite",
        {
            "destination": [base_mask, 0],
            "source": [patch_mask, 0],
            "x": box_x,
            "y": box_y,
            "operation": "add",
        },
        title="Marked Region",
    )
    segs = g.add(
        "MaskToSEGS",
        {
            "mask": [marked_mask, 0],
            "combined": False,
            "crop_factor": params.crop_factor,
            "bbox_fill": False,
            "drop_size": 10,
            "contour_fill": False,
        },
        title="Marked Region SEGS",
    )

    detailer = g.add(
        "DetailerForEach",
        {
            "image": [load, 0],
            "segs": [segs, 0],
            "model": list(model_ref),
            "clip": list(clip_ref),
            "vae": list(vae_ref),
            "positive": [positive, 0],
            "negative": [negative, 0],
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
            "wildcard": "",
            "cycle": 1,
            "inpaint_model": False,
            "noise_mask_feather": 20,
        },
        title="Hand Detailer (manual)",
    )

    save = g.add(
        "SaveImage",
        {"images": [detailer, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )
    return g.as_prompt(), save


@dataclass
class DrawnMaskHandDetailerParams:
    """Same tunables as `ManualHandDetailerParams` minus the tapped-point/box
    geometry — there's no box to size or center here, since the mask comes
    from a freehand drawing instead (see `build_hand_detailer_drawn_mask`,
    backing the Telegram WebApp mask editor)."""

    guide_size: int = 512
    max_size: int = 1024
    steps: int = 25
    cfg: float = 5.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.5
    feather: int = 10
    seed: int | None = None
    filename_prefix: str = "comfytelegram_hand"
    #: Same role as `ManualHandDetailerParams.crop_factor`.
    crop_factor: float = 3.0


@dataclass
class DrawnMaskFixParams:
    """Same shape as `DrawnMaskHandDetailerParams`, for the general-purpose
    "🩹 Fix Artifact" drawn-mask flow (`build_fix_drawn_mask`) instead of a
    hand specifically — see `_build_drawn_mask_detailer`. A higher default
    `denoise` than the hand variant: touching up a hand only needs to
    correct finger shape/count within content that's already roughly right,
    while removing an arbitrary unwanted artifact (a stray object, a
    watermark, a background glitch) needs the model free to replace the
    masked region's content much more substantially. A wider `crop_factor`
    too, so the detailer's guide crop pulls in more surrounding context to
    blend the replacement against — useful for a lot more than the small,
    already-hand-shaped region a hand crop needs. `steps`/`cfg`/`scheduler`
    match a real `krita-ai-diffusion` "remove object" job's own values,
    captured from this server's `/history` for an actual krita-submitted
    job — see `_build_anima_fix_drawn_mask`."""

    guide_size: int = 512
    max_size: int = 1024
    steps: int = 30
    cfg: float = 4.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    denoise: float = 0.75
    feather: int = 10
    seed: int | None = None
    filename_prefix: str = "comfytelegram_fix"
    crop_factor: float = 4.0
    #: A dedicated (non-diffusion) content-aware fill model — MAT, trained on
    #: Places2 for exactly this "erase and continue the background" task —
    #: run over the masked region *before* the diffusion pass, the same
    #: `krita-ai-diffusion`'s "remove object" mode does via its
    #: `ComfyUI-Inpaint-Nodes` custom pack. Structurally replaces the masked
    #: pixels with plausible background texture up front, so the diffusion
    #: pass that follows has nothing left describing the old object's shape
    #: to preserve/embellish — without it, `DetailerForEach` was denoising
    #: from the *original* pixels under an (empty) text prompt alone, which
    #: often reconstructed a cleaner-looking version of the very thing being
    #: masked out instead of erasing it.
    fill_model: str = "MAT_Places512_G_fp16.safetensors"
    #: `_build_anima_fix_drawn_mask` only — all four below mirror a real
    #: krita job's own values exactly (captured from /history, not guessed).
    #: The drawn mask is grown+blurred (`INPAINT_ExpandMask`) then stabilized
    #: (`INPAINT_StabilizeMask`) into the "main" mask used for the ControlNet
    #: condition, the sampler's noise mask, and the post-sample color-match
    #: exclusion — a raw, unprocessed mask left a visible seam at its edges.
    mask_grow: int = 25
    mask_blur: int = 42
    mask_stabilize_epsilon: float = 0.01
    #: A *separate*, much lighter grow (no blur) of the raw mask, used only
    #: for the content-aware fill step — krita keeps this one tight so MAT
    #: doesn't get asked to erase more than necessary.
    fill_mask_grow: int = 4
    #: `_build_anima_fix_drawn_mask` only — the longest side (in px) the
    #: *entire* source image is downscaled to before sampling, matching
    #: krita's own `scale_to_initial`/`ScaledExtent` (its working resolution
    #: is decoupled from the canvas's native one; ours wasn't, and a first
    #: version of this pipeline sampled a raw, un-cropped 4096x4096 source
    #: directly, which was clocked running full 30-step diffusion over it).
    #: `generation.post_process` computes the actual `(work_w, work_h)` from
    #: this and the real image size, preserving aspect ratio and never
    #: upscaling past the source's own resolution. An *earlier* version of
    #: this fix cropped to a region around the mask instead of downscaling
    #: the whole image — cheaper, but wrong: it throws away the surrounding
    #: scene context the model otherwise uses to infer "this is fabric,
    #: keep it fabric" for the masked hole. Confirmed on a real removal
    #: case pulled from this server's `/history`: cropped to a small region
    #: around the mask, the diffusion pass painted a literal window showing
    #: background *through* the hoodie; sampling the whole (downscaled)
    #: image instead, same mask, same seed, it didn't.
    work_max_size: int = 1280


def build_hand_detailer_drawn_mask(
    source_filename: str,
    mask_filename: str,
    base: PostProcessBaseParams,
    params: DrawnMaskHandDetailerParams,
) -> tuple[dict[str, Any], str]:
    """Build a hand-detail graph from a mask the user drew freehand (the
    Telegram WebApp mask editor), instead of YOLO/SAM detection
    (`build_hand_detailer`) or a fixed box around a tapped point
    (`build_hand_detailer_manual`). See `_build_drawn_mask_detailer` for the
    shared graph shape behind this and `build_fix_drawn_mask`."""
    return _build_drawn_mask_detailer(source_filename, mask_filename, base, params, label="Hand")


def build_fix_drawn_mask(
    source_filename: str,
    mask_filename: str,
    base: PostProcessBaseParams,
    params: DrawnMaskFixParams,
    *,
    image_size: tuple[int, int],
    work_size: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    """Build a general-purpose "🩹 Fix Artifact" graph from a freehand-drawn
    mask. `image_size` is the real source image's own (width, height);
    `work_size` (computed by `generation.post_process` from `image_size` and
    `DrawnMaskFixParams.work_max_size`, preserving aspect ratio) is the
    resolution `_build_anima_fix_drawn_mask` actually downscales the *whole*
    source image to before sampling, upscaling back to `image_size` before
    the final composite. Both ignored by the `_build_drawn_mask_detailer`
    fallback below, which does its own cropping via `MaskToSEGS`'s
    `crop_factor` instead of a resize.

    When `base` is a split-loader (Anima) checkpoint with
    `anima_lllite_inpaint_patch` set, delegates to
    `_build_anima_fix_drawn_mask` — a hand-built pipeline that closely
    mirrors `krita-ai-diffusion`'s own real graph for this exact task
    (captured from this server's `/history` for an actual krita job, not
    inferred), since Impact Pack's `DetailerForEach` turned out to diverge
    from it in ways that mattered for quality: no "background scenery"
    positive-prompt hint, no mask grow/stabilize, no post-sample color
    match to correct drift between the refined patch and its surroundings,
    and — the one that mattered most — sampling a small crop around the
    mask instead of the whole (downscaled) scene, which starved the model
    of the surrounding context it needs to know what to continue.

    Otherwise (no inpainting-aware path for this checkpoint at all) falls
    back to the same graph shape as `build_hand_detailer_drawn_mask` plus a
    `DrawnMaskFixParams.fill_model` content-aware fill pass — see
    `_build_drawn_mask_detailer`'s `content_aware_fill`."""
    if base.loader == "split" and base.anima_lllite_inpaint_patch:
        return _build_anima_fix_drawn_mask(
            source_filename, mask_filename, base, params, image_size, work_size
        )
    return _build_drawn_mask_detailer(
        source_filename, mask_filename, base, params, label="Fix", content_aware_fill=True
    )


def _build_anima_fix_drawn_mask(
    source_filename: str,
    mask_filename: str,
    base: PostProcessBaseParams,
    params: DrawnMaskFixParams,
    image_size: tuple[int, int],
    work_size: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    """ "🩹 Fix Artifact" for a split-loader (Anima) checkpoint with
    `anima_lllite_inpaint_patch` set — a hand-built pipeline instead of
    `DetailerForEach`, because krita-ai-diffusion's own real graph for this
    exact task (pulled from this server's `/history` for an actual job
    krita submitted, side by side with this bot's own attempt on the same
    source image and mask) turned out not to use `DetailerForEach`/
    `MaskToSEGS` at all, and does several things this bot's
    `DetailerForEach`-based attempt didn't:

    - Its positive prompt was never blank — `base.positive_prompt` is
      expected to already carry `<profile prefix>, background scenery`
      (see `generation._fix_artifact_override_base`) rather than "".
    - The drawn mask is grown, blurred, and stabilized
      (`INPAINT_ExpandMask` → `INPAINT_StabilizeMask`) into a "main" mask
      used for the ControlNet condition, the sampler's noise mask, and the
      post-sample color-match exclusion — not the raw mask directly.
    - A second, separately (lightly) grown mask feeds just the
      content-aware fill step.
    - `ETN_control_apply`'s `image` is the *raw* source, not the
      content-aware-filled one — an earlier attempt fed it the filled image
      on the theory that the raw one would leak the removed object's pixels
      into the ControlNet's structural conditioning; krita's own real graph
      feeds it the raw source, so the `anima-lllite-inpainting-v2.safetensors`
      weights' own metadata must already handle that internally — matched
      here now that there's a verified reference to check against.
    - `INPAINT_ColorMatch` runs on the decoded result before compositing,
      matched against the content-aware-filled image and excluding the
      main mask — corrects color/exposure drift between the freshly
      sampled patch and its surroundings that `DetailerForEach` never did,
      which was showing up as a visibly mismatched, muddy patch.
    - It samples over the *whole* image, not a small crop around the mask.
      This turned out to be the one that mattered most: an earlier version
      of this function cropped to a region around the mask instead (cheaper
      than a full-image resize, and DetailerForEach's own crop_factor
      approach was the obvious template to reach for) — confirmed wrong on
      a real removal case pulled from this server's own `/history`: cropped
      to a small region, the diffusion pass painted a literal window
      showing background *through* a hoodie the mask sat inside of, no
      matter the seed, checkpoint, or prompt wording tried. Same mask, same
      seed, sampling the *whole* (downscaled) image instead, it didn't —
      the surrounding scene apparently gives the model the context it
      needs to know "this is fabric, keep it fabric" for the masked hole,
      context a tight crop had thrown away. Matches krita's own real
      behavior: it samples its entire working canvas too, which it can
      afford because that working resolution is decoupled from the source
      image's native one (`ScaledExtent`/`scale_to_initial` in its
      `workflow.py`) — this bot has no such decoupling built in elsewhere,
      so `work_size` (computed by `generation.post_process` from the real
      source image's dimensions and `DrawnMaskFixParams.work_max_size`,
      preserving aspect ratio and never upscaling past the source's own
      resolution) downscales the *whole* image to a sane working resolution
      here instead, the same role krita's own scaling plays — a genuine
      resize, not a crop, so no scene context is lost. (A first version of
      this function had no resize *or* crop at all, and was clocked running
      a full 30-step pass over a raw, un-cropped 4096x4096 source — also
      wrong, just for runtime instead of quality.)

    krita's own graph ends by handing a small patch + mask back to its
    Krita-side client, which pastes it onto the canvas layer itself — this
    bot has no such client, so this graph instead scales the color-matched
    result and the main mask back up to the source's own resolution and
    composites onto the *original* full-resolution source with
    `ImageCompositeMasked`, guaranteeing pixels outside the mask stay
    identical to the source regardless of any drift introduced by
    downscaling and re-upscaling.

    Still not guaranteed to cleanly erase — this weight file behaves like a
    general "fill this hole with something plausible" model rather than a
    specialized "erase to background" one — but this is now a faithful
    match to what krita-ai-diffusion itself actually submits, not a
    from-scratch approximation."""
    g = PromptGraph()
    work_w, work_h = work_size
    full_w, full_h = image_size

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    raw_mask_full = g.add(
        "LoadImageMask", {"image": mask_filename, "channel": "red"}, title="Drawn Mask"
    )
    load_small = g.add(
        "ImageScale",
        {
            "image": [load, 0],
            "width": work_w,
            "height": work_h,
            "upscale_method": "lanczos",
            "crop": "disabled",
        },
        title="Downscale Source To Working Resolution",
    )
    raw_mask_full_img = g.add(
        "MaskToImage", {"mask": [raw_mask_full, 0]}, title="Drawn Mask As Image"
    )
    raw_mask_small_img = g.add(
        "ImageScale",
        {
            "image": [raw_mask_full_img, 0],
            "width": work_w,
            "height": work_h,
            "upscale_method": "bilinear",
            "crop": "disabled",
        },
        title="Downscale Mask To Working Resolution",
    )
    raw_mask = g.add(
        "ImageToMask", {"image": [raw_mask_small_img, 0], "channel": "red"}, title="Drawn Mask"
    )

    model_ref, _clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)
    diff_diffusion = g.add(
        "DifferentialDiffusion", {"model": list(model_ref)}, title="Differential Diffusion"
    )
    model_ref = (diff_diffusion, 0)

    grown_mask = g.add(
        "INPAINT_ExpandMask",
        {
            "mask": [raw_mask, 0],
            "grow": params.mask_grow,
            "blur": params.mask_blur,
            "blur_type": "linear",
        },
        title="Grow + Blur Mask",
    )
    main_mask = g.add(
        "INPAINT_StabilizeMask",
        {"mask": [grown_mask, 0], "epsilon": params.mask_stabilize_epsilon},
        title="Stabilize Mask",
    )
    fill_mask = g.add(
        "INPAINT_ExpandMask",
        {"mask": [raw_mask, 0], "grow": params.fill_mask_grow, "blur": 0, "blur_type": "gaussian"},
        title="Fill Mask",
    )

    fill_model = g.add(
        "INPAINT_LoadInpaintModel",
        {"model_name": params.fill_model},
        title="Content-Aware Fill Model",
    )
    seed = _resolve_seed(params.seed)
    prefilled = g.add(
        "INPAINT_InpaintWithModel",
        {
            "inpaint_model": [fill_model, 0],
            "image": [load_small, 0],
            "mask": [fill_mask, 0],
            "seed": seed,
        },
        title="Content-Aware Fill",
    )

    control_load = g.add(
        "ETN_control_load",
        {"model": list(model_ref), "weights": base.anima_lllite_inpaint_patch},
        title="Load Anima Inpaint ControlNet (LLLite)",
    )
    patched_model = g.add(
        "ETN_control_apply",
        {
            "model": [control_load, 0],
            "control_net": [control_load, 1],
            "image": [load_small, 0],
            "strength": base.anima_lllite_inpaint_patch_strength,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "mask": [main_mask, 0],
        },
        title="Apply Anima Inpaint ControlNet (LLLite)",
    )

    latent = g.add(
        "VAEEncode", {"vae": list(vae_ref), "pixels": [prefilled, 0]}, title="VAE Encode"
    )
    masked_latent = g.add(
        "SetLatentNoiseMask",
        {"samples": [latent, 0], "mask": [main_mask, 0]},
        title="Set Latent Noise Mask",
    )

    guider = g.add(
        "CFGGuider",
        {
            "model": [patched_model, 0],
            "positive": [positive, 0],
            "negative": [negative, 0],
            "cfg": params.cfg,
        },
        title="CFG Guider",
    )
    scheduler = g.add(
        "BasicScheduler",
        {
            "model": [patched_model, 0],
            "scheduler": params.scheduler,
            "steps": params.steps,
            "denoise": 1.0,
        },
        title="Scheduler",
    )
    noise = g.add("RandomNoise", {"noise_seed": seed}, title="Random Noise")
    sampler_select = g.add("KSamplerSelect", {"sampler_name": params.sampler_name}, title="Sampler")
    sampled = g.add(
        "SamplerCustomAdvanced",
        {
            "noise": [noise, 0],
            "guider": [guider, 0],
            "sampler": [sampler_select, 0],
            "sigmas": [scheduler, 0],
            "latent_image": [masked_latent, 0],
        },
        title="Sample",
    )
    decoded = g.add(
        "VAEDecode", {"vae": list(vae_ref), "samples": [sampled, 1]}, title="VAE Decode"
    )

    color_matched = g.add(
        "INPAINT_ColorMatch",
        {
            "target": [decoded, 0],
            "reference": [prefilled, 0],
            "exclude_mask": [main_mask, 0],
            "strength": 1.0,
        },
        title="Color Match",
    )

    # Scale the small refined result, and the mask it's composited through,
    # back up to the source's own resolution — the one place besides `load`
    # itself that this graph touches full resolution again. Explicit
    # width/height (not ImageCompositeMasked's own resize_source) so the
    # mask is guaranteed to match source's resolution too, not just the
    # image.
    result_full = g.add(
        "ImageScale",
        {
            "image": [color_matched, 0],
            "width": full_w,
            "height": full_h,
            "upscale_method": "lanczos",
            "crop": "disabled",
        },
        title="Upscale Result To Source Resolution",
    )
    main_mask_img = g.add("MaskToImage", {"mask": [main_mask, 0]}, title="Main Mask As Image")
    main_mask_full_img = g.add(
        "ImageScale",
        {
            "image": [main_mask_img, 0],
            "width": full_w,
            "height": full_h,
            "upscale_method": "bilinear",
            "crop": "disabled",
        },
        title="Upscale Main Mask To Source Resolution",
    )
    main_mask_full = g.add(
        "ImageToMask", {"image": [main_mask_full_img, 0], "channel": "red"}, title="Main Mask"
    )
    composited = g.add(
        "ImageCompositeMasked",
        {
            "destination": [load, 0],
            "source": [result_full, 0],
            "x": 0,
            "y": 0,
            "resize_source": False,
            "mask": [main_mask_full, 0],
        },
        title="Composite Onto Source",
    )

    save = g.add(
        "SaveImage",
        {"images": [composited, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )
    return g.as_prompt(), save


def _build_drawn_mask_detailer(
    source_filename: str,
    mask_filename: str,
    base: PostProcessBaseParams,
    params: DrawnMaskHandDetailerParams | DrawnMaskFixParams,
    *,
    label: str,
    content_aware_fill: bool = False,
) -> tuple[dict[str, Any], str]:
    """Shared graph shape behind `build_hand_detailer_drawn_mask` and
    `build_fix_drawn_mask` — both `source_filename` and `mask_filename` must
    already exist in ComfyUI's `input` directory (upload both first via
    `ComfyClient.upload_image`); the mask is a plain grayscale PNG (white =
    inpaint, black = keep), read via the core `LoadImageMask` node's red
    channel. `label` ("Hand"/"Fix") only affects node titles shown in the
    ComfyUI UI, not graph behavior.

    `content_aware_fill` (only set by `build_fix_drawn_mask` — `params` must
    be a `DrawnMaskFixParams` when it's True) runs `INPAINT_LoadInpaintModel`
    + `INPAINT_InpaintWithModel` (ComfyUI-Inpaint-Nodes' MAT/LaMa-family
    models — non-diffusion, trained specifically on content-aware "erase and
    continue the background") over the masked region first, and feeds
    *that* image into `DetailerForEach` instead of the raw source. Skipped
    for the hand-detail variant, where the masked region should be corrected
    against the subject prompt, not structurally erased first.

    Within that same `content_aware_fill` branch, `base.anima_lllite_inpaint_patch`
    (set — currently only by the Anima profiles) additionally patches
    `model_ref` via `ETN_control_load`+`ETN_control_apply`
    (`comfyui-tooling-nodes` — the same node pack `krita-ai-diffusion` itself
    uses for this exact weight format, per its `apply_controlnet_lllite`;
    *not* ComfyUI core's `ModelPatchLoader`+`AnimaLLLiteApply`, which is an
    unrelated, superficially similar mechanism that also lists this file in
    its dropdown without actually being able to interpret it correctly —
    confirmed by a first pass wired against those core nodes producing very
    poor results) before it reaches `DetailerForEach`, so the diffusion
    refine pass that follows the MAT/LaMa prefill is itself inpainting-aware
    (conditioned on the drawn mask directly) instead of plain noise-masked
    img2img on a checkpoint that's never seen a masked hole. Its `image`
    input is `detailer_image` (the MAT/LaMa-filled image), not the raw
    `load` source — confirmed against `comfyui-tooling-nodes`' own
    `ETN_control_apply` implementation: whether it blanks the masked region
    of the RGB before using it as structural conditioning depends on an
    `inpaint_masked_input` flag baked into the *weights file's own
    metadata*, which this graph has no way to inspect or trust, so a second
    pass wired with the raw source image reliably reconstructed the exact
    object the mask was drawn over — the ControlNet was reading its
    literal, undamaged pixels as "this is what belongs here" structural
    guidance. Feeding the already-erased image instead removes that signal
    regardless of what the metadata says. No SDXL/SD1.5 equivalent is wired
    in yet — those need either a custom Impact Pack `detailer_hook` for the
    Fooocus inpaint patch, or an inpaint ControlNet through the
    `tile_controlnet`-style path instead.

    Two more deltas in that same Anima branch, both matched against
    `krita-ai-diffusion`'s own `workflow.inpaint()` rather than guessed:
    `model_ref` is wrapped in a `DifferentialDiffusion` node (krita applies
    this unconditionally before any inpaint operation), and the local
    `denoise` used by `DetailerForEach` below is forced to `1.0` regardless
    of `params.denoise` — krita's own `detect_inpaint` only turns on this
    ControlNet-conditioned inpaint path for Anima at all when generation
    strength is `1.0`, and testing confirmed why: at the params' own default
    (0.75) the masked region came back almost entirely untouched for this
    arch/sampler combination. Even matching krita's approach this closely,
    results on stylized/illustrated content are inconsistent — this weight
    file behaves like a general "fill this hole with something plausible"
    model rather than a specialized "erase to background" one, so it can
    still invent a *different* object in the masked region instead of
    leaving it blank. Known and accepted for now rather than solved.

    Same downstream shape as `build_hand_detailer_manual` (`MaskToSEGS` ->
    `DetailerForEach`) and the same reasoning for skipping a detection-check
    `PreviewImage` node — a mask the user drew by hand is never empty."""
    g = PromptGraph()

    load = g.add("LoadImage", {"image": source_filename}, title="Source Image")
    model_ref, clip_ref, vae_ref, positive, negative = _build_model_clip_vae(g, base)

    mask = g.add(
        "LoadImageMask",
        {"image": mask_filename, "channel": "red"},
        title="Drawn Mask",
    )
    seed = _resolve_seed(params.seed)

    detailer_image: NodeRef = (load, 0)
    denoise = params.denoise
    if content_aware_fill:
        assert isinstance(params, DrawnMaskFixParams)
        fill_model = g.add(
            "INPAINT_LoadInpaintModel",
            {"model_name": params.fill_model},
            title="Content-Aware Fill Model",
        )
        prefilled = g.add(
            "INPAINT_InpaintWithModel",
            {"inpaint_model": [fill_model, 0], "image": [load, 0], "mask": [mask, 0], "seed": seed},
            title="Content-Aware Fill",
        )
        detailer_image = (prefilled, 0)

        if base.loader == "split" and base.anima_lllite_inpaint_patch:
            # krita-ai-diffusion applies this unconditionally before any
            # inpaint operation (`w.differential_diffusion(model)`) — it
            # changes how `DetailerForEach`'s noise_mask is interpreted so
            # the masked region actually gets fully re-sampled rather than
            # staying anchored to the (here: content-aware-filled) input
            # latent. Scoped to this branch, not applied generally, since
            # it's only been verified for this exact path.
            diff_diffusion_model = g.add(
                "DifferentialDiffusion", {"model": list(model_ref)}, title="Differential Diffusion"
            )
            model_ref = (diff_diffusion_model, 0)
            # krita-ai-diffusion only engages this ControlNet-conditioned
            # inpaint path for Anima at all when its generation strength is
            # 1.0 (`detect_inpaint`'s `use_inpaint_model = strength == 1.0`
            # for Arch.anima) — confirmed necessary by testing too: at 0.75
            # the masked region came back almost untouched for this arch.
            denoise = 1.0
            control_load = g.add(
                "ETN_control_load",
                {"model": list(model_ref), "weights": base.anima_lllite_inpaint_patch},
                title="Load Anima Inpaint ControlNet (LLLite)",
            )
            patched_model = g.add(
                "ETN_control_apply",
                {
                    "model": [control_load, 0],
                    "control_net": [control_load, 1],
                    # The already-content-aware-filled image (object erased),
                    # not the raw source — see this function's docstring for
                    # why: whether the node itself blanks the masked region
                    # before using it as structural conditioning depends on a
                    # metadata flag baked into the weights file, which we
                    # can't verify from here, so the object's shape must
                    # already be gone from what we hand it regardless.
                    "image": list(detailer_image),
                    "strength": base.anima_lllite_inpaint_patch_strength,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                    "mask": [mask, 0],
                },
                title="Apply Anima Inpaint ControlNet (LLLite)",
            )
            model_ref = (patched_model, 0)

    segs = g.add(
        "MaskToSEGS",
        {
            "mask": [mask, 0],
            "combined": False,
            "crop_factor": params.crop_factor,
            "bbox_fill": False,
            "drop_size": 10,
            "contour_fill": False,
        },
        title="Drawn Region SEGS",
    )

    detailer = g.add(
        "DetailerForEach",
        {
            "image": list(detailer_image),
            "segs": [segs, 0],
            "model": list(model_ref),
            "clip": list(clip_ref),
            "vae": list(vae_ref),
            "positive": [positive, 0],
            "negative": [negative, 0],
            "guide_size": params.guide_size,
            "guide_size_for": True,
            "max_size": params.max_size,
            "seed": seed,
            "steps": params.steps,
            "cfg": params.cfg,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            "denoise": denoise,
            "feather": params.feather,
            "noise_mask": True,
            "force_inpaint": True,
            "wildcard": "",
            "cycle": 1,
            "inpaint_model": False,
            "noise_mask_feather": 20,
        },
        title=f"{label} Detailer (drawn mask)",
    )

    save = g.add(
        "SaveImage",
        {"images": [detailer, 0], "filename_prefix": params.filename_prefix},
        title="Save Image",
    )
    return g.as_prompt(), save
