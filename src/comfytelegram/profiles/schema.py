"""Pydantic schema for the per-model "smart defaults" JSON profile format.

A profile describes generation defaults tuned for one checkpoint (or a
family of checkpoints matched by glob), e.g. "this Illustrious-based furry
merge wants cfg<=5 and clip_skip=-2, this Pony-derived model wants a score_9
prefix and a specific negative embedding". See `model_profiles/README.md`
for the on-disk format and `model_profiles/*.json` for worked examples.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from comfytelegram.workflows.builder import LoraSpec


class LoraDefault(BaseModel):
    """One LoRA a model profile knows about — not necessarily applied to
    every request; see `default_enabled`."""

    name: str = Field(..., description="LoRA filename as ComfyUI's LoraLoader knows it")
    strength_model: float = 1.0
    strength_clip: float = 1.0
    default_enabled: bool = Field(
        True, description="Applied automatically unless the user opts out"
    )

    def to_spec(self) -> LoraSpec:
        """Drop `default_enabled` (a profile-resolution-only concern) to get
        the `LoraSpec` `workflows.builder` actually wires into the graph."""
        return LoraSpec(
            name=self.name, strength_model=self.strength_model, strength_clip=self.strength_clip
        )


class ProfileDefaults(BaseModel):
    """All fields optional: unset fields fall back to GenerationParams' own defaults."""

    cfg: float | None = None
    steps: int | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    clip_skip: int | None = None
    width: int | None = None
    height: int | None = None
    batch_size: int | None = None
    upscale_denoise: float | None = None


class ModelProfile(BaseModel):
    """One profile: which checkpoints it applies to (`match`), plus the
    generation defaults, prompt prefixes, and LoRAs to apply for them. See
    `model_profiles/README.md` for the on-disk JSON format and worked
    examples."""

    match: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Glob patterns (fnmatch, case-insensitive) matched against the checkpoint "
            "filename, e.g. ['*illustrious*', 'furrytoonmix_*']"
        ),
    )
    display_name: str = Field(..., description="Shown to the user in the /model picker")
    description: str = ""

    defaults: ProfileDefaults = Field(default_factory=ProfileDefaults)

    positive_prompt_prefix: str = Field(
        "", description="Prepended to the user's prompt, e.g. quality tags this model wants"
    )
    negative_prompt_prefix: str = Field(
        "", description="Used as the negative prompt whenever the user doesn't supply one"
    )

    loras: list[LoraDefault] = Field(default_factory=list)

    prompt_style: Literal["tags", "natural"] = Field(
        "natural",
        description=(
            "Whether this checkpoint expects comma-separated booru tags or a "
            "prose-style prompt. Gates the '🐛 Show Prompt' button's tag-health check "
            "(handlers.py's SHOW_PROMPT_CALLBACK_KIND branch) to tag-style prompts only."
        ),
    )

    tag_dictionary: Literal["danbooru", "e621"] | None = Field(
        None,
        description=(
            "Which booru tag dictionary '/tags' and '/tagcheck' default to for this "
            "checkpoint — 'e621' for furry-trained models, 'danbooru' for anime/manga-"
            "trained ones. Unset searches/checks against both. Independent of "
            "prompt_style: a checkpoint can be tag-trained without this being set, in "
            "which case both dictionaries are searched."
        ),
    )

    loader: Literal["checkpoint", "split"] = Field(
        "checkpoint",
        description=(
            "'checkpoint' (the default) loads a single-file model through "
            "CheckpointLoaderSimple, matched by 'match' the normal way. 'split' is for "
            "architectures shipped as separate UNET/text-encoder/VAE files — e.g. Anima, "
            "loaded through UNETLoader+CLIPLoader+VAELoader instead — where 'match' targets "
            "the UNET filename and clip_name/vae_name/model_sampling_shift below supply the "
            "rest of the graph."
        ),
    )
    clip_name: str = Field(
        "", description="loader='split' only: CLIPLoader's text-encoder filename"
    )
    clip_type: str = Field(
        "stable_diffusion", description="loader='split' only: CLIPLoader's `type` input"
    )
    vae_name: str = Field("", description="loader='split' only: VAELoader's filename")
    model_sampling_shift: float | None = Field(
        None,
        description=(
            "loader='split' only: shift for a ModelSamplingAuraFlow node inserted after "
            "the UNET load (Anima's AuraFlow-style sampling). Leave unset for split "
            "architectures that don't need it."
        ),
    )

    tile_controlnet: str | None = Field(
        None,
        description=(
            "Filename of a ControlNet Tile model (in ComfyUI's models/controlnet) to "
            "condition the post-processing upscale pass on, for adding detail across "
            "the whole image instead of just the face/hand-detailer regions. Must be "
            "trained for this checkpoint's base architecture (e.g. an SDXL tile "
            "ControlNet won't work on an Anima/AuraFlow checkpoint). Unset (the "
            "default) skips the branch entirely."
        ),
    )
    tile_controlnet_strength: float = Field(
        0.4, description="tile_controlnet only: ControlNetApplyAdvanced's strength"
    )

    anima_lllite_inpaint_patch: str | None = Field(
        None,
        description=(
            "loader='split' only: filename of a ModelPatchLoader-compatible "
            "ControlNet-LLLite inpainting patch (e.g. Anima's "
            "'anima-lllite-inpainting-v2.safetensors', in ComfyUI's "
            "models/model_patches) applied via AnimaLLLiteApply before the "
            "'🩹 Fix Artifact' removal pass samples, so it's actually "
            "inpainting-aware instead of running plain noise-masked img2img. "
            "Unset (the default) skips the branch entirely."
        ),
    )
    anima_lllite_inpaint_patch_strength: float = Field(
        1.0, description="anima_lllite_inpaint_patch only: AnimaLLLiteApply's strength"
    )

    fix_artifact_checkpoint: str | None = Field(
        None,
        description=(
            "If set, '🩹 Fix Artifact' always loads THIS exact checkpoint filename "
            "(plus this profile's loader/clip_name/clip_type/vae_name/"
            "model_sampling_shift/loras/negative_prompt_prefix/"
            "anima_lllite_inpaint_patch) instead of the image's own original "
            "checkpoint — for a checkpoint that's actually inpainting-aware (e.g. "
            "an Anima profile with anima_lllite_inpaint_patch set), when the "
            "image's own checkpoint has no inpainting-aware path wired at all yet "
            "and produces unreliable removal results regardless of tuning. Unlike "
            "'match', this is a literal filename, not a glob — 'match' picks which "
            "profile applies to a checkpoint you already have; this instead names "
            "which exact installed file to switch to. The positive prompt is still "
            "forced empty for this pass either way (see generation.post_process's "
            "'fix_drawn' branch) — style/subject continuity for the patched region "
            "comes from the content-aware fill and (if configured) the inpainting "
            "patch, not the prompt. At most one profile should set this; if "
            "several do, the first one in load order wins. Unset (the default) "
            "keeps '🩹 Fix Artifact' on each image's own checkpoint."
        ),
    )
