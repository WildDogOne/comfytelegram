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
    trigger_words: str = Field(
        "",
        description=(
            "Free text — the word(s)/phrase this LoRA actually needs in the prompt to "
            "activate. When this LoRA ends up active for a generation (default_enabled, "
            "after any /lora override), resolve_generation_params folds this straight into "
            "positive_prompt automatically — right after positive_prompt_prefix and before "
            "the user's own text — so a LoRA that needs a specific trigger word doesn't "
            "silently do nothing (or need that word retyped by hand every single time) just "
            "because nobody remembered to include it. Deliberately manual-only to fill in, "
            "though: nothing in this codebase ever *writes* to this field automatically "
            "(specifically including lora_discovery.py's auto-registration, which only ever "
            "sets name/strength_model/strength_clip/default_enabled) — auto-filling it from "
            "CivitAI's own trainedWords would be easy to get wrong silently (a LoRA can be "
            "re-uploaded/renamed under the same hash, CivitAI's own list is sometimes "
            "incomplete or stale, and a custom-trained LoRA has no CivitAI entry to pull from "
            "at all), and a wrong trigger word silently saved as if verified — and then "
            "silently injected into every prompt — is worse than an empty field that's "
            "obviously still TODO. Left empty (the default, folds in as nothing) if you "
            "haven't filled it in yet."
        ),
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
    detailer_denoise: float | None = None
    detailer_cfg: float | None = None
    detailer_steps: int | None = None


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

    civitai_base_models: list[str] = Field(
        default_factory=list,
        description=(
            "CivitAI 'baseModel' strings (exact match, case-insensitive) this profile "
            "accepts for boot-time LoRA auto-discovery (see lora_discovery.py) — e.g. "
            "['SDXL 1.0'] for a plain SDXL profile, ['Pony'] for a Pony-derived one, "
            "['Illustrious'] for an Illustrious-derived one. A LoRA file found under "
            "Settings.comfyui_loras_dir that no profile's `loras` list already mentions "
            "gets identified via CivitAI's hash lookup and, if its reported base model "
            "matches an entry here, appended to this profile's `loras` (disabled by "
            "default) automatically. Empty (the default) opts this profile out of "
            "auto-discovery entirely — existing profiles are unaffected until this is "
            "set. A LoRA CivitAI has no hash record for at all (most commonly a "
            "privately/custom-trained one) can never be matched this way regardless of "
            "this setting and still needs to be added to `loras` by hand."
        ),
    )

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
    detail_prompt_tile_controlnet: bool = Field(
        False,
        description=(
            "Whether '✏️ Detail Prompt' also conditions its DetailerForEach pass on "
            "tile_controlnet, the same ControlNetLoader+ControlNetApplyAdvanced wiring "
            "the '🔍 Upscale 4x'/'🧵 Homogenize' passes already use — an experiment for a "
            "checkpoint whose detailer strays off-structure at any denoise high enough "
            "to still add real detail, to see whether anchoring it to the source image "
            "steadies that. False (the default) skips the branch entirely, same as "
            "tile_controlnet being unset — it's a no-op regardless of this flag if "
            "tile_controlnet itself isn't also set. '🖌️ Draw Mask'/'🩹 Fix Artifact' never "
            "read this at all, even though they share the exact same graph-building "
            "function as '✏️ Detail Prompt' — see generation.post_process's "
            "is_detail_prompt parameter for how that distinction is actually made, "
            "since post_process's own kind ('hand_drawn') is identical for both."
        ),
    )

    anima_lllite_inpaint_patch: str | None = Field(
        None,
        description=(
            "loader='split' only: weights filename for ETN_control_load/"
            "ETN_control_apply (comfyui-tooling-nodes — the same node pack "
            "krita-ai-diffusion itself uses for this weight format), e.g. "
            "Anima's 'anima-lllite-inpainting-v2.safetensors', staged under "
            "ComfyUI's models/controlnet/ (a plain ControlNetLoader scans "
            "the same folder). Applied before the '🩹 Fix Artifact' removal "
            "pass samples, so it's actually inpainting-aware instead of "
            "running plain noise-masked img2img. Do NOT stage this under "
            "models/model_patches/ for ComfyUI core's ModelPatchLoader/"
            "AnimaLLLiteApply instead — that's a different, unrelated "
            "mechanism that also lists this file in its dropdown without "
            "actually being able to interpret it (confirmed against a live "
            "server: wiring those core nodes produced very poor results). "
            "Unset (the default) skips the branch entirely."
        ),
    )
    anima_lllite_inpaint_patch_strength: float = Field(
        1.0, description="anima_lllite_inpaint_patch only: ETN_control_apply's strength"
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

    fix_artifact_positive_prefix: str | None = Field(
        None,
        description=(
            "'🩹 Fix Artifact' only: quality tags to put in front of "
            "'background scenery' for that pass, instead of this profile's "
            "normal positive_prompt_prefix. Exists because a removal pass wants "
            "different conditioning than ordinary generation — matched against a "
            "real krita-ai-diffusion 'remove object' job on this server, whose "
            "style prompt carried booru score tags this profile's generation "
            "prefix deliberately does not. Unset (the default) falls back to "
            "positive_prompt_prefix."
        ),
    )
    fix_artifact_negative_prefix: str | None = Field(
        None,
        description=(
            "'🩹 Fix Artifact' only: the negative prompt for that pass, instead "
            "of this profile's normal negative_prompt_prefix. Same reasoning as "
            "fix_artifact_positive_prefix — the captured krita job's negative "
            "prompt suppressed low-score output and stray text/signatures, which "
            "is exactly what a removal pass tends to hallucinate into the hole. "
            "Unset (the default) falls back to negative_prompt_prefix."
        ),
    )
