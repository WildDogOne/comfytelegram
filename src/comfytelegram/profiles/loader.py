"""Load model profiles from disk and resolve one against a chosen checkpoint."""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from comfytelegram.profiles.schema import ModelProfile
from comfytelegram.workflows.builder import GenerationParams

#: `positive_prompt_prefix`/`negative_prompt_prefix` live on `ModelProfile`
#: itself, not `ProfileDefaults` (they're not `GenerationParams` fields —
#: see `resolve_generation_params` below), but the /settings menu lets the
#: user override them the same way as any numeric default, so they need to
#: be routed differently in `apply_profile_override` rather than merged
#: straight into `.defaults`.
PROMPT_OVERRIDE_FIELDS = {"positive_prompt_prefix", "negative_prompt_prefix"}

#: `apply_checkpoint_switch`'s `new_profile=None` branch resets every other
#: `GenerationParams` field to its generic default — these are the ones a
#: checkpoint switch must never touch: the image's own identity (`seed`,
#: `filename_prefix`), its scene content (prompt/raw-prompt pairs), and
#: `loras` (reset separately, to `[]`, rather than `GenerationParams`' own
#: default_factory — spelled out here for clarity even though the two are
#: actually the same value).
_CHECKPOINT_SWITCH_PRESERVED_FIELDS = {
    "checkpoint",
    "positive_prompt",
    "negative_prompt",
    "raw_positive_prompt",
    "raw_negative_prompt",
    "seed",
    "filename_prefix",
    "loras",
}

logger = logging.getLogger(__name__)


def load_profile_files(directory: Path) -> list[tuple[Path, dict[str, Any], ModelProfile]]:
    """Like `load_profiles`, but keeps each file's path and raw parsed JSON
    dict alongside the validated `ModelProfile` — `load_profiles` itself is
    built on this and just drops the extra two. `lora_discovery.py` needs
    both: the raw dict is what it mutates and writes back (so every field
    it doesn't touch round-trips through `json.dump` exactly as parsed,
    not re-derived from `ModelProfile.model_dump()`, which would drop
    unknown fields and re-order things), and the path is where.

    Files starting with `_` (e.g. `_schema.json`) are skipped — that prefix
    is reserved for non-profile documentation/schema files living alongside
    the profiles. A file that fails to validate is logged and skipped rather
    than raising, so one bad profile can't take the whole bot down.
    """
    if not directory.is_dir():
        logger.warning("Model profiles directory %s does not exist", directory)
        return []

    results: list[tuple[Path, dict[str, Any], ModelProfile]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text())
            profile = ModelProfile.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Skipping invalid model profile %s: %s", path, exc)
            continue
        results.append((path, data, profile))
    return results


def load_profiles(directory: Path) -> list[ModelProfile]:
    """Parse every `*.json` file in `directory` as a ModelProfile. See
    `load_profile_files` for the version that also keeps each file's path
    and raw JSON."""
    return [profile for _, _, profile in load_profile_files(directory)]


def resolve_profile(checkpoint_name: str, profiles: list[ModelProfile]) -> ModelProfile | None:
    """First profile whose `match` glob patterns hit `checkpoint_name` (case-insensitive)."""
    needle = checkpoint_name.lower()
    for profile in profiles:
        if any(fnmatch.fnmatch(needle, pattern.lower()) for pattern in profile.match):
            return profile
    return None


def apply_profile_override(
    profile: ModelProfile | None, checkpoint: str, override_fields: dict[str, Any]
) -> ModelProfile | None:
    """Layer a user's stored per-(chat, checkpoint) override (see `storage.py`,
    already validated against `ProfileDefaults` before being persisted) on top
    of the shipped profile for that checkpoint.

    If no shipped profile matched but the user still has an override stored
    (they set one before any `model_profiles/*.json` existed for this
    checkpoint), synthesize a minimal profile from just the override so it
    still applies.
    """
    if not override_fields:
        return profile
    base = profile or ModelProfile(match=[checkpoint], display_name=checkpoint)
    prompt_updates = {k: v for k, v in override_fields.items() if k in PROMPT_OVERRIDE_FIELDS}
    defaults_updates = {k: v for k, v in override_fields.items() if k not in PROMPT_OVERRIDE_FIELDS}
    merged_defaults = base.defaults.model_copy(update=defaults_updates)
    return base.model_copy(update={"defaults": merged_defaults, **prompt_updates})


def _architecture_fields(profile: ModelProfile) -> dict[str, Any]:
    """The `GenerationParams` fields that describe *how to load* `profile`'s
    checkpoint (loader/clip/vae/model-sampling-shift/tile-ControlNet/Anima
    inpaint patch), as opposed to a tunable generation default — shared by
    `resolve_generation_params` (fresh generation) and
    `apply_checkpoint_switch` (switching an existing image's post-processing
    pipeline to a different checkpoint) so a new architecture-fact field
    only needs adding here once. `detail_prompt_tile_controlnet` is the one
    exception living in this dict despite not being an architecture fact
    itself (it's a tunable experiment toggle) — it rides alongside
    `tile_controlnet(_strength)` for the same reason those do: there's
    nowhere else for a per-checkpoint flag like this to live, and
    `generation.post_process` re-resolves all three live off the current
    profile anyway (see `_refresh_detail_prompt_tile_controlnet`), so what's
    frozen in here at resolution time barely matters for it."""
    return {
        "loader": profile.loader,
        "clip_name": profile.clip_name,
        "clip_type": profile.clip_type,
        "vae_name": profile.vae_name,
        "model_sampling_shift": profile.model_sampling_shift,
        "tile_controlnet": profile.tile_controlnet,
        "tile_controlnet_strength": profile.tile_controlnet_strength,
        "detail_prompt_tile_controlnet": profile.detail_prompt_tile_controlnet,
        "anima_lllite_inpaint_patch": profile.anima_lllite_inpaint_patch,
        "anima_lllite_inpaint_patch_strength": profile.anima_lllite_inpaint_patch_strength,
    }


def apply_checkpoint_switch(
    params: GenerationParams, new_checkpoint: str, new_profile: ModelProfile | None
) -> GenerationParams:
    """`handlers.py`'s "🔀 Switch Model" button: rebuild `params` for
    `new_checkpoint`, keeping the image's own scene content (positive/
    negative prompt, both raw and resolved) exactly as it is — the picture
    being detailed/upscaled hasn't changed just because a different
    checkpoint is now doing the technical work — while replacing every
    architecture fact (`_architecture_fields`) and tunable generation
    default (`ProfileDefaults`, via the same `model_dump(exclude_none=True)`
    merge `resolve_generation_params` uses) with `new_profile`'s own, same
    as a fresh generation against that checkpoint would get. `loras` resets
    to `new_profile`'s own `default_enabled` set — the original image's
    LoRAs were validated for the *old* checkpoint, not this one, so
    silently carrying them over risks an incompatible or simply wrong
    result; `handlers.py`'s "🎛 LoRAs" button is the follow-up for picking
    exactly which of the new checkpoint's LoRAs should actually be active.
    `new_profile=None` (no shipped profile matches `new_checkpoint`) falls
    back to `GenerationParams`' own generic defaults for everything —
    mirrors `resolve_generation_params`'s own `profile is None` branch.
    Derives the reset field set from `dataclass_fields(GenerationParams)`
    itself, minus `_CHECKPOINT_SWITCH_PRESERVED_FIELDS`, rather than listing
    every field by hand — so a field added to `GenerationParams` later (the
    next `detailer_*`-style tunable, say) is reset to its generic default
    here automatically instead of silently keeping the old checkpoint's
    stale value."""
    if new_profile is None:
        defaults = GenerationParams(
            checkpoint=new_checkpoint, positive_prompt="", negative_prompt=""
        )
        reset_fields = {
            f.name: getattr(defaults, f.name)
            for f in dataclass_fields(GenerationParams)
            if f.name not in _CHECKPOINT_SWITCH_PRESERVED_FIELDS
        }
        return replace(params, checkpoint=new_checkpoint, loras=[], **reset_fields)
    active_loras = [lora.to_spec() for lora in new_profile.loras if lora.default_enabled]
    field_defaults = new_profile.defaults.model_dump(exclude_none=True)
    return replace(
        params,
        checkpoint=new_checkpoint,
        loras=active_loras,
        **_architecture_fields(new_profile),
        **field_defaults,
    )


def apply_lora_overrides(
    profile: ModelProfile | None, overrides: dict[str, bool]
) -> ModelProfile | None:
    """Layer a chat's `/lora` toggle state (see `storage.get_lora_overrides`)
    on top of the profile's own `LoraDefault.default_enabled` flags. A LoRA
    name with no stored override keeps whatever the profile's JSON says;
    only `resolve_generation_params`'s `default_enabled` filter reads this,
    so this has no effect on a checkpoint with no matched profile (nothing
    to toggle) or on post-processing, which stays locked to whatever LoRAs
    actually made the original image rather than reacting to a later
    toggle (see `generation._refresh_tunable_defaults`'s docstring for why
    that split exists)."""
    if profile is None or not overrides:
        return profile
    updated_loras = [
        lora.model_copy(update={"default_enabled": overrides[lora.name]})
        if lora.name in overrides
        else lora
        for lora in profile.loras
    ]
    return profile.model_copy(update={"loras": updated_loras})


def apply_lora_strength_overrides(
    profile: ModelProfile | None, overrides: dict[str, dict[str, float]]
) -> ModelProfile | None:
    """Layer a chat's `/lora` per-LoRA `strength_model`/`strength_clip`
    overrides (see `storage.get_lora_strength_overrides`) on top of the
    profile's own configured values — the same override shape as
    `apply_lora_overrides`, just for strength instead of enabled/disabled,
    and kept as a separate function/table since the two are edited from
    different `/lora` screens. A LoRA name absent from `overrides`, or a
    field absent from its entry, keeps the profile's own value; a partial
    entry (just one of the two fields) only touches that one."""
    if profile is None or not overrides:
        return profile
    updated_loras = [
        lora.model_copy(update=overrides[lora.name]) if lora.name in overrides else lora
        for lora in profile.loras
    ]
    return profile.model_copy(update={"loras": updated_loras})


def join_nonempty(parts: list[str], sep: str = ", ") -> str:
    """Join the non-empty strings in `parts` — shared by prompt assembly here
    and in handlers.py's character-prompt folding."""
    return sep.join(p for p in parts if p)


def resolve_generation_params(
    checkpoint: str,
    user_prompt: str,
    profile: ModelProfile | None,
    *,
    overrides: dict[str, Any] | None = None,
    extra_negative_prompt: str = "",
    raw_positive_prompt: str = "",
    raw_negative_prompt: str = "",
) -> GenerationParams:
    """Combine a model profile's defaults with the user's prompt and any explicit
    overrides (highest priority, e.g. a user-set /cfg or /steps command) into a
    ready-to-build GenerationParams.

    `extra_negative_prompt` is appended after the profile's own negative
    prefix — e.g. an active saved character's negative prompt (see
    `storage.py`'s `character` table), which isn't part of the model
    profile at all.

    `raw_positive_prompt`/`raw_negative_prompt` carry straight through onto
    the returned `GenerationParams` unchanged — they're not folded into
    `positive_prompt`/`negative_prompt` at all, just the caller's own
    record of what the user actually typed before any profile/character
    prompt got mixed in (see `GenerationParams.raw_positive_prompt`).

    Every *active* LoRA's `trigger_words` (default_enabled after `/lora`'s
    own toggle overrides — see the `active_loras` filter below, shared with
    `loras`) gets folded into `positive_prompt` too, right after the
    profile's own `positive_prompt_prefix` and before `user_prompt` — a
    LoRA that needs a specific word/phrase to actually activate is useless
    if that word never reaches the prompt, and typing it by hand every time
    is exactly the tedium `trigger_words` (manual-only — see
    `LoraDefault.trigger_words`'s docstring for why it's never auto-filled)
    exists to remove. Purely additive to `positive_prompt`; never touches
    `raw_positive_prompt` (still exactly what the user typed) or the
    `LoraSpec`s the graph itself is built from (`LoraDefault.to_spec()`
    still drops it, same as ever) — "🐛 Show Prompt" on the resulting image
    shows the folded-in result via `positive_prompt`, so an injected
    trigger word is never invisible.
    """
    overrides = dict(overrides or {})

    if profile is not None:
        active_loras = [lora for lora in profile.loras if lora.default_enabled]
        lora_trigger_words = join_nonempty([lora.trigger_words for lora in active_loras])
        positive_prompt = join_nonempty(
            [profile.positive_prompt_prefix, lora_trigger_words, user_prompt]
        )
        negative_prompt = profile.negative_prompt_prefix
        loras = [lora.to_spec() for lora in active_loras]
        field_defaults = profile.defaults.model_dump(exclude_none=True)
        # Architecture facts about the checkpoint, not a tunable generation
        # default — same reasoning as PROMPT_OVERRIDE_FIELDS living outside
        # `.defaults` — so these come straight off the profile (via
        # `_architecture_fields`) rather than through `field_defaults`/
        # ProfileDefaults.
        architecture_fields = _architecture_fields(profile)
    else:
        positive_prompt = user_prompt
        negative_prompt = ""
        loras = []
        field_defaults = {}
        architecture_fields = {}

    negative_prompt = join_nonempty([negative_prompt, extra_negative_prompt])

    kwargs: dict[str, Any] = {
        "checkpoint": checkpoint,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "loras": loras,
        "raw_positive_prompt": raw_positive_prompt,
        "raw_negative_prompt": raw_negative_prompt,
        **architecture_fields,
        **field_defaults,
    }
    kwargs.update(overrides)
    return GenerationParams(**kwargs)
