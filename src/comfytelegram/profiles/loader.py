"""Load model profiles from disk and resolve one against a chosen checkpoint."""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from comfytelegram.profiles.schema import ModelProfile, ProfileDefaults
from comfytelegram.workflows.builder import GenerationParams

#: `ProfileDefaults`'s field names — the set of keys a stored per-chat
#: override (or the /override command) is allowed to touch.
OVERRIDABLE_FIELDS = set(ProfileDefaults.model_fields)

logger = logging.getLogger(__name__)


def load_profiles(directory: Path) -> list[ModelProfile]:
    """Parse every `*.json` file in `directory` as a ModelProfile.

    Files starting with `_` (e.g. `_schema.json`) are skipped — that prefix
    is reserved for non-profile documentation/schema files living alongside
    the profiles. A file that fails to validate is logged and skipped rather
    than raising, so one bad profile can't take the whole bot down.
    """
    if not directory.is_dir():
        logger.warning("Model profiles directory %s does not exist", directory)
        return []

    profiles: list[ModelProfile] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text())
            profiles.append(ModelProfile.model_validate(data))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Skipping invalid model profile %s: %s", path, exc)
    return profiles


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
    merged_defaults = base.defaults.model_copy(update=override_fields)
    return base.model_copy(update={"defaults": merged_defaults})


def _join_nonempty(parts: list[str], sep: str = ", ") -> str:
    return sep.join(p for p in parts if p)


def resolve_generation_params(
    checkpoint: str,
    user_prompt: str,
    profile: ModelProfile | None,
    *,
    overrides: dict[str, Any] | None = None,
) -> GenerationParams:
    """Combine a model profile's defaults with the user's prompt and any explicit
    overrides (highest priority, e.g. a user-set /cfg or /steps command) into a
    ready-to-build GenerationParams.
    """
    overrides = dict(overrides or {})

    if profile is not None:
        positive_prompt = _join_nonempty([profile.positive_prompt_prefix, user_prompt])
        negative_prompt = profile.negative_prompt_prefix
        loras = [lora.to_spec() for lora in profile.loras if lora.default_enabled]
        field_defaults = profile.defaults.model_dump(exclude_none=True)
    else:
        positive_prompt = user_prompt
        negative_prompt = ""
        loras = []
        field_defaults = {}

    kwargs: dict[str, Any] = {
        "checkpoint": checkpoint,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "loras": loras,
        **field_defaults,
    }
    kwargs.update(overrides)
    return GenerationParams(**kwargs)
