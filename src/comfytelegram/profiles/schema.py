"""Pydantic schema for the per-model "smart defaults" JSON profile format.

A profile describes generation defaults tuned for one checkpoint (or a
family of checkpoints matched by glob), e.g. "this Illustrious-based furry
merge wants cfg<=5 and clip_skip=-2, this Pony-derived model wants a score_9
prefix and a specific negative embedding". See `model_profiles/README.md`
for the on-disk format and `model_profiles/*.json` for worked examples.
"""

from __future__ import annotations

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
