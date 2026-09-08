from pathlib import Path

import pytest

from comfytelegram.profiles import (
    ModelProfile,
    ProfileDefaults,
    load_profiles,
    resolve_generation_params,
    resolve_profile,
)

PROFILES_DIR = Path(__file__).resolve().parent.parent / "model_profiles"


@pytest.fixture(scope="module")
def profiles():
    loaded = load_profiles(PROFILES_DIR)
    assert loaded, "expected the shipped example profiles to load"
    return loaded


def test_all_shipped_profiles_load(profiles):
    names = {p.display_name for p in profiles}
    assert names == {
        "FurryToonMix XL (Illustrious)",
        "SDXL Base",
        "Pony Diffusion XL",
        "Animagine XL",
    }


def test_resolve_profile_matches_furrytoonmix(profiles):
    profile = resolve_profile("furrytoonmix_xlIllustriousV2.safetensors", profiles)
    assert profile is not None
    assert profile.display_name == "FurryToonMix XL (Illustrious)"


def test_resolve_profile_matches_pony_case_insensitively(profiles):
    profile = resolve_profile("PonyDiffusionXL_v6.safetensors", profiles)
    assert profile is not None
    assert profile.display_name == "Pony Diffusion XL"


def test_resolve_profile_no_match_returns_none(profiles):
    assert resolve_profile("some_unknown_checkpoint.safetensors", profiles) is None


def _synthetic_profile() -> ModelProfile:
    """A profile with fixed, non-tunable values for testing merge logic —
    deliberately not one of the shipped model_profiles/*.json files, since
    those are meant to be user-editable and a prior version of this test
    broke the moment their exact numbers were tuned (see git history / the
    furrytoonmix_illustrious.json steps mismatch this test used to pin)."""
    return ModelProfile(
        match=["synthetic_test_ckpt*"],
        display_name="Synthetic Test Profile",
        defaults=ProfileDefaults(cfg=5.0, steps=40, clip_skip=-2),
        positive_prompt_prefix="masterpiece,best quality",
        negative_prompt_prefix="low quality",
    )


def test_resolve_generation_params_applies_profile_defaults():
    profile = _synthetic_profile()
    params = resolve_generation_params("synthetic_test_ckpt.safetensors", "a fox", profile)
    assert params.cfg == 5.0
    assert params.steps == 40
    assert params.clip_skip == -2
    assert params.positive_prompt.startswith("masterpiece,best quality")
    assert params.positive_prompt.endswith("a fox")
    assert params.negative_prompt == "low quality"
    assert params.loras == []


def test_resolve_generation_params_overrides_win():
    profile = _synthetic_profile()
    params = resolve_generation_params(
        "synthetic_test_ckpt.safetensors",
        "a fox",
        profile,
        overrides={"cfg": 9.0},
    )
    assert params.cfg == 9.0


def test_resolve_generation_params_without_profile_uses_raw_prompt():
    params = resolve_generation_params("unknown.safetensors", "a fox", None)
    assert params.positive_prompt == "a fox"
    assert params.negative_prompt == ""
    assert params.loras == []


def test_resolve_generation_params_appends_extra_negative():
    profile = _synthetic_profile()
    params = resolve_generation_params(
        "synthetic_test_ckpt.safetensors",
        "a fox",
        profile,
        extra_negative_prompt="extra limbs, blurry",
    )
    assert params.negative_prompt == "low quality, extra limbs, blurry"


def test_resolve_generation_params_extra_negative_without_profile():
    params = resolve_generation_params(
        "unknown.safetensors", "a fox", None, extra_negative_prompt="blurry"
    )
    assert params.negative_prompt == "blurry"
