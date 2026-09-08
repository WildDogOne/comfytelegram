from pathlib import Path

import pytest

from comfytelegram.profiles import load_profiles, resolve_generation_params, resolve_profile

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


def test_resolve_generation_params_applies_profile_defaults(profiles):
    profile = resolve_profile("furrytoonmix_xlIllustriousV2.safetensors", profiles)
    params = resolve_generation_params(
        "furrytoonmix_xlIllustriousV2.safetensors", "a fox", profile
    )
    assert params.cfg == 5.0
    assert params.steps == 40
    assert params.clip_skip == -2
    assert params.positive_prompt.startswith("masterpiece,best quality")
    assert params.positive_prompt.endswith("a fox")
    assert params.loras == []  # all shipped with default_enabled: false


def test_resolve_generation_params_overrides_win(profiles):
    profile = resolve_profile("furrytoonmix_xlIllustriousV2.safetensors", profiles)
    params = resolve_generation_params(
        "furrytoonmix_xlIllustriousV2.safetensors",
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
