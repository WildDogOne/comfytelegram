from pathlib import Path

import pytest

from comfytelegram.profiles import (
    LoraDefault,
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
        "Anima Aesthetic",
        "Anima Turbo",
        "AutismMix SDXL",
    }


def test_resolve_profile_matches_furrytoonmix(profiles):
    profile = resolve_profile("furrytoonmix_xlIllustriousV2.safetensors", profiles)
    assert profile is not None
    assert profile.display_name == "FurryToonMix XL (Illustrious)"
    assert profile.prompt_style == "tags"


def test_sdxl_base_defaults_to_natural_prompt_style(profiles):
    profile = resolve_profile("sd_xl_base_1.0.safetensors", profiles)
    assert profile is not None
    assert profile.prompt_style == "natural"


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


def test_resolve_generation_params_defaults_to_checkpoint_loader_without_profile():
    params = resolve_generation_params("unknown.safetensors", "a fox", None)
    assert params.loader == "checkpoint"
    assert params.clip_name == ""
    assert params.vae_name == ""
    assert params.model_sampling_shift is None
    assert params.tile_controlnet is None
    assert params.anima_lllite_inpaint_patch is None
    assert params.upscale_denoise is None


def test_resolve_generation_params_carries_upscale_denoise_from_profile_defaults():
    profile = ModelProfile(
        match=["furrytoonmix_*"],
        display_name="FurryToonMix Test",
        defaults=ProfileDefaults(upscale_denoise=0.5),
    )
    params = resolve_generation_params("furrytoonmix_xlIllustriousV2.safetensors", "a fox", profile)
    assert params.upscale_denoise == 0.5


def test_resolve_generation_params_carries_tile_controlnet_from_profile():
    profile = ModelProfile(
        match=["illustriousxl*"],
        display_name="Illustrious Test",
        tile_controlnet="xinsir_tile_sdxl.safetensors",
        tile_controlnet_strength=0.55,
    )
    params = resolve_generation_params("illustriousxl_v10.safetensors", "a fox", profile)
    assert params.tile_controlnet == "xinsir_tile_sdxl.safetensors"
    assert params.tile_controlnet_strength == 0.55


def test_resolve_generation_params_carries_split_loader_fields_from_profile():
    profile = ModelProfile(
        match=["anima*"],
        display_name="Anima Test",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        clip_type="stable_diffusion",
        vae_name="qwen_image_vae.safetensors",
        model_sampling_shift=3.0,
    )
    params = resolve_generation_params("anima-aesthetic-v1.safetensors", "a fox", profile)
    assert params.loader == "split"
    assert params.clip_name == "qwen_3_06b_base.safetensors"
    assert params.clip_type == "stable_diffusion"
    assert params.vae_name == "qwen_image_vae.safetensors"
    assert params.model_sampling_shift == 3.0


def test_resolve_generation_params_carries_anima_lllite_inpaint_patch_from_profile():
    profile = ModelProfile(
        match=["anima*"],
        display_name="Anima Test",
        loader="split",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
        anima_lllite_inpaint_patch_strength=0.8,
    )
    params = resolve_generation_params("anima-aesthetic-v1.safetensors", "a fox", profile)
    assert params.anima_lllite_inpaint_patch == "anima-lllite-inpainting-v2.safetensors"
    assert params.anima_lllite_inpaint_patch_strength == 0.8


def test_fix_artifact_checkpoint_defaults_to_none():
    profile = ModelProfile(match=["*ckpt*"], display_name="X")
    assert profile.fix_artifact_checkpoint is None


def test_anima_aesthetic_profile_sets_fix_artifact_checkpoint(profiles):
    """The Fix Artifact button should always route to Anima Aesthetic
    regardless of which checkpoint generated the image being fixed — see
    generation._fix_artifact_override_base. If this ever stops matching the
    staged UNET filename, "🩹 Fix Artifact" would silently start rejecting
    every request with a ComfyUI "value not in list" error."""
    anima = next(p for p in profiles if p.display_name == "Anima Aesthetic")
    assert anima.fix_artifact_checkpoint == "anima_aestheticV11.safetensors"
    assert anima.loader == "split"
    assert anima.anima_lllite_inpaint_patch == "anima-lllite-inpainting-v2.safetensors"


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


def test_resolve_generation_params_carries_raw_prompt_through_unfolded():
    profile = _synthetic_profile()
    params = resolve_generation_params(
        "synthetic_test_ckpt.safetensors",
        "aria, a fox",
        profile,
        extra_negative_prompt="bad anatomy, blurry",
        raw_positive_prompt="a fox",
        raw_negative_prompt="blurry",
    )
    assert params.raw_positive_prompt == "a fox"
    assert params.raw_negative_prompt == "blurry"
    # unlike positive_prompt/negative_prompt, the raw fields never see the
    # profile's prefixes or the character text folded into user_prompt/
    # extra_negative_prompt above.
    assert "masterpiece" not in params.raw_positive_prompt
    assert "aria" not in params.raw_positive_prompt
    assert "bad anatomy" not in params.raw_negative_prompt


def test_resolve_generation_params_defaults_raw_prompt_to_empty():
    params = resolve_generation_params("unknown.safetensors", "a fox", None)
    assert params.raw_positive_prompt == ""
    assert params.raw_negative_prompt == ""


def test_lora_default_to_spec_drops_default_enabled_flag():
    lora = LoraDefault(name="a.safetensors", strength_model=0.8, strength_clip=0.9, default_enabled=False)
    spec = lora.to_spec()
    assert spec.name == "a.safetensors"
    assert spec.strength_model == 0.8
    assert spec.strength_clip == 0.9


def test_resolve_generation_params_only_applies_default_enabled_loras():
    profile = ModelProfile(
        match=["*ckpt*"],
        display_name="X",
        loras=[
            LoraDefault(name="on.safetensors", default_enabled=True),
            LoraDefault(name="off.safetensors", default_enabled=False),
        ],
    )
    params = resolve_generation_params("ckpt.safetensors", "a fox", profile)
    assert [lora.name for lora in params.loras] == ["on.safetensors"]
