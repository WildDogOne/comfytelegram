from comfytelegram.profiles import (
    LoraDefault,
    ModelProfile,
    ProfileDefaults,
    apply_lora_overrides,
    apply_lora_strength_overrides,
    apply_profile_override,
)


def test_no_override_returns_profile_unchanged():
    profile = ModelProfile(match=["*ckpt*"], display_name="X", defaults=ProfileDefaults(cfg=5.0))
    result = apply_profile_override(profile, "ckpt.safetensors", {})
    assert result is profile


def test_override_merges_onto_existing_profile():
    profile = ModelProfile(
        match=["*ckpt*"], display_name="X", defaults=ProfileDefaults(cfg=5.0, steps=40)
    )
    result = apply_profile_override(profile, "ckpt.safetensors", {"cfg": 8.0})
    assert result is not None
    assert result.defaults.cfg == 8.0
    assert result.defaults.steps == 40  # untouched field preserved
    assert result.display_name == "X"  # rest of the profile preserved


def test_override_synthesizes_profile_when_none_matched():
    result = apply_profile_override(None, "unknown.safetensors", {"cfg": 6.0})
    assert result is not None
    assert result.defaults.cfg == 6.0
    assert result.display_name == "unknown.safetensors"


def test_override_routes_prompt_prefixes_onto_profile_not_defaults():
    profile = ModelProfile(
        match=["*ckpt*"],
        display_name="X",
        positive_prompt_prefix="masterpiece",
        defaults=ProfileDefaults(cfg=5.0),
    )
    result = apply_profile_override(
        profile, "ckpt.safetensors", {"positive_prompt_prefix": "best quality", "cfg": 8.0}
    )
    assert result.positive_prompt_prefix == "best quality"
    assert result.defaults.cfg == 8.0
    # prompt-prefix override key must not leak into .defaults
    assert not hasattr(result.defaults, "positive_prompt_prefix")


def test_override_synthesizes_profile_for_negative_prompt_prefix_only():
    result = apply_profile_override(
        None, "unknown.safetensors", {"negative_prompt_prefix": "blurry"}
    )
    assert result is not None
    assert result.negative_prompt_prefix == "blurry"


def _profile_with_loras() -> ModelProfile:
    return ModelProfile(
        match=["*ckpt*"],
        display_name="X",
        loras=[
            LoraDefault(
                name="styleA.safetensors",
                default_enabled=True,
                strength_model=0.8,
                strength_clip=0.9,
            ),
            LoraDefault(
                name="styleB.safetensors",
                default_enabled=False,
                strength_model=1.0,
                strength_clip=1.0,
            ),
        ],
    )


def test_apply_lora_overrides_with_no_overrides_returns_profile_unchanged():
    profile = _profile_with_loras()
    assert apply_lora_overrides(profile, {}) is profile


def test_apply_lora_overrides_with_no_profile_returns_none():
    assert apply_lora_overrides(None, {"styleA.safetensors": False}) is None


def test_apply_lora_overrides_flips_only_named_loras():
    profile = _profile_with_loras()
    result = apply_lora_overrides(
        profile, {"styleA.safetensors": False, "styleB.safetensors": True}
    )
    assert [lora.default_enabled for lora in result.loras] == [False, True]
    # names/strengths untouched, only default_enabled flips
    assert [lora.name for lora in result.loras] == ["styleA.safetensors", "styleB.safetensors"]


def test_apply_lora_overrides_leaves_unmentioned_loras_at_their_profile_default():
    profile = _profile_with_loras()
    result = apply_lora_overrides(profile, {"styleA.safetensors": False})
    assert result.loras[0].default_enabled is False
    assert result.loras[1].default_enabled is False  # profile's own default, untouched

    result = apply_lora_overrides(profile, {"styleB.safetensors": True})
    assert result.loras[0].default_enabled is True  # profile's own default, untouched
    assert result.loras[1].default_enabled is True


def test_apply_lora_strength_overrides_with_no_overrides_returns_profile_unchanged():
    profile = _profile_with_loras()
    assert apply_lora_strength_overrides(profile, {}) is profile


def test_apply_lora_strength_overrides_with_no_profile_returns_none():
    assert (
        apply_lora_strength_overrides(None, {"styleA.safetensors": {"strength_model": 0.5}}) is None
    )


def test_apply_lora_strength_overrides_applies_only_named_fields():
    profile = _profile_with_loras()
    result = apply_lora_strength_overrides(profile, {"styleA.safetensors": {"strength_model": 0.5}})
    assert result.loras[0].strength_model == 0.5
    assert result.loras[0].strength_clip == 0.9  # untouched field preserved
    assert result.loras[1].strength_model == 1.0  # unmentioned lora untouched


def test_apply_lora_strength_overrides_can_set_both_fields():
    profile = _profile_with_loras()
    result = apply_lora_strength_overrides(
        profile, {"styleA.safetensors": {"strength_model": 0.5, "strength_clip": 0.6}}
    )
    assert result.loras[0].strength_model == 0.5
    assert result.loras[0].strength_clip == 0.6
    # enabled state and name untouched by a strength override
    assert result.loras[0].default_enabled is True
    assert result.loras[0].name == "styleA.safetensors"
