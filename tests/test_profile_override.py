from comfytelegram.profiles import ModelProfile, ProfileDefaults, apply_profile_override


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
