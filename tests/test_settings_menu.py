from dataclasses import fields as dataclass_fields

from comfytelegram.profiles import ModelProfile
from comfytelegram.settings_menu import (
    FIELDS,
    FIELDS_BY_KEY,
    NumericFieldMeta,
    TextFieldMeta,
    _coerce_field_value,
    _curate,
    _field_value,
    _format_value,
    _home_keyboard,
    _numeric_submenu_keyboard,
    _text_submenu_keyboard,
    _truncate,
)
from comfytelegram.workflows import GenerationParams


def test_every_numeric_or_enum_field_key_is_a_real_generation_params_field():
    # TextFieldMeta fields (positive/negative prompt prefix) live on
    # ModelProfile, not GenerationParams — see TextFieldMeta's docstring.
    valid = {f.name for f in dataclass_fields(GenerationParams)}
    for meta in FIELDS:
        if isinstance(meta, TextFieldMeta):
            assert meta.key in ModelProfile.model_fields
        else:
            assert meta.key in valid


def test_format_value_drops_trailing_zero_for_whole_floats():
    assert _format_value(5.0) == "5"
    assert _format_value(5.5) == "5.5"
    assert _format_value(30) == "30"
    assert _format_value("euler") == "euler"


def test_curate_keeps_preferred_order_when_enough_match():
    # _curate only trusts the curated subset once at least 4 preferred names
    # actually exist server-side — below that it assumes curation missed too
    # much and falls back to the raw list (see test below).
    choices = ["z", "euler", "y", "euler_ancestral", "x", "dpmpp_2m", "ddim"]
    preferred = ("euler", "euler_ancestral", "dpmpp_2m", "ddim")
    result = _curate(choices, preferred, max_items=12)
    assert result == ["euler", "euler_ancestral", "dpmpp_2m", "ddim"]


def test_curate_falls_back_to_raw_list_when_curation_is_too_sparse():
    choices = ["some_exotic_sampler", "another_one"]
    preferred = ("euler", "euler_ancestral", "dpmpp_2m", "ddim")  # none of these are in choices
    result = _curate(choices, preferred, max_items=12)
    assert result == choices


def test_curate_with_no_preferred_list_shows_raw_choices_capped():
    choices = [f"opt{i}" for i in range(20)]
    result = _curate(choices, (), max_items=9)
    assert result == choices[:9]


def test_numeric_submenu_has_stepper_presets_and_navigation():
    meta = FIELDS_BY_KEY["cfg"]
    assert isinstance(meta, NumericFieldMeta)
    keyboard = _numeric_submenu_keyboard(meta, 5.0)
    all_buttons = [b for row in keyboard.inline_keyboard for b in row]
    callback_data = [b.callback_data for b in all_buttons]

    assert "st:d:cfg:-0.5" in callback_data
    assert "st:d:cfg:0.5" in callback_data
    assert any(cd.startswith("st:v:cfg:") for cd in callback_data)
    assert "st:c:cfg" in callback_data  # custom value
    assert "st:home" in callback_data  # back
    assert "st:r:cfg" in callback_data  # reset


def test_home_keyboard_marks_overridden_fields():
    keyboard = _home_keyboard("some_ckpt.safetensors", None, override_fields={"cfg": 8.0})
    all_buttons = [b for row in keyboard.inline_keyboard for b in row]
    cfg_button = next(b for b in all_buttons if b.callback_data == "st:f:cfg")
    steps_button = next(b for b in all_buttons if b.callback_data == "st:f:steps")
    assert cfg_button.text.startswith("★")
    assert not steps_button.text.startswith("★")


def test_home_keyboard_has_reset_all_and_close():
    keyboard = _home_keyboard("some_ckpt.safetensors", None, override_fields={})
    all_callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "st:ra" in all_callback_data
    assert "st:close" in all_callback_data


def test_truncate_leaves_short_strings_alone():
    assert _truncate("short") == "short"


def test_truncate_shortens_long_strings_with_ellipsis():
    result = _truncate("a" * 50, n=10)
    assert result.endswith("…")
    assert len(result) == 10


def test_field_value_for_text_field_reads_off_profile_not_params():
    meta = FIELDS_BY_KEY["positive_prompt_prefix"]
    assert isinstance(meta, TextFieldMeta)
    assert _field_value(meta, "ckpt.safetensors", None) == ""

    profile = ModelProfile(
        match=["*ckpt*"], display_name="X", positive_prompt_prefix="masterpiece, best quality"
    )
    assert _field_value(meta, "ckpt.safetensors", profile) == "masterpiece, best quality"


def test_text_submenu_has_edit_reset_and_back():
    meta = FIELDS_BY_KEY["negative_prompt_prefix"]
    keyboard = _text_submenu_keyboard(meta)
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "st:c:negative_prompt_prefix" in callback_data
    assert "st:r:negative_prompt_prefix" in callback_data
    assert "st:home" in callback_data


def test_coerce_field_value_accepts_any_text_for_prompt_fields():
    assert _coerce_field_value("positive_prompt_prefix", "  masterpiece  ") == {
        "positive_prompt_prefix": "masterpiece"
    }
    assert _coerce_field_value("negative_prompt_prefix", "") == {"negative_prompt_prefix": ""}


def test_coerce_field_value_still_validates_numeric_fields():
    assert _coerce_field_value("cfg", "6.5") == {"cfg": 6.5}
