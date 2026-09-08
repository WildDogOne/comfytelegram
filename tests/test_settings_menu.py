from dataclasses import fields as dataclass_fields

from comfytelegram.settings_menu import (
    FIELDS,
    FIELDS_BY_KEY,
    NumericFieldMeta,
    _curate,
    _format_value,
    _home_keyboard,
    _numeric_submenu_keyboard,
)
from comfytelegram.workflows import GenerationParams


def test_every_field_key_is_a_real_generation_params_field():
    valid = {f.name for f in dataclass_fields(GenerationParams)}
    for meta in FIELDS:
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
