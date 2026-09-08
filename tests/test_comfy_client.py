from comfytelegram.comfy_client import _enum_choices


def test_enum_choices_extracts_choice_list():
    node_info = {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}}
    assert _enum_choices(node_info, "ckpt_name") == ["a.safetensors", "b.safetensors"]


def test_enum_choices_missing_input_returns_empty():
    node_info = {"input": {"required": {}}}
    assert _enum_choices(node_info, "ckpt_name") == []


def test_enum_choices_non_enum_input_returns_empty():
    # a non-enum required input looks like ["INT", {...}] — spec[0] isn't a list
    node_info = {"input": {"required": {"seed": ["INT", {"default": 0}]}}}
    assert _enum_choices(node_info, "seed") == []


def test_enum_choices_missing_input_section_returns_empty():
    assert _enum_choices({}, "ckpt_name") == []
