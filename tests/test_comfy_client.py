import pytest

from comfytelegram.comfy_client import ComfyClient, _enum_choices


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


@pytest.mark.asyncio
async def test_list_checkpoints_merges_checkpoint_and_unet_loader_enums(monkeypatch):
    """`/model` offers both single-file checkpoints and split-loader UNET
    filenames (e.g. Anima) from one merged list — see `ModelProfile.loader`."""
    client = ComfyClient("http://example", "ws://example")

    async def fake_get_node_info(class_type: str) -> dict:
        if class_type == "CheckpointLoaderSimple":
            return {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}}
        if class_type == "UNETLoader":
            return {"input": {"required": {"unet_name": [["anima-aesthetic-v1.safetensors"], {}]}}}
        raise AssertionError(f"unexpected class_type {class_type!r}")

    monkeypatch.setattr(client, "get_node_info", fake_get_node_info)

    assert await client.list_checkpoints() == [
        "a.safetensors",
        "b.safetensors",
        "anima-aesthetic-v1.safetensors",
    ]
