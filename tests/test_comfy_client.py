from unittest.mock import AsyncMock

import pytest

from comfytelegram.comfy_client import (
    ComfyClient,
    ComfyUIError,
    NodeNotInstalledError,
    _enum_choices,
)


def test_enum_choices_extracts_choice_list():
    node_info = {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}}
    assert _enum_choices(node_info, "ckpt_name") == ["a.safetensors", "b.safetensors"]


def test_enum_choices_extracts_new_combo_options():
    """Newer ComfyUI builds an enum as `["COMBO", {"options": [...]}]`
    instead of putting the list at `[0]` — both shapes are live in the wild
    (checkpoints still use the old one), so both must be read."""
    node_info = {
        "input": {
            "required": {
                "bg_removal_name": [["birefnet.safetensors"], {"tooltip": "the model"}],
                "other_name": ["COMBO", {"options": ["x", "y"], "multiselect": False}],
            }
        }
    }
    assert _enum_choices(node_info, "bg_removal_name") == ["birefnet.safetensors"]
    assert _enum_choices(node_info, "other_name") == ["x", "y"]


def test_enum_choices_missing_input_returns_empty():
    node_info = {"input": {"required": {}}}
    assert _enum_choices(node_info, "ckpt_name") == []


def test_enum_choices_non_enum_input_returns_empty():
    # a non-enum required input looks like ["INT", {...}] — spec[0] isn't a list
    node_info = {"input": {"required": {"seed": ["INT", {"default": 0}]}}}
    assert _enum_choices(node_info, "seed") == []


def test_enum_choices_missing_input_section_returns_empty():
    assert _enum_choices({}, "ckpt_name") == []


class _FakeResp:
    """Minimal aiohttp-response stand-in for the `async with session.get(...)`
    shape `get_node_info` uses."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_get_node_info_unknown_class_raises_node_not_installed(monkeypatch):
    """ComfyUI answers `GET /object_info/<Unknown>` with 200 and `{}` —
    unwrapping must raise, not return `{}` or throw `KeyError`, so callers
    can tell "this ComfyUI doesn't have the node" from "the node offers
    nothing"."""
    client = ComfyClient("http://example", "ws://example")

    class _FakeSession:
        def get(self, url: str):
            return _FakeResp({})

    # `session` is a read-only property over the context-managed client
    monkeypatch.setattr(client, "_session", _FakeSession())
    with pytest.raises(NodeNotInstalledError):
        await client.get_node_info("LoadBackgroundRemovalModel")
    # and it stays catchable as a plain ComfyUIError for "couldn't ask" callers
    with pytest.raises(ComfyUIError):
        await client.get_node_info("LoadBackgroundRemovalModel")


@pytest.mark.asyncio
async def test_list_background_removal_models_reads_the_combo(monkeypatch):
    """The staging check for "✂️ Remove BG" is this combo (built from
    `models/background_removal/`), so the handler side never has to parse
    the node spec itself."""
    client = ComfyClient("http://example", "ws://example")
    monkeypatch.setattr(
        client,
        "get_node_info",
        AsyncMock(
            return_value={
                "input": {
                    "required": {
                        "bg_removal_name": ["COMBO", {"options": ["birefnet.safetensors"]}]
                    }
                }
            }
        ),
    )
    assert await client.list_background_removal_models() == ["birefnet.safetensors"]


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
