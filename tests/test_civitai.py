import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfytelegram.civitai import (
    CIVITAI_HASH_LOOKUP_URL,
    CivitaiLookupError,
    fetch_civitai_info,
    hash_lora_file,
    resolve_lora_path,
)


def test_resolve_lora_path_finds_a_direct_child_file(tmp_path: Path):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    lora_file = loras_dir / "styleA.safetensors"
    lora_file.write_bytes(b"data")

    assert resolve_lora_path(loras_dir, "styleA.safetensors") == lora_file


def test_resolve_lora_path_finds_a_nested_subdirectory_file(tmp_path: Path):
    loras_dir = tmp_path / "loras"
    (loras_dir / "characters").mkdir(parents=True)
    lora_file = loras_dir / "characters" / "styleA.safetensors"
    lora_file.write_bytes(b"data")

    assert resolve_lora_path(loras_dir, "characters/styleA.safetensors") == lora_file


def test_resolve_lora_path_returns_none_for_a_missing_file(tmp_path: Path):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()

    assert resolve_lora_path(loras_dir, "does_not_exist.safetensors") is None


def test_resolve_lora_path_refuses_to_escape_the_loras_dir(tmp_path: Path):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    outside_file = tmp_path / "secret.safetensors"
    outside_file.write_bytes(b"data")

    assert resolve_lora_path(loras_dir, "../secret.safetensors") is None


@pytest.mark.asyncio
async def test_hash_lora_file_matches_hashlib_sha256(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    content = b"some lora file bytes" * 1000
    path.write_bytes(content)

    result = await hash_lora_file(path)

    assert result == hashlib.sha256(content).hexdigest()


def _response(status: int, json_body=None):
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_body)
    resp.raise_for_status = MagicMock()
    return resp


def _session_patch(response):
    def _get(url):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.get = _get
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=session_ctx)


@pytest.mark.asyncio
async def test_fetch_civitai_info_returns_none_on_404():
    response = _response(404)
    with patch("comfytelegram.civitai.aiohttp.ClientSession", _session_patch(response)):
        result = await fetch_civitai_info("deadbeef")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_civitai_info_parses_a_match():
    # Shaped exactly like a real response (confirmed live against
    # civitai.com/api/v1/model-versions/by-hash/<hash>): `modelId` is
    # top-level, and `model` itself never carries an `id` field at all.
    body = {
        "id": 456,
        "modelId": 123,
        "baseModel": "SDXL 1.0",
        "trainedWords": ["cool style", "trigger2"],
        "model": {"name": "Cool Style", "type": "LORA", "nsfw": False, "poi": False},
    }
    response = _response(200, body)
    with patch("comfytelegram.civitai.aiohttp.ClientSession", _session_patch(response)):
        result = await fetch_civitai_info("deadbeef")

    assert result is not None
    assert result.model_name == "Cool Style"
    assert result.base_model == "SDXL 1.0"
    assert result.trained_words == ["cool style", "trigger2"]
    assert result.civitai_url == "https://civitai.com/models/123?modelVersionId=456"


@pytest.mark.asyncio
async def test_fetch_civitai_info_handles_missing_optional_fields():
    body = {"id": 456, "baseModel": None, "trainedWords": None, "model": None}
    response = _response(200, body)
    with patch("comfytelegram.civitai.aiohttp.ClientSession", _session_patch(response)):
        result = await fetch_civitai_info("deadbeef")

    assert result is not None
    assert result.model_name == "Unknown"
    assert result.base_model == "Unknown"
    assert result.trained_words == []
    assert result.civitai_url == "https://civitai.com"


@pytest.mark.asyncio
async def test_fetch_civitai_info_raises_civitai_lookup_error_on_network_error():
    import aiohttp

    def _raise(*args, **kwargs):
        raise aiohttp.ClientConnectionError("down")

    with (
        patch("comfytelegram.civitai.aiohttp.ClientSession", MagicMock(side_effect=_raise)),
        pytest.raises(CivitaiLookupError),
    ):
        await fetch_civitai_info("deadbeef")


def test_hash_lookup_url_is_keyed_by_sha256():
    assert CIVITAI_HASH_LOOKUP_URL.format(sha256="abc123") == (
        "https://civitai.com/api/v1/model-versions/by-hash/abc123"
    )
