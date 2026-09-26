import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from comfytelegram.civitai import CivitaiLookupError, CivitaiLoraInfo
from comfytelegram.lora_discovery import (
    _scan_lora_files,
    discover_new_loras,
    reload_profiles_and_discover,
)
from comfytelegram.settings import Settings
from comfytelegram.storage import Storage


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, telegram_bot_token="test-token", **overrides)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def _write_profile(path: Path, **fields) -> None:
    base = {"match": ["*ckpt*"], "display_name": "X", "loras": []}
    base.update(fields)
    path.write_text(json.dumps(base, indent=2))


def test_scan_lora_files_finds_recognized_extensions_recursively(tmp_path: Path):
    (tmp_path / "characters").mkdir()
    (tmp_path / "top.safetensors").write_bytes(b"a")
    (tmp_path / "characters" / "nested.pt").write_bytes(b"b")
    (tmp_path / "preview.png").write_bytes(b"c")  # not a recognized model extension
    (tmp_path / "notes.txt").write_text("hi")

    result = _scan_lora_files(tmp_path)

    assert result == ["characters/nested.pt", "top.safetensors"]


@pytest.mark.asyncio
async def test_discover_does_nothing_when_no_profile_opts_in(tmp_path: Path, storage: Storage):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "new.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write_profile(profiles_dir / "a.json")  # civitai_base_models left unset/empty

    with patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock()) as hash_mock:
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is False
    hash_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_discover_does_nothing_when_no_new_files(tmp_path: Path, storage: Storage):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "known.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write_profile(
        profiles_dir / "a.json",
        civitai_base_models=["SDXL 1.0"],
        loras=[{"name": "known.safetensors", "strength_model": 1.0, "strength_clip": 1.0}],
    )

    with patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock()) as hash_mock:
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is False
    hash_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_discover_registers_a_matched_lora_into_the_opted_in_profile(
    tmp_path: Path, storage: Storage
):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "new.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, civitai_base_models=["SDXL 1.0"])

    info = CivitaiLoraInfo(
        model_name="Cool Style",
        base_model="SDXL 1.0",
        trained_words=["cool style"],
        civitai_url="https://civitai.com/models/1",
    )
    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock(return_value="hash1")),
        patch("comfytelegram.lora_discovery.fetch_civitai_info", AsyncMock(return_value=info)),
        patch("comfytelegram.lora_discovery.asyncio.sleep", AsyncMock()),
    ):
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is True
    written = json.loads(profile_path.read_text())
    assert written["loras"] == [
        {
            "name": "new.safetensors",
            "strength_model": 1.0,
            "strength_clip": 1.0,
            "default_enabled": False,
        }
    ]
    # other fields survive the rewrite untouched
    assert written["display_name"] == "X"
    assert written["civitai_base_models"] == ["SDXL 1.0"]

    cached = storage.get_lora_civitai_cache("new.safetensors")
    assert cached["found"] is True
    assert cached["model_name"] == "Cool Style"


@pytest.mark.asyncio
async def test_discover_matches_base_model_case_insensitively_across_profiles(
    tmp_path: Path, storage: Storage
):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "new.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    a_path = profiles_dir / "a.json"
    b_path = profiles_dir / "b.json"
    c_path = profiles_dir / "c.json"
    _write_profile(a_path, civitai_base_models=["sdxl 1.0"])
    _write_profile(b_path, civitai_base_models=["Pony"])  # should NOT match
    _write_profile(c_path, civitai_base_models=["SDXL 1.0", "Pony"])

    info = CivitaiLoraInfo(
        model_name="Cool Style",
        base_model="SDXL 1.0",
        trained_words=[],
        civitai_url="https://civitai.com/models/1",
    )
    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock(return_value="hash1")),
        patch("comfytelegram.lora_discovery.fetch_civitai_info", AsyncMock(return_value=info)),
        patch("comfytelegram.lora_discovery.asyncio.sleep", AsyncMock()),
    ):
        await discover_new_loras(loras_dir, profiles_dir, storage)

    assert json.loads(a_path.read_text())["loras"]
    assert json.loads(b_path.read_text())["loras"] == []
    assert json.loads(c_path.read_text())["loras"]


@pytest.mark.asyncio
async def test_discover_skips_and_caches_a_civitai_miss(tmp_path: Path, storage: Storage):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "custom.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, civitai_base_models=["SDXL 1.0"])

    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock(return_value="hash1")),
        patch("comfytelegram.lora_discovery.fetch_civitai_info", AsyncMock(return_value=None)),
        patch("comfytelegram.lora_discovery.asyncio.sleep", AsyncMock()),
    ):
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is False
    assert json.loads(profile_path.read_text())["loras"] == []
    cached = storage.get_lora_civitai_cache("custom.safetensors")
    assert cached["found"] is False


@pytest.mark.asyncio
async def test_discover_skips_a_base_model_no_profile_accepts(tmp_path: Path, storage: Storage):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "new.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, civitai_base_models=["SDXL 1.0"])

    info = CivitaiLoraInfo(
        model_name="Flux Style", base_model="Flux.1 D", trained_words=[], civitai_url="https://x"
    )
    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock(return_value="hash1")),
        patch("comfytelegram.lora_discovery.fetch_civitai_info", AsyncMock(return_value=info)),
        patch("comfytelegram.lora_discovery.asyncio.sleep", AsyncMock()),
    ):
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is False
    assert json.loads(profile_path.read_text())["loras"] == []


@pytest.mark.asyncio
async def test_discover_reuses_the_cache_instead_of_rehashing(tmp_path: Path, storage: Storage):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "new.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, civitai_base_models=["SDXL 1.0"])

    info = CivitaiLoraInfo(
        model_name="Cool Style", base_model="SDXL 1.0", trained_words=[], civitai_url="https://x"
    )
    storage.set_lora_civitai_cache("new.safetensors", "cachedhash", info)

    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock()) as hash_mock,
        patch("comfytelegram.lora_discovery.fetch_civitai_info", AsyncMock()) as fetch_mock,
    ):
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is True
    hash_mock.assert_not_awaited()
    fetch_mock.assert_not_awaited()
    assert json.loads(profile_path.read_text())["loras"]


@pytest.mark.asyncio
async def test_discover_stops_the_run_on_a_civitai_lookup_error_without_caching(
    tmp_path: Path, storage: Storage
):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "a.safetensors").write_bytes(b"data")
    (loras_dir / "b.safetensors").write_bytes(b"data2")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, civitai_base_models=["SDXL 1.0"])

    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock(return_value="hash1")),
        patch(
            "comfytelegram.lora_discovery.fetch_civitai_info",
            AsyncMock(side_effect=CivitaiLookupError("down")),
        ),
        patch("comfytelegram.lora_discovery.asyncio.sleep", AsyncMock()),
    ):
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is False
    assert storage.get_lora_civitai_cache("a.safetensors") is None
    assert storage.get_lora_civitai_cache("b.safetensors") is None
    assert json.loads(profile_path.read_text())["loras"] == []


@pytest.mark.asyncio
async def test_discover_reports_changed_false_when_the_write_fails(
    tmp_path: Path, storage: Storage
):
    """A profile file mounted read-only (the actual bug this test pins —
    docker-compose.example.yml's `model_profiles` mount used to be `:ro`,
    which silently discarded every match this job ever computed) must not
    be reported as a successful write: `main.py`'s `_start_lora_discovery`
    only reloads `bot_data['profiles']` when this returns True, and a
    write that never happened has nothing to reload."""
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "new.safetensors").write_bytes(b"data")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, civitai_base_models=["SDXL 1.0"])

    info = CivitaiLoraInfo(
        model_name="Cool Style",
        base_model="SDXL 1.0",
        trained_words=[],
        civitai_url="https://civitai.com/models/1",
    )
    with (
        patch("comfytelegram.lora_discovery.hash_lora_file", AsyncMock(return_value="hash1")),
        patch("comfytelegram.lora_discovery.fetch_civitai_info", AsyncMock(return_value=info)),
        patch("comfytelegram.lora_discovery.asyncio.sleep", AsyncMock()),
        patch.object(Path, "write_text", side_effect=OSError("Read-only file system")),
    ):
        changed = await discover_new_loras(loras_dir, profiles_dir, storage)

    assert changed is False
    # the match was still computed and cached — just never persisted
    assert storage.get_lora_civitai_cache("new.safetensors")["found"] is True
    assert json.loads(profile_path.read_text())["loras"] == []


@pytest.mark.asyncio
async def test_reload_profiles_and_discover_skips_scan_without_loras_dir(
    tmp_path: Path, storage: Storage
):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write_profile(profiles_dir / "a.json", display_name="A")
    settings = _settings(model_profiles_dir=profiles_dir, comfyui_loras_dir=None)

    with patch("comfytelegram.lora_discovery.discover_new_loras", AsyncMock()) as discover_mock:
        profiles, discovered = await reload_profiles_and_discover(settings, storage)

    discover_mock.assert_not_awaited()
    assert discovered is False
    assert [p.display_name for p in profiles] == ["A"]


@pytest.mark.asyncio
async def test_reload_profiles_and_discover_runs_discovery_when_loras_dir_set(
    tmp_path: Path, storage: Storage
):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write_profile(profiles_dir / "a.json", display_name="A")
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    settings = _settings(model_profiles_dir=profiles_dir, comfyui_loras_dir=loras_dir)

    with patch(
        "comfytelegram.lora_discovery.discover_new_loras", AsyncMock(return_value=True)
    ) as discover_mock:
        profiles, discovered = await reload_profiles_and_discover(settings, storage)

    discover_mock.assert_awaited_once_with(loras_dir, profiles_dir, storage)
    assert discovered is True
    assert [p.display_name for p in profiles] == ["A"]


@pytest.mark.asyncio
async def test_reload_profiles_and_discover_always_rereads_profiles_from_disk(
    tmp_path: Path, storage: Storage
):
    """Unlike `_start_lora_discovery`'s boot-time reload (only when
    discovery itself wrote something), this must re-read every profile
    file unconditionally — the whole point is picking up a hand-edited
    profile discovery never touched at all."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "a.json"
    _write_profile(profile_path, display_name="Before Edit")
    settings = _settings(model_profiles_dir=profiles_dir, comfyui_loras_dir=None)

    profiles, _ = await reload_profiles_and_discover(settings, storage)
    assert profiles[0].display_name == "Before Edit"

    _write_profile(profile_path, display_name="After Hand Edit")
    profiles, _ = await reload_profiles_and_discover(settings, storage)
    assert profiles[0].display_name == "After Hand Edit"
