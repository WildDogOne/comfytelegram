from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfytelegram.civitai import CivitaiLoraInfo
from comfytelegram.lora_menu import (
    _NO_LORAS_TEXT,
    _display_name,
    _home_keyboard,
    _lora_field_keyboard,
    _lora_field_text,
    _render,
    _truncate,
    handle_lora_custom_value_message,
    lora_callback,
    lora_command,
)
from comfytelegram.profiles import LoraDefault, ModelProfile
from comfytelegram.storage import Storage
from comfytelegram.topics import NO_TOPIC


def _profile_with_loras() -> ModelProfile:
    return ModelProfile(
        match=["*ckpt*"],
        display_name="Test Model",
        loras=[
            LoraDefault(name="styleA.safetensors", strength_model=0.8, default_enabled=True),
            LoraDefault(name="styleB.safetensors", strength_model=1.0, default_enabled=False),
        ],
    )


def _context(
    storage: Storage, profiles: list[ModelProfile], loras_dir: Path | None = None
) -> MagicMock:
    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, comfyui_loras_dir=loras_dir),
        "storage": storage,
        "profiles": profiles,
    }
    context.chat_data = {}
    return context


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def test_truncate_leaves_short_strings_alone():
    assert _truncate("short.safetensors") == "short.safetensors"


def test_truncate_shortens_long_strings_with_ellipsis():
    result = _truncate("a" * 80, n=10)
    assert result.endswith("…")
    assert len(result) == 10


def test_display_name_strips_subdirectories():
    assert _display_name("illustrious/yakovlev-vad/yakovlev-vad.safetensors") == (
        "yakovlev-vad.safetensors"
    )


def test_display_name_leaves_a_flat_filename_alone():
    assert _display_name("styleA.safetensors") == "styleA.safetensors"


def test_home_keyboard_shows_only_the_filename_not_the_full_path():
    profile = ModelProfile(
        match=["*ckpt*"],
        display_name="Test Model",
        loras=[
            LoraDefault(
                name="illustrious/yakovlev-vad/yakovlev-vad.safetensors",
                strength_model=1.0,
                default_enabled=True,
            )
        ],
    )
    keyboard = _home_keyboard(profile, show_info=False)
    label = keyboard.inline_keyboard[0][0].text
    assert "yakovlev-vad.safetensors" in label
    assert "illustrious/" not in label


def test_render_with_no_loras_shows_the_no_loras_message_with_reload_and_close():
    text, keyboard = _render("ckpt.safetensors", None, show_info=False)
    assert text == _NO_LORAS_TEXT
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert callback_data == ["lr:reload", "lr:close"]

    empty_profile = ModelProfile(match=["*ckpt*"], display_name="X")
    text, _ = _render("ckpt.safetensors", empty_profile, show_info=False)
    assert text == _NO_LORAS_TEXT


def test_home_keyboard_marks_each_lora_enabled_state_and_shows_strength():
    keyboard = _home_keyboard(_profile_with_loras(), show_info=False)
    all_buttons = [b for row in keyboard.inline_keyboard for b in row]
    style_a = next(b for b in all_buttons if b.callback_data == "lr:t:0")
    style_b = next(b for b in all_buttons if b.callback_data == "lr:t:1")
    assert style_a.text.startswith("✅")
    assert "styleA.safetensors" in style_a.text
    assert "0.8" in style_a.text
    assert style_b.text.startswith("◻️")
    assert "styleB.safetensors" in style_b.text


def test_home_keyboard_has_reset_all_reload_and_close():
    keyboard = _home_keyboard(_profile_with_loras(), show_info=False)
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "lr:ra" in callback_data
    assert "lr:reload" in callback_data
    assert "lr:close" in callback_data


def test_home_keyboard_omits_info_buttons_when_not_configured():
    keyboard = _home_keyboard(_profile_with_loras(), show_info=False)
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert not any(cd.startswith("lr:i:") for cd in callback_data)


def test_home_keyboard_shows_info_buttons_when_configured():
    keyboard = _home_keyboard(_profile_with_loras(), show_info=True)
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "lr:i:0" in callback_data
    assert "lr:i:1" in callback_data


def test_home_keyboard_gives_each_toggle_its_own_full_width_row():
    """Telegram splits a row's width evenly across its buttons with no way
    to weight one wider — a toggle sharing a row with ⚙️/ℹ️ truncated even
    short filenames to a handful of characters (seen in an actual
    screenshot). The toggle must be alone in its row; ⚙️/ℹ️ share a
    separate, narrower one below it."""
    keyboard = _home_keyboard(_profile_with_loras(), show_info=True)
    toggle_row = next(row for row in keyboard.inline_keyboard if row[0].callback_data == "lr:t:0")
    assert len(toggle_row) == 1
    secondary_row = next(
        row for row in keyboard.inline_keyboard if row[0].callback_data == "lr:f:0"
    )
    assert [b.callback_data for b in secondary_row] == ["lr:f:0", "lr:i:0"]


@pytest.mark.asyncio
async def test_lora_command_with_no_checkpoint_asks_to_pick_a_model(storage: Storage):
    update = MagicMock()
    update.effective_chat.id = 1
    update.effective_message.reply_text = AsyncMock()
    context = _context(storage, [])

    await lora_command(update, context)

    update.effective_message.reply_text.assert_awaited_once()
    assert "/model" in update.effective_message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_lora_command_with_no_configured_loras_says_so(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    update = MagicMock()
    update.effective_chat.id = 1
    update.effective_message.reply_text = AsyncMock()
    context = _context(storage, [ModelProfile(match=["*ckpt*"], display_name="X")])

    await lora_command(update, context)

    update.effective_message.reply_text.assert_awaited_once()
    assert update.effective_message.reply_text.await_args.args[0] == _NO_LORAS_TEXT


@pytest.mark.asyncio
async def test_lora_command_shows_toggles_for_the_selected_checkpoint(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    update = MagicMock()
    update.effective_chat.id = 1
    update.effective_message.reply_text = AsyncMock()
    context = _context(storage, [_profile_with_loras()])

    await lora_command(update, context)

    await_args = update.effective_message.reply_text.await_args
    assert "Test Model" in await_args.args[0]
    callback_data = [
        b.callback_data for row in await_args.kwargs["reply_markup"].inline_keyboard for b in row
    ]
    assert "lr:t:0" in callback_data
    assert "lr:t:1" in callback_data


@pytest.mark.asyncio
async def test_toggle_action_flips_and_persists_the_override(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:t:1"  # styleB, currently default_enabled=False
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    assert storage.get_lora_overrides(1, "ckpt.safetensors") == {"styleB.safetensors": True}
    query.answer.assert_awaited_once()
    assert "on" in query.answer.await_args.args[0]

    # tapping again flips it back off
    await lora_callback(update, context)
    assert storage.get_lora_overrides(1, "ckpt.safetensors") == {"styleB.safetensors": False}


@pytest.mark.asyncio
async def test_reset_action_clears_stored_overrides(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    storage.set_lora_override(1, "ckpt.safetensors", "styleA.safetensors", False)
    query = AsyncMock()
    query.data = "lr:ra"
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    assert storage.get_lora_overrides(1, "ckpt.safetensors") == {}


@pytest.mark.asyncio
async def test_close_action_deletes_the_message(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:close"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    query.message.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_toggle_with_no_checkpoint_selected_shows_an_alert(storage: Storage):
    query = AsyncMock()
    query.data = "lr:t:0"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [])

    await lora_callback(update, context)

    query.answer.assert_awaited_once_with("No model selected — use /model first.", show_alert=True)


@pytest.mark.asyncio
async def test_info_action_without_loras_dir_configured_shows_an_alert(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:i:0"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()], loras_dir=None)

    await lora_callback(update, context)

    query.answer.assert_awaited_once_with(
        "CivitAI info isn't set up — set COMFYUI_LORAS_DIR.", show_alert=True
    )
    query.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_info_action_reports_a_missing_file(storage: Storage, tmp_path: Path):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:i:0"  # styleA.safetensors — never created under loras_dir
    status = AsyncMock()
    query.message.reply_text = AsyncMock(return_value=status)
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()], loras_dir=loras_dir)

    await lora_callback(update, context)

    status.edit_text.assert_awaited_once()
    text = status.edit_text.await_args.args[0]
    assert "Couldn't find" in text
    assert "styleA.safetensors" in text


@pytest.mark.asyncio
async def test_info_action_hashes_queries_and_caches_a_civitai_match(
    storage: Storage, tmp_path: Path
):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "styleA.safetensors").write_bytes(b"fake lora bytes")
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:i:0"
    status = AsyncMock()
    query.message.reply_text = AsyncMock(return_value=status)
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()], loras_dir=loras_dir)

    info = CivitaiLoraInfo(
        model_name="Cool Style",
        base_model="SDXL 1.0",
        trained_words=["cool style", "trigger2"],
        civitai_url="https://civitai.com/models/123?modelVersionId=456",
    )
    with patch("comfytelegram.lora_menu.fetch_civitai_info", AsyncMock(return_value=info)):
        await lora_callback(update, context)

    text = status.edit_text.await_args.args[0]
    assert "Cool Style" in text
    assert "SDXL 1.0" in text
    assert "cool style, trigger2" in text
    assert "civitai.com/models/123" in text

    # cached, so a second tap doesn't hash/query again
    cached = storage.get_lora_civitai_cache("styleA.safetensors")
    assert cached is not None
    assert cached["found"] is True
    assert cached["model_name"] == "Cool Style"

    with patch("comfytelegram.lora_menu.fetch_civitai_info", AsyncMock(side_effect=AssertionError)):
        await lora_callback(update, context)  # would raise if it re-fetched instead of caching


@pytest.mark.asyncio
async def test_info_action_caches_a_civitai_miss_too(storage: Storage, tmp_path: Path):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "styleA.safetensors").write_bytes(b"fake lora bytes")
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:i:0"
    status = AsyncMock()
    query.message.reply_text = AsyncMock(return_value=status)
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()], loras_dir=loras_dir)

    with patch("comfytelegram.lora_menu.fetch_civitai_info", AsyncMock(return_value=None)):
        await lora_callback(update, context)

    text = status.edit_text.await_args.args[0]
    assert "No CivitAI match" in text
    cached = storage.get_lora_civitai_cache("styleA.safetensors")
    assert cached is not None
    assert cached["found"] is False


@pytest.mark.asyncio
async def test_refresh_action_bypasses_the_cache(storage: Storage, tmp_path: Path):
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / "styleA.safetensors").write_bytes(b"fake lora bytes")
    storage.set_checkpoint(1, "ckpt.safetensors")
    storage.set_lora_civitai_cache("styleA.safetensors", "stale-hash", None)

    query = AsyncMock()
    query.data = "lr:ir:0"
    status = AsyncMock()
    query.message.reply_text = AsyncMock(return_value=status)
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()], loras_dir=loras_dir)

    info = CivitaiLoraInfo(
        model_name="Cool Style",
        base_model="SDXL 1.0",
        trained_words=[],
        civitai_url="https://civitai.com/models/123",
    )
    with patch("comfytelegram.lora_menu.fetch_civitai_info", AsyncMock(return_value=info)) as m:
        await lora_callback(update, context)

    m.assert_awaited_once()
    text = status.edit_text.await_args.args[0]
    assert "Cool Style" in text
    assert storage.get_lora_civitai_cache("styleA.safetensors")["found"] is True


def test_lora_field_text_shows_both_strengths():
    lora = LoraDefault(
        name="illustrious/medieval.safetensors", strength_model=0.8, strength_clip=0.9
    )
    text = _lora_field_text(lora)
    assert "medieval.safetensors" in text
    assert "illustrious/" not in text
    assert "0.8" in text
    assert "0.9" in text


def test_lora_field_keyboard_has_steppers_custom_and_nav_but_no_presets():
    lora = LoraDefault(name="styleA.safetensors", strength_model=0.8, strength_clip=0.9)
    keyboard = _lora_field_keyboard(0, lora)
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]

    assert "lr:d:m:0:-0.05" in callback_data
    assert "lr:d:m:0:0.05" in callback_data
    assert "lr:c:m:0" in callback_data
    assert "lr:d:c:0:-0.05" in callback_data
    assert "lr:d:c:0:0.05" in callback_data
    assert "lr:c:c:0" in callback_data
    assert "lr:pr:0" in callback_data
    assert "lr:home" in callback_data
    # no preset-value row — removed for looking too busy/chaotic
    assert not any(cd.startswith("lr:v:") for cd in callback_data)


@pytest.mark.asyncio
async def test_open_field_screen_action_shows_the_lora_strengths(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:f:0"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    query.edit_message_text.assert_awaited_once()
    text = query.edit_message_text.await_args.args[0]
    assert "styleA.safetensors" in text
    assert "0.8" in text


@pytest.mark.asyncio
async def test_home_action_navigates_back(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:home"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    text = query.edit_message_text.await_args.args[0]
    assert "Test Model" in text


@pytest.mark.asyncio
async def test_noop_action_just_acks(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:noop"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    query.answer.assert_awaited_once_with()
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_stepper_action_adjusts_and_persists_strength(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:d:m:0:0.05"  # styleA's strength_model, currently 0.8
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    assert storage.get_lora_strength_overrides(1, "ckpt.safetensors") == {
        "styleA.safetensors": {"strength_model": 0.85}
    }
    text = query.edit_message_text.await_args.args[0]
    assert "0.85" in text


@pytest.mark.asyncio
async def test_preset_action_no_longer_exists(storage: Storage):
    """`lr:v:...` used to set a preset value directly; the preset row was
    removed for looking too busy, and nothing emits this callback_data
    anymore — it must fall through to the generic "unknown action" reply
    rather than silently doing something (e.g. matching `action in
    ("d", "c")` by accident)."""
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:v:c:1:0.5"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    query.answer.assert_awaited_once_with("Unknown action.", show_alert=True)
    assert storage.get_lora_strength_overrides(1, "ckpt.safetensors") == {}


@pytest.mark.asyncio
async def test_reset_one_lora_action_clears_only_that_loras_strength(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    storage.set_lora_strength_override(
        1, "ckpt.safetensors", "styleA.safetensors", "strength_model", 0.5
    )
    storage.set_lora_strength_override(
        1, "ckpt.safetensors", "styleB.safetensors", "strength_model", 0.5
    )
    query = AsyncMock()
    query.data = "lr:pr:0"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    assert storage.get_lora_strength_overrides(1, "ckpt.safetensors") == {
        "styleB.safetensors": {"strength_model": 0.5}
    }


@pytest.mark.asyncio
async def test_reset_all_action_also_clears_strength_overrides(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    storage.set_lora_override(1, "ckpt.safetensors", "styleA.safetensors", False)
    storage.set_lora_strength_override(
        1, "ckpt.safetensors", "styleA.safetensors", "strength_model", 0.5
    )
    query = AsyncMock()
    query.data = "lr:ra"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    assert storage.get_lora_overrides(1, "ckpt.safetensors") == {}
    assert storage.get_lora_strength_overrides(1, "ckpt.safetensors") == {}


@pytest.mark.asyncio
async def test_custom_action_sets_pending_and_prompts(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:c:m:0"
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    await lora_callback(update, context)

    assert context.chat_data["awaiting_lora_strength"][NO_TOPIC] == (
        "ckpt.safetensors",
        "styleA.safetensors",
        "strength_model",
        0,
    )
    text = query.edit_message_text.await_args.args[0]
    assert "Model strength" in text


@pytest.mark.asyncio
async def test_custom_value_message_applies_the_typed_strength(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    context = _context(storage, [_profile_with_loras()])
    message = AsyncMock()
    message.message_thread_id = None
    message.text = "0.42"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    context.chat_data["awaiting_lora_strength"] = {
        NO_TOPIC: ("ckpt.safetensors", "styleA.safetensors", "strength_model", 0)
    }

    handled = await handle_lora_custom_value_message(update, context)

    assert handled is True
    assert storage.get_lora_strength_overrides(1, "ckpt.safetensors") == {
        "styleA.safetensors": {"strength_model": 0.42}
    }
    message.reply_text.assert_awaited_once()
    assert "0.42" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_custom_value_message_rejects_a_non_numeric_reply(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    context = _context(storage, [_profile_with_loras()])
    message = AsyncMock()
    message.message_thread_id = None
    message.text = "not a number"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    context.chat_data["awaiting_lora_strength"] = {
        NO_TOPIC: ("ckpt.safetensors", "styleA.safetensors", "strength_model", 0)
    }

    handled = await handle_lora_custom_value_message(update, context)

    assert handled is True
    assert storage.get_lora_strength_overrides(1, "ckpt.safetensors") == {}
    assert "Invalid value" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_custom_value_message_returns_false_when_nothing_pending(storage: Storage):
    context = _context(storage, [_profile_with_loras()])
    message = AsyncMock()
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message

    handled = await handle_lora_custom_value_message(update, context)

    assert handled is False


@pytest.mark.asyncio
async def test_reload_action_updates_profiles_and_shows_the_refreshed_home_screen(
    storage: Storage,
):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:reload"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    reloaded_profile = ModelProfile(
        match=["*ckpt*"],
        display_name="Reloaded Model",
        loras=[LoraDefault(name="new_style.safetensors", default_enabled=False)],
    )
    with patch(
        "comfytelegram.lora_menu.reload_profiles_and_discover",
        AsyncMock(return_value=([reloaded_profile], False)),
    ) as mock_reload:
        await lora_callback(update, context)

    mock_reload.assert_awaited_once()
    assert context.bot_data["profiles"] == [reloaded_profile]
    final_text = query.edit_message_text.await_args_list[-1].args[0]
    assert "Reloaded Model" in final_text
    query.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_reload_action_announces_newly_discovered_loras(storage: Storage):
    storage.set_checkpoint(1, "ckpt.safetensors")
    query = AsyncMock()
    query.data = "lr:reload"
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context(storage, [_profile_with_loras()])

    with patch(
        "comfytelegram.lora_menu.reload_profiles_and_discover",
        AsyncMock(return_value=([_profile_with_loras()], True)),
    ):
        await lora_callback(update, context)

    query.message.reply_text.assert_awaited_once()
    assert "New LoRA" in query.message.reply_text.await_args.args[0]
