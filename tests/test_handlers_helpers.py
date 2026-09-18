from unittest.mock import AsyncMock, MagicMock

import pytest

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.handlers import (
    _MAIN_KEYBOARD,
    _again_keyboard,
    _characters_keyboard,
    _consume_awaiting_character_edit,
    _consume_awaiting_character_rename,
    _extract_file_id,
    _generate_from_prompt_keyboard,
    _post_process_keyboard,
    _resolve_effective_prompt,
    _resolve_tag_sources,
    _run_reporting_errors,
    _split_negative_prompt,
    _upscale_confirm_keyboard,
    start,
)
from comfytelegram.profiles import ModelProfile
from comfytelegram.tags import TagSource


def test_split_negative_prompt_pulls_out_comma_separated_tags():
    positive, negative = _split_negative_prompt("1girl, outdoors, -blurry, -watermark")
    assert positive == "1girl, outdoors"
    assert negative == "blurry, watermark"


def test_split_negative_prompt_handles_space_separated_tokens():
    positive, negative = _split_negative_prompt("a cute cat -blurry -watermark")
    assert positive == "a cute cat"
    assert negative == "blurry, watermark"


def test_split_negative_prompt_leaves_mid_word_hyphens_alone():
    positive, negative = _split_negative_prompt("a well-lit room")
    assert positive == "a well-lit room"
    assert negative == ""


def test_split_negative_prompt_handles_leading_negative_token():
    positive, negative = _split_negative_prompt("-lonely, a cat")
    assert positive == "a cat"
    assert negative == "lonely"


def test_split_negative_prompt_with_no_negatives_is_unchanged():
    positive, negative = _split_negative_prompt("a cat in a garden")
    assert positive == "a cat in a garden"
    assert negative == ""


def test_split_negative_prompt_handles_a_dash_block_separator():
    positive, negative = _split_negative_prompt(
        "1girl, outdoors\n---\nblurry, watermark, bad anatomy"
    )
    assert positive == "1girl, outdoors"
    assert negative == "blurry, watermark, bad anatomy"


def test_split_negative_prompt_block_separator_normalizes_newline_separated_tags():
    positive, negative = _split_negative_prompt("1girl\noutdoors\n---\nblurry\nwatermark")
    assert positive == "1girl, outdoors"
    assert negative == "blurry, watermark"


def test_split_negative_prompt_block_separator_ignores_mid_word_hyphens():
    positive, negative = _split_negative_prompt("a well-lit room\n---\nblurry")
    assert positive == "a well-lit room"
    assert negative == "blurry"


def test_split_negative_prompt_treats_newlines_like_commas_without_a_block_separator():
    positive, negative = _split_negative_prompt("1girl\noutdoors\n-blurry\n-watermark")
    assert positive == "1girl, outdoors"
    assert negative == "blurry, watermark"


def test_resolve_effective_prompt_without_a_character():
    effective_prompt, extra_negative = _resolve_effective_prompt("1girl, -blurry", None)
    assert effective_prompt == "1girl"
    assert extra_negative == "blurry"


def test_resolve_effective_prompt_folds_in_the_active_character():
    character = {"positive_prompt": "aria, red hair", "negative_prompt": "bad anatomy"}
    effective_prompt, extra_negative = _resolve_effective_prompt("outdoors, -blurry", character)
    assert effective_prompt == "aria, red hair, outdoors"
    assert extra_negative == "bad anatomy, blurry"


def test_post_process_keyboard_scopes_every_button_to_result_id():
    keyboard = _post_process_keyboard("abc123")
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "pp:upscale:abc123" in callback_data
    assert "pp:face:abc123" in callback_data
    assert "pp:hand:abc123" in callback_data
    assert "pp:analyze_only:abc123" in callback_data
    assert "pp:analyze:abc123" in callback_data
    assert "pp:deep_analyze:abc123" in callback_data
    assert "pp:show_prompt:abc123" in callback_data


def test_upscale_confirm_keyboard_scopes_both_buttons_to_result_id():
    keyboard = _upscale_confirm_keyboard("abc123")
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "pp:upscale_confirmed:abc123" in callback_data
    assert "pp:upscale_cancelled:abc123" in callback_data


def test_again_keyboard_scopes_button_to_snapshot_id():
    keyboard = _again_keyboard("snap123")
    button = keyboard.inline_keyboard[0][0]
    assert button.callback_data == "again:snap123"


def test_generate_from_prompt_keyboard_scopes_button_to_prompt_id():
    keyboard = _generate_from_prompt_keyboard("prompt123", "1girl, outdoors")
    generate_button, copy_button = keyboard.inline_keyboard[0]
    assert generate_button.callback_data == "genp:prompt123"
    assert copy_button.copy_text.text == "1girl, outdoors"


def test_generate_from_prompt_keyboard_omits_copy_button_past_telegram_limit():
    keyboard = _generate_from_prompt_keyboard("prompt123", "x" * 257)
    (row,) = keyboard.inline_keyboard
    assert len(row) == 1
    assert row[0].callback_data == "genp:prompt123"


def test_generate_from_prompt_keyboard_includes_copy_button_at_telegram_limit():
    keyboard = _generate_from_prompt_keyboard("prompt123", "x" * 256)
    _generate_button, copy_button = keyboard.inline_keyboard[0]
    assert copy_button.copy_text.text == "x" * 256


def test_characters_keyboard_marks_active_character():
    characters = [{"name": "fox"}, {"name": "wolf"}]
    keyboard = _characters_keyboard(characters, active="fox")
    labels = {b.text: b.callback_data for row in keyboard.inline_keyboard for b in row}
    assert labels["✅ fox"] == "char:activate:fox"
    assert labels["wolf"] == "char:activate:wolf"
    assert "char:clear" in labels.values()


def test_characters_keyboard_has_no_clear_button_when_none_active():
    keyboard = _characters_keyboard([{"name": "fox"}], active=None)
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "char:clear" not in callback_data


def test_characters_keyboard_includes_an_edit_button_per_character():
    characters = [{"name": "fox"}, {"name": "wolf"}]
    keyboard = _characters_keyboard(characters, active="fox")
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "char:edit:fox" in callback_data
    assert "char:edit:wolf" in callback_data


def test_characters_keyboard_includes_a_rename_button_per_character():
    characters = [{"name": "fox"}, {"name": "wolf"}]
    keyboard = _characters_keyboard(characters, active="fox")
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "char:rename:fox" in callback_data
    assert "char:rename:wolf" in callback_data


@pytest.mark.asyncio
async def test_consume_awaiting_character_edit_returns_false_when_not_pending():
    update = MagicMock()
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {"storage": MagicMock()}

    assert await _consume_awaiting_character_edit(update, context) is False
    context.bot_data["storage"].save_character.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_character_edit_saves_the_new_prompt():
    message = AsyncMock()
    message.text = "new positive | new negative"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_edit": "fox"}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_edit(update, context) is True

    storage.save_character.assert_called_once_with(42, "fox", "new positive", "new negative")
    assert "awaiting_character_edit" not in context.chat_data
    message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_awaiting_character_edit_rejects_an_empty_positive_prompt():
    message = AsyncMock()
    message.text = "| just a negative"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_edit": "fox"}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_edit(update, context) is True

    storage.save_character.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_character_rename_returns_false_when_not_pending():
    update = MagicMock()
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {"storage": MagicMock()}

    assert await _consume_awaiting_character_rename(update, context) is False
    context.bot_data["storage"].rename_character.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_character_rename_renames():
    message = AsyncMock()
    message.text = "vixen"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.side_effect = lambda chat_id, name: (
        {"positive_prompt": "old", "negative_prompt": ""} if name == "fox" else None
    )
    context = MagicMock()
    context.chat_data = {"awaiting_character_rename": "fox"}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_rename(update, context) is True

    storage.rename_character.assert_called_once_with(42, "fox", "vixen")
    assert "awaiting_character_rename" not in context.chat_data
    message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_awaiting_character_rename_rejects_an_invalid_name():
    message = AsyncMock()
    message.text = "not a valid name!"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_rename": "fox"}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_rename(update, context) is True

    storage.rename_character.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_character_rename_rejects_a_name_already_taken():
    message = AsyncMock()
    message.text = "wolf"
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_rename": "fox"}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_rename(update, context) is True

    storage.rename_character.assert_not_called()


def test_extract_file_id_prefers_largest_photo_size():
    sent = MagicMock(document=None)
    sent.photo = [MagicMock(file_id="small"), MagicMock(file_id="large")]
    assert _extract_file_id(sent) == "large"


def test_extract_file_id_falls_back_to_document():
    sent = MagicMock(photo=[])
    sent.document = MagicMock(file_id="doc123")
    assert _extract_file_id(sent) == "doc123"


def test_extract_file_id_raises_without_photo_or_document():
    sent = MagicMock(photo=[], document=None)
    with pytest.raises(ValueError):
        _extract_file_id(sent)


@pytest.mark.asyncio
async def test_start_installs_the_main_keyboard():
    message = AsyncMock()
    update = MagicMock()
    update.effective_message = message
    context = MagicMock()
    context.bot_data = {"settings": MagicMock(allowed_user_ids=None)}

    await start(update, context)

    message.reply_text.assert_awaited_once()
    _, kwargs = message.reply_text.await_args
    assert kwargs["reply_markup"] is _MAIN_KEYBOARD


@pytest.mark.asyncio
async def test_run_reporting_errors_returns_result_on_success():
    status_message = AsyncMock()

    async def _ok():
        return "done"

    result = await _run_reporting_errors(status_message, "Generation", "generation", _ok())

    assert result == "done"
    status_message.edit_text.assert_not_called()


@pytest.mark.asyncio
async def test_run_reporting_errors_reports_comfyui_error_and_returns_none():
    status_message = AsyncMock()

    async def _fail():
        raise ComfyUIError("boom")

    result = await _run_reporting_errors(status_message, "Generation", "generation", _fail())

    assert result is None
    status_message.edit_text.assert_awaited_once_with("Generation failed: boom")


@pytest.mark.asyncio
async def test_run_reporting_errors_reports_unexpected_exception_generically():
    status_message = AsyncMock()

    async def _fail():
        raise RuntimeError("kaboom")

    result = await _run_reporting_errors(status_message, "Upscaling", "post-processing", _fail())

    assert result is None
    status_message.edit_text.assert_awaited_once_with("Upscaling failed with an unexpected error.")


def test_resolve_tag_sources_prefix_override_wins_over_profile():
    profile = ModelProfile(match=["*"], display_name="x", tag_dictionary="danbooru")
    sources, remainder = _resolve_tag_sources("e621:fox ears", "ckpt.safetensors", [profile])
    assert sources == [TagSource.E621]
    assert remainder == "fox ears"


def test_resolve_tag_sources_prefix_override_is_case_insensitive():
    sources, remainder = _resolve_tag_sources("DANBOORU:1girl", None, [])
    assert sources == [TagSource.DANBOORU]
    assert remainder == "1girl"


def test_resolve_tag_sources_uses_profile_tag_dictionary_when_no_override():
    profile = ModelProfile(match=["furry*"], display_name="x", tag_dictionary="e621")
    sources, remainder = _resolve_tag_sources("fox", "furrytoonmix.safetensors", [profile])
    assert sources == [TagSource.E621]
    assert remainder == "fox"


def test_resolve_tag_sources_searches_both_when_profile_leaves_it_unset():
    profile = ModelProfile(match=["*"], display_name="x")
    sources, remainder = _resolve_tag_sources("fox", "ckpt.safetensors", [profile])
    assert sources == [TagSource.DANBOORU, TagSource.E621]
    assert remainder == "fox"


def test_resolve_tag_sources_searches_both_when_no_checkpoint_selected():
    sources, remainder = _resolve_tag_sources("fox", None, [])
    assert sources == [TagSource.DANBOORU, TagSource.E621]
    assert remainder == "fox"
