from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.generation import GeneratedImage
from comfytelegram.handlers import (
    _MAIN_KEYBOARD,
    ANALYZE_PROMPT_CALLBACK_KIND,
    SHOW_PROMPT_CALLBACK_KIND,
    TAGCHECK_TOKEN_LIMIT,
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
    _serialize_generation_params,
    _split_negative_prompt,
    _strip_prompt_weight,
    _tag_prompt_text,
    _tag_results_keyboard,
    _tagcheck_lines,
    _upscale_confirm_keyboard,
    postprocess_callback,
    start,
)
from comfytelegram.profiles import ModelProfile
from comfytelegram.tags import TagResult, TagSource
from comfytelegram.topics import NO_TOPIC
from comfytelegram.workflows import GenerationParams


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
    assert f"pp:{ANALYZE_PROMPT_CALLBACK_KIND}:abc123" in callback_data
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


def test_tag_prompt_text_replaces_underscores_with_spaces():
    assert _tag_prompt_text("blue_eyes") == "blue eyes"


def test_tag_prompt_text_leaves_spaceless_tags_unchanged():
    assert _tag_prompt_text("smile") == "smile"


def test_tag_results_keyboard_copies_spaces_but_displays_underscores():
    result = TagResult(source=TagSource.DANBOORU, name="blue_eyes", category=0, post_count=1000)
    keyboard = _tag_results_keyboard([result])
    (button,) = keyboard.inline_keyboard[0]
    assert button.text == "📋 blue_eyes"
    assert button.copy_text.text == "blue eyes"


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
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_edit": {NO_TOPIC: "fox"}}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_edit(update, context) is True

    storage.save_character.assert_called_once_with(42, "fox", "new positive", "new negative")
    assert "awaiting_character_edit" not in context.chat_data
    message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_awaiting_character_edit_rejects_an_empty_positive_prompt():
    message = AsyncMock()
    message.text = "| just a negative"
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_edit": {NO_TOPIC: "fox"}}
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
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.side_effect = lambda chat_id, name: (
        {"positive_prompt": "old", "negative_prompt": ""} if name == "fox" else None
    )
    context = MagicMock()
    context.chat_data = {"awaiting_character_rename": {NO_TOPIC: "fox"}}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_rename(update, context) is True

    storage.rename_character.assert_called_once_with(42, "fox", "vixen")
    assert "awaiting_character_rename" not in context.chat_data
    message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_awaiting_character_rename_rejects_an_invalid_name():
    message = AsyncMock()
    message.text = "not a valid name!"
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_rename": {NO_TOPIC: "fox"}}
    context.bot_data = {"storage": storage}

    assert await _consume_awaiting_character_rename(update, context) is True

    storage.rename_character.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_character_rename_rejects_a_name_already_taken():
    message = AsyncMock()
    message.text = "wolf"
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42

    storage = MagicMock()
    storage.get_character.return_value = {"positive_prompt": "old", "negative_prompt": ""}
    context = MagicMock()
    context.chat_data = {"awaiting_character_rename": {NO_TOPIC: "fox"}}
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


def _tag_result(name: str, post_count: int, source: TagSource = TagSource.DANBOORU) -> TagResult:
    return TagResult(source=source, name=name, category=0, post_count=post_count)


def test_tagcheck_lines_marks_a_known_common_tag():
    tags_db = MagicMock()
    tags_db.lookup_exact.return_value = _tag_result("1girl", 500_000)
    settings = MagicMock(tag_rare_threshold=100)

    lines = _tagcheck_lines("1girl", [TagSource.DANBOORU], tags_db, settings)

    assert lines == ["✅ 1girl — 500,000 posts (danbooru)"]


def test_tagcheck_lines_flags_a_rare_tag():
    tags_db = MagicMock()
    tags_db.lookup_exact.return_value = _tag_result("obscure_tag", 5)
    settings = MagicMock(tag_rare_threshold=100)

    lines = _tagcheck_lines("obscure_tag", [TagSource.DANBOORU], tags_db, settings)

    assert lines == ["⚠️ obscure_tag — rare (5 posts, danbooru)"]


def test_tagcheck_lines_flags_an_unknown_tag_with_a_suggestion():
    tags_db = MagicMock()
    tags_db.lookup_exact.return_value = None
    tags_db.search.return_value = [_tag_result("1girl", 500_000)]
    settings = MagicMock(tag_rare_threshold=100)

    lines = _tagcheck_lines("1gril", [TagSource.DANBOORU], tags_db, settings)

    assert lines == ['❌ 1gril — not a known tag — did you mean "1girl"?']


def test_strip_prompt_weight_unwraps_an_explicit_weight():
    assert _strip_prompt_weight("(yellow markings:1.2)") == "yellow markings"


def test_strip_prompt_weight_unwraps_plain_emphasis():
    assert _strip_prompt_weight("(masterpiece)") == "masterpiece"


def test_strip_prompt_weight_unwraps_nested_emphasis():
    assert _strip_prompt_weight("((masterpiece))") == "masterpiece"


def test_strip_prompt_weight_unwraps_bracket_de_emphasis():
    assert _strip_prompt_weight("[lowres:0.8]") == "lowres"


def test_strip_prompt_weight_leaves_a_plain_tag_alone():
    assert _strip_prompt_weight("1girl") == "1girl"


def test_strip_prompt_weight_leaves_a_tag_with_a_literal_paren_alone():
    assert _strip_prompt_weight("hat_(costume)") == "hat_(costume)"


def test_tagcheck_lines_looks_up_the_tag_inside_weight_syntax():
    tags_db = MagicMock()
    tags_db.lookup_exact.return_value = _tag_result("yellow markings", 5_000)
    settings = MagicMock(tag_rare_threshold=100)

    lines = _tagcheck_lines("(yellow markings:1.2)", [TagSource.DANBOORU], tags_db, settings)

    tags_db.lookup_exact.assert_called_once_with("yellow markings", [TagSource.DANBOORU])
    assert lines == ["✅ (yellow markings:1.2) — 5,000 posts (danbooru)"]


def test_tagcheck_lines_truncates_at_the_token_limit():
    tags_db = MagicMock()
    tags_db.lookup_exact.return_value = _tag_result("t", 500_000)
    settings = MagicMock(tag_rare_threshold=100)
    prompt = ", ".join(f"tag{i}" for i in range(TAGCHECK_TOKEN_LIMIT + 5))

    lines = _tagcheck_lines(prompt, [TagSource.DANBOORU], tags_db, settings)

    assert len(lines) == TAGCHECK_TOKEN_LIMIT + 1
    assert lines[-1] == f"…5 more token(s) omitted (limit {TAGCHECK_TOKEN_LIMIT})."


def _pending_result_mock(profile: ModelProfile, positive: str, negative: str = "") -> MagicMock:
    params = GenerationParams(
        checkpoint="fluffyfurry.safetensors", positive_prompt=positive, negative_prompt=negative
    )
    storage = MagicMock()
    storage.get_pending_result.return_value = {
        "base_params": _serialize_generation_params(params),
        "chat_id": 1,
        "file_id": "file123",
        "filename": "source.png",
    }
    return storage


@pytest.mark.asyncio
async def test_show_prompt_never_runs_a_tag_check():
    query = AsyncMock()
    query.data = f"pp:{SHOW_PROMPT_CALLBACK_KIND}:abc123"
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    profile = ModelProfile(
        match=["fluffyfurry*"], display_name="x", prompt_style="tags", tag_dictionary="e621"
    )
    storage = _pending_result_mock(profile, "fox, blue_eyes", "blurry")
    tags_db = MagicMock()
    tags_db.stats.return_value = {"e621": 100}

    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, tag_rare_threshold=100),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [profile],
        "tags_db": tags_db,
    }

    await postprocess_callback(update, context)

    assert query.message.reply_text.await_count == 1
    prompt_message = query.message.reply_text.await_args_list[0].args[0]
    assert "fox, blue_eyes" in prompt_message
    tags_db.lookup_exact.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_prompt_runs_a_tag_check_for_a_tag_style_profile():
    query = AsyncMock()
    query.data = f"pp:{ANALYZE_PROMPT_CALLBACK_KIND}:abc123"
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    profile = ModelProfile(
        match=["fluffyfurry*"], display_name="x", prompt_style="tags", tag_dictionary="e621"
    )
    storage = _pending_result_mock(profile, "fox, blue_eyes", "blurry")
    tags_db = MagicMock()
    tags_db.stats.return_value = {"e621": 100}
    tags_db.lookup_exact.return_value = _tag_result("fox", 500_000, TagSource.E621)

    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, tag_rare_threshold=100),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [profile],
        "tags_db": tags_db,
    }

    await postprocess_callback(update, context)

    assert query.message.reply_text.await_count == 1
    tagcheck_message = query.message.reply_text.await_args_list[0].args[0]
    assert "Tag check (positive)" in tagcheck_message
    assert "Tag check (negative)" in tagcheck_message
    assert "fox" in tagcheck_message
    assert "blurry" in tagcheck_message


@pytest.mark.asyncio
async def test_analyze_prompt_reports_unavailable_for_a_natural_language_profile():
    query = AsyncMock()
    query.data = f"pp:{ANALYZE_PROMPT_CALLBACK_KIND}:abc123"
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    profile = ModelProfile(match=["anima*"], display_name="x", prompt_style="natural")
    storage = _pending_result_mock(profile, "a fox in the snow")
    tags_db = MagicMock()
    tags_db.stats.return_value = {"e621": 100}

    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, tag_rare_threshold=100),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [profile],
        "tags_db": tags_db,
    }

    await postprocess_callback(update, context)

    assert query.message.reply_text.await_count == 1
    message = query.message.reply_text.await_args_list[0].args[0]
    assert "No tag check available" in message
    tags_db.lookup_exact.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_prompt_reports_unavailable_when_no_tag_data_imported():
    query = AsyncMock()
    query.data = f"pp:{ANALYZE_PROMPT_CALLBACK_KIND}:abc123"
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    profile = ModelProfile(
        match=["fluffyfurry*"], display_name="x", prompt_style="tags", tag_dictionary="e621"
    )
    storage = _pending_result_mock(profile, "fox")
    tags_db = MagicMock()
    tags_db.stats.return_value = {"e621": 0, "danbooru": 0}

    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, tag_rare_threshold=100),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [profile],
        "tags_db": tags_db,
    }

    await postprocess_callback(update, context)

    assert query.message.reply_text.await_count == 1
    message = query.message.reply_text.await_args_list[0].args[0]
    assert "No tag check available" in message


def _postprocess_context(storage: MagicMock, profile: ModelProfile) -> MagicMock:
    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [profile],
    }
    context.bot.get_file = AsyncMock(
        return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=bytearray(b"orig")))
    )
    return context


@pytest.mark.asyncio
async def test_face_detail_with_no_detection_sends_a_warning_instead_of_the_image():
    query = AsyncMock()
    query.data = "pp:face:abc123"
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    profile = ModelProfile(match=["*"], display_name="x")
    storage = _pending_result_mock(profile, "a fox")
    context = _postprocess_context(storage, profile)

    unchanged_result = GeneratedImage(
        data=b"orig",
        filename="out.png",
        full_params=GenerationParams(
            checkpoint="fluffyfurry.safetensors", positive_prompt="a fox", negative_prompt=""
        ),
        unchanged=True,
    )
    with patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=unchanged_result)):
        await postprocess_callback(update, context)

    assert query.message.reply_text.await_args_list[-1].args[0] == (
        "⚠️ No face detected — image unchanged."
    )
    query.message.reply_photo.assert_not_called()
    storage.store_pending_result.assert_not_called()


@pytest.mark.asyncio
async def test_hand_detail_with_a_real_change_sends_the_image_normally():
    query = AsyncMock()
    query.data = "pp:hand:abc123"
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    profile = ModelProfile(match=["*"], display_name="x")
    storage = _pending_result_mock(profile, "a fox")
    context = _postprocess_context(storage, profile)

    changed_result = GeneratedImage(
        data=b"refined",
        filename="out.png",
        full_params=GenerationParams(
            checkpoint="fluffyfurry.safetensors", positive_prompt="a fox", negative_prompt=""
        ),
        unchanged=False,
    )
    with patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=changed_result)):
        await postprocess_callback(update, context)

    query.message.reply_photo.assert_awaited_once()
    storage.store_pending_result.assert_called_once()
    assert not any("unchanged" in call.args[0] for call in query.message.reply_text.await_args_list)
