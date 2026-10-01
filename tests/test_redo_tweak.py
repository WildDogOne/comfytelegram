""" "🎚️ Redo…" — a drawn-mask redo with a denoise picked first, instead of
replaying the stored one (`handlers.redo_tweak_callback`)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest
from test_handlers_helpers import _pending_result_mock, _postprocess_context

from comfytelegram.generation import GeneratedImage, resolve_drawn_mask_denoise
from comfytelegram.handlers import (
    HAND_REDO_CALLBACK_KIND,
    REDO_TWEAK_CALLBACK_PREFIX,
    _carried_extra_rows,
    _consume_awaiting_redo_denoise,
    _parse_denoise,
    postprocess_callback,
    redo_tweak_callback,
)
from comfytelegram.profiles import ModelProfile
from comfytelegram.profiles.schema import ProfileDefaults
from comfytelegram.workflows import (
    DrawnMaskFixParams,
    DrawnMaskHandDetailerParams,
    GenerationParams,
)

_REDO_ROW = {
    "source_file_id": "presource123",
    "source_filename": "presource.png",
    "mask_png": b"stored-mask-bytes",
}


def _callback(data: str) -> tuple[MagicMock, AsyncMock]:
    query = AsyncMock()
    query.data = data
    query.message.message_thread_id = None
    query.message.chat_id = 1
    query.message.message_id = 50
    query.message.reply_to_message = None
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1
    return update, query


def _redone() -> GeneratedImage:
    return GeneratedImage(
        data=b"redone",
        filename="out.png",
        full_params=GenerationParams(
            checkpoint="fluffyfurry.safetensors", positive_prompt="a fox", negative_prompt=""
        ),
    )


def _context(redo: dict | None = None) -> MagicMock:
    profile = ModelProfile(match=["*"], display_name="x")
    storage = _pending_result_mock(profile, "a fox")
    storage.get_inpaint_redo.return_value = redo
    context = _postprocess_context(storage, profile)
    context.chat_data = {}
    context.bot.delete_message = AsyncMock()
    return context


def test_parse_denoise():
    assert _parse_denoise("0.42") == 0.42
    assert _parse_denoise(" 0,5 ") == 0.5
    assert _parse_denoise("1") == 1.0
    assert _parse_denoise("0") is None
    assert _parse_denoise("1.2") is None
    assert _parse_denoise("abc") is None


@pytest.mark.asyncio
async def test_redo_result_carries_the_tweak_button_and_it_survives_a_page_flip():
    update, query = _callback(f"pp:{HAND_REDO_CALLBACK_KIND}:abc123")
    context = _context(dict(_REDO_ROW))
    with patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=_redone())):
        await postprocess_callback(update, context)

    keyboard = query.message.reply_photo.await_args.kwargs["reply_markup"]
    new_result_id = context.bot_data["storage"].store_inpaint_redo.call_args.args[0]
    tweak = [
        b
        for row in keyboard.inline_keyboard
        for b in row
        if (b.callback_data or "").startswith(REDO_TWEAK_CALLBACK_PREFIX)
    ]
    assert [b.callback_data for b in tweak] == [f"rt:hand:{new_result_id}"]
    carried = _carried_extra_rows(keyboard)
    assert len(carried) == 1 and len(carried[0]) == 3


@pytest.mark.asyncio
async def test_tweak_opens_picker_marking_the_current_denoise():
    update, query = _callback("rt:hand:abc123")
    context = _context({**_REDO_ROW, "detail_denoise": 0.35})
    await redo_tweak_callback(update, context)

    text = query.message.reply_text.await_args.args[0]
    assert "currently 0.35" in text
    keyboard = query.message.reply_text.await_args.kwargs["reply_markup"]
    labels = [b.text for row in keyboard.inline_keyboard for b in row]
    assert "• 0.35" in labels
    assert "rt:hand:abc123:0.35" in [
        b.callback_data for row in keyboard.inline_keyboard for b in row
    ]


@pytest.mark.asyncio
async def test_tweak_pick_redoes_at_that_denoise_and_stores_it_for_the_next_redo():
    update, query = _callback("rt:fix:abc123:0.6")
    # In a group the picker quotes the result image — replies go there,
    # since the picker itself is deleted before the redo runs.
    result_photo = AsyncMock()
    query.message.reply_to_message = result_photo
    context = _context({**_REDO_ROW, "detail_denoise": 0.3})
    with patch(
        "comfytelegram.handlers.post_process", new=AsyncMock(return_value=_redone())
    ) as post_process_mock:
        await redo_tweak_callback(update, context)

    args, kwargs = post_process_mock.await_args
    assert args[1] == "fix_drawn"
    assert kwargs["mask_bytes"] == b"stored-mask-bytes"
    assert kwargs["denoise"] == 0.6
    storage = context.bot_data["storage"]
    assert storage.store_inpaint_redo.call_args.kwargs["detail_denoise"] == 0.6
    context.bot.delete_message.assert_awaited_once_with(chat_id=1, message_id=50)
    result_photo.reply_photo.assert_awaited_once()
    query.message.reply_photo.assert_not_awaited()


@pytest.mark.asyncio
async def test_tweak_pick_still_redoes_when_the_picker_cant_be_deleted():
    update, query = _callback("rt:hand:abc123:0.6")
    context = _context(dict(_REDO_ROW))
    context.bot.delete_message.side_effect = BadRequest("Message can't be deleted")
    with patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=_redone())):
        await redo_tweak_callback(update, context)
    # Private chat: no quoted result, so the (undeletable) picker is the anchor.
    query.message.reply_photo.assert_awaited_once()


@pytest.mark.asyncio
async def test_tweak_reports_expired_mask():
    update, query = _callback("rt:detail:abc123")
    context = _context(None)
    await redo_tweak_callback(update, context)
    assert "expired" in query.message.reply_text.await_args.args[0]
    assert "✏️ Detail Prompt" in query.message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_custom_denoise_is_consumed_from_the_next_message():
    update, query = _callback("rt:hand:abc123:custom")
    query.message.reply_text.return_value = MagicMock(message_id=51)
    context = _context(dict(_REDO_ROW))
    await redo_tweak_callback(update, context)

    message = AsyncMock()
    message.chat_id = 1
    message.reply_text.return_value = AsyncMock(message_id=53)
    message.text = "nope"
    message.message_thread_id = None
    message.api_kwargs = {}
    text_update = MagicMock()
    text_update.effective_message = message
    # Unparseable: asks again and keeps waiting.
    assert await _consume_awaiting_redo_denoise(text_update, context)
    assert "Couldn't read" in message.reply_text.await_args.args[0]
    context.bot.delete_message.assert_not_awaited()

    message.text = "0.42"
    with patch(
        "comfytelegram.handlers.post_process", new=AsyncMock(return_value=_redone())
    ) as post_process_mock:
        assert await _consume_awaiting_redo_denoise(text_update, context)
    assert post_process_mock.await_args.kwargs["denoise"] == 0.42
    # The picker, the "send the denoise" prompt and the retry prompt are gone.
    assert [c.kwargs["message_id"] for c in context.bot.delete_message.await_args_list] == [
        50,
        51,
        53,
    ]
    # Consumed — the next message is a prompt again.
    assert not await _consume_awaiting_redo_denoise(text_update, context)


def test_resolve_drawn_mask_denoise_fallbacks():
    params = GenerationParams(checkpoint="x.safetensors", positive_prompt="", negative_prompt="")
    plain = ModelProfile(match=["*"], display_name="x")
    tuned = ModelProfile(
        match=["*"], display_name="x", defaults=ProfileDefaults(detailer_denoise=0.33)
    )
    assert resolve_drawn_mask_denoise("hand_drawn", params, [plain], 0.7) == 0.7
    assert resolve_drawn_mask_denoise("hand_drawn", params, [tuned]) == 0.33
    assert (
        resolve_drawn_mask_denoise("hand_drawn", params, [plain])
        == DrawnMaskHandDetailerParams().denoise
    )
    assert resolve_drawn_mask_denoise("fix_drawn", params, [plain]) == DrawnMaskFixParams().denoise


def test_fix_drawn_denoise_override_reaches_the_detailer_graph():
    from test_generation import _solid_mask_png, _solid_png

    from comfytelegram.generation import _build_fix_drawn_post_process, _to_post_process_base

    params = GenerationParams(checkpoint="x.safetensors", positive_prompt="", negative_prompt="")
    graph, _ = _build_fix_drawn_post_process(
        _to_post_process_base(params),
        "src.png",
        "mask.png",
        _solid_png(64, 64, (0, 0, 0)),
        _solid_mask_png(64, 64, 255),
        [],
        0.55,
    )
    denoises = [n["inputs"]["denoise"] for n in graph.values() if "denoise" in n["inputs"]]
    assert denoises == [0.55]
