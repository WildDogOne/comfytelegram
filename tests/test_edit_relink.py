"""Sending an edited image back as a *reply* to the bot result it came from
re-attaches that result's settings to it — the round trip for an image
edited in an app that dropped `png_metadata`'s chunk.

See `handlers._link_edit_to_result`, and `TELEGRAM_ONLY_FILENAME_PREFIX`
for why the new row must never fetch its source from ComfyUI.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from comfytelegram.handlers import (
    TELEGRAM_ONLY_FILENAME_PREFIX,
    _fetch_source_image,
    _post_process_keyboard,
    document_message,
    photo_message,
    postprocess_callback,
)
from comfytelegram.params_serde import serialize_generation_params
from comfytelegram.png_metadata import extract_metadata
from comfytelegram.workflows import GenerationParams

_PARAMS = serialize_generation_params(
    GenerationParams(
        checkpoint="ponyxl.safetensors",
        positive_prompt="score_9, a fox",
        negative_prompt="worst quality",
        steps=28,
        cfg=6.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        width=832,
        height=1216,
        clip_skip=-2,
    )
)


def _bot_result(result_id: str = "orig1", page: int = 1) -> MagicMock:
    replied = MagicMock()
    replied.reply_markup = _post_process_keyboard(result_id, page)
    return replied


def _context(pending: dict | None) -> MagicMock:
    context = MagicMock()
    storage = MagicMock()
    storage.get_pending_result.return_value = pending
    storage.get_checkpoint.return_value = "ponyxl.safetensors"
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "comfy_client": AsyncMock(),
    }
    context.bot = AsyncMock()
    return context


def _pending() -> dict:
    return {
        "chat_id": 42,
        "file_id": "orig-photo",
        "filename": "comfytelegram_00042_.png",
        "base_params": _PARAMS,
    }


def _photo_update(reply_to) -> tuple[MagicMock, AsyncMock]:
    message = AsyncMock()
    message.photo = [MagicMock(file_id="small"), MagicMock(file_id="edited-photo")]
    message.reply_to_message = reply_to
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42
    update.effective_user.id = 1
    return update, message


def _document_update(reply_to) -> tuple[MagicMock, AsyncMock]:
    message = AsyncMock()
    message.document = MagicMock(file_id="edited-doc", file_size=1000, file_name="edit.png")
    message.reply_to_message = reply_to
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42
    update.effective_user.id = 1
    return update, message


@pytest.mark.asyncio
async def test_a_document_replying_to_a_result_inherits_its_settings_without_analysis():
    update, message = _document_update(_bot_result("orig1"))
    context = _context(_pending())

    await document_message(update, context)

    storage = context.bot_data["storage"]
    storage.get_pending_result.assert_called_once_with("orig1")
    result_id, chat_id, file_id, filename, stored = storage.store_pending_result.call_args.args
    assert (chat_id, file_id, stored) == (42, "edited-doc", _PARAMS)
    assert filename.startswith(TELEGRAM_ONLY_FILENAME_PREFIX)
    assert result_id != "orig1"  # a fresh row: the old one's redo state stays behind
    context.bot.get_file.assert_not_awaited()  # nothing downloaded, nothing analyzed

    keyboard = message.reply_text.await_args.kwargs["reply_markup"]
    callbacks = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert f"pp:upscale:{result_id}" in callbacks
    assert "ponyxl.safetensors" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_a_photo_reply_links_too_and_mentions_the_compression():
    """An edit often comes back as a photo; that's fine, just lossy."""
    update, message = _photo_update(_bot_result("orig1", page=2))
    context = _context(_pending())

    with patch("comfytelegram.handlers.analyze_caption_deep", new=AsyncMock()) as deep:
        await photo_message(update, context)

    deep.assert_not_awaited()
    args = context.bot_data["storage"].store_pending_result.call_args.args
    assert args[2] == "edited-photo"  # the largest size
    assert "compressed" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_a_reply_to_an_expired_result_says_so_instead_of_analyzing():
    update, message = _photo_update(_bot_result("gone"))
    context = _context(None)

    with patch("comfytelegram.handlers.analyze_caption_deep", new=AsyncMock()) as deep:
        await photo_message(update, context)

    deep.assert_not_awaited()
    context.bot_data["storage"].store_pending_result.assert_not_called()
    assert "expired" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply_to",
    [
        None,
        # a reply to something that isn't a result (an analyzer reply, a
        # forum topic's opening message, ...) is just an ordinary upload
        MagicMock(
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("🎨 Generate", callback_data="gp:abc")]]
            )
        ),
        MagicMock(reply_markup=None),
    ],
)
async def test_an_upload_not_replying_to_a_result_is_analyzed_as_before(reply_to):
    update, _message = _photo_update(reply_to)
    context = _context(_pending())
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(b"image-bytes")
    context.bot.get_file.return_value = tg_file

    with (
        patch("comfytelegram.handlers.analyze_tags", new=AsyncMock(return_value="1girl")),
        patch(
            "comfytelegram.handlers.analyze_caption_deep",
            new=AsyncMock(return_value=("a girl", "")),
        ) as deep,
    ):
        await photo_message(update, context)

    deep.assert_awaited_once()
    context.bot_data["storage"].store_pending_result.assert_not_called()


@pytest.mark.asyncio
async def test_a_telegram_only_source_never_consults_comfyui():
    """ComfyUI may well have a file by the original name — the unedited
    image. Using it would silently undo the edit."""
    client = AsyncMock()
    bot = AsyncMock()
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(b"edited")
    bot.get_file.return_value = tg_file
    on_fallback = AsyncMock()

    data = await _fetch_source_image(
        client, bot, f"{TELEGRAM_ONLY_FILENAME_PREFIX}edited_x.png", "fid", on_fallback=on_fallback
    )

    assert data == b"edited"
    client.get_image_bytes.assert_not_awaited()
    on_fallback.assert_not_awaited()  # not a fallback: the intended source


@pytest.mark.asyncio
async def test_downloading_a_photo_linked_edit_gives_a_png_with_its_settings():
    jpeg = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(jpeg, format="JPEG")

    query = AsyncMock()
    query.data = "pp:archive:new1"
    query.message = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1
    pending = _pending() | {
        "file_id": "edited-photo",
        "filename": f"{TELEGRAM_ONLY_FILENAME_PREFIX}edited_new1.png",
    }
    context = _context(pending)
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(jpeg.getvalue())
    context.bot.get_file.return_value = tg_file

    await postprocess_callback(update, context)

    context.bot_data["comfy_client"].get_image_bytes.assert_not_awaited()
    context.bot.get_file.assert_awaited_once_with("edited-photo")
    sent = query.message.reply_document.await_args.kwargs
    assert sent["filename"] == "edited_new1.png"
    metadata = extract_metadata(sent["document"].getvalue())
    assert metadata["params"] == _PARAMS
