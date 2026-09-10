from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfytelegram.handlers import photo_message


def _settings_mock() -> MagicMock:
    return MagicMock(allowed_user_ids=None)


def _photo_update_mock(*, chat_id: int = 1, user_id: int = 1) -> tuple[MagicMock, AsyncMock]:
    message = AsyncMock()
    message.photo = [MagicMock(file_id="small"), MagicMock(file_id="large")]
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    return update, message


def _context_mock(*, checkpoint: str | None = "ckpt.safetensors") -> MagicMock:
    context = MagicMock()
    storage = MagicMock()
    storage.get_checkpoint.return_value = checkpoint
    client = AsyncMock()
    context.bot_data = {"settings": _settings_mock(), "storage": storage, "comfy_client": client}
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(b"image-bytes")
    context.bot = AsyncMock()
    context.bot.get_file.return_value = tg_file
    return context


@pytest.mark.asyncio
async def test_photo_message_sends_both_derived_prompts_with_generate_buttons():
    update, message = _photo_update_mock()
    context = _context_mock()

    with (
        patch("comfytelegram.handlers.analyze_tags", new=AsyncMock(return_value="1girl, outdoors")),
        patch(
            "comfytelegram.handlers.analyze_caption", new=AsyncMock(return_value="a girl outside")
        ),
    ):
        await photo_message(update, context)

    context.bot.get_file.assert_awaited_once_with("large")  # largest photo size
    storage = context.bot_data["storage"]
    assert storage.store_derived_prompt.call_count == 2

    texts = [call.args[0] for call in message.reply_text.await_args_list]
    assert any(t.startswith("🏷️ WD14 tags:\n1girl, outdoors") for t in texts)
    assert any(t.startswith("💬 Qwen-VL caption:\na girl outside") for t in texts)


@pytest.mark.asyncio
async def test_photo_message_reports_one_analyzer_failing_without_blocking_the_other():
    update, message = _photo_update_mock()
    context = _context_mock()

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("WD14 model files missing")

    with (
        patch("comfytelegram.handlers.analyze_tags", new=_boom),
        patch(
            "comfytelegram.handlers.analyze_caption", new=AsyncMock(return_value="a girl outside")
        ),
    ):
        await photo_message(update, context)

    storage = context.bot_data["storage"]
    storage.store_derived_prompt.assert_called_once()  # only the caption succeeded

    texts = [call.args[0] for call in message.reply_text.await_args_list]
    assert any("🏷️ WD14 tags: failed" in t for t in texts)
    assert any(t.startswith("💬 Qwen-VL caption:\na girl outside") for t in texts)


@pytest.mark.asyncio
async def test_photo_message_stops_when_no_checkpoint_available():
    update, message = _photo_update_mock()
    context = _context_mock(checkpoint=None)
    context.bot_data["comfy_client"].list_checkpoints.return_value = []

    await photo_message(update, context)

    message.reply_text.assert_awaited_once_with("ComfyUI reports no checkpoints installed.")
    context.bot.get_file.assert_not_called()
