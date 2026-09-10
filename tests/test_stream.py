from unittest.mock import AsyncMock, MagicMock

import pytest

from comfytelegram.handlers import (
    _MAIN_KEYBOARD,
    _finish_stream,
    _resolve_checkpoint_or_default,
    stop_command,
    stream_command,
)


def _settings_mock() -> MagicMock:
    return MagicMock(allowed_user_ids=None)


def _update_mock(*, chat_id: int = 1, user_id: int = 1) -> tuple[MagicMock, AsyncMock]:
    message = AsyncMock()
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    return update, message


@pytest.mark.asyncio
async def test_resolve_checkpoint_or_default_returns_stored_checkpoint():
    message = AsyncMock()
    storage = MagicMock()
    storage.get_checkpoint.return_value = "ckpt.safetensors"
    client = AsyncMock()
    context = MagicMock()
    context.bot_data = {}

    result = await _resolve_checkpoint_or_default(message, 1, storage, client, context)

    assert result == "ckpt.safetensors"
    client.list_checkpoints.assert_not_called()
    message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_checkpoint_or_default_falls_back_to_first_available():
    message = AsyncMock()
    storage = MagicMock()
    storage.get_checkpoint.return_value = None
    client = AsyncMock()
    client.list_checkpoints.return_value = ["a.safetensors", "b.safetensors"]
    context = MagicMock()
    context.bot_data = {}

    result = await _resolve_checkpoint_or_default(message, 1, storage, client, context)

    assert result == "a.safetensors"
    storage.set_checkpoint.assert_called_once_with(1, "a.safetensors")
    assert context.bot_data["available_checkpoints"] == ["a.safetensors", "b.safetensors"]
    message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_checkpoint_or_default_reports_when_none_installed():
    message = AsyncMock()
    storage = MagicMock()
    storage.get_checkpoint.return_value = None
    client = AsyncMock()
    client.list_checkpoints.return_value = []
    context = MagicMock()
    context.bot_data = {}

    result = await _resolve_checkpoint_or_default(message, 1, storage, client, context)

    assert result is None
    message.reply_text.assert_awaited_once_with("ComfyUI reports no checkpoints installed.")


@pytest.mark.asyncio
async def test_stream_command_requires_a_prompt():
    update, message = _update_mock()
    message.text = "/stream"
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock()}

    await stream_command(update, context)

    assert message.reply_text.await_args.args[0].startswith("Usage: /stream")
    assert "active_streams" not in context.bot_data


@pytest.mark.asyncio
async def test_stream_command_rejects_when_already_running():
    existing_task = MagicMock()
    existing_task.done.return_value = False
    update, message = _update_mock()
    message.text = "/stream a fox"
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock(), "active_streams": {1: existing_task}}

    await stream_command(update, context)

    message.reply_text.assert_awaited_once_with(
        "A stream is already running in this chat — /stop it first."
    )
    assert context.bot_data["active_streams"][1] is existing_task


@pytest.mark.asyncio
async def test_stop_command_reports_no_active_stream():
    update, message = _update_mock()
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock()}

    await stop_command(update, context)

    message.reply_text.assert_awaited_once_with("No stream is running in this chat.")


@pytest.mark.asyncio
async def test_stop_command_reports_no_active_stream_once_task_is_done():
    task = MagicMock()
    task.done.return_value = True
    update, message = _update_mock()
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock(), "active_streams": {1: task}}

    await stop_command(update, context)

    message.reply_text.assert_awaited_once_with("No stream is running in this chat.")
    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_stop_command_cancels_running_stream():
    task = MagicMock()
    task.done.return_value = False
    update, message = _update_mock()
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock(), "active_streams": {1: task}}

    await stop_command(update, context)

    task.cancel.assert_called_once()
    message.reply_text.assert_awaited_once_with("Stopping the stream…")


@pytest.mark.asyncio
async def test_finish_stream_is_a_noop_without_a_status_message():
    """No status message exists yet if the stream never got past checkpoint
    resolution (e.g. ComfyUI unreachable) — nothing to reply into."""
    await _finish_stream(None, "Stream stopped after 0 image(s).")


@pytest.mark.asyncio
async def test_finish_stream_replies_and_restores_the_main_keyboard():
    status_message = AsyncMock()

    await _finish_stream(status_message, "Stream finished — hit the 100-image limit.")

    status_message.reply_text.assert_awaited_once()
    args, kwargs = status_message.reply_text.await_args
    assert args[0] == "Stream finished — hit the 100-image limit."
    assert kwargs["reply_markup"] is _MAIN_KEYBOARD
    status_message.edit_text.assert_not_called()
