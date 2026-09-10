import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from comfytelegram.handlers import (
    _MAIN_KEYBOARD,
    STREAM_CANCEL_CALLBACK_DATA,
    _consume_awaiting_stream_prompt,
    _finish_stream,
    _resolve_checkpoint_or_default,
    stop_command,
    stream_cancel_callback,
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


def _callback_update_mock(*, chat_id: int = 1, user_id: int = 1) -> tuple[MagicMock, AsyncMock]:
    query = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    return update, query


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
async def test_stream_command_asks_for_a_prompt_when_none_given():
    update, message = _update_mock()
    message.text = "/stream"
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock()}
    context.chat_data = {}

    await stream_command(update, context)

    assert message.reply_text.await_args.args[0].startswith("What should the stream generate?")
    assert context.chat_data["awaiting_stream_prompt"] is True
    assert "active_streams" not in context.bot_data
    _, kwargs = message.reply_text.await_args
    buttons = [b.callback_data for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert buttons == [STREAM_CANCEL_CALLBACK_DATA]


@pytest.mark.asyncio
async def test_stream_cancel_callback_clears_the_flag_and_edits_the_message():
    update, query = _callback_update_mock()
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock()}
    context.chat_data = {"awaiting_stream_prompt": True}

    await stream_cancel_callback(update, context)

    assert "awaiting_stream_prompt" not in context.chat_data
    query.answer.assert_awaited_once_with("Cancelled.")
    query.edit_message_text.assert_awaited_once()
    assert query.edit_message_text.await_args.args[0] == "Cancelled — no stream started."


@pytest.mark.asyncio
async def test_stream_cancel_callback_is_a_noop_once_already_handled():
    update, query = _callback_update_mock()
    context = MagicMock()
    context.bot_data = {"settings": _settings_mock()}
    context.chat_data = {}

    await stream_cancel_callback(update, context)

    query.answer.assert_awaited_once_with("Nothing to cancel.")
    query.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_stream_prompt_ignores_unrelated_text():
    update, message = _update_mock()
    message.text = "just a normal prompt"
    context = MagicMock()
    context.chat_data = {}

    result = await _consume_awaiting_stream_prompt(update, context)

    assert result is False
    message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_consume_awaiting_stream_prompt_starts_the_stream():
    update, message = _update_mock()
    message.text = "a fox in the snow"
    context = MagicMock()
    context.chat_data = {"awaiting_stream_prompt": True}
    context.bot_data = {}

    with patch("comfytelegram.handlers._run_stream", new=AsyncMock()):
        result = await _consume_awaiting_stream_prompt(update, context)

    assert result is True
    assert "awaiting_stream_prompt" not in context.chat_data
    task = context.bot_data["active_streams"][1]
    assert isinstance(task, asyncio.Task)
    await task


@pytest.mark.asyncio
async def test_consume_awaiting_stream_prompt_reports_empty_message():
    update, message = _update_mock()
    message.text = "   "
    context = MagicMock()
    context.chat_data = {"awaiting_stream_prompt": True}

    result = await _consume_awaiting_stream_prompt(update, context)

    assert result is True
    message.reply_text.assert_awaited_once_with("Cancelled — no prompt received.")


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
