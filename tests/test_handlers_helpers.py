from unittest.mock import AsyncMock, MagicMock

import pytest

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.handlers import (
    _again_keyboard,
    _characters_keyboard,
    _extract_file_id,
    _post_process_keyboard,
    _run_reporting_errors,
)


def test_post_process_keyboard_scopes_every_button_to_result_id():
    keyboard = _post_process_keyboard("abc123")
    callback_data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert "pp:upscale:abc123" in callback_data
    assert "pp:face:abc123" in callback_data
    assert "pp:analyze:abc123" in callback_data


def test_again_keyboard_scopes_button_to_snapshot_id():
    keyboard = _again_keyboard("snap123")
    button = keyboard.inline_keyboard[0][0]
    assert button.callback_data == "again:snap123"


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
