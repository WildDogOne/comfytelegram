import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.constants import ChatType
from telegram.error import BadRequest, RetryAfter

from comfytelegram import progress
from comfytelegram.progress import (
    DRAFT_BOARDS_KEY,
    DRAFT_DONE_TEXT,
    DraftBoards,
    start_chat_status,
    start_reply_status,
)


@pytest.fixture(autouse=True)
def _fast_drafts(monkeypatch):
    monkeypatch.setattr(progress, "DRAFT_MIN_INTERVAL", 0.0)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def _message(bot: MagicMock, *, chat_type: str = ChatType.PRIVATE, chat_id: int = 7) -> AsyncMock:
    message = AsyncMock()
    message.chat = MagicMock(type=chat_type, id=chat_id)
    message.is_topic_message = False
    message.message_thread_id = None
    message.get_bot = MagicMock(return_value=bot)
    return message


def _bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    bot.send_message = AsyncMock()
    return bot


def _drafts(bot: MagicMock) -> list[tuple[int, str]]:
    return [(c.args[1], c.args[2]) for c in bot.send_message_draft.await_args_list]


@pytest.mark.asyncio
async def test_private_chat_progress_goes_to_a_draft_and_ends_with_done():
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Generating… 0%")
    await _settle()
    await status.update("Generating… 50%")
    await _settle()
    await status.delete()
    await _settle()

    drafts = _drafts(bot)
    assert [text for _, text in drafts] == ["Generating… 0%", "Generating… 50%", DRAFT_DONE_TEXT]
    assert len({draft_id for draft_id, _ in drafts}) == 1
    message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_draft_drops_the_trailing_ellipsis_telegram_draws_itself():
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Detailing (drawn mask)…")
    await _settle()
    await status.close()

    assert _drafts(bot)[0][1] == "Detailing (drawn mask)"


@pytest.mark.asyncio
async def test_two_jobs_share_one_draft_and_the_survivor_moves_to_a_new_id():
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    first = await start_reply_status(bot_data, message, "Generating… 10%")
    second = await start_reply_status(bot_data, message, "Upscaling…")
    await _settle()
    first_id, text = _drafts(bot)[-1]
    assert text == "Generating… 10%\nUpscaling"

    await first.delete()
    await _settle()
    new_id, text = _drafts(bot)[-1]
    assert text == "Upscaling"
    assert new_id != first_id
    await second.close()


@pytest.mark.asyncio
async def test_lasting_text_is_sent_as_a_reply_in_draft_mode():
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}
    markup = MagicMock()

    status = await start_reply_status(bot_data, message, "Upscaling…")
    await _settle()
    await status.edit_text("Upscaling failed: boom", reply_markup=markup)
    await _settle()

    message.reply_text.assert_awaited_once_with("Upscaling failed: boom", reply_markup=markup)
    assert _drafts(bot)[-1][1] == DRAFT_DONE_TEXT


@pytest.mark.asyncio
async def test_message_sent_moves_the_draft_below_it():
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Streaming 1/3…")
    await _settle()
    status.message_sent()
    await status.update("Streaming 2/3…")
    await _settle()

    (first_id, _), (second_id, text) = _drafts(bot)[0], _drafts(bot)[-1]
    assert text == "Streaming 2/3"
    assert second_id != first_id
    await status.close()


@pytest.mark.asyncio
async def test_unchanged_draft_is_re_sent_as_a_keep_alive(monkeypatch):
    monkeypatch.setattr(progress, "DRAFT_KEEPALIVE_SECONDS", 0.01)
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Queued…")
    await asyncio.sleep(0.05)
    await status.close()

    assert [text for _, text in _drafts(bot)].count("Queued") >= 2


@pytest.mark.asyncio
async def test_job_whose_task_ended_without_closing_is_dropped():
    bot = _bot()
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    async def handler_that_forgets_to_close() -> None:
        await start_reply_status(bot_data, message, "Generating…")

    await asyncio.create_task(handler_that_forgets_to_close())
    other = await start_reply_status(bot_data, message, "Upscaling…")
    await _settle()

    assert _drafts(bot)[-1][1] == "Upscaling"
    await other.close()


@pytest.mark.asyncio
async def test_rate_limited_draft_is_retried():
    bot = _bot()
    bot.send_message_draft.side_effect = [RetryAfter(0), True, True]
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Generating…")
    await asyncio.sleep(0.01)
    await _settle()

    assert [text for _, text in _drafts(bot)][:2] == ["Generating", "Generating"]
    assert status.uses_draft
    await status.close()


@pytest.mark.asyncio
async def test_failed_draft_falls_back_to_a_status_message():
    bot = _bot()
    bot.send_message_draft.side_effect = BadRequest("drafts not allowed")
    message = _message(bot)
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Generating… 0%")
    await _settle()
    await status.update("Generating… 50%")

    message.reply_text.assert_awaited_once_with("Generating… 50%", disable_notification=True)
    assert not status.uses_draft


@pytest.mark.asyncio
async def test_group_chat_keeps_the_edited_status_message():
    bot = _bot()
    message = _message(bot, chat_type=ChatType.SUPERGROUP, chat_id=-100)
    status_message = AsyncMock()
    message.reply_text.return_value = status_message
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_reply_status(bot_data, message, "Generating… 0%")
    await status.update("Generating… 50%")
    await status.delete()

    message.reply_text.assert_awaited_once_with("Generating… 0%", disable_notification=True)
    status_message.edit_text.assert_awaited_once_with("Generating… 50%")
    status_message.delete.assert_awaited_once()
    bot.send_message_draft.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_draft_boards_it_is_the_old_status_message():
    bot = _bot()
    message = _message(bot)
    status_message = AsyncMock()
    message.reply_text.return_value = status_message

    status = await start_reply_status({}, message, "Refining hand…")
    await status.edit_text("Refining hand failed: boom")

    status_message.edit_text.assert_awaited_once_with("Refining hand failed: boom")
    bot.send_message_draft.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_message_is_edited_but_never_deleted_in_edit_mode():
    bot = _bot()
    message = _message(bot, chat_type=ChatType.GROUP, chat_id=-5)
    existing = AsyncMock()

    status = await start_reply_status({}, message, None, existing=existing)
    await status.update("Streaming 1/100… 40%")
    await status.delete()

    existing.edit_text.assert_awaited_once_with("Streaming 1/100… 40%")
    existing.delete.assert_not_awaited()
    message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_status_uses_a_draft_for_a_private_chat_id():
    bot = _bot()
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    status = await start_chat_status(bot_data, bot, 42, "Detailing…", message_thread_id=3)
    await _settle()
    await status.edit_text("Detailing failed: boom")

    bot.send_message_draft.assert_any_await(
        42, bot.send_message_draft.await_args_list[0].args[1], "Detailing", message_thread_id=3
    )
    bot.send_message.assert_awaited_once_with(42, "Detailing failed: boom", message_thread_id=3)


@pytest.mark.asyncio
async def test_chat_status_posts_a_message_for_a_group():
    bot = _bot()
    bot_data = {DRAFT_BOARDS_KEY: DraftBoards()}

    await start_chat_status(bot_data, bot, -100, "Generating regions… 0%")

    bot.send_message.assert_awaited_once_with(
        -100, "Generating regions… 0%", message_thread_id=None, disable_notification=True
    )
    bot.send_message_draft.assert_not_awaited()
