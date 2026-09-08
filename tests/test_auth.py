from unittest.mock import AsyncMock, MagicMock

import pytest

from comfytelegram.auth import (
    is_authorized,
    reject_if_unauthorized,
    reject_if_unauthorized_callback,
)
from comfytelegram.settings import Settings


def _settings(allowed_user_ids=None) -> Settings:
    # _env_file=None: don't let the real repo-root .env (if present) leak
    # into these tests — Settings should only reflect what's passed here.
    return Settings(
        _env_file=None, telegram_bot_token="test-token", allowed_user_ids=allowed_user_ids or []
    )


def _update(user_id):
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id) if user_id is not None else None
    update.effective_message = AsyncMock()
    return update


def test_is_authorized_allows_anyone_when_allowlist_empty():
    settings = _settings(allowed_user_ids=[])
    assert is_authorized(settings, 12345) is True
    assert is_authorized(settings, None) is True


def test_is_authorized_restricts_to_allowlisted_ids():
    settings = _settings(allowed_user_ids=[111, 222])
    assert is_authorized(settings, 111) is True
    assert is_authorized(settings, 333) is False
    assert is_authorized(settings, None) is False


@pytest.mark.asyncio
async def test_reject_if_unauthorized_allows_listed_user_without_replying():
    settings = _settings(allowed_user_ids=[42])
    update = _update(42)
    rejected = await reject_if_unauthorized(update, settings)
    assert rejected is False
    update.effective_message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_reject_if_unauthorized_rejects_and_replies():
    settings = _settings(allowed_user_ids=[42])
    update = _update(99)
    rejected = await reject_if_unauthorized(update, settings)
    assert rejected is True
    update.effective_message.reply_text.assert_awaited_once_with(
        "You're not authorized to use this bot."
    )


@pytest.mark.asyncio
async def test_reject_if_unauthorized_callback_allows_listed_user_without_answering():
    settings = _settings(allowed_user_ids=[42])
    query = AsyncMock()
    rejected = await reject_if_unauthorized_callback(query, 42, settings)
    assert rejected is False
    query.answer.assert_not_called()


@pytest.mark.asyncio
async def test_reject_if_unauthorized_callback_rejects_with_alert_toast():
    settings = _settings(allowed_user_ids=[42])
    query = AsyncMock()
    rejected = await reject_if_unauthorized_callback(query, 99, settings)
    assert rejected is True
    query.answer.assert_awaited_once_with("Not authorized.", show_alert=True)
