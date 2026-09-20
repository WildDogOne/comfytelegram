import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlencode

import pytest

from comfytelegram.auth import (
    is_authorized,
    reject_if_unauthorized,
    reject_if_unauthorized_callback,
    validate_webapp_init_data,
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


def _signed_init_data(bot_token: str, **fields: str) -> str:
    """Build an `initData` string signed exactly the way Telegram's own
    WebApp client would, for testing `validate_webapp_init_data` against a
    known-good signature."""
    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    hash_ = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": hash_})


def test_validate_webapp_init_data_accepts_a_correctly_signed_payload():
    init_data = _signed_init_data("test-token", auth_date="1700000000", query_id="abc")
    assert validate_webapp_init_data(init_data, "test-token") == {
        "auth_date": "1700000000",
        "query_id": "abc",
    }


def test_validate_webapp_init_data_rejects_wrong_bot_token():
    init_data = _signed_init_data("test-token", auth_date="1700000000")
    assert validate_webapp_init_data(init_data, "a-different-token") is None


def test_validate_webapp_init_data_rejects_tampered_fields():
    init_data = _signed_init_data("test-token", auth_date="1700000000", query_id="abc")
    tampered = init_data.replace("query_id=abc", "query_id=xyz")
    assert validate_webapp_init_data(tampered, "test-token") is None


def test_validate_webapp_init_data_rejects_missing_hash():
    assert validate_webapp_init_data("auth_date=1700000000", "test-token") is None


def test_validate_webapp_init_data_rejects_malformed_input():
    assert validate_webapp_init_data("%zz", "test-token") is None
