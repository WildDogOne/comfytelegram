"""Timeout and retry configuration on the built `Application`.

These pin values that only ever surface as a failure on a slow network with
a large file, which makes them exactly the kind of thing a later refactor
would silently drop.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from telegram.error import BadRequest, NetworkError
from telegram.request import HTTPXRequest

from comfytelegram.main import (
    MEDIA_WRITE_TIMEOUT_SECONDS,
    READ_TIMEOUT_SECONDS,
    TELEGRAM_REQUEST_RETRIES,
    RetryingRequest,
    _start_lora_discovery,
    build_application,
)
from comfytelegram.settings import Settings


def _application(tmp_path):
    return build_application(
        Settings(
            telegram_bot_token="123:abc",
            state_db_path=tmp_path / "state.sqlite3",
            tags_db_path=tmp_path / "tags.sqlite3",
        )
    )


def test_media_uploads_get_far_longer_than_ptbs_20_second_default(tmp_path):
    """A 4x-upscaled PNG is ~20MB; PTB's default 20s media write timeout
    demands a sustained ~8 Mbit/s uplink and otherwise dies mid-body,
    throwing away a ComfyUI run that had already completed."""
    request = _application(tmp_path).bot.request

    assert request._media_write_timeout == MEDIA_WRITE_TIMEOUT_SECONDS
    assert MEDIA_WRITE_TIMEOUT_SECONDS >= 300


def test_read_timeout_survives_telegram_ingesting_a_large_upload(tmp_path):
    """The response to a large `sendDocument` only comes once Telegram has
    taken the whole file, which is well past the 5s PTB defaults to."""
    request = _application(tmp_path).bot.request

    assert request._client.timeout.read == READ_TIMEOUT_SECONDS
    assert READ_TIMEOUT_SECONDS > 5


def test_messages_are_quiet_by_default(tmp_path):
    """Menus, statuses and replies shouldn't buzz the phone; only finished
    images opt back in (see `test_result_images_still_notify`)."""
    assert _application(tmp_path).bot.defaults.disable_notification is True


def test_api_requests_go_through_the_retrying_request(tmp_path):
    """A transient Telegram 5xx on `sendPhoto` killed a finished 4-image
    batch live ("NetworkError: Internal Server Error", GPU work already
    done, nothing sent) — PTB 22 removed `auto_retry`, so the retry must be
    wired into the request object itself."""
    assert isinstance(_application(tmp_path).bot.request, RetryingRequest)


def _patch_flaky_post(monkeypatch, post_impl):
    """Replace `HTTPXRequest.post` (the real network layer) with
    `post_impl`, and `asyncio.sleep` so backoff waits don't slow the test."""

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(HTTPXRequest, "post", post_impl)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)


def test_transient_network_errors_are_retried_then_succeed(monkeypatch):
    """A 500 from Telegram is usually gone a second later; the request is
    re-sent with backoff instead of failing the whole update."""
    attempts = []

    async def flaky_post(self, url, **kwargs):
        attempts.append(url)
        if len(attempts) <= TELEGRAM_REQUEST_RETRIES:
            raise NetworkError("Internal Server Error")
        return {"ok": True}

    _patch_flaky_post(monkeypatch, flaky_post)

    result = asyncio.run(RetryingRequest().post("https://api.example/sendPhoto"))

    assert result == {"ok": True}
    assert len(attempts) == TELEGRAM_REQUEST_RETRIES + 1


def test_retry_budget_is_finite_and_reraises(monkeypatch):
    """A genuinely down API must not hang a handler forever — after
    `TELEGRAM_REQUEST_RETRIES` retries the original error surfaces (and
    `_error_handler` tells the chat)."""
    attempts = []

    async def always_500(self, url, **kwargs):
        attempts.append(url)
        raise NetworkError("Internal Server Error")

    _patch_flaky_post(monkeypatch, always_500)

    with pytest.raises(NetworkError):
        asyncio.run(RetryingRequest().post("https://api.example/sendPhoto"))

    assert len(attempts) == TELEGRAM_REQUEST_RETRIES + 1


def test_client_errors_are_not_retried(monkeypatch):
    """`BadRequest` & friends are deterministic — the bot's own mistake,
    repeating it only wastes the handler's time."""
    attempts = []

    async def bad_post(self, url, **kwargs):
        attempts.append(url)
        raise BadRequest("some client-side mistake")

    _patch_flaky_post(monkeypatch, bad_post)

    with pytest.raises(BadRequest):
        asyncio.run(RetryingRequest().post("https://api.example/sendPhoto"))

    assert len(attempts) == 1


def test_relay_upload_gets_its_own_generous_timeout():
    """`_relay_create_job` POSTs a full-resolution source PNG (~20MB after
    an upscale) to a relay on a public host, and aiohttp's `total` covers
    uploading the body — so the 15s total the small poll/delete calls use
    aborted mid-upload. They must not share one timeout again."""
    from comfytelegram.handlers import (
        _INPAINT_RELAY_TIMEOUT,
        _INPAINT_RELAY_UPLOAD_TIMEOUT,
    )

    assert _INPAINT_RELAY_UPLOAD_TIMEOUT.total >= 120
    assert _INPAINT_RELAY_UPLOAD_TIMEOUT.total > _INPAINT_RELAY_TIMEOUT.total
    # an unreachable relay should still fail fast, which is what the short
    # shared timeout was really protecting
    assert _INPAINT_RELAY_UPLOAD_TIMEOUT.connect is not None
    assert _INPAINT_RELAY_UPLOAD_TIMEOUT.connect <= 30


def test_lora_discovery_is_skipped_without_comfyui_loras_dir():
    """No `COMFYUI_LORAS_DIR` means no filesystem access to hash anything
    against — the background task must not even be created, not just
    no-op once running (that'd still cost an event-loop scheduling round
    trip on every startup for nothing)."""
    application = MagicMock()
    application.bot_data = {}
    settings = MagicMock(comfyui_loras_dir=None)

    _start_lora_discovery(application, settings)

    assert "lora_discovery_task" not in application.bot_data
