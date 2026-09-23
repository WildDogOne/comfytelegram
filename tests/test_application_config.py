"""Timeout configuration on the built `Application`.

These pin values that only ever surface as a failure on a slow network with
a large file, which makes them exactly the kind of thing a later refactor
would silently drop.
"""

from comfytelegram.main import (
    MEDIA_WRITE_TIMEOUT_SECONDS,
    READ_TIMEOUT_SECONDS,
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
