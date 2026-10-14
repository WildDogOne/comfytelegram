"""Compressing the mask-editor's reference copy before uploading it.

This used to happen on the relay, which meant the full-resolution PNG
crossed the internet only to be re-encoded and discarded at the far end —
and, at ~19MB after an upscale, routinely didn't finish inside the upload
timeout. See `handlers._to_display_jpeg`.
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from comfytelegram.handlers import _relay_create_job, _to_display_jpeg


def _noisy_png(size=(512, 512)) -> bytes:
    """A PNG that doesn't compress away to nothing — a flat colour would
    make the size comparison meaningless, since PNG wins on those."""
    import random

    random.seed(0)
    image = Image.new("RGB", size)
    image.putdata(
        [
            (random.randrange(256), random.randrange(256), random.randrange(256))
            for _ in range(size[0] * size[1])
        ]
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_display_jpeg_keeps_the_exact_pixel_dimensions():
    """Non-negotiable: the editor sizes its mask canvas off
    `naturalWidth/Height`, and `post_process` scales that mask back onto
    the real image — resizing here would silently shift the geometry the
    mask comes back in."""
    source = _noisy_png((640, 384))

    encoded = _to_display_jpeg(source)

    assert Image.open(io.BytesIO(encoded)).size == (640, 384)
    assert encoded.startswith(b"\xff\xd8\xff")


def test_display_jpeg_is_substantially_smaller_than_the_source_png():
    source = _noisy_png()

    assert len(_to_display_jpeg(source)) < len(source) / 2


def test_display_jpeg_passes_undecodable_bytes_through_untouched():
    """An oversized upload beats no mask editor at all."""
    junk = b"not an image at all"

    assert _to_display_jpeg(junk) is junk


def _parts(form) -> dict:
    """`{name: (value, content_type, filename)}` for an `aiohttp.FormData`."""
    return {
        options["name"]: (value, headers.get("Content-Type"), options.get("filename"))
        for options, headers, value in form._fields
    }


def _session_patch(captured: dict):
    """Stand in for `aiohttp.ClientSession`, recording the POST's multipart
    parts and headers and returning a token."""
    response = AsyncMock()
    response.json.return_value = {"token": "tok123"}
    response.raise_for_status = MagicMock()

    def _post(url, *, data, headers):
        captured["url"] = url
        captured["parts"] = _parts(data)
        captured["headers"] = headers
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = _post
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


@pytest.mark.asyncio
async def test_relay_upload_sends_the_compressed_copy_and_says_what_it_is():
    """The relay no longer re-encodes, so it serves these bytes back
    verbatim and needs to know what they are."""
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")
    source = _noisy_png()
    captured: dict = {}

    with patch("comfytelegram.handlers.aiohttp.ClientSession", _session_patch(captured)):
        token = await _relay_create_job(settings, source)

    assert token == "tok123"
    assert captured["headers"] == {"Authorization": "Bearer s"}
    data, content_type, filename = captured["parts"]["image"]
    assert (content_type, filename) == ("image/jpeg", "source.jpg")
    assert data.startswith(b"\xff\xd8\xff")
    assert len(data) < len(source)


@pytest.mark.asyncio
async def test_relay_upload_omits_the_meta_part_for_a_plain_mask_job():
    """The common case ("🖌️ Draw Mask"/"🩹 Fix Artifact") doesn't pass
    `meta` at all — the relay defaults a missing part to `mode="mask"`."""
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")
    captured: dict = {}

    with patch("comfytelegram.handlers.aiohttp.ClientSession", _session_patch(captured)):
        await _relay_create_job(settings, _noisy_png())

    assert set(captured["parts"]) == {"image"}


@pytest.mark.asyncio
async def test_relay_upload_sends_meta_as_a_json_part_however_large():
    """ "✏️ Detail Prompt" passes `meta` — in the body, not a header, so
    long suggestion lists and non-ASCII prompts both just go through."""
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")
    captured: dict = {}
    meta = {
        "mode": "mask_prompt",
        "positive": "五本指",
        "loras": [{"name": f"folder/lora_{i}", "strength": "1", "preview": ""} for i in range(500)],
    }

    with patch("comfytelegram.handlers.aiohttp.ClientSession", _session_patch(captured)):
        await _relay_create_job(settings, _noisy_png(), meta=meta)

    value, content_type, _ = captured["parts"]["meta"]
    assert content_type == "application/json"
    assert json.loads(value) == meta


@pytest.mark.asyncio
async def test_relay_upload_labels_a_pass_through_as_png():
    """When Pillow can't decode the source, the original bytes go up
    unchanged — and must not be announced as a JPEG."""
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")
    captured: dict = {}

    with (
        patch("comfytelegram.handlers.aiohttp.ClientSession", _session_patch(captured)),
        patch("comfytelegram.handlers._to_display_jpeg", side_effect=lambda src: src),
    ):
        await _relay_create_job(settings, b"\x89PNG\r\n\x1a\n undecodable")

    assert captured["parts"]["image"] == (
        b"\x89PNG\r\n\x1a\n undecodable",
        "image/png",
        "source.png",
    )
