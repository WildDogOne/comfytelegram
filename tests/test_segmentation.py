"""The bot's half of the mask editor's "✨ Smart" select: encoding the image
(`segmentation.py`) and handing the embedding to inpaint_relay
(`handlers._segmentation_meta`/`_relay_upload_embedding`). The decoder half
runs in the browser and isn't covered here."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from comfytelegram import segmentation
from comfytelegram.handlers import _relay_upload_embedding, _segmentation_meta
from comfytelegram.segmentation import EMBEDDING_SHAPE, encode_image, sam_input_size


def _png(size) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((4096, 4096), (1024, 1024)),
        ((832, 1216), (701, 1024)),
        ((1216, 832), (1024, 701)),
        ((512, 256), (1024, 512)),
    ],
)
def test_sam_input_size_scales_the_longest_side_to_1024(size, expected):
    assert sam_input_size(*size) == expected


def test_encode_image_feeds_raw_rgb_at_sam_size_and_returns_float16():
    """The MobileSAM export normalizes and pads itself, so it wants 0-255
    HWC floats at the resized size — and the page decodes exactly
    `EMBEDDING_SHAPE` little-endian halves."""
    fake = MagicMock()
    fake.run.return_value = [np.full(EMBEDDING_SHAPE, 0.5, dtype=np.float32)]

    with patch.object(segmentation, "_load_encoder", return_value=fake):
        result = encode_image(_png((1216, 832)), Path("enc.onnx"))

    pixels = fake.run.call_args.args[1]["input_image"]
    assert pixels.shape == (701, 1024, 3)
    assert pixels.dtype == np.float32
    assert pixels[0, 0].tolist() == [10.0, 20.0, 30.0]
    assert (result.width, result.height) == (1024, 701)
    decoded = np.frombuffer(result.data, dtype="<f2")
    assert decoded.size == np.prod(EMBEDDING_SHAPE)
    assert np.all(decoded == 0.5)


def test_segmentation_meta_is_off_without_a_staged_encoder(tmp_path):
    settings = MagicMock(sam_encoder_path=tmp_path / "missing.onnx")

    assert _segmentation_meta(settings, _png((64, 64))) is None


def test_segmentation_meta_describes_the_decoder_and_sam_frame(tmp_path):
    encoder = tmp_path / "enc.onnx"
    encoder.write_bytes(b"model")
    settings = MagicMock(sam_encoder_path=encoder, sam_decoder_url="https://models.example/d.onnx")

    meta = _segmentation_meta(settings, _png((832, 1216)))

    assert meta == {"decoder_url": "https://models.example/d.onnx", "width": 701, "height": 1024}


def test_segmentation_meta_is_off_for_an_unreadable_image(tmp_path):
    """No smart button beats no mask editor at all."""
    encoder = tmp_path / "enc.onnx"
    encoder.write_bytes(b"model")
    settings = MagicMock(sam_encoder_path=encoder)

    assert _segmentation_meta(settings, b"not an image") is None


def _put_session(captured: dict, status_error: Exception | None = None):
    response = MagicMock()
    response.raise_for_status = MagicMock(side_effect=status_error)

    def _put(url, *, data, headers):
        captured.update(url=url, data=data, headers=headers)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.put = _put
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


@pytest.mark.asyncio
async def test_embedding_upload_puts_the_encoded_bytes_with_the_shared_secret():
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")
    embedding = segmentation.ImageEmbedding(b"\x00\x3c" * 4, 1024, 1024)
    captured: dict = {}

    with (
        patch("comfytelegram.handlers.encode_image", return_value=embedding),
        patch("comfytelegram.handlers.aiohttp.ClientSession", _put_session(captured)),
    ):
        await _relay_upload_embedding(settings, "tok1", b"image")

    assert captured["url"] == "https://relay.example/jobs/tok1/embedding"
    assert captured["data"] == embedding.data
    assert captured["headers"]["Authorization"] == "Bearer s"


@pytest.mark.asyncio
async def test_embedding_upload_failures_are_swallowed():
    """It runs as an unawaited background task — raising would only end up
    as an unretrieved-exception warning, and the page copes with a missing
    embedding by leaving the smart button disabled."""
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")

    with patch("comfytelegram.handlers.encode_image", side_effect=RuntimeError("boom")):
        await _relay_upload_embedding(settings, "tok1", b"image")

    captured: dict = {}
    with (
        patch(
            "comfytelegram.handlers.encode_image",
            return_value=segmentation.ImageEmbedding(b"", 1, 1),
        ),
        patch(
            "comfytelegram.handlers.aiohttp.ClientSession",
            _put_session(captured, status_error=RuntimeError("404")),
        ),
    ):
        await _relay_upload_embedding(settings, "tok1", b"image")
