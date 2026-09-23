"""The archive round trip: an image sent back to the bot as a *file* gets
its post-processing buttons restored from the metadata embedded in it.

See `png_metadata` for why the metadata is ours rather than ComfyUI's, and
`handlers.document_message` for why this is bound to
`filters.Document.IMAGE` rather than `filters.PHOTO`.
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfytelegram.handlers import IMPORT_MAX_FILE_BYTES, document_message, postprocess_callback
from comfytelegram.params_serde import serialize_generation_params
from comfytelegram.png_metadata import build_metadata, embed_metadata, extract_metadata
from comfytelegram.workflows import GenerationParams, LoraSpec


def _png(text: dict[str, str] | None = None) -> bytes:
    info = PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _params() -> GenerationParams:
    return GenerationParams(
        checkpoint="ponyxl.safetensors",
        positive_prompt="score_9, a fox",
        negative_prompt="worst quality",
        steps=28,
        cfg=6.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        width=832,
        height=1216,
        clip_skip=-2,
        loras=[LoraSpec(name="detail.safetensors", strength_model=0.8, strength_clip=0.8)],
        raw_positive_prompt="a fox",
    )


def _stamped_png(params: GenerationParams | None = None, **kwargs) -> bytes:
    params = params or _params()
    return embed_metadata(
        _png(),
        build_metadata(
            params=serialize_generation_params(params),
            kind=kwargs.pop("kind", "txt2img"),
            filename=kwargs.pop("filename", "comfytelegram_00042_.png"),
            seed=kwargs.pop("seed", 777),
        ),
    )


def _update(*, file_size: int = 1000, file_name: str = "img.png"):
    message = AsyncMock()
    message.document = MagicMock(file_id="doc-1", file_size=file_size, file_name=file_name)
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 42
    update.effective_user.id = 1
    return update, message


def _context(data: bytes, *, checkpoints=("ponyxl.safetensors",)):
    context = MagicMock()
    storage = MagicMock()
    storage.get_checkpoint.return_value = "ponyxl.safetensors"
    client = AsyncMock()
    client.list_checkpoints.return_value = list(checkpoints)
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "comfy_client": client,
    }
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(data)
    context.bot = AsyncMock()
    context.bot.get_file.return_value = tg_file
    return context


@pytest.mark.asyncio
async def test_import_restores_a_pending_result_with_the_full_original_params():
    """The point of the whole feature: an image the database has never seen
    gets a `pending_result` row built from its own bytes, so every
    post-processing button works again."""
    update, message = _update()
    context = _context(_stamped_png())

    await document_message(update, context)

    storage = context.bot_data["storage"]
    storage.store_pending_result.assert_called_once()
    result_id, chat_id, file_id, filename, stored = storage.store_pending_result.call_args.args
    assert chat_id == 42
    assert file_id == "doc-1"  # the uploaded document, re-downloadable later
    assert filename == "comfytelegram_00042_.png"
    assert stored == serialize_generation_params(_params())

    keyboard = message.reply_text.await_args.kwargs["reply_markup"]
    callbacks = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert f"pp:upscale:{result_id}" in callbacks
    assert f"pp:face:{result_id}" in callbacks
    assert f"pp:fix_draw:{result_id}" in callbacks


@pytest.mark.asyncio
async def test_import_reply_shows_the_settings_it_recovered():
    update, message = _update()
    context = _context(_stamped_png())

    await document_message(update, context)

    text = message.reply_text.await_args.args[0]
    assert "ponyxl.safetensors" in text
    assert "28 steps" in text
    assert "cfg 6.5" in text
    assert "dpmpp_2m/karras" in text
    assert "832×1216" in text
    assert "Seed: 777" in text
    assert "detail.safetensors" in text
    assert "score_9, a fox" in text


@pytest.mark.asyncio
async def test_import_warns_when_the_checkpoint_is_not_installed_here():
    """The buttons would otherwise all look live and then fail at submit
    time on a model this ComfyUI doesn't have."""
    update, message = _update()
    context = _context(_stamped_png(), checkpoints=("something_else.safetensors",))

    await document_message(update, context)

    assert "doesn't have" in message.reply_text.await_args.args[0]
    # still imported — a missing model is a warning, not a refusal
    context.bot_data["storage"].store_pending_result.assert_called_once()


@pytest.mark.asyncio
async def test_import_stays_silent_about_the_checkpoint_when_comfyui_is_unreachable():
    """An unreachable server is a different problem, which the buttons
    report in their own way — inferring "model missing" from it would be
    wrong, and the import itself doesn't need ComfyUI at all."""
    update, message = _update()
    context = _context(_stamped_png())
    context.bot_data["comfy_client"].list_checkpoints.side_effect = aiohttp.ClientError("down")

    await document_message(update, context)

    context.bot_data["storage"].store_pending_result.assert_called_once()
    assert "doesn't have" not in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_a_foreign_comfyui_image_reports_its_workflow_instead_of_importing():
    """A Krita AI Diffusion export (or any other front-end's PNG) has
    ComfyUI's `prompt` chunk but not ours — readable enough to show and to
    generate from, not enough to restore post-processing buttons."""
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "other.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a wolf in snow"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "seed": 123,
                "steps": 20,
                "cfg": 7.0,
            },
        },
    }
    update, message = _update()
    context = _context(_png({"prompt": json.dumps(graph)}))

    await document_message(update, context)

    context.bot_data["storage"].store_pending_result.assert_not_called()
    texts = [call.args[0] for call in message.reply_text.await_args_list]
    assert any("other.safetensors" in t and "20 steps" in t for t in texts)
    assert any("a wolf in snow" in t for t in texts)
    # the recovered prompt is offered for generation
    context.bot_data["storage"].store_derived_prompt.assert_called_once()
    assert context.bot_data["storage"].store_derived_prompt.call_args.args[3] == "a wolf in snow"


@pytest.mark.asyncio
async def test_a_png_with_no_metadata_at_all_falls_back_to_image_analysis():
    """Uploading any image as a file should still do something useful."""
    update, _message = _update()
    context = _context(_png())

    with (
        patch("comfytelegram.handlers.analyze_tags", new=AsyncMock(return_value="1girl")),
        patch(
            "comfytelegram.handlers.analyze_caption_deep",
            new=AsyncMock(return_value=("a girl", "blurry")),
        ),
    ):
        await document_message(update, context)

    context.bot_data["storage"].store_pending_result.assert_not_called()
    assert context.bot_data["storage"].store_derived_prompt.call_count == 2


@pytest.mark.asyncio
async def test_a_non_png_file_is_rejected_with_an_explanation():
    update, message = _update(file_name="img.jpg")
    context = _context(b"\xff\xd8\xff\xe0 jpeg bytes")

    await document_message(update, context)

    context.bot_data["storage"].store_pending_result.assert_not_called()
    edited = message.reply_text.return_value.edit_text.await_args.args[0]
    assert "isn't a PNG" in edited


@pytest.mark.asyncio
async def test_an_oversized_file_is_refused_before_any_download():
    """Telegram's Bot API refuses `getFile` over 20MB, so say so up front
    rather than surfacing an opaque failure."""
    update, message = _update(file_size=IMPORT_MAX_FILE_BYTES + 1)
    context = _context(_stamped_png())

    await document_message(update, context)

    context.bot.get_file.assert_not_awaited()
    assert "20MB" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_archive_button_resends_the_original_png_with_metadata_embedded():
    """ "📥 Download file" has to re-fetch from ComfyUI, not from the stored
    Telegram file_id — that id points at the JPEG Telegram re-encoded the
    photo into, which has no chunks left."""
    query = AsyncMock()
    query.data = "pp:archive:res1"
    query.message = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    params = _params()
    context = MagicMock()
    storage = MagicMock()
    storage.get_pending_result.return_value = {
        "chat_id": 42,
        "file_id": "tg-photo-id",
        "filename": "comfytelegram_00042_.png",
        "base_params": serialize_generation_params(params),
    }
    client = AsyncMock()
    # what ComfyUI still has on disk: no chunk of ours, but its own graph
    client.get_image_bytes.return_value = _png(
        {"prompt": json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 4242}}})}
    )
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "comfy_client": client,
    }

    await postprocess_callback(update, context)

    client.get_image_bytes.assert_awaited_once_with("comfytelegram_00042_.png", "", "output")
    sent = query.message.reply_document.await_args.kwargs
    assert sent["filename"] == "comfytelegram_00042_.png"
    metadata = extract_metadata(sent["document"].getvalue())
    assert metadata["params"] == serialize_generation_params(params)
    assert metadata["seed"] == 4242  # recovered from ComfyUI's own chunk
