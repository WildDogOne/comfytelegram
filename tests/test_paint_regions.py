""" "🎨 Paint Regions": the editor meta and starting mask
(`_paint_regions_meta`/`_paint_regions_initial_mask`), opening it from the
keyboard, recovering the seed (`_txt2img_seed`), and the submitted
painting turning into a regional generation (`_run_paint_regions_job`, via
`_process_one_inpaint_job`)."""

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from test_detail_prompt import _callback_update, _context

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.generation import GeneratedImage
from comfytelegram.handlers import (
    PAINT_REGIONS_CALLBACK_KIND,
    PAINT_REGIONS_JOB_KIND,
    _paint_regions_initial_mask,
    _paint_regions_meta,
    _process_one_inpaint_job,
    _serialize_generation_params,
    _txt2img_seed,
    postprocess_callback,
)
from comfytelegram.profiles import LoraDefault, ModelProfile
from comfytelegram.regional import painted_regions
from comfytelegram.storage import Storage
from comfytelegram.workflows import GenerationParams

_PROFILE = ModelProfile(
    match=["*"],
    display_name="p",
    positive_prompt_prefix="masterpiece",
    loras=[LoraDefault(name="chars/marlow.safetensors", default_enabled=False)],
)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def _params(**kw) -> GenerationParams:
    fields = {
        "checkpoint": "ckpt.safetensors",
        "positive_prompt": "masterpiece, 2boys, cuddling",
        "negative_prompt": "lowres",
        "raw_positive_prompt": "2boys, cuddling",
        "raw_negative_prompt": "lowres",
        "width": 64,
        "height": 48,
    }
    fields.update(kw)
    return GenerationParams(**fields)


def _png(pixels: np.ndarray, *, graph: dict | None = None) -> bytes:
    info = PngInfo()
    if graph is not None:
        info.add_text("prompt", json.dumps(graph))
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, "PNG", pnginfo=info)
    return out.getvalue()


def _left_painting(width=64, height=48, cols=20) -> bytes:
    pixels = np.zeros((height, width), dtype=np.uint8)
    pixels[:, :cols] = 255
    return _png(pixels)


def _white(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L")) >= 128


# --- opening the editor ----------------------------------------------------------


def test_meta_prefills_a_plain_prompt_as_the_whole_image(storage):
    meta = _paint_regions_meta(_params(), storage, 1, 7, [_PROFILE])

    assert meta["mode"] == "regions"
    assert (meta["positive"], meta["negative"]) == ("2boys, cuddling", "lowres")
    assert meta["regions"] == ["", ""]
    assert "64×48" in meta["readonly"]["text"]
    assert [lora["name"] for lora in meta["loras"]] == ["chars/marlow"]


def test_meta_splits_a_regional_prompt_back_into_its_parts(storage):
    params = _params(raw_positive_prompt="2boys, park\n[left] red hair\n[right] black hair")

    meta = _paint_regions_meta(params, storage, 1, None, [_PROFILE])

    assert meta["positive"] == "2boys, park"
    assert meta["regions"] == ["red hair", "black hair"]


def test_initial_mask_is_the_left_half_or_the_first_region():
    source = _png(np.zeros((40, 100, 3), dtype=np.uint8))

    left = _white(_paint_regions_initial_mask(source, None))
    assert left.shape == (40, 100)
    assert left[:, :50].all() and not left[:, 50:].any()

    # A painted image opens with its own painting, scaled to the source
    # (which may since have been upscaled).
    painted, rest = painted_regions(_left_painting(), "a", "b", 64, 48)
    assert _white(_paint_regions_initial_mask(source, painted))[:, :25].all()
    assert not _white(_paint_regions_initial_mask(source, rest))[:, :25].any()


@pytest.mark.asyncio
async def test_button_opens_the_editor_as_a_regions_job(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, _query = _callback_update(f"pp:{PAINT_REGIONS_CALLBACK_KIND}:abc")
    context = _context(storage)
    context.bot_data["settings"] = MagicMock(allowed_user_ids=None, inpaint_relay_url="https://r")

    with patch("comfytelegram.handlers._open_mask_editor", new=AsyncMock()) as open_editor:
        await postprocess_callback(update, context)
        kwargs = open_editor.await_args.kwargs
        mask = await kwargs["initial_mask_for"](_png(np.zeros((10, 20, 3), dtype=np.uint8)))

    assert kwargs["job_kind"] == PAINT_REGIONS_JOB_KIND
    assert kwargs["meta_overrides"]["mode"] == "regions"
    assert _white(mask)[:, :10].all() and not _white(mask)[:, 10:].any()


@pytest.mark.asyncio
async def test_button_without_a_relay_says_so(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, query = _callback_update(f"pp:{PAINT_REGIONS_CALLBACK_KIND}:abc")

    await postprocess_callback(update, _context(storage))

    query.message.reply_text.assert_awaited_once_with("Mask drawing isn't configured on this bot.")


# --- the seed --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_comes_from_a_txt2img_graph_only():
    pixels = np.zeros((4, 4), dtype=np.uint8)
    txt2img = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1234}},
    }
    upscale = {"2": {"class_type": "KSampler", "inputs": {"seed": 99}}}
    client = MagicMock(get_image_bytes=AsyncMock(return_value=_png(pixels, graph=txt2img)))

    assert await _txt2img_seed(client, "a.png") == 1234
    client.get_image_bytes.return_value = _png(pixels, graph=upscale)
    assert await _txt2img_seed(client, "a.png") is None
    client.get_image_bytes.side_effect = ComfyUIError("gone")
    assert await _txt2img_seed(client, "a.png") is None


# --- the submission --------------------------------------------------------------


async def _submit(storage, result: dict, *, profiles=(_PROFILE,), seed=1234):
    storage.store_pending_result(
        "abc", 42, "file", "a.png", _serialize_generation_params(_params())
    )
    application = MagicMock()
    application.bot.send_message = AsyncMock(return_value=AsyncMock())
    application.bot_data = {"profiles": list(profiles)}
    settings = MagicMock(inpaint_relay_url="https://r", telegram_bot_token="t")
    job = {
        "token": "tok",
        "chat_id": 42,
        "message_thread_id": 9,
        "result_id": "abc",
        "kind": PAINT_REGIONS_JOB_KIND,
    }
    image = GeneratedImage(data=b"img", filename="o.png", full_params=_params())
    with (
        patch(
            "comfytelegram.handlers._relay_poll_result",
            new=AsyncMock(return_value={"init_data": "x", **result}),
        ),
        patch(
            "comfytelegram.handlers.validate_webapp_init_data",
            return_value={"user": json.dumps({"id": 7})},
        ),
        patch("comfytelegram.handlers._relay_delete_job", new=AsyncMock()),
        patch("comfytelegram.handlers._txt2img_seed", new=AsyncMock(return_value=seed)),
        patch("comfytelegram.handlers.generate", new=AsyncMock(return_value=[image])) as gen,
        patch("comfytelegram.handlers._send_and_store_bot_result", new=AsyncMock()) as send,
    ):
        await _process_one_inpaint_job(application, settings, storage, MagicMock(), job)
    return gen, send, application.bot.send_message


@pytest.mark.asyncio
async def test_a_painting_generates_with_the_painted_side_and_its_inverse(storage):
    gen, send, send_message = await _submit(
        storage,
        {
            "mask": _left_painting(),
            "positive": "2boys, cuddling",
            "negative": "bad hands",
            "region_prompts": ["red hair", "black hair"],
        },
    )

    args, kwargs = gen.await_args.args, gen.await_args.kwargs
    assert args[1:3] == ("ckpt.safetensors", "2boys, cuddling")
    assert "bad hands" in kwargs["extra_negative_prompt"]
    painted, rest = kwargs["regions"]
    assert (painted.prompt, painted.invert_mask) == ("red hair", False)
    assert (rest.prompt, rest.invert_mask) == ("black hair", True)
    assert painted.mask_png == rest.mask_png
    # Same size as the image painted over, and its seed.
    assert kwargs["overrides"] == {"width": 64, "height": 48, "seed": 1234}
    assert kwargs["raw_positive_prompt"] == "2boys, cuddling\n[left] red hair\n[right] black hair"
    send.assert_awaited_once()
    assert send.await_args.kwargs["message_thread_id"] == 9
    assert send_message.await_args.args[1] == "Done — 1 image(s)."
    assert storage.list_inpaint_jobs() == []


@pytest.mark.asyncio
async def test_without_a_seed_the_run_rolls_a_fresh_one(storage):
    gen, _send, _msg = await _submit(
        storage,
        {"mask": _left_painting(), "positive": "x", "region_prompts": ["a", "b"]},
        seed=None,
    )

    assert "seed" not in gen.await_args.kwargs["overrides"]


@pytest.mark.asyncio
async def test_an_empty_painting_is_refused_without_generating(storage):
    gen, _send, send_message = await _submit(
        storage,
        {"mask": _png(np.zeros((48, 64), dtype=np.uint8)), "region_prompts": ["a", "b"]},
    )

    gen.assert_not_awaited()
    assert "not none or all of it" in send_message.await_args.args[1]


@pytest.mark.asyncio
async def test_unknown_names_are_dropped_with_a_warning_and_known_loras_kept(storage):
    gen, _send, send_message = await _submit(
        storage,
        {
            "mask": _left_painting(),
            "positive": "2boys $nobody",
            "region_prompts": ["<lora:marlow:0.7> red hair", "<lora:missing> black hair"],
        },
    )

    warnings = [call.args[1] for call in send_message.await_args_list]
    assert any("$nobody" in text for text in warnings)
    assert any("<lora:missing>" in text for text in warnings)
    painted, rest = gen.await_args.kwargs["regions"]
    assert [lora.name for lora in painted.loras] == ["chars/marlow.safetensors"]
    assert rest.loras == [] and "missing" not in rest.prompt
