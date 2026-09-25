"""The "✏️ Detail Prompt" flow: draw a mask and type a prompt/denoise in
one webapp visit, then run the detailer on that exact mask+prompt
immediately — see handlers.py's `DETAIL_PROMPT_CALLBACK_KIND` and
`generation.DETAIL_PROMPT_KINDS`.

It shares the relay-job/poller machinery "🖌️ Draw Mask"/"🩹 Fix Artifact"
use (`_DRAWN_MASK_KINDS["detail"]`, `post_process(kind="hand_drawn")`), so
these tests focus on what's actually specific to it: the prompt is
one-shot (never saved on the image for a later, unrelated detailer tap —
see `storage.py`'s removed `pending_result.detail_prompt`), and only
"🔁 Redo (same mask)" (`DETAIL_REDO_CALLBACK_KIND`) ever replays it, via
`storage.py`'s `inpaint_redo`."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_generation import _solid_mask_png, _solid_png, _StubUploadingComfyClient

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.generation import GeneratedImage, post_process
from comfytelegram.handlers import (
    DETAIL_PROMPT_CALLBACK_KIND,
    DETAIL_REDO_CALLBACK_KIND,
    _detail_prompt_readonly_info,
    _process_one_inpaint_job,
    _serialize_generation_params,
    postprocess_callback,
)
from comfytelegram.storage import Storage
from comfytelegram.workflows import GenerationParams


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


# ------------------------------------------------------------ generation


def _detailer_prompts(graph: dict) -> tuple[str, str]:
    """The positive/negative text actually wired into the queued detailer
    node — read through its own `positive`/`negative` links rather than by
    collecting every CLIPTextEncode, so this can't pass on a graph that
    encodes the override but wires the original in."""
    detailer = next(
        node for node in graph.values() if node["class_type"] in ("FaceDetailer", "DetailerForEach")
    )
    return (
        graph[detailer["inputs"]["positive"][0]]["inputs"]["text"],
        graph[detailer["inputs"]["negative"][0]]["inputs"]["text"],
    )


def _detailer_denoise(graph: dict) -> float:
    """The denoise value actually wired into the queued detailer node —
    unlike positive/negative, it's a plain scalar input, not a link."""
    detailer = next(
        node for node in graph.values() if node["class_type"] in ("FaceDetailer", "DetailerForEach")
    )
    return detailer["inputs"]["denoise"]


@pytest.mark.asyncio
async def test_detail_prompt_replaces_what_the_drawn_mask_detailer_conditions_on():
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox in a forest, castle, sunset",
        negative_prompt="worst quality",
    )

    await post_process(
        client,
        "hand_drawn",
        source,
        "source.png",
        params,
        mask_bytes=mask,
        detail_prompt="detailed face, freckles",
        detail_negative_prompt="blurry",
    )

    positive, negative = _detailer_prompts(client.queued_graph)
    assert positive == "detailed face, freckles"
    assert negative == "blurry"


@pytest.mark.asyncio
async def test_detail_prompt_sides_are_independent():
    """Overriding only the positive must not wipe a negative prompt the
    user never mentioned."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox in a forest",
        negative_prompt="worst quality",
    )

    await post_process(
        client,
        "hand_drawn",
        source,
        "source.png",
        params,
        mask_bytes=mask,
        detail_prompt="detailed face",
    )

    positive, negative = _detailer_prompts(client.queued_graph)
    assert positive == "detailed face"
    assert negative == "worst quality"


@pytest.mark.asyncio
async def test_detail_denoise_overrides_the_detailers_own_default():
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    await post_process(
        client, "hand_drawn", source, "source.png", params, mask_bytes=mask, denoise=0.9
    )

    assert _detailer_denoise(client.queued_graph) == 0.9


@pytest.mark.asyncio
async def test_upscale_ignores_the_detail_prompt():
    """A prompt written for one masked region would be actively wrong
    applied to every tile of a whole-image pass — see DETAIL_PROMPT_KINDS."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox in a forest",
        negative_prompt="worst quality",
    )

    await post_process(
        client, "upscale", source, "source.png", params, detail_prompt="detailed face"
    )

    texts = [
        node["inputs"]["text"]
        for node in client.queued_graph.values()
        if node["class_type"] == "CLIPTextEncode"
    ]
    assert "a fox in a forest" in texts
    assert "detailed face" not in texts


@pytest.mark.asyncio
async def test_face_auto_detect_ignores_a_detail_prompt_override():
    """Auto-detect Face/Hand Detail never collect a prompt at all — no
    caller passes one, but the contract itself (DETAIL_PROMPT_KINDS) should
    still refuse it defensively."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt="worst quality"
    )

    await post_process(
        client, "face", source, "source.png", params, detail_prompt="detailed face", denoise=0.9
    )

    positive, _negative = _detailer_prompts(client.queued_graph)
    assert positive == "a fox"
    detailer = next(
        n
        for n in client.queued_graph.values()
        if n["class_type"] in ("FaceDetailer", "DetailerForEach")
    )
    assert detailer["inputs"]["denoise"] != 0.9


# -------------------------------------------------------------- handlers


def _context(storage, **bot_data):
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, inpaint_relay_url=None),
        "storage": storage,
        # get_image_bytes raises so `_fetch_source_image` falls back to
        # `_download_telegram_file`, stubbed below via `context.bot.get_file`.
        "comfy_client": MagicMock(get_image_bytes=AsyncMock(side_effect=ComfyUIError("no file"))),
        "profiles": [],
        **bot_data,
    }
    context.bot.get_file = AsyncMock(
        return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=bytearray(b"orig")))
    )
    return context


def _callback_update(data: str):
    query = AsyncMock()
    query.data = data
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1
    return update, query


def _params(positive="a fox in a forest", negative="worst quality") -> GenerationParams:
    return GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt=positive, negative_prompt=negative
    )


def _pending() -> dict:
    return {
        "base_params": _serialize_generation_params(_params()),
        "chat_id": 1,
        "file_id": "file123",
        "filename": "source.png",
    }


def test_detail_prompt_readonly_info_is_the_drawn_mask_detailers_own_settings():
    from comfytelegram.workflows import DrawnMaskHandDetailerParams

    info = _detail_prompt_readonly_info()

    params = DrawnMaskHandDetailerParams()
    assert info == {
        "steps": params.steps,
        "cfg": params.cfg,
        "sampler_name": params.sampler_name,
        "scheduler": params.scheduler,
        "denoise": params.denoise,
    }


@pytest.mark.asyncio
async def test_detail_prompt_button_uploads_and_opens_the_mask_editor():
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    context = _context(storage)
    context.bot_data["settings"] = MagicMock(
        allowed_user_ids=None,
        inpaint_relay_url="https://inpaint.example.com",
        inpaint_relay_shared_secret="shh",
    )

    status_message = AsyncMock()
    query.message.reply_text.return_value = status_message

    with patch(
        "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok1")
    ) as create_job:
        await postprocess_callback(update, context)

    assert create_job.await_args.args[0] is context.bot_data["settings"]
    meta = create_job.await_args.kwargs["meta"]
    assert meta["mode"] == "mask_prompt"
    # Pre-filled with the image's own current prompt, to edit down rather
    # than type from scratch — see _pending()/_params()'s defaults.
    assert meta["positive"] == "a fox in a forest"
    assert meta["negative"] == "worst quality"
    assert "steps" in meta["readonly"]

    storage.store_inpaint_job.assert_called_once_with(
        "tok1", "abc123", 1, query.message.message_thread_id, kind="detail"
    )
    status_message.edit_text.assert_awaited_once()
    button = status_message.edit_text.await_args.kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.web_app.url == "https://inpaint.example.com/jobs/tok1"


@pytest.mark.asyncio
async def test_detail_prompt_button_reports_when_relay_not_configured():
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    context = _context(storage)  # settings.inpaint_relay_url is None by default

    await postprocess_callback(update, context)

    query.message.reply_text.assert_awaited_once_with("Mask drawing isn't configured on this bot.")
    storage.store_inpaint_job.assert_not_called()


@pytest.mark.asyncio
async def test_process_one_inpaint_job_detail_kind_runs_the_one_shot_prompt():
    """A `kind="detail"` job's submission carries a mask *and* a
    prompt/denoise — both go straight into `post_process`, and neither is
    saved anywhere on the image (see storage.py: no more
    `pending_result.detail_prompt`)."""
    storage = MagicMock()
    storage.get_pending_result.return_value = {
        "chat_id": 42,
        "file_id": "presource123",
        "filename": "presource.png",
        "base_params": _serialize_generation_params(_params()),
    }
    application = MagicMock()
    application.bot.send_message = AsyncMock(return_value=AsyncMock())
    application.bot.get_file = AsyncMock(
        return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=bytearray(b"orig")))
    )
    application.bot.send_photo = AsyncMock(return_value=MagicMock())
    settings = MagicMock(inpaint_relay_url="https://inpaint.example.com")
    client = MagicMock(get_image_bytes=AsyncMock(side_effect=ComfyUIError("no file")))
    job = {
        "token": "tok1",
        "chat_id": 42,
        "message_thread_id": None,
        "result_id": "abc123",
        "kind": "detail",
    }

    generated = GeneratedImage(data=b"refined", filename="out.png", full_params=_params())

    with (
        patch(
            "comfytelegram.handlers._relay_poll_result",
            new=AsyncMock(
                return_value={
                    "mask": b"drawn-mask-bytes",
                    "positive": "two hands",
                    "negative": "jewellery",
                    "denoise": 0.42,
                    "init_data": "raw-init-data",
                }
            ),
        ),
        patch("comfytelegram.handlers.validate_webapp_init_data", return_value={"user": "1"}),
        patch("comfytelegram.handlers._relay_delete_job", new=AsyncMock()),
        patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=generated)) as pp,
    ):
        await _process_one_inpaint_job(application, settings, storage, client, job)

    assert pp.await_args.kwargs["detail_prompt"] == "two hands"
    assert pp.await_args.kwargs["detail_negative_prompt"] == "jewellery"
    assert pp.await_args.kwargs["denoise"] == 0.42
    assert pp.await_args.args[1] == "hand_drawn"

    # One-shot: nothing about the prompt is stored on the image itself.
    assert "detail_prompt" not in storage.store_pending_result.call_args.kwargs

    # ...but it *is* kept alongside the mask, for a possible redo.
    storage.store_inpaint_redo.assert_called_once()
    redo_kwargs = storage.store_inpaint_redo.call_args.kwargs
    assert redo_kwargs["detail_prompt"] == "two hands"
    assert redo_kwargs["detail_negative_prompt"] == "jewellery"
    assert redo_kwargs["detail_denoise"] == 0.42


@pytest.mark.asyncio
async def test_hand_draw_job_never_carries_a_prompt():
    """ "🖌️ Draw Mask" jobs never collect a prompt — the relay always
    reports `positive`/`negative`/`denoise` as None for them, and that
    propagates straight through as "no override"."""
    storage = MagicMock()
    storage.get_pending_result.return_value = {
        "chat_id": 42,
        "file_id": "presource123",
        "filename": "presource.png",
        "base_params": _serialize_generation_params(_params()),
    }
    application = MagicMock()
    application.bot.send_message = AsyncMock(return_value=AsyncMock())
    application.bot.get_file = AsyncMock(
        return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=bytearray(b"orig")))
    )
    application.bot.send_photo = AsyncMock(return_value=MagicMock())
    settings = MagicMock(inpaint_relay_url="https://inpaint.example.com")
    client = MagicMock(get_image_bytes=AsyncMock(side_effect=ComfyUIError("no file")))
    job = {
        "token": "tok1",
        "chat_id": 42,
        "message_thread_id": None,
        "result_id": "abc123",
        "kind": "hand",
    }
    generated = GeneratedImage(data=b"refined", filename="out.png", full_params=_params())

    with (
        patch(
            "comfytelegram.handlers._relay_poll_result",
            new=AsyncMock(
                return_value={
                    "mask": b"drawn-mask-bytes",
                    "positive": None,
                    "negative": None,
                    "denoise": None,
                    "init_data": "raw-init-data",
                }
            ),
        ),
        patch("comfytelegram.handlers.validate_webapp_init_data", return_value={"user": "1"}),
        patch("comfytelegram.handlers._relay_delete_job", new=AsyncMock()),
        patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=generated)) as pp,
    ):
        await _process_one_inpaint_job(application, settings, storage, client, job)

    assert pp.await_args.kwargs["detail_prompt"] is None
    assert pp.await_args.kwargs["detail_negative_prompt"] is None
    assert pp.await_args.kwargs["denoise"] is None


@pytest.mark.asyncio
async def test_detail_redo_replays_the_stored_mask_and_prompt():
    update, _query = _callback_update(f"pp:{DETAIL_REDO_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    storage.get_inpaint_redo.return_value = {
        "source_file_id": "srcfile",
        "source_filename": "source.png",
        "mask_png": b"mask-bytes",
        "detail_prompt": "two hands",
        "detail_negative_prompt": "jewellery",
        "detail_denoise": 0.42,
    }
    context = _context(storage)

    generated = GeneratedImage(data=b"redone", filename="out.png", full_params=_params())
    with (
        patch("comfytelegram.handlers.post_process", AsyncMock(return_value=generated)) as pp,
        patch("comfytelegram.handlers._send_result_image", AsyncMock()) as send,
    ):
        send.return_value.photo = [MagicMock(file_id="new-file-id")]
        await postprocess_callback(update, context)

    assert pp.await_args.kwargs["detail_prompt"] == "two hands"
    assert pp.await_args.kwargs["detail_negative_prompt"] == "jewellery"
    assert pp.await_args.kwargs["denoise"] == 0.42
    # The redo's own result carries the override forward again, so a
    # second redo of *that* result can chain indefinitely.
    storage.store_inpaint_redo.assert_called_once()
    assert storage.store_inpaint_redo.call_args.kwargs["detail_prompt"] == "two hands"


@pytest.mark.asyncio
async def test_detail_redo_reports_expired_mask():
    update, query = _callback_update(f"pp:{DETAIL_REDO_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    storage.get_inpaint_redo.return_value = None
    context = _context(storage)

    await postprocess_callback(update, context)

    query.message.reply_text.assert_awaited_once_with(
        "That mask has expired — draw a new one with ✏️ Detail Prompt."
    )


@pytest.mark.asyncio
async def test_face_tap_never_receives_a_detail_prompt_override():
    """Auto-detect Face Detail has no prompt to inherit any more — it
    always uses the image's own scene prompt."""
    update, _query = _callback_update("pp:face:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    context = _context(storage)

    generated = GeneratedImage(
        data=_solid_png(4, 4, (1, 1, 1)), filename="out.png", full_params=_params()
    )
    with (
        patch("comfytelegram.handlers.post_process", AsyncMock(return_value=generated)) as pp,
        patch("comfytelegram.handlers._send_result_image", AsyncMock()) as send,
    ):
        send.return_value.photo = [MagicMock(file_id="new-file-id")]
        await postprocess_callback(update, context)

    assert "detail_prompt" not in pp.await_args.kwargs
    assert "denoise" not in pp.await_args.kwargs
    stored = storage.store_pending_result.call_args
    assert "detail_prompt" not in stored.kwargs
