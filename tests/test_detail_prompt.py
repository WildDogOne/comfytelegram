"""The "✏️ Detail Prompt" flow: a per-image prompt override that only the
region detailers (face/hand/fix) condition on — see handlers.py's
`DETAIL_PROMPT_CALLBACK_KIND` and `generation.DETAIL_PROMPT_KINDS`."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_generation import _solid_png, _StubUploadingComfyClient

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.generation import GeneratedImage, post_process
from comfytelegram.handlers import (
    DETAIL_PROMPT_CALLBACK_KIND,
    DETAIL_PROMPT_RESET_WORD,
    DETAIL_RESET_CALLBACK_KIND,
    _consume_awaiting_detail_prompt,
    _serialize_generation_params,
    detail_prompt_cancel_callback,
    postprocess_callback,
)
from comfytelegram.storage import Storage
from comfytelegram.workflows import GenerationParams


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


# --------------------------------------------------------------- storage


def test_detail_prompt_defaults_to_unset(storage: Storage):
    storage.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    row = storage.get_pending_result("abc123")
    assert row["detail_prompt"] is None
    assert row["detail_negative_prompt"] is None


def test_set_detail_prompt_roundtrips(storage: Storage):
    storage.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    assert storage.set_detail_prompt("abc123", "two hands", "jewellery") is True

    row = storage.get_pending_result("abc123")
    assert row["detail_prompt"] == "two hands"
    assert row["detail_negative_prompt"] == "jewellery"
    # The override lives beside base_params, never inside it — that blob
    # also drives upscale/homogenize and "🐛 Show Prompt".
    assert row["base_params"] == {}


def test_set_detail_prompt_clears_with_none(storage: Storage):
    storage.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    storage.set_detail_prompt("abc123", "two hands", "jewellery")
    storage.set_detail_prompt("abc123", None, None)

    row = storage.get_pending_result("abc123")
    assert row["detail_prompt"] is None
    assert row["detail_negative_prompt"] is None


def test_set_detail_prompt_reports_an_expired_row(storage: Storage):
    """The button outlives the row it points at (PENDING_RESULT_TTL_SECONDS),
    so the caller needs to be able to say so rather than silently no-op."""
    assert storage.set_detail_prompt("gone", "two hands", None) is False


def test_store_pending_result_carries_an_inherited_override(storage: Storage):
    """A detail prompt set once has to survive a chain of passes — each
    result is a new row, so the override is copied onto it explicitly."""
    storage.store_pending_result(
        "derived",
        1,
        "FILE_ID",
        "out.png",
        {},
        detail_prompt="two hands",
        detail_negative_prompt=None,
    )
    row = storage.get_pending_result("derived")
    assert row["detail_prompt"] == "two hands"
    assert row["detail_negative_prompt"] is None


def test_detail_prompt_survives_reopen(tmp_path: Path):
    """The guarded ALTER TABLE has to run against a database created before
    these columns existed — `state.sqlite3` is a bind-mounted file that
    outlives any image build (see storage.py's `_add_column_if_missing`)."""
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    s1.set_detail_prompt("abc123", "two hands", None)
    s1.close()

    s2 = Storage(db_path)
    assert s2.get_pending_result("abc123")["detail_prompt"] == "two hands"
    s2.close()


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


@pytest.mark.asyncio
async def test_detail_prompt_replaces_what_the_detailer_conditions_on():
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox in a forest, castle, sunset",
        negative_prompt="worst quality",
    )

    await post_process(
        client,
        "face",
        source,
        "source.png",
        params,
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
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox in a forest",
        negative_prompt="worst quality",
    )

    await post_process(client, "face", source, "source.png", params, detail_prompt="detailed face")

    positive, negative = _detailer_prompts(client.queued_graph)
    assert positive == "detailed face"
    assert negative == "worst quality"


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


# -------------------------------------------------------------- handlers


def _context(storage, **bot_data):
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        # get_image_bytes raises so `_fetch_source_image` falls back to
        # `_download_telegram_file` — the source these tests actually stub
        # via `context.bot.get_file`.
        "comfy_client": MagicMock(get_image_bytes=AsyncMock(side_effect=ComfyUIError("no file"))),
        "profiles": [],
        **bot_data,
    }
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


def _pending(detail_prompt=None, detail_negative_prompt=None) -> dict:
    return {
        "base_params": _serialize_generation_params(_params()),
        "chat_id": 1,
        "file_id": "file123",
        "filename": "source.png",
        "detail_prompt": detail_prompt,
        "detail_negative_prompt": detail_negative_prompt,
    }


@pytest.mark.asyncio
async def test_detail_prompt_button_offers_the_current_prompt_to_edit():
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    context = _context(storage)

    await postprocess_callback(update, context)

    text = query.message.reply_text.await_args.args[0]
    # The starting point is the image's own prompt, in paste-back-able form
    # — the usual reason to open this is to cut most of it away.
    assert "a fox in a forest\n---\nworst quality" in text
    assert context.chat_data["awaiting_detail_prompt"] == {0: "abc123"}
    copy_button = query.message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard[0][0]
    assert copy_button.copy_text.text == "a fox in a forest\n---\nworst quality"


@pytest.mark.asyncio
async def test_detail_prompt_entry_splits_negatives(storage: Storage):
    storage.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    update = MagicMock()
    update.effective_message = AsyncMock()
    update.effective_message.text = "two hands, five fingers\n---\nextra fingers"
    update.effective_message.message_thread_id = None
    update.effective_message.api_kwargs = {}
    context = _context(storage)
    context.chat_data = {"awaiting_detail_prompt": {0: "abc123"}}

    assert await _consume_awaiting_detail_prompt(update, context) is True

    row = storage.get_pending_result("abc123")
    assert row["detail_prompt"] == "two hands, five fingers"
    assert row["detail_negative_prompt"] == "extra fingers"


@pytest.mark.asyncio
async def test_detail_prompt_entry_leaves_the_unmentioned_side_alone(storage: Storage):
    storage.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    update = MagicMock()
    update.effective_message = AsyncMock()
    update.effective_message.text = "two hands"
    update.effective_message.message_thread_id = None
    update.effective_message.api_kwargs = {}
    context = _context(storage)
    context.chat_data = {"awaiting_detail_prompt": {0: "abc123"}}

    await _consume_awaiting_detail_prompt(update, context)

    row = storage.get_pending_result("abc123")
    assert row["detail_prompt"] == "two hands"
    assert row["detail_negative_prompt"] is None


@pytest.mark.asyncio
async def test_detail_prompt_reset_clears_the_override(storage: Storage):
    storage.store_pending_result("abc123", 1, "FILE_ID", "out.png", {})
    storage.set_detail_prompt("abc123", "two hands", "extra fingers")
    update = MagicMock()
    update.effective_message = AsyncMock()
    update.effective_message.text = DETAIL_PROMPT_RESET_WORD.upper()
    update.effective_message.message_thread_id = None
    update.effective_message.api_kwargs = {}
    context = _context(storage)
    context.chat_data = {"awaiting_detail_prompt": {0: "abc123"}}

    await _consume_awaiting_detail_prompt(update, context)

    row = storage.get_pending_result("abc123")
    assert row["detail_prompt"] is None
    assert row["detail_negative_prompt"] is None


@pytest.mark.asyncio
async def test_detail_prompt_consumer_ignores_an_ordinary_prompt(storage: Storage):
    """Nothing pending means the text is a generation prompt — the consumer
    has to hand it straight back to `generate_message`."""
    update = MagicMock()
    update.effective_message = AsyncMock()
    update.effective_message.text = "a fox"
    update.effective_message.message_thread_id = None
    update.effective_message.api_kwargs = {}
    context = _context(storage)

    assert await _consume_awaiting_detail_prompt(update, context) is False
    update.effective_message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_detail_prompt_cancel_clears_the_pending_entry():
    update, _query = _callback_update("detail:cancel")
    context = _context(MagicMock())
    context.chat_data = {"awaiting_detail_prompt": {0: "abc123"}}

    await detail_prompt_cancel_callback(update, context)

    assert "awaiting_detail_prompt" not in context.chat_data


@pytest.mark.asyncio
async def test_face_tap_passes_the_override_through_and_inherits_it():
    update, _query = _callback_update("pp:face:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending("detailed face", "blurry")
    context = _context(storage)
    context.bot.get_file = AsyncMock()
    context.bot.get_file.return_value.download_as_bytearray = AsyncMock(
        return_value=bytearray(_solid_png(4, 4, (0, 0, 0)))
    )

    generated = GeneratedImage(
        data=_solid_png(4, 4, (1, 1, 1)), filename="out.png", full_params=_params()
    )
    with (
        patch("comfytelegram.handlers.post_process", AsyncMock(return_value=generated)) as pp,
        patch("comfytelegram.handlers._send_result_image", AsyncMock()) as send,
    ):
        send.return_value.photo = [MagicMock(file_id="new-file-id")]
        await postprocess_callback(update, context)

    assert pp.await_args.kwargs["detail_prompt"] == "detailed face"
    assert pp.await_args.kwargs["detail_negative_prompt"] == "blurry"
    # ...and the refined image keeps it, so the next detailer tap on the
    # result doesn't need it re-entered.
    stored = storage.store_pending_result.call_args.kwargs
    assert stored["detail_prompt"] == "detailed face"
    assert stored["detail_negative_prompt"] == "blurry"


def _keyboard_of(query) -> list[list[str | None]]:
    markup = query.message.reply_text.await_args.kwargs["reply_markup"]
    return [[b.callback_data for b in row] for row in markup.inline_keyboard]


@pytest.mark.asyncio
async def test_no_reset_button_when_there_is_nothing_to_reset():
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending()
    await postprocess_callback(update, _context(storage))

    flat = [data for row in _keyboard_of(query) for data in row]
    assert f"pp:{DETAIL_RESET_CALLBACK_KIND}:abc123" not in flat


@pytest.mark.asyncio
async def test_reset_button_appears_once_an_override_is_set():
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending("detailed face", None)
    await postprocess_callback(update, _context(storage))

    flat = [data for row in _keyboard_of(query) for data in row]
    assert f"pp:{DETAIL_RESET_CALLBACK_KIND}:abc123" in flat


@pytest.mark.asyncio
async def test_reset_button_clears_the_override_and_closes_the_entry():
    """The typed reset word is only valid while the entry is open — send a
    prompt first and it lands in `generate_message` as an ordinary prompt,
    starting a full generation of the word "reset". The button can't be
    mistimed that way, and it has to close the entry behind it so the next
    ordinary prompt isn't swallowed either."""
    update, query = _callback_update(f"pp:{DETAIL_RESET_CALLBACK_KIND}:abc123")
    storage = MagicMock()
    storage.get_pending_result.return_value = _pending("detailed face", "blurry")
    context = _context(storage)
    context.chat_data = {"awaiting_detail_prompt": {0: "abc123"}}

    await postprocess_callback(update, context)

    storage.set_detail_prompt.assert_called_once_with("abc123", None, None)
    assert "awaiting_detail_prompt" not in context.chat_data
    assert "cleared" in query.edit_message_text.await_args.args[0]
