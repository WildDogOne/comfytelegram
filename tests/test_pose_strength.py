"""`/pose` strength on the fly: "🦴 Pose strength…" under a posed image
(re-render it on the same noise at another strength) and the chat-wide
strength it leaves behind for new images."""

import io
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from comfytelegram.generation import GeneratedImage, generate, repeat
from comfytelegram.handlers import (
    POSE_STRENGTH_CALLBACK_KIND,
    POSE_STRENGTH_CALLBACK_PREFIX,
    POSE_STRENGTH_CHAT_TARGET,
    _chat_pose_strength,
    _consume_awaiting_pose_strength,
    _deserialize_generation_params,
    _parse_pose_strength,
    _pose_keyboard,
    _pose_strength_keyboard,
    _pose_strength_rows,
    _serialize_generation_params,
    pose_strength_callback,
    postprocess_callback,
)
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.storage import Storage
from comfytelegram.topics import NO_TOPIC
from comfytelegram.workflows import GenerationParams


def _skeleton() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 96), "black").save(out, format="PNG")
    return out.getvalue()


def _params(**kwargs) -> GenerationParams:
    defaults = {
        "pose_png": _skeleton(),
        "pose_controlnet": "openpose.safetensors",
        "batch_size": 3,
    }
    return GenerationParams("c.safetensors", "pos", "neg", **{**defaults, **kwargs})


def _node(graph: dict, class_type: str) -> dict:
    ((node,),) = [[v for v in graph.values() if v["class_type"] == class_type]]
    return node


def _profile(**kwargs) -> ModelProfile:
    return ModelProfile(
        match=["*"],
        display_name="Pose Model",
        defaults=ProfileDefaults(width=1024, height=1024),
        pose_controlnet="openpose.safetensors",
        **kwargs,
    )


# --- generation -----------------------------------------------------------------


async def _graphs_of(coro_factory):
    graphs = []

    async def run_graph(_client, graph, *args, **kwargs):
        graphs.append(graph)
        return [], {}

    with patch("comfytelegram.generation._run_graph", new=run_graph):
        await coro_factory()
    return graphs


@pytest.mark.asyncio
async def test_generate_uses_the_chats_pose_strength_over_the_profiles():
    client = MagicMock(upload_image=AsyncMock())
    profile = _profile(pose_controlnet_strength=0.8)

    graphs = await _graphs_of(
        lambda: generate(client, "c", "p", profile, pose_png=_skeleton(), pose_strength=0.55)
    )

    assert _node(graphs[0], "ControlNetApplyAdvanced")["inputs"]["strength"] == 0.55


@pytest.mark.asyncio
async def test_generate_without_a_chat_strength_keeps_the_profiles():
    client = MagicMock(upload_image=AsyncMock())
    profile = _profile(pose_controlnet_strength=0.7)

    graphs = await _graphs_of(lambda: generate(client, "c", "p", profile, pose_png=_skeleton()))

    assert _node(graphs[0], "ControlNetApplyAdvanced")["inputs"]["strength"] == 0.7


@pytest.mark.asyncio
async def test_repeat_can_pin_the_seed():
    client = MagicMock(upload_image=AsyncMock())

    pinned = await _graphs_of(lambda: repeat(client, _params(), seed=1234))
    fresh = await _graphs_of(lambda: repeat(client, _params(seed=1234)))

    assert _node(pinned[0], "KSampler")["inputs"]["seed"] == 1234
    # Without `seed`, a stored seed is still dropped — "🔁 Generate Again".
    assert _node(fresh[0], "KSampler")["inputs"]["seed"] != 1234


# --- keyboard -------------------------------------------------------------------


def test_the_button_is_only_under_txt2img_images_made_with_a_pose():
    posed = GeneratedImage(data=b"", filename="a.png", full_params=_params(), noise_index=0)
    unposed = replace(posed, full_params=_params(pose_png=None))
    upscaled = replace(posed, noise_index=None)

    (row,) = _pose_strength_rows("rid", posed)
    assert row[0].callback_data == f"pp:{POSE_STRENGTH_CALLBACK_KIND}:rid"
    assert _pose_strength_rows("rid", unposed) == []
    assert _pose_strength_rows("rid", upscaled) == []


def test_picker_marks_the_current_strength():
    keyboard = _pose_strength_keyboard("rid", 0.8)
    labels = [b.text for row in keyboard.inline_keyboard for b in row]

    assert "• 0.8" in labels and "0.6" in labels
    assert keyboard.inline_keyboard[0][0].callback_data == f"{POSE_STRENGTH_CALLBACK_PREFIX}rid:0.3"
    assert all(len(b.callback_data.encode()) <= 64 for row in keyboard.inline_keyboard for b in row)


def test_pose_keyboard_offers_the_strength_only_with_a_pose():
    with_pose = _pose_keyboard(has_pose=True, waiting=False, strength=0.65)
    without = _pose_keyboard(has_pose=False, waiting=True, strength=0.65)

    assert with_pose.inline_keyboard[-1][0].text == "🎚 Strength 0.65"
    assert with_pose.inline_keyboard[-1][0].callback_data == (
        f"{POSE_STRENGTH_CALLBACK_PREFIX}{POSE_STRENGTH_CHAT_TARGET}"
    )
    assert all("Strength" not in b.text for row in without.inline_keyboard for b in row)


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.65", 0.65), ("0,5", 0.5), ("2", 2.0), ("0", None), ("3", None)]
)
def test_parse_pose_strength(raw, expected):
    assert _parse_pose_strength(raw) == expected


# --- storage --------------------------------------------------------------------


def test_chat_strength_lives_with_the_pose(tmp_path: Path):
    storage = Storage(tmp_path / "state.sqlite3")

    assert storage.set_pose_strength(1, 0.6) is False  # no pose to set it on
    storage.set_pose(1, b"pose")
    assert storage.set_pose_strength(1, 0.6) is True
    storage.set_pose(1, b"another pose")  # a new pose keeps the pick
    assert storage.get_pose_strength(1) == 0.6
    storage.clear_pose(1)
    assert storage.get_pose_strength(1) is None


def test_chat_strength_falls_back_to_the_profile_then_the_default(tmp_path: Path):
    storage = Storage(tmp_path / "state.sqlite3")
    profiles = [_profile(pose_controlnet_strength=0.7)]

    assert _chat_pose_strength(storage, profiles, 1) == 0.8  # no checkpoint picked
    storage.set_checkpoint(1, "c.safetensors")
    assert _chat_pose_strength(storage, profiles, 1) == 0.7
    storage.set_pose(1, b"pose")
    storage.set_pose_strength(1, 0.45)
    assert _chat_pose_strength(storage, profiles, 1) == 0.45


# --- Telegram flow ----------------------------------------------------------------


def _context(storage: Storage) -> MagicMock:
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "profiles": [_profile()],
        "comfy_client": MagicMock(),
    }
    context.bot.delete_message = AsyncMock()
    return context


def _store_posed_image(storage: Storage, *, noise_index: int | None = 2) -> None:
    storage.store_pending_result(
        "rid",
        1,
        "file-id",
        "out_00003_.png",
        _serialize_generation_params(_params(pose_controlnet_strength=0.8)),
        noise_index=noise_index,
    )


def _query(data: str):
    query = AsyncMock()
    query.data = data
    query.message.chat_id = 1
    query.message.chat.id = 1
    query.message.message_id = 50
    query.message.message_thread_id = None
    query.message.reply_to_message = None
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1
    update.effective_chat.id = 1
    return update, query


@pytest.mark.asyncio
async def test_button_opens_the_picker_at_the_images_strength(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    _store_posed_image(storage)
    update, query = _query(f"pp:{POSE_STRENGTH_CALLBACK_KIND}:rid")

    await postprocess_callback(update, _context(storage))

    text = query.message.reply_text.await_args.args[0]
    assert "now 0.8" in text
    keyboard = query.message.reply_text.await_args.kwargs["reply_markup"]
    assert any(b.text == "• 0.8" for row in keyboard.inline_keyboard for b in row)


def _batch(n: int) -> list[GeneratedImage]:
    return [
        GeneratedImage(
            data=f"img{i}".encode(),
            filename=f"out{i}.png",
            full_params=_params(pose_controlnet_strength=0.6),
            noise_index=i,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_a_pick_rerenders_that_image_on_the_same_noise(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    storage.set_pose(1, b"pose")
    _store_posed_image(storage, noise_index=2)
    update, query = _query(f"{POSE_STRENGTH_CALLBACK_PREFIX}rid:0.6")
    context = _context(storage)

    with (
        patch("comfytelegram.handlers._txt2img_seed", new=AsyncMock(return_value=1234)),
        patch("comfytelegram.handlers.repeat", new=AsyncMock(return_value=_batch(3))) as rerun,
        patch(
            "comfytelegram.handlers._send_and_store_result", new=AsyncMock(return_value="new")
        ) as send,
    ):
        await pose_strength_callback(update, context)

    params = rerun.await_args.args[1]
    assert rerun.await_args.kwargs["seed"] == 1234
    assert params.pose_controlnet_strength == 0.6
    assert params.batch_size == 3
    sent = send.await_args.args[3]
    assert sent.data == b"img2"  # the same slice of the batch
    assert sent.noise_index == 2 and sent.full_params.batch_size == 3
    assert storage.get_pose_strength(1) is None  # the image's, not the chat's
    context.bot.delete_message.assert_awaited_once_with(chat_id=1, message_id=50)
    # A "Done" reply with "🔁 Generate Again" for a full batch at 0.6.
    done = query.message.reply_text.await_args_list[-1]
    assert "Generate Again tries 3" in done.args[0]
    (snapshot_id,) = [
        b.callback_data.split(":", 1)[1]
        for row in done.kwargs["reply_markup"].inline_keyboard
        for b in row
    ]
    snapshot = storage.get_generation_snapshot(snapshot_id)
    again = _deserialize_generation_params(snapshot["params"])
    assert again.pose_controlnet_strength == 0.6 and again.batch_size == 3


@pytest.mark.asyncio
async def test_without_a_seed_it_renders_one_fresh_image_and_says_so(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    _store_posed_image(storage, noise_index=2)
    update, query = _query(f"{POSE_STRENGTH_CALLBACK_PREFIX}rid:0.5")

    with (
        patch("comfytelegram.handlers._txt2img_seed", new=AsyncMock(return_value=None)),
        patch("comfytelegram.handlers.repeat", new=AsyncMock(return_value=_batch(1))) as rerun,
        patch(
            "comfytelegram.handlers._send_and_store_result", new=AsyncMock(return_value="new")
        ) as send,
    ):
        await pose_strength_callback(update, _context(storage))

    assert rerun.await_args.kwargs["seed"] is None
    assert rerun.await_args.args[1].batch_size == 1
    sent = send.await_args.args[3]
    assert sent.full_params.batch_size == 1 and sent.noise_index == 0
    assert "new seed" in query.message.reply_text.await_args_list[-1].args[0]


@pytest.mark.asyncio
async def test_chat_target_only_sets_the_strength(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    storage.set_pose(1, b"pose")
    update, query = _query(f"{POSE_STRENGTH_CALLBACK_PREFIX}{POSE_STRENGTH_CHAT_TARGET}:0.7")

    with patch("comfytelegram.handlers.repeat", new=AsyncMock()) as rerun:
        await pose_strength_callback(update, _context(storage))

    rerun.assert_not_awaited()
    assert storage.get_pose_strength(1) == 0.7
    assert "0.7 for new images" in query.message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_chat_picker_needs_a_pose(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    update, query = _query(f"{POSE_STRENGTH_CALLBACK_PREFIX}{POSE_STRENGTH_CHAT_TARGET}")

    await pose_strength_callback(update, _context(storage))

    assert "No pose is set" in query.message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_custom_value_is_read_from_the_next_message(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    storage.set_pose(1, b"pose")
    context = _context(storage)
    update, query = _query(f"{POSE_STRENGTH_CALLBACK_PREFIX}{POSE_STRENGTH_CHAT_TARGET}:custom")
    query.message.reply_text.return_value = MagicMock(message_id=51)
    await pose_strength_callback(update, context)

    message = AsyncMock()
    message.text = "nonsense"
    message.chat_id = 1
    message.message_thread_id = None
    message.reply_text.return_value = MagicMock(message_id=52)
    typed = MagicMock(effective_message=message)
    assert await _consume_awaiting_pose_strength(typed, context) is True
    assert storage.get_pose_strength(1) is None  # asked again
    assert context.chat_data["awaiting_pose_strength"][NO_TOPIC]

    message.text = "0.65"
    assert await _consume_awaiting_pose_strength(typed, context) is True
    assert storage.get_pose_strength(1) == 0.65
    deleted = {call.kwargs["message_id"] for call in context.bot.delete_message.await_args_list}
    assert deleted == {50, 51, 52}
    assert await _consume_awaiting_pose_strength(typed, context) is False
