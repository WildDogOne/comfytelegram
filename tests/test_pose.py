"""`/pose`: OpenPose skeleton guidance — image validation and sizing,
the ControlNet branch in `build_txt2img`, serialization, storage, and the
Telegram flow."""

import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from comfytelegram.generation import generate, repeat
from comfytelegram.handlers import (
    POSE_CLEAR_CALLBACK_DATA,
    _consume_pose_upload,
    _pose_for_generation,
    pose_callback,
    pose_command,
)
from comfytelegram.params_serde import (
    deserialize_generation_params,
    serialize_generation_params,
)
from comfytelegram.pose import PoseImageError, normalize_pose_image, pose_size, size_for_pose
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.regional import parse_regional_prompt
from comfytelegram.storage import Storage
from comfytelegram.topics import NO_TOPIC
from comfytelegram.workflows import GenerationParams, build_txt2img, pose_image_filename


def _skeleton(width: int = 512, height: int = 768) -> bytes:
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    draw.line(
        [(width // 2, height // 5), (width // 2, height * 3 // 5)], fill=(0, 200, 80), width=8
    )
    draw.line(
        [(width // 3, height // 3), (width * 2 // 3, height // 3)], fill=(200, 80, 0), width=8
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _photo() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 150, 120)).save(out, format="JPEG")
    return out.getvalue()


def _node(graph: dict, class_type: str) -> tuple[str, dict]:
    ((node_id, node),) = [(k, v) for k, v in graph.items() if v["class_type"] == class_type]
    return node_id, node


def _profile(**kwargs) -> ModelProfile:
    return ModelProfile(
        match=["*"],
        display_name="Pose Model",
        defaults=ProfileDefaults(width=1024, height=1024),
        **kwargs,
    )


# --- image ----------------------------------------------------------------------


def test_normalize_accepts_a_skeleton_and_keeps_its_size():
    pose = normalize_pose_image(_skeleton(512, 768))

    assert pose_size(pose) == (512, 768)


def test_normalize_scales_a_huge_pose_down():
    assert max(pose_size(normalize_pose_image(_skeleton(3000, 2000)))) == 1536


@pytest.mark.parametrize("data", [b"not an image", _photo()])
def test_normalize_rejects_what_isnt_a_skeleton(data):
    with pytest.raises(PoseImageError):
        normalize_pose_image(data)


def test_size_for_pose_keeps_the_pixel_count_at_the_pose_aspect():
    width, height = size_for_pose(_skeleton(512, 768), 1024, 1024)

    assert (width % 64, height % 64) == (0, 0)
    assert abs(width / height - 512 / 768) < 0.05
    assert abs(width * height - 1024 * 1024) / (1024 * 1024) < 0.1


# --- graph ----------------------------------------------------------------------


def _params(**kwargs) -> GenerationParams:
    return GenerationParams(
        "c.safetensors",
        "pos",
        "neg",
        pose_png=_skeleton(),
        pose_controlnet="openpose.safetensors",
        **kwargs,
    )


def test_txt2img_conditions_the_sampler_on_the_pose():
    params = _params(pose_controlnet_strength=0.7, pose_controlnet_end=0.6)
    graph, _ = build_txt2img(params)

    _, load = _node(graph, "LoadImage")
    loader_id, loader = _node(graph, "ControlNetLoader")
    apply_id, apply = _node(graph, "ControlNetApplyAdvanced")
    _, sampler = _node(graph, "KSampler")
    assert load["inputs"]["image"] == pose_image_filename(params.pose_png)
    assert loader["inputs"]["control_net_name"] == "openpose.safetensors"
    assert apply["inputs"]["control_net"] == [loader_id, 0]
    assert (apply["inputs"]["strength"], apply["inputs"]["end_percent"]) == (0.7, 0.6)
    assert sampler["inputs"]["positive"] == [apply_id, 0]
    assert sampler["inputs"]["negative"] == [apply_id, 1]


@pytest.mark.parametrize(
    "changes", [{"pose_png": None}, {"pose_controlnet": None}], ids=["no pose", "no controlnet"]
)
def test_txt2img_without_pose_or_controlnet_has_no_controlnet(changes):
    graph, _ = build_txt2img(replace(_params(), **changes))

    assert not any(node["class_type"] == "ControlNetApplyAdvanced" for node in graph.values())


def test_regional_pose_keeps_the_plain_global_prompt_as_base_cond():
    regions = parse_regional_prompt("[left] a\n[right] b").regions
    graph, _ = build_txt2img(_params(regions=regions))

    _, couple = _node(graph, "AttentionCouplePPM")
    apply_id, apply = _node(graph, "ControlNetApplyAdvanced")
    _, sampler = _node(graph, "KSampler")
    assert couple["inputs"]["base_cond"] == apply["inputs"]["positive"]
    assert sampler["inputs"]["positive"] == [apply_id, 0]


def test_pose_survives_serialization():
    params = _params(pose_controlnet_strength=0.7)

    data = serialize_generation_params(params)
    json.dumps(data)
    restored = deserialize_generation_params(data)

    assert restored.pose_png == params.pose_png
    assert (restored.pose_controlnet, restored.pose_controlnet_strength) == (
        "openpose.safetensors",
        0.7,
    )


def test_params_from_before_poses_existed_have_none():
    data = serialize_generation_params(GenerationParams("c", "p", ""))
    for key in ("pose_png", "pose_controlnet", "pose_controlnet_strength", "pose_controlnet_end"):
        del data[key]

    assert deserialize_generation_params(data).pose_png is None


# --- generate -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_uploads_the_pose_and_takes_its_aspect():
    pose = _skeleton(512, 768)
    client = MagicMock()
    client.upload_image = AsyncMock()
    graphs = []

    async def run_graph(_client, graph, *args, **kwargs):
        graphs.append(graph)
        return [], {}

    profile = _profile(pose_controlnet="openpose.safetensors")
    with patch("comfytelegram.generation._run_graph", new=run_graph):
        await generate(client, "c", "two people", profile, pose_png=pose)

    client.upload_image.assert_awaited_once_with(
        pose, filename=pose_image_filename(pose), overwrite=True
    )
    _, latent = _node(graphs[0], "EmptyLatentImage")
    assert latent["inputs"]["width"] < latent["inputs"]["height"]


@pytest.mark.asyncio
async def test_generate_keeps_a_size_the_caller_pinned():
    client = MagicMock()
    client.upload_image = AsyncMock()
    graphs = []

    async def run_graph(_client, graph, *args, **kwargs):
        graphs.append(graph)
        return [], {}

    profile = _profile(pose_controlnet="openpose.safetensors")
    with patch("comfytelegram.generation._run_graph", new=run_graph):
        await generate(
            client,
            "c",
            "p",
            profile,
            pose_png=_skeleton(512, 768),
            overrides={"width": 1024, "height": 1024},
        )

    _, latent = _node(graphs[0], "EmptyLatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1024, 1024)


@pytest.mark.asyncio
async def test_repeat_replays_the_pose_the_image_was_made_with():
    client = MagicMock()
    client.upload_image = AsyncMock()

    async def run_graph(*args, **kwargs):
        return [], {}

    params = _params()
    with patch("comfytelegram.generation._run_graph", new=run_graph):
        await repeat(client, params)

    client.upload_image.assert_awaited_once_with(
        params.pose_png, filename=pose_image_filename(params.pose_png), overwrite=True
    )


# --- storage --------------------------------------------------------------------


def test_pose_storage_roundtrip(tmp_path: Path):
    storage = Storage(tmp_path / "state.sqlite3")
    try:
        assert storage.get_pose(1) is None
        storage.set_pose(1, b"one")
        storage.set_pose(1, b"two")
        storage.set_pose(2, b"other")
        assert storage.get_pose(1) == b"two"
        storage.clear_pose(1)
        assert storage.get_pose(1) is None
        assert storage.get_pose(2) == b"other"
    finally:
        storage.close()


# --- handlers -------------------------------------------------------------------


def _update(*, text: str | None = None, caption: str | None = None):
    message = AsyncMock()
    message.message_thread_id = None
    message.text = text
    message.caption = caption
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    update.effective_user.id = 1
    return update, message


def _context(tmp_path: Path, upload: bytes = b"", chat_data: dict | None = None):
    storage = Storage(tmp_path / "state.sqlite3")
    context = MagicMock()
    context.chat_data = {} if chat_data is None else chat_data
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "profiles": [_profile(pose_controlnet="openpose.safetensors")],
    }
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(upload)
    context.bot.get_file = AsyncMock(return_value=tg_file)
    return context, storage


@pytest.mark.asyncio
async def test_pose_command_waits_for_an_upload(tmp_path):
    update, message = _update(text="/pose")
    context, _storage = _context(tmp_path)

    await pose_command(update, context)

    assert context.chat_data["awaiting_pose"][NO_TOPIC] is True
    message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_pose_off_clears_it(tmp_path):
    update, message = _update(text="/pose off")
    context, storage = _context(tmp_path)
    storage.set_pose(1, _skeleton())

    await pose_command(update, context)

    assert storage.get_pose(1) is None
    assert message.reply_text.await_args.args[0] == "🦴 Pose cleared."


@pytest.mark.asyncio
async def test_an_upload_while_waiting_becomes_the_pose(tmp_path):
    update, message = _update()
    context, storage = _context(
        tmp_path, upload=_skeleton(), chat_data={"awaiting_pose": {NO_TOPIC: True}}
    )

    assert await _consume_pose_upload(update, context, "file") is True

    assert pose_size(storage.get_pose(1)) == (512, 768)
    assert "awaiting_pose" not in context.chat_data
    assert message.reply_text.await_args.args[0].startswith("🦴 Pose set (512x768)")


@pytest.mark.asyncio
async def test_a_photo_captioned_pose_becomes_the_pose(tmp_path):
    update, _message = _update(caption="/pose")
    context, storage = _context(tmp_path, upload=_skeleton())

    assert await _consume_pose_upload(update, context, "file") is True
    assert storage.get_pose(1) is not None


@pytest.mark.asyncio
async def test_an_ordinary_upload_is_left_alone(tmp_path):
    update, _message = _update(caption="look at this")
    context, storage = _context(tmp_path, upload=_skeleton())

    assert await _consume_pose_upload(update, context, "file") is False
    assert storage.get_pose(1) is None


@pytest.mark.asyncio
async def test_a_non_skeleton_upload_is_refused_and_keeps_waiting(tmp_path):
    update, message = _update()
    context, storage = _context(
        tmp_path, upload=_photo(), chat_data={"awaiting_pose": {NO_TOPIC: True}}
    )

    assert await _consume_pose_upload(update, context, "file") is True

    assert storage.get_pose(1) is None
    assert context.chat_data["awaiting_pose"][NO_TOPIC] is True
    assert message.reply_text.await_args.args[0].startswith("⚠️")


@pytest.mark.asyncio
async def test_pose_for_generation_skips_a_checkpoint_without_pose_controlnet(tmp_path):
    _, message = _update()
    _ctx, storage = _context(tmp_path)
    storage.set_pose(1, _skeleton())

    assert await _pose_for_generation(message, storage, 1, _profile()) is None
    assert "no pose ControlNet" in message.reply_text.await_args.args[0]

    supported = _profile(pose_controlnet="openpose.safetensors")
    assert await _pose_for_generation(message, storage, 1, supported) == storage.get_pose(1)


@pytest.mark.asyncio
async def test_clear_button_clears_the_pose(tmp_path):
    context, storage = _context(tmp_path, chat_data={"awaiting_pose": {NO_TOPIC: True}})
    storage.set_pose(1, _skeleton())
    query = AsyncMock()
    query.data = POSE_CLEAR_CALLBACK_DATA
    query.message.chat.id = 1
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1

    await pose_callback(update, context)

    assert storage.get_pose(1) is None
    assert "awaiting_pose" not in context.chat_data
