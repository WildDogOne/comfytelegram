"""`/pose`'s keypoint side: the OpenPose JSON model and renderer, DWPose
output decoding, and the detect/edit flows through inpaint_relay's pose
editor."""

import io
import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from comfytelegram import pose_detect
from comfytelegram.handlers import (
    POSE_EDIT_CALLBACK_DATA,
    POSE_JOB_KIND,
    _consume_pose_upload,
    _process_one_inpaint_job,
    _relay_poll_result,
    pose_callback,
)
from comfytelegram.pose import (
    KEYPOINT_COLORS,
    LIMBS,
    MAX_PEOPLE,
    NUM_KEYPOINTS,
    POSE_MAX_SIDE,
    Pose,
    PoseDataError,
    canvas_size,
    default_person,
    normalize_pose_image,
    parse_pose_json,
    pose_size,
    render_pose,
)
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.storage import Storage
from comfytelegram.topics import NO_TOPIC


def _two_people(width: int = 1024, height: int = 768) -> Pose:
    return Pose(
        width,
        height,
        (
            default_person(width, height, center_x=0.3),
            default_person(width, height, center_x=0.7),
        ),
    )


def _pixel(png: bytes, xy: tuple[float, float]) -> tuple[int, int, int]:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return image.getpixel((round(xy[0]), round(xy[1])))


# --- model ------------------------------------------------------------------


def test_json_roundtrip_keeps_missing_keypoints_missing():
    person = list(default_person(800, 1200))
    person[4] = None  # right wrist hidden
    pose = Pose(800, 1200, (tuple(person),))

    data = json.loads(pose.to_json())
    assert data["canvas_width"] == 800
    assert data["people"][0]["pose_keypoints_2d"][4 * 3 : 4 * 3 + 3] == [0.0, 0.0, 0.0]

    back = parse_pose_json(pose.to_json())
    assert back.width == 800 and back.height == 1200
    assert back.people[0][4] is None
    assert back.people[0][1] == pytest.approx(person[1], abs=0.01)


def test_parse_ignores_hands_faces_and_people_with_no_keypoints():
    empty = {"pose_keypoints_2d": [0, 0, 0] * NUM_KEYPOINTS}
    full = json.loads(_two_people().to_json())["people"][0]
    full["hand_left_keypoints_2d"] = [1, 2, 3]
    data = {"canvas_width": 1024, "canvas_height": 768, "people": [empty, full], "animals": []}

    pose = parse_pose_json(json.dumps(data))

    assert len(pose.people) == 1


def test_parse_clamps_far_off_canvas_keypoints_to_just_past_the_edge():
    values = [500, 400, 1] * NUM_KEYPOINTS
    values[0] = 1e9
    data = {"canvas_width": 1000, "canvas_height": 800, "people": [{"pose_keypoints_2d": values}]}

    assert parse_pose_json(json.dumps(data)).people[0][0] == (1500.0, 400.0)


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        "[]",
        {"canvas_width": 10, "canvas_height": 800, "people": []},
        {"canvas_width": 800, "canvas_height": True, "people": []},
        {"canvas_width": 800, "canvas_height": 800, "people": [{"pose_keypoints_2d": [1, 2, 1]}]},
        {
            "canvas_width": 800,
            "canvas_height": 800,
            "people": [{"pose_keypoints_2d": [float("nan")] * NUM_KEYPOINTS * 3}],
        },
        {
            "canvas_width": 800,
            "canvas_height": 800,
            "people": [{"pose_keypoints_2d": [1, 1, 1] * NUM_KEYPOINTS}] * (MAX_PEOPLE + 1),
        },
    ],
)
def test_parse_rejects_malformed_poses(data):
    text = data if isinstance(data, str) else json.dumps(data)
    with pytest.raises(PoseDataError):
        parse_pose_json(text)


def test_pose_data_error_is_reported_like_an_image_error():
    from comfytelegram.pose import PoseImageError

    assert issubclass(PoseDataError, PoseImageError)


def test_default_person_fits_the_canvas_and_faces_the_viewer():
    person = default_person(832, 1216)

    assert all(0 <= x <= 832 and 0 <= y <= 1216 for x, y in person)
    # Front-facing: the figure's right shoulder (2) is on the image's left.
    assert person[2][0] < person[1][0] < person[5][0]


def test_canvas_size_keeps_the_aspect_and_caps_the_long_side():
    assert canvas_size(832, 1216) == (832, 1216)
    width, height = canvas_size(4000, 3000)
    assert width == POSE_MAX_SIDE
    assert height == round(3000 * POSE_MAX_SIDE / 4000)


# --- rendering ----------------------------------------------------------------


def test_render_uses_the_same_limb_colours_for_every_person():
    """Colour says which limb, never whose: ControlNet's OpenPose models
    were trained on annotators that draw everyone with one palette."""
    pose = _two_people()
    png = render_pose(pose)

    # The twelve body limbs — the head's (neck-nose, eyes, ears) are shorter
    # than the joint circles drawn over them.
    for (i, j), color in list(zip(LIMBS, KEYPOINT_COLORS, strict=False))[:12]:
        dim = tuple(int(c * 0.6) for c in color)
        for person in pose.people:
            # 40% of the way along the limb: on it, clear of both joints.
            a, b = person[i], person[j]
            mid = (a[0] + (b[0] - a[0]) * 0.4, a[1] + (b[1] - a[1]) * 0.4)
            assert _pixel(png, mid) == dim, (i, j)


def test_render_draws_joints_in_their_own_colour_and_skips_missing_ones():
    person = list(default_person(1024, 1024))
    hidden_wrist = person[4]
    person[4] = None
    png = render_pose(Pose(1024, 1024, (tuple(person),)))

    assert _pixel(png, person[7]) == KEYPOINT_COLORS[7]
    assert _pixel(png, hidden_wrist) == (0, 0, 0)


def test_render_scales_a_big_canvas_down_and_passes_the_upload_check():
    png = render_pose(_two_people(3000, 2000))

    assert max(pose_size(png)) == POSE_MAX_SIDE
    assert pose_size(normalize_pose_image(png)) == pose_size(png)


def test_render_line_width_follows_the_canvas_like_a_512px_annotator():
    """At 1024px every line is twice as wide as `draw_bodypose`'s 4px at 512."""
    neck, hip = (512.0, 200.0), (512.0, 800.0)
    person = [None] * NUM_KEYPOINTS
    person[1], person[8] = neck, hip
    png = render_pose(Pose(1024, 1024, (tuple(person),)))
    image = Image.open(io.BytesIO(png)).convert("RGB")

    row = [image.getpixel((x, 500)) for x in range(490, 535)]
    lit = sum(1 for pixel in row if pixel != (0, 0, 0))
    assert 15 <= lit <= 18  # 2 x 8px half-width, give or take a pixel of rasterizing


# --- detection decoding ---------------------------------------------------------


def _anchor_index(x: int, y: int, stride: int) -> int:
    """Index of YOLOX's anchor at grid cell (x, y) of the given stride."""
    offset = 0
    for s in (8, 16, 32):
        cells = 640 // s
        if s == stride:
            return offset + y * cells + x
        offset += cells * cells
    raise ValueError(stride)


def test_decode_detections_finds_a_confident_person_box():
    anchors = sum((640 // s) ** 2 for s in (8, 16, 32))
    output = np.zeros((1, anchors, 85), dtype=np.float32)
    idx = _anchor_index(10, 5, 32)
    # Centre at cell (10, 5) + 0.5, size exp(log(4)) * 32 = 128 x exp(log(8)) * 32 = 256.
    output[0, idx, :4] = [0.5, 0.5, math.log(4), math.log(8)]
    output[0, idx, 4] = 0.9  # objectness
    output[0, idx, 5] = 0.9  # P(person)
    # A weaker, overlapping detection of the same person, which NMS drops.
    neighbour = _anchor_index(21, 11, 16)
    output[0, neighbour, :4] = [0.0, 0.0, math.log(8), math.log(16)]
    output[0, neighbour, 4] = 0.8
    output[0, neighbour, 5] = 0.8

    boxes = pose_detect.decode_detections(output, ratio=0.5)

    assert boxes.shape == (1, 4)
    # (336, 176) centre, 128x256, scaled back by the 0.5 letterbox ratio.
    np.testing.assert_allclose(boxes[0], [544, 96, 800, 608], atol=0.01)


def test_decode_detections_ignores_unlikely_people():
    anchors = sum((640 // s) ** 2 for s in (8, 16, 32))
    output = np.zeros((1, anchors, 85), dtype=np.float32)
    output[0, 0, 4] = 0.5
    output[0, 0, 5] = 0.5  # 0.25 < the 0.3 person threshold

    assert pose_detect.decode_detections(output, ratio=1.0).shape == (0, 4)


def test_decode_keypoints_maps_simcc_peaks_back_into_the_box():
    k = 133
    simcc_x = np.zeros((1, k, 288 * 2), dtype=np.float32)
    simcc_y = np.zeros((1, k, 384 * 2), dtype=np.float32)
    simcc_x[0, :, 288] = 0.9  # input-space x = 144, the crop's centre
    simcc_y[0, :, 192] = 0.7  # input-space y = 96, a quarter down
    center = np.array([500.0, 400.0])
    scale = np.array([300.0, 400.0])

    keypoints, scores = pose_detect.decode_keypoints(simcc_x, simcc_y, center, scale)

    np.testing.assert_allclose(keypoints[0], [500.0, 300.0])
    assert scores[0] == pytest.approx(0.7)


def test_to_openpose_reorders_coco_and_adds_the_neck():
    keypoints = np.array([[float(i), float(i) * 10] for i in range(17)])
    scores = np.full(17, 0.9)
    scores[9] = 0.1  # COCO left wrist: too unsure to keep

    person = pose_detect.to_openpose(keypoints, scores, factor=2.0)

    assert len(person) == NUM_KEYPOINTS
    assert person[0] == (0.0, 0.0)  # nose
    assert person[1] == ((5 + 6), (50 + 60))  # neck: shoulder midpoint, x2
    assert person[2] == (12.0, 120.0)  # right shoulder = COCO 6
    assert person[7] is None  # left wrist = COCO 9, dropped
    assert person[17] == (6.0, 60.0)  # left ear = COCO 3


# --- Telegram flows ---------------------------------------------------------------


def _photo_bytes() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (640, 480), (180, 150, 120)).save(out, format="JPEG")
    return out.getvalue()


def _context(tmp_path: Path, *, upload: bytes = b"", relay: bool = True, detection: bool = True):
    storage = Storage(tmp_path / "state.sqlite3")
    detector = tmp_path / "det.onnx"
    estimator = tmp_path / "est.onnx"
    if detection:
        detector.write_bytes(b"x")
        estimator.write_bytes(b"x")
    settings = MagicMock(
        allowed_user_ids=None,
        inpaint_relay_url="https://relay.example" if relay else None,
        inpaint_relay_shared_secret="secret",
        pose_detector_path=detector,
        pose_estimator_path=estimator,
    )
    context = MagicMock()
    context.chat_data = {"awaiting_pose": {NO_TOPIC: True}}
    context.bot_data = {
        "settings": settings,
        "storage": storage,
        "profiles": [
            ModelProfile(
                match=["*"],
                display_name="Pose Model",
                defaults=ProfileDefaults(width=1024, height=1024),
                pose_controlnet="openpose.safetensors",
            )
        ],
    }
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(upload)
    context.bot.get_file = AsyncMock(return_value=tg_file)
    context.bot.delete_message = AsyncMock()
    return context, storage


def _upload_update():
    message = AsyncMock()
    message.message_thread_id = None
    message.caption = None
    message.chat.id = 1
    status = AsyncMock()
    status.message_id = 77
    message.reply_text.return_value = status
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    update.effective_user.id = 1
    return update, message, status


@pytest.mark.asyncio
async def test_a_photo_upload_becomes_a_detected_editable_pose(tmp_path):
    update, message, _status = _upload_update()
    context, storage = _context(tmp_path, upload=_photo_bytes())
    detected = _two_people(640, 480)

    with patch("comfytelegram.handlers.detect_poses", return_value=detected) as detect:
        assert await _consume_pose_upload(update, context, "photo-file-id") is True

    detect.assert_called_once()
    pose_json, background = storage.get_pose_source(1)
    assert parse_pose_json(pose_json) == parse_pose_json(detected.to_json())
    assert background == "photo-file-id"
    assert pose_size(storage.get_pose(1)) == (640, 480)
    caption = message.reply_photo.await_args.kwargs["caption"]
    assert "2 people" in caption
    buttons = message.reply_photo.await_args.kwargs["reply_markup"].inline_keyboard[0]
    assert buttons[0].callback_data == POSE_EDIT_CALLBACK_DATA
    assert buttons[0].text == "✏️ Edit pose"


@pytest.mark.asyncio
async def test_without_detection_a_photo_upload_is_still_refused(tmp_path):
    update, message, _status = _upload_update()
    context, storage = _context(tmp_path, upload=_photo_bytes(), detection=False)

    with patch("comfytelegram.handlers.detect_poses") as detect:
        assert await _consume_pose_upload(update, context, "photo-file-id") is True

    detect.assert_not_called()
    assert storage.get_pose(1) is None
    assert message.reply_text.await_args.args[0].startswith("⚠️")


@pytest.mark.asyncio
async def test_nobody_found_opens_the_editor_over_the_photo(tmp_path):
    update, _message, status = _upload_update()
    photo = _photo_bytes()
    context, storage = _context(tmp_path, upload=photo)

    with (
        patch("comfytelegram.handlers.detect_poses", return_value=Pose(640, 480, ())),
        patch(
            "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok")
        ) as create,
    ):
        await _consume_pose_upload(update, context, "photo-file-id")

    assert storage.get_pose(1) is None  # nothing set until the editor's Done
    image, meta = create.await_args.args[1], create.await_args.kwargs["meta"]
    assert image == photo
    assert meta["mode"] == POSE_JOB_KIND
    assert meta["pose_has_background"] is True
    assert len(meta["pose"]["people"]) == 1
    (job,) = storage.list_inpaint_jobs()
    assert job["kind"] == POSE_JOB_KIND
    assert job["background_file_id"] == "photo-file-id"
    assert job["editor_message_id"] == 77
    button = status.edit_text.await_args.kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.web_app.url == "https://relay.example/jobs/tok"


def _edit_update():
    query = AsyncMock()
    query.data = POSE_EDIT_CALLBACK_DATA
    query.message.chat.id = 1
    query.message.message_thread_id = 9
    status = AsyncMock()
    status.message_id = 78
    query.message.reply_text.return_value = status
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = 1
    return update, query


@pytest.mark.asyncio
async def test_edit_button_reopens_the_stored_pose_over_its_photo(tmp_path):
    context, storage = _context(tmp_path, upload=b"background-bytes")
    pose = _two_people(640, 480)
    storage.set_pose(1, render_pose(pose), pose_json=pose.to_json(), background_file_id="bg")
    update, _query = _edit_update()

    with patch(
        "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok")
    ) as create:
        await pose_callback(update, context)

    assert create.await_args.args[1] == b"background-bytes"
    meta = create.await_args.kwargs["meta"]
    assert parse_pose_json(json.dumps(meta["pose"])) == parse_pose_json(pose.to_json())
    assert meta["pose_has_background"] is True
    (job,) = storage.list_inpaint_jobs()
    assert job["message_thread_id"] == 9
    assert job["background_file_id"] == "bg"


@pytest.mark.asyncio
async def test_edit_button_without_keypoints_traces_over_the_uploaded_skeleton(tmp_path):
    context, storage = _context(tmp_path)
    skeleton = render_pose(_two_people(512, 768))
    storage.set_pose(1, skeleton)  # uploaded as an image: no keypoints
    update, _query = _edit_update()

    with patch(
        "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok")
    ) as create:
        await pose_callback(update, context)

    assert create.await_args.args[1] == skeleton
    meta = create.await_args.kwargs["meta"]
    assert (meta["pose"]["canvas_width"], meta["pose"]["canvas_height"]) == (512, 768)
    assert len(meta["pose"]["people"]) == 1
    (job,) = storage.list_inpaint_jobs()
    assert job["background_file_id"] is None


@pytest.mark.asyncio
async def test_edit_button_with_no_pose_starts_a_blank_portrait(tmp_path):
    context, _storage = _context(tmp_path)
    update, _query = _edit_update()

    with patch(
        "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok")
    ) as create:
        await pose_callback(update, context)

    meta = create.await_args.kwargs["meta"]
    assert meta["pose_has_background"] is False
    assert Image.open(io.BytesIO(create.await_args.args[1])).size == (832, 1216)


def _poller(storage: Storage, pose_json: str | None):
    application = MagicMock()
    application.bot.send_message = AsyncMock()
    application.bot.send_photo = AsyncMock()
    application.bot.delete_message = AsyncMock()
    application.bot_data = {
        "settings": MagicMock(inpaint_relay_url="https://relay.example"),
        "profiles": [],
    }
    job = {
        "token": "tok",
        "result_id": "",
        "chat_id": 1,
        "message_thread_id": 4,
        "kind": POSE_JOB_KIND,
        "editor_message_id": None,
        "background_file_id": "bg",
    }
    result = {"mask": None, "pose": pose_json, "init_data": "init"}
    patches = (
        patch("comfytelegram.handlers._relay_poll_result", new=AsyncMock(return_value=result)),
        patch("comfytelegram.handlers._relay_delete_job", new=AsyncMock()),
        patch("comfytelegram.handlers.validate_webapp_init_data", return_value={"user": "{}"}),
    )
    return application, job, patches


@pytest.mark.asyncio
async def test_a_submitted_pose_becomes_the_chats_pose(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    pose = _two_people(1232, 821)
    application, job, patches = _poller(storage, pose.to_json())

    with patches[0], patches[1], patches[2]:
        await _process_one_inpaint_job(application, MagicMock(), storage, MagicMock(), job)

    pose_json, background = storage.get_pose_source(1)
    assert parse_pose_json(pose_json) == parse_pose_json(pose.to_json())
    assert background == "bg"
    assert pose_size(storage.get_pose(1)) == (1232, 821)
    sent = application.bot.send_photo.await_args
    assert sent.kwargs["message_thread_id"] == 4
    assert "2 people, 1232x821" in sent.kwargs["caption"]


@pytest.mark.asyncio
async def test_a_bad_or_empty_submitted_pose_keeps_the_old_one(tmp_path):
    storage = Storage(tmp_path / "state.sqlite3")
    storage.set_pose(1, b"old")
    for pose_json in ("{nope", Pose(800, 800, ()).to_json()):
        application, job, patches = _poller(storage, pose_json)
        with patches[0], patches[1], patches[2]:
            await _process_one_inpaint_job(application, MagicMock(), storage, MagicMock(), job)
        application.bot.send_photo.assert_not_awaited()
        assert application.bot.send_message.await_args.kwargs["message_thread_id"] == 4

    assert storage.get_pose(1) == b"old"


@pytest.mark.asyncio
async def test_relay_poll_result_passes_a_pose_through_without_a_mask():
    response = MagicMock(status=200)
    response.json = AsyncMock(
        return_value={
            "status": "submitted",
            "mask_base64": None,
            "pose": '{"canvas_width": 64}',
            "init_data": "init",
        }
    )
    response.raise_for_status = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    settings = MagicMock(inpaint_relay_url="https://relay.example")

    with patch("comfytelegram.handlers.aiohttp.ClientSession", return_value=session):
        result = await _relay_poll_result(settings, "tok")

    assert result["mask"] is None
    assert result["pose"] == '{"canvas_width": 64}'


def test_a_joint_two_people_claim_goes_to_the_more_confident_one():
    """A hug: the person in front's box covers the other, so both get the
    same shoulder and hand."""
    a_keys = np.zeros((17, 2))
    b_keys = np.zeros((17, 2))
    a_keys[6] = [1398, 3053]  # right shoulder, claimed by both
    b_keys[6] = [1421, 3084]
    a_keys[5] = [3275, 2773]  # left shoulder, only near-ish
    b_keys[5] = [1703, 3000]
    a_scores = np.full(17, 0.9)
    b_scores = np.full(17, 0.9)
    a_scores[6], b_scores[6] = 0.43, 0.51
    people = [(a_keys, a_scores), (b_keys, b_scores)]

    pose_detect.dedupe_shared_keypoints(people, diagonal=7476.0)

    assert a_scores[6] == 0.0 and b_scores[6] == 0.51
    assert a_scores[5] == 0.9 and b_scores[5] == 0.9
    # Every other joint is at (0, 0) for both: equally confident, so the
    # first keeps them — but only ever one of the two.
    assert all((a_scores[k] == 0) != (b_scores[k] == 0) for k in range(17) if k not in (5, 6))


def test_one_hidden_shoulder_still_gets_a_neck_under_the_head():
    keypoints = np.zeros((17, 2))
    keypoints[0] = [2060, 2392]  # nose
    keypoints[3] = [2251, 2346]  # left ear
    keypoints[6] = [1421, 3084]  # right shoulder
    scores = np.zeros(17)
    scores[[0, 3, 6]] = 0.6

    person = pose_detect.to_openpose(keypoints, scores, factor=0.5)

    assert person[1] == pytest.approx(((2060 + 2251) / 4, 3084 / 2))


def test_no_shoulders_means_no_neck():
    keypoints = np.zeros((17, 2))
    scores = np.zeros(17)
    scores[0] = 0.9

    assert pose_detect.to_openpose(keypoints, scores, factor=1.0)[1] is None
