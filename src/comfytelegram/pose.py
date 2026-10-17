"""OpenPose guidance: the Telegram-independent half of `/pose`.

A pose is a rendered OpenPose skeleton (coloured limbs on black) that
`build_txt2img` feeds through the profile's `pose_controlnet`
(`GenerationParams.pose_png`). Like a painted region mask it travels as the
bytes themselves — `params_serde`, `pending_result`, PNG metadata — so
"🔁 Generate Again" and a re-imported archive copy still have it, and it's
uploaded to ComfyUI's input folder under a content-hash name before every
submission (`generation._upload_graph_inputs`).

Verified live with `scripts/smoke_test_pose.py` on FurryToonMix (Illustrious)
+ `illustriousXL_v10_openpose`: the pose held at every strength tried
(0.6-1.0), strength 0.8 followed raised arms closely without the odd
artifacts 1.0 started to show, `end_percent` 0.6-1.0 made next to no
difference, and it combined with `/rp`'s `AttentionCouplePPM` with no extra
wiring — each region's character stayed on its own skeleton.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError

#: A pose image bigger than this on its longer side is scaled down on the
#: way in. ControlNet resizes the hint to the latent anyway, and the bytes
#: ride along in every `pending_result` row and PNG metadata chunk.
POSE_MAX_SIDE = 1536

#: Below this share of near-black pixels an upload probably isn't a rendered
#: skeleton at all (a photo of a person, say), which ControlNet would read
#: as noise. Rendered skeletons sit around 95%+.
_MIN_BLACK_SHARE = 0.6
_BLACK_LEVEL = 40

_SIZE_MULTIPLE = 64


class PoseImageError(ValueError):
    """An upload that can't be used as a pose — the message is shown to the
    user as-is."""


def normalize_pose_image(data: bytes) -> bytes:
    """Validate an uploaded pose image and return it as an RGB PNG, scaled
    down to `POSE_MAX_SIDE` if needed. Raises `PoseImageError` for something
    that isn't an image, or doesn't look like a skeleton on black."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise PoseImageError("That doesn't look like an image I can read.") from exc

    image = image.convert("RGB")
    if max(image.size) > POSE_MAX_SIDE:
        image.thumbnail((POSE_MAX_SIDE, POSE_MAX_SIDE), Image.Resampling.LANCZOS)

    gray = image.convert("L")
    histogram = gray.histogram()
    black_share = sum(histogram[:_BLACK_LEVEL]) / (image.width * image.height)
    if black_share < _MIN_BLACK_SHARE:
        raise PoseImageError(
            "That doesn't look like an OpenPose skeleton (coloured limbs on a black "
            "background). Send a rendered pose image — e.g. one exported from an "
            "OpenPose editor."
        )

    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def pose_size(pose_png: bytes) -> tuple[int, int]:
    """The pose image's pixel dimensions."""
    return Image.open(io.BytesIO(pose_png)).size


def size_for_pose(pose_png: bytes, width: int, height: int) -> tuple[int, int]:
    """A generation size with the pose's aspect ratio and about the pixel
    count of `width`x`height` (the profile's/`/settings`' size), snapped to
    64. ControlNet fits its hint to the latent's size, so a pose on a
    canvas of another shape comes out cropped or squashed."""
    pose_w, pose_h = pose_size(pose_png)
    scale = (width * height / (pose_w * pose_h)) ** 0.5
    return (
        max(_SIZE_MULTIPLE, round(pose_w * scale / _SIZE_MULTIPLE) * _SIZE_MULTIPLE),
        max(_SIZE_MULTIPLE, round(pose_h * scale / _SIZE_MULTIPLE) * _SIZE_MULTIPLE),
    )


# --- keypoints ----------------------------------------------------------------
#
# The pose editor (inpaint_relay's `static/pose.html`) and the detector
# (`pose_detect.py`) both work in keypoints, not pixels: the OpenPose
# "BODY_18" layout every SD pose ControlNet was trained on, in the JSON shape
# OpenPose itself, ControlNet's annotators and the common web editors all
# read and write (`{"canvas_width", "canvas_height", "people":
# [{"pose_keypoints_2d": [x, y, confidence] * 18}]}`, pixel coordinates). A
# keypoint is either placed or missing — OpenPose's confidence 0 — which is
# how a skeleton says "this arm is hidden".
#
# Colours identify the *limb*, never the person: `render_pose` draws every
# person with the same 17 limb / 18 joint colours, exactly like
# comfyui_controlnet_aux's `draw_bodypose` (the renderer the ControlNets'
# training data came from), so red is always neck-to-right-shoulder and
# blue always the left leg. Giving each person their own colours would put
# limbs in colours the model reads as other body parts. People are told
# apart by where they are; who is who is the prompt's job (`/rp` regions).

#: Names in OpenPose BODY_18 order (`Pose.people[i][k]` is `KEYPOINT_NAMES[k]`).
KEYPOINT_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

#: Limbs as 0-based keypoint index pairs, in `draw_bodypose`'s `limbSeq`
#: order (there 1-based) — limb i is drawn in `KEYPOINT_COLORS[i]`.
LIMBS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

#: `draw_bodypose`'s `colors`: joint k is drawn in colour k, limb i in
#: colour i at 60% brightness.
KEYPOINT_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)

#: More than this many people in one pose is rejected. ControlNet has no
#: limit, but nobody poses a crowd by hand and a cap keeps a hostile
#: submission small.
MAX_PEOPLE = 12

#: Canvas dimensions a submitted pose may claim, in pixels.
_MIN_CANVAS_SIDE = 64
_MAX_CANVAS_SIDE = 8192

#: `draw_bodypose`'s stick half-width and joint radius, at the 512px
#: detect resolution its output is usually rendered at before ComfyUI
#: scales the hint up to the latent — so `render_pose` scales both by the
#: canvas's shorter side over this, matching what an annotator's skeleton
#: looks like at full size.
_REFERENCE_SIDE = 512
_STICK_WIDTH = 4
_JOINT_RADIUS = 4

Keypoint = tuple[float, float] | None


class PoseDataError(PoseImageError):
    """A pose JSON (from the editor) that can't be used — subclasses
    `PoseImageError` so every caller reporting one reports both."""


@dataclass(frozen=True)
class Pose:
    """Body keypoints for everyone in a pose, in pixel coordinates on a
    `width`x`height` canvas. Each person is `NUM_KEYPOINTS` long, a missing
    keypoint `None`."""

    width: int
    height: int
    people: tuple[tuple[Keypoint, ...], ...]

    def to_json(self) -> str:
        """The OpenPose JSON form — what the editor reads and storage keeps."""
        return json.dumps(
            {
                "canvas_width": self.width,
                "canvas_height": self.height,
                "people": [
                    {
                        "pose_keypoints_2d": [
                            value
                            for point in person
                            for value in (
                                (round(point[0], 2), round(point[1], 2), 1.0)
                                if point is not None
                                else (0.0, 0.0, 0.0)
                            )
                        ]
                    }
                    for person in self.people
                ],
            },
            separators=(",", ":"),
        )


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseDataError("A keypoint coordinate isn't a number.")
    value = float(value)
    if not math.isfinite(value):
        raise PoseDataError("A keypoint coordinate isn't a number.")
    return value


def parse_pose_json(text: str | bytes) -> Pose:
    """Parse and validate an OpenPose JSON pose (see `Pose.to_json`).
    Hand/face/animal keys are ignored — only the body is used. Keypoints
    far off the canvas are clamped to just past its edge, so a dragged-off
    joint still renders as an arm leaving the frame. Raises
    `PoseDataError` for anything malformed."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise PoseDataError("The pose data isn't valid JSON.") from exc
    if not isinstance(data, dict):
        raise PoseDataError("The pose data isn't an OpenPose JSON object.")
    width, height = data.get("canvas_width"), data.get("canvas_height")
    if not all(
        isinstance(side, int)
        and not isinstance(side, bool)
        and _MIN_CANVAS_SIDE <= side <= _MAX_CANVAS_SIDE
        for side in (width, height)
    ):
        raise PoseDataError("The pose's canvas size is missing or out of range.")
    raw_people = data.get("people") or []
    if not isinstance(raw_people, list):
        raise PoseDataError("The pose's people list is malformed.")
    if len(raw_people) > MAX_PEOPLE:
        raise PoseDataError(f"A pose can have at most {MAX_PEOPLE} people.")

    people = []
    for raw in raw_people:
        values = raw.get("pose_keypoints_2d") if isinstance(raw, dict) else None
        if not isinstance(values, list) or len(values) != NUM_KEYPOINTS * 3:
            raise PoseDataError(f"Each person needs exactly {NUM_KEYPOINTS} body keypoints.")
        person: list[Keypoint] = []
        for k in range(NUM_KEYPOINTS):
            x, y, confidence = (_finite(v) for v in values[k * 3 : k * 3 + 3])
            if confidence <= 0:
                person.append(None)
                continue
            person.append(
                (
                    min(max(x, -0.5 * width), 1.5 * width),
                    min(max(y, -0.5 * height), 1.5 * height),
                )
            )
        if any(point is not None for point in person):
            people.append(tuple(person))
    return Pose(width, height, tuple(people))


def canvas_size(width: int, height: int) -> tuple[int, int]:
    """A pose canvas for an image of `width`x`height`: the same aspect,
    scaled down to `POSE_MAX_SIDE` if bigger — the size `render_pose` will
    draw at anyway, so keypoints never live in a bigger space than the hint
    they end up in."""
    scale = min(1.0, POSE_MAX_SIDE / max(width, height))
    return (
        max(_MIN_CANVAS_SIDE, round(width * scale)),
        max(_MIN_CANVAS_SIDE, round(height * scale)),
    )


def default_person(width: int, height: int, *, center_x: float = 0.5) -> tuple[Keypoint, ...]:
    """A standing, front-facing figure filling most of a `width`x`height`
    canvas's height, centred at `center_x` (a fraction of the width) — what
    a blank editor starts from. Same proportions as the editor's own "add
    person" figure (`pose.html`'s `DEFAULT_PERSON`)."""
    # (x offset from the centre line, y) as fractions of the figure height,
    # in BODY_18 order; right is the figure's right, i.e. image left.
    shape = (
        (0.0, 0.10),
        (0.0, 0.20),
        (-0.09, 0.20),
        (-0.12, 0.35),
        (-0.13, 0.49),
        (0.09, 0.20),
        (0.12, 0.35),
        (0.13, 0.49),
        (-0.06, 0.50),
        (-0.06, 0.70),
        (-0.06, 0.90),
        (0.06, 0.50),
        (0.06, 0.70),
        (0.06, 0.90),
        (-0.02, 0.08),
        (0.02, 0.08),
        (-0.04, 0.09),
        (0.04, 0.09),
    )
    figure = height * 0.9
    top = (height - figure) / 2
    return tuple((width * center_x + dx * figure, top + dy * figure) for dx, dy in shape)


def _limb_polygon(
    a: tuple[float, float], b: tuple[float, float], half_width: float
) -> list[tuple[float, float]]:
    """The filled ellipse `draw_bodypose` draws for a limb (`cv2.ellipse2Poly`
    with one point per degree): centred between the two joints, as long as
    they are apart, `half_width` across."""
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    half_length = math.hypot(a[0] - b[0], a[1] - b[1]) / 2
    angle = math.atan2(a[1] - b[1], a[0] - b[0])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for degree in range(0, 360, 4):
        t = math.radians(degree)
        ex, ey = half_length * math.cos(t), half_width * math.sin(t)
        points.append((cx + ex * cos_a - ey * sin_a, cy + ex * sin_a + ey * cos_a))
    return points


def render_pose(pose: Pose) -> bytes:
    """Draw `pose` as a ControlNet OpenPose hint: coloured limbs and joints
    on black, as an RGB PNG of the canvas's aspect, scaled down to
    `POSE_MAX_SIDE` if the canvas is bigger. Drawn the way
    comfyui_controlnet_aux's `draw_bodypose` draws (same colours, 60%
    limbs, joints on top; see `_REFERENCE_SIDE` for the line widths)."""
    scale = min(1.0, POSE_MAX_SIDE / max(pose.width, pose.height))
    width = max(1, round(pose.width * scale))
    height = max(1, round(pose.height * scale))
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    line_scale = min(width, height) / _REFERENCE_SIDE
    half_width = max(1.0, _STICK_WIDTH * line_scale)
    radius = max(1.0, _JOINT_RADIUS * line_scale)

    people = [
        [(p[0] * scale, p[1] * scale) if p is not None else None for p in person]
        for person in pose.people
    ]
    # Limb by limb across everyone, then joints — the reference's order,
    # which decides what overlaps what where people touch.
    for (i, j), color in zip(LIMBS, KEYPOINT_COLORS, strict=False):
        dim = tuple(int(c * 0.6) for c in color)
        for person in people:
            if person[i] is not None and person[j] is not None:
                draw.polygon(_limb_polygon(person[i], person[j], half_width), fill=dim)
    for person in people:
        for point, color in zip(person, KEYPOINT_COLORS, strict=True):
            if point is not None:
                x, y = point
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
