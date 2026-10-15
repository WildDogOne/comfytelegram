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

from PIL import Image, UnidentifiedImageError

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
