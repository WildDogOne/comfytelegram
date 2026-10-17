"""Pose detection: body keypoints for everyone in an image, as a `pose.Pose`.

DWPose, the detector ControlNet's own OpenPose annotators default to: a
YOLOX person detector finds everyone, then the DWPose (RTMPose) top-down
estimator places keypoints in each person's box. Ported from
comfyui_controlnet_aux's `dwpose/dw_onnx` (onnxruntime, CPU) with Pillow in
place of OpenCV — same preprocessing, thresholds and COCO -> OpenPose
keypoint reordering, so a pose found here matches what its DWPose node
finds. The estimator is the whole-body one (133 keypoints); only the 17
body ones are used, plus OpenPose's neck, which COCO doesn't have (see
`_neck`). Two deliberate departures from the reference, both for people
overlapping: a joint two people both claim goes to the more confident one
(`dedupe_shared_keypoints`), and a neck is still placed when one shoulder
is hidden.

Model files are staged by hand (`Settings.pose_detector_path`/
`pose_estimator_path`, both from https://huggingface.co/yzd-v/DWPose), same
as the WD14 tagger and MobileSAM; detection is simply off without them.
"""

from __future__ import annotations

import functools
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from comfytelegram.pose import NUM_KEYPOINTS, Keypoint, Pose, canvas_size

_DETECTOR_INPUT = (640, 640)  # (height, width)
_ESTIMATOR_INPUT = (288, 384)  # (width, height)
#: YOLOX's letterbox padding value.
_PAD_VALUE = 114
_NMS_THRESHOLD = 0.45
_CANDIDATE_THRESHOLD = 0.1
#: A detection below this confidence isn't a person.
_PERSON_THRESHOLD = 0.3
#: A keypoint below this confidence is left out (OpenPose's "not visible").
_KEYPOINT_THRESHOLD = 0.3
#: The whole-image fallback (no person box found) only counts as a person
#: with at least this many confident keypoints — the estimator always
#: answers something, and on a frame with nobody in it that's a scatter of
#: low-confidence guesses, a few of which clear `_KEYPOINT_THRESHOLD`.
_FALLBACK_MIN_KEYPOINTS = 6
_BOX_PADDING = 1.25
_SIMCC_SPLIT_RATIO = 2.0
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

#: COCO body keypoint index for each OpenPose BODY_18 one; `None` is the
#: neck (shoulder midpoint).
_COCO_FOR_OPENPOSE = (0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)
_COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER = 5, 6
#: Nose, eyes and ears — where `to_openpose` takes the head's centre from.
_COCO_HEAD = (0, 1, 2, 3, 4)
#: Two people's same joint closer than this (a share of the image's
#: diagonal) is one joint claimed twice — see `dedupe_shared_keypoints`.
_SHARED_KEYPOINT_DISTANCE = 0.02


@functools.lru_cache(maxsize=2)
def _session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression (controlnet_aux's `nms`)."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def decode_detections(output: np.ndarray, ratio: float) -> np.ndarray:
    """Person boxes (`(n, 4)` xyxy, in the original image's pixels) from
    YOLOX's raw `(1, anchors, 85)` output. COCO class 0 is person; the
    reference's NMS is per class, so the other 79 never affect it."""
    height, width = _DETECTOR_INPUT
    grids, strides = [], []
    for stride in (8, 16, 32):
        xv, yv = np.meshgrid(np.arange(width // stride), np.arange(height // stride))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))
    grid = np.concatenate(grids)
    stride = np.concatenate(strides)

    predictions = output[0].astype(np.float32)
    centers = (predictions[:, :2] + grid) * stride
    sizes = np.exp(predictions[:, 2:4]) * stride
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1) / ratio
    scores = predictions[:, 4] * predictions[:, 5]

    candidates = scores > _CANDIDATE_THRESHOLD
    boxes, scores = boxes[candidates], scores[candidates]
    kept = _nms(boxes, scores, _NMS_THRESHOLD) if len(scores) else []
    boxes, scores = boxes[kept], scores[kept]
    return boxes[scores > _PERSON_THRESHOLD].astype(np.float32).reshape(-1, 4)


def _detector_input(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """YOLOX's letterboxed input: resized to fit 640x640, top-left aligned,
    padded with 114, raw 0-255 BGR, NCHW."""
    height, width = _DETECTOR_INPUT
    ratio = min(height / bgr.shape[0], width / bgr.shape[1])
    new_w, new_h = int(bgr.shape[1] * ratio), int(bgr.shape[0] * ratio)
    resized = np.asarray(
        Image.fromarray(bgr).resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.uint8
    )
    padded = np.full((height, width, 3), _PAD_VALUE, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded.transpose(2, 0, 1)[None].astype(np.float32), ratio


def _box_crop(bgr: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The estimator's input for one person: their box padded by 25%, grown
    to the input's aspect, scaled to fit it (`top_down_affine` — a pure
    scale + shift, so Pillow's AFFINE transform does what `cv2.warpAffine`
    does there), normalized. Returns `(input HWC, center, scale)`, the last
    two to map the keypoints back."""
    out_w, out_h = _ESTIMATOR_INPUT
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * _BOX_PADDING
    aspect = out_w / out_h
    if scale[0] > scale[1] * aspect:
        scale = np.array([scale[0], scale[0] / aspect], dtype=np.float32)
    else:
        scale = np.array([scale[1] * aspect, scale[1]], dtype=np.float32)

    s = out_w / scale[0]
    # OpenCV maps pixel indices (pixel i is at i); Pillow's transform works
    # in continuous coordinates (pixel i spans [i, i+1]) — hence the 0.5s.
    offset_x = center[0] + 0.5 - (out_w / 2 + 0.5) / s
    offset_y = center[1] + 0.5 - (out_h / 2 + 0.5) / s
    crop = Image.fromarray(bgr).transform(
        (out_w, out_h),
        Image.Transform.AFFINE,
        (1 / s, 0, offset_x, 0, 1 / s, offset_y),
        resample=Image.Resampling.BILINEAR,
    )
    normalized = (np.asarray(crop, dtype=np.float32) - _MEAN) / _STD
    return normalized, center, scale


def decode_keypoints(
    simcc_x: np.ndarray, simcc_y: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One person's keypoints `(K, 2)` in image pixels and their scores
    `(K,)` from the estimator's SimCC outputs (`(1, K, W*2)`/`(1, K, H*2)`):
    each axis' argmax, scored by the weaker of the two maxima."""
    x_locs = np.argmax(simcc_x[0], axis=1)
    y_locs = np.argmax(simcc_y[0], axis=1)
    scores = np.minimum(np.amax(simcc_x[0], axis=1), np.amax(simcc_y[0], axis=1))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32) / _SIMCC_SPLIT_RATIO
    keypoints = locs / np.array(_ESTIMATOR_INPUT, dtype=np.float32) * scale + center - scale / 2
    return keypoints, scores


def dedupe_shared_keypoints(people: list[tuple[np.ndarray, np.ndarray]], diagonal: float) -> None:
    """Where two people's same joint landed in (almost) the same spot, keep
    it only for the more confident one, zeroing the other's score in place.

    The estimator places every joint inside each person's box, and
    overlapping people — a hug — have overlapping boxes, so the person in
    front easily claims the other's shoulder or the hand resting on them:
    on a real hug photo both got the same right shoulder (39px apart at
    4033px wide) and the same hand. The owner is reliably the one scoring
    it higher. Keypoint-level, not whole-pose, NMS: the poses as a whole
    are genuinely different people."""
    limit = _SHARED_KEYPOINT_DISTANCE * diagonal
    for a in range(len(people)):
        for b in range(a + 1, len(people)):
            (ka, sa), (kb, sb) = people[a], people[b]
            for k in range(min(len(sa), len(sb), 17)):
                if sa[k] <= _KEYPOINT_THRESHOLD or sb[k] <= _KEYPOINT_THRESHOLD:
                    continue
                if np.hypot(*(ka[k] - kb[k])) < limit:
                    if sa[k] >= sb[k]:
                        sb[k] = 0.0
                    else:
                        sa[k] = 0.0


def _neck(keypoints: np.ndarray, scores: np.ndarray) -> tuple[float, float] | None:
    """OpenPose's neck, which COCO doesn't have: the shoulders' midpoint,
    as in the reference. With only one shoulder (the other hidden — behind
    the person's own head, in a hug) the reference leaves it out, which
    cuts the head off from the body entirely, so instead: under the head's
    centre, at the visible shoulder's height."""
    left, right = _COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER
    has_left, has_right = scores[left] > _KEYPOINT_THRESHOLD, scores[right] > _KEYPOINT_THRESHOLD
    if has_left and has_right:
        x, y = (keypoints[left] + keypoints[right]) / 2
        return float(x), float(y)
    head = [k for k in _COCO_HEAD if scores[k] > _KEYPOINT_THRESHOLD]
    if (has_left or has_right) and head:
        shoulder = keypoints[left if has_left else right]
        return float(np.mean([keypoints[k][0] for k in head])), float(shoulder[1])
    return None


def to_openpose(keypoints: np.ndarray, scores: np.ndarray, factor: float) -> tuple[Keypoint, ...]:
    """One person's COCO keypoints as OpenPose BODY_18, scaled by `factor`
    (image pixels -> pose canvas), low-confidence ones left out."""
    person: list[Keypoint] = []
    for coco in _COCO_FOR_OPENPOSE:
        if coco is None:
            neck = _neck(keypoints, scores)
            person.append((neck[0] * factor, neck[1] * factor) if neck else None)
        elif scores[coco] > _KEYPOINT_THRESHOLD:
            x, y = keypoints[coco]
            person.append((float(x) * factor, float(y) * factor))
        else:
            person.append(None)
    assert len(person) == NUM_KEYPOINTS
    return tuple(person)


def detect_poses(image_bytes: bytes, detector_path: Path, estimator_path: Path) -> Pose:
    """Everyone's body pose in `image_bytes`, on a canvas of the image's
    aspect (`pose.canvas_size`). No people found is an empty pose, not an
    error. Blocking (a second or two on a desktop CPU, mostly the
    detector) — call through `asyncio.to_thread`."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB")
    canvas_w, canvas_h = canvas_size(*rgb.size)
    factor = canvas_w / rgb.width
    # DWPose's models expect BGR (comfyui_controlnet_aux flips ComfyUI's RGB
    # before calling them); the same mean/std is then applied as-is.
    bgr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[:, :, ::-1])

    detector = _session(str(detector_path))
    det_input, ratio = _detector_input(bgr)
    (raw,) = detector.run(None, {detector.get_inputs()[0].name: det_input})[:1]
    boxes = decode_detections(raw, ratio)
    fallback = len(boxes) == 0
    if fallback:
        # YOLOX misses most close-ups and many stylized/anthro characters
        # outright (person scores around 0.01 on our own furry images —
        # the reference node gives up there). The estimator still often
        # places a confident upper body on the whole frame, so try that as
        # one person before reporting nobody.
        boxes = np.array([[0, 0, rgb.width, rgb.height]], dtype=np.float32)

    estimator = _session(str(estimator_path))
    crops = [_box_crop(bgr, box) for box in boxes]
    batch = np.stack([crop for crop, _, _ in crops]).transpose(0, 3, 1, 2)
    simcc_x, simcc_y = estimator.run(None, {estimator.get_inputs()[0].name: batch})[:2]

    decoded = [
        decode_keypoints(simcc_x[i : i + 1], simcc_y[i : i + 1], center, scale)
        for i, (_, center, scale) in enumerate(crops)
    ]
    dedupe_shared_keypoints(decoded, float(np.hypot(rgb.width, rgb.height)))

    people = []
    for keypoints, scores in decoded:
        person = to_openpose(keypoints, scores, factor)
        found = sum(point is not None for point in person)
        if found >= (_FALLBACK_MIN_KEYPOINTS if fallback else 1):
            people.append(person)
    return Pose(canvas_w, canvas_h, tuple(people))
