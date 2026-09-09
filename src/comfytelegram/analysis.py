"""Image-to-prompt analysis: backs the "🔬 Analyze & Regenerate" button.

Two independent analyzers, chosen per-checkpoint by a model profile's
`prompt_style` field (see profiles/schema.py):

- "tags": WD14 tagger (ONNX, run locally via onnxruntime) — for
  booru/danbooru-tag-trained checkpoints like Pony/Illustrious/Animagine
  merges, whose training captions are comma-separated tags, not prose.
- "natural": Qwen-VL, called through a local Ollama server — for
  checkpoints (e.g. stock SDXL) that expect natural-language prompts.

Both take raw image bytes and return a single string meant to be fed
straight into `generate()` as the new `user_prompt` — the checkpoint's
profile still supplies its own quality-tag prefix/negative/LoRAs on top,
exactly as it would for a normal typed prompt.

The WD14 model files (model.onnx + selected_tags.csv) are *not* downloaded
automatically. Hugging Face serves large model files from an LFS/Xet CDN on
a different hostname than huggingface.co itself, which a restricted-egress
environment may allow while blocking the CDN — that's exactly what
happened developing this against the bot's own sandbox. Fetch them once
from a machine with normal internet access and place them at
`Settings.wd14_model_dir` (see env.example's WD14_* vars).
"""

from __future__ import annotations

import asyncio
import base64
import csv
import functools
import io
import logging
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np
import onnxruntime as ort
from PIL import Image

from comfytelegram.settings import Settings

logger = logging.getLogger(__name__)

# WD14-family tag categories (selected_tags.csv's "category" column) — see
# https://huggingface.co/SmilingWolf/wd-vit-tagger-v3. Category 9 ("rating",
# e.g. general/sensitive/questionable/explicit) describes content rating
# rather than image content, so it's never surfaced into a prompt.
_CATEGORY_GENERAL = 0
_CATEGORY_CHARACTER = 4

# Tags that are themselves kaomoji — the underscore is part of the tag, not
# a multi-word separator, so don't turn it into a space like every other tag.
_KAOMOJI = frozenset(
    {
        "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
        ">_<", "3_3", "6_9", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
    }
)

_OLLAMA_VISION_PROMPT = (
    "Describe this image as a concise, comma-separated Stable Diffusion prompt: "
    "subject, appearance, pose, clothing, setting, lighting, art style. Output "
    "only the prompt text, no commentary or preamble."
)


def _format_tag(name: str) -> str:
    return name if name in _KAOMOJI else name.replace("_", " ")


def select_tags(tags: list[tuple[str, int, float]], threshold: float) -> str:
    """Pick WD14 tags at or above `threshold`, formatted as a comma-separated
    prompt fragment: character tags (category 4) first, then general tags
    (category 0), each sorted by descending confidence. Pure function (no
    model/session involved) so it's unit-testable without the ~370MB ONNX
    file — see `_run_wd14` for the part that actually needs it."""
    general = sorted(
        (t for t in tags if t[1] == _CATEGORY_GENERAL and t[2] >= threshold), key=lambda t: -t[2]
    )
    character = sorted(
        (t for t in tags if t[1] == _CATEGORY_CHARACTER and t[2] >= threshold), key=lambda t: -t[2]
    )
    return ", ".join(_format_tag(name) for name, _category, _score in character + general)


class _WD14Model:
    """Bundles the loaded ONNX session with its tag-label list and resolved
    input shape, so callers don't have to re-derive those on every request."""

    def __init__(self, session: ort.InferenceSession, tags: list[tuple[str, int]]):
        self.session = session
        self.tags = tags
        self.input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        # A dynamic axis comes back as a string (e.g. "batch"/"height")
        # rather than an int — fall back to WD14's usual 448x448 input then.
        self.target_size = shape[1] if isinstance(shape[1], int) else 448


@functools.lru_cache(maxsize=1)
def _load_wd14_model(model_path: str, tags_path: str) -> _WD14Model:
    """Loading is a real cost (reading a few-hundred-MB file into an ONNX
    session), so `lru_cache` makes it happen once per process and get reused
    across every "Analyze & Regenerate" tap rather than per-request."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    with open(tags_path, newline="", encoding="utf-8") as f:
        tags = [(row["name"], int(row["category"])) for row in csv.DictReader(f)]
    return _WD14Model(session, tags)


def _require_wd14_files(settings: Settings) -> tuple[Path, Path]:
    model_path = settings.wd14_model_dir / "model.onnx"
    tags_path = settings.wd14_model_dir / "selected_tags.csv"
    if not model_path.exists() or not tags_path.exists():
        raise RuntimeError(
            f"WD14 tagger files not found in {settings.wd14_model_dir}. Download "
            f"model.onnx and selected_tags.csv from "
            f"https://huggingface.co/{settings.wd14_model_repo}/tree/main and place "
            f"them there — see analysis.py's module docstring for why this isn't "
            f"done automatically."
        )
    return model_path, tags_path


def _prepare_wd14_image(image_bytes: bytes, target_size: int) -> np.ndarray:
    """Resize-with-padding to a square, matching WD14's training
    preprocessing: an RGBA source is flattened onto a white background
    first, then letterboxed onto a white square (never cropped, so nothing
    outside the original frame gets inferred from the padding), resized,
    and channel-flipped to the BGR order the model was trained on."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    flattened = Image.new("RGBA", image.size, (255, 255, 255))
    flattened.alpha_composite(image)
    flattened = flattened.convert("RGB")

    max_dim = max(flattened.size)
    square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    square.paste(flattened, ((max_dim - flattened.width) // 2, (max_dim - flattened.height) // 2))
    resized = square.resize((target_size, target_size), Image.BICUBIC)

    array = np.asarray(resized, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(array, axis=0)


def _run_wd14(image_bytes: bytes, settings: Settings) -> str:
    """Synchronous WD14 inference — run off the event loop via
    `asyncio.to_thread` by `analyze_tags` below, since onnxruntime has no
    async API and a 448x448 CPU forward pass isn't free."""
    model_path, tags_path = _require_wd14_files(settings)
    model = _load_wd14_model(str(model_path), str(tags_path))

    image_array = _prepare_wd14_image(image_bytes, model.target_size)
    # wd-vit-tagger-v3's exported graph already ends in a sigmoid — this is
    # already [0, 1] per-tag probabilities, not raw logits. Verified against
    # a real inference run: applying a second sigmoid on top (an easy
    # mistake, since older WD14 exports *did* need one) squashes every
    # value toward ~0.5-0.7, pushing the entire tag vocabulary over any
    # reasonable threshold.
    probs = model.session.run(None, {model.input_name: image_array})[0][0]

    scored = [(name, category, float(prob)) for (name, category), prob in zip(model.tags, probs)]
    return select_tags(scored, settings.wd14_tag_threshold)


async def analyze_tags(image_bytes: bytes, settings: Settings) -> str:
    """WD14-tag the image and return the derived prompt fragment. Raises
    `RuntimeError` if the model files aren't staged (see
    `_require_wd14_files`), or whatever onnxruntime/PIL raise for a
    corrupt/unreadable image."""
    return await asyncio.to_thread(_run_wd14, image_bytes, settings)


async def analyze_caption(image_bytes: bytes, settings: Settings) -> str:
    """Caption the image via a local Ollama server running a vision-capable
    model (Qwen-VL by default — see Settings.ollama_vision_model), for
    checkpoints that expect natural-language prompts rather than tags."""
    payload = {
        "model": settings.ollama_vision_model,
        "messages": [
            {
                "role": "user",
                "content": _OLLAMA_VISION_PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "stream": False,
        # Ollama otherwise keeps the model resident in VRAM for a few
        # minutes after answering (its default keep-alive), stacked on top
        # of whatever checkpoint ComfyUI is already keeping loaded for
        # this same request's upcoming generate() call — request an
        # immediate unload instead, since that VRAM is needed right after
        # this call returns, not idle-cached for a follow-up caption.
        "keep_alive": 0,
    }
    # A vision-capable model can take well over a minute on a cold load (a
    # 27B model observed taking ~70s the first time, faster once Ollama
    # keeps it warm) — generous rather than tight, but still bounded so a
    # stuck Ollama server can't hang the handler forever.
    timeout = aiohttp.ClientTimeout(total=300)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(f"{settings.ollama_http_base}/api/chat", json=payload) as resp,
    ):
        resp.raise_for_status()
        data = await resp.json()
    return data["message"]["content"].strip()


async def analyze_image(image_bytes: bytes, style: Literal["tags", "natural"], settings: Settings) -> str:
    """Dispatch to the analyzer matching a checkpoint's `prompt_style` (see
    `ModelProfile.prompt_style` in profiles/schema.py)."""
    if style == "tags":
        return await analyze_tags(image_bytes, settings)
    return await analyze_caption(image_bytes, settings)
