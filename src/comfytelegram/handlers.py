"""Telegram-facing handlers: /start, /help, /model, plain-text generation
requests, and the inline-keyboard callbacks for model selection and
post-processing.

Shared objects (settings, the ComfyUI client, loaded profiles, storage)
live in `context.bot_data`, populated once at startup in `main.py`.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import uuid
from collections.abc import Awaitable
from typing import Any, TypeVar

from PIL import Image, ImageDraw, ImageFont
from telegram import (
    CopyTextButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import InlineKeyboardButtonLimit
from telegram.ext import ContextTypes

from comfytelegram.analysis import (
    analyze_caption,
    analyze_caption_deep,
    analyze_tags,
)
from comfytelegram.auth import reject_if_unauthorized, reject_if_unauthorized_callback
from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.generation import GeneratedImage, generate, post_process, repeat
from comfytelegram.message_text import message_text
from comfytelegram.profiles import (
    ModelProfile,
    apply_profile_override,
    join_nonempty,
    resolve_profile,
)
from comfytelegram.settings import Settings
from comfytelegram.settings_menu import _safe_edit_message, handle_custom_value_message
from comfytelegram.storage import Storage
from comfytelegram.tags import TagDatabase, TagResult, TagSource, category_label
from comfytelegram.topics import pop_pending, set_pending
from comfytelegram.workflows import GenerationParams, LoraSpec

logger = logging.getLogger(__name__)

T = TypeVar("T")

POSTPROCESS_KEYBOARD_LABELS = {
    "upscale": "🔍 Upscale 4x",
    "face": "✨ Face Detail",
    "hand": "🖐️ Hand Detail",
}
POSTPROCESS_STATUS_LABELS = {
    "upscale": "Upscaling",
    "face": "Refining face",
    "hand": "Refining hand",
}
ANALYZE_ONLY_CALLBACK_KIND = "analyze_only"
DEEP_ANALYZE_CALLBACK_KIND = "deep_analyze"
SHOW_PROMPT_CALLBACK_KIND = "show_prompt"
ANALYZE_PROMPT_CALLBACK_KIND = "analyze_prompt"
UPSCALE_CONFIRM_CALLBACK_KIND = "upscale_confirmed"
UPSCALE_CANCEL_CALLBACK_KIND = "upscale_cancelled"
#: "🖐️ Hand Detail" doesn't post-process immediately — it asks auto vs.
#: manual first (see `_hand_mode_keyboard`), since the YOLO bbox detector
#: `HAND_AUTO_CALLBACK_KIND` runs often can't find a hand at all.
HAND_AUTO_CALLBACK_KIND = "hand_auto"
HAND_MANUAL_CALLBACK_KIND = "hand_manual"
#: Grid resolution for "✋ Tap to mark" (see `_draw_hand_point_grid`/
#: `_hand_point_keyboard`) — coarse enough to keep the keyboard to
#: `HAND_POINT_GRID_SIZE` rows of `HAND_POINT_GRID_SIZE` buttons each.
HAND_POINT_GRID_SIZE = 4
#: Callback_data prefix for a tapped grid cell (`hp:<result_id>:<row>:<col>`)
#: — its own namespace, separate from `pp:`, so `postprocess_callback`'s
#: `query.data.split(":", 2)` parsing doesn't need to account for the extra
#: row/col fields (see `hand_point_callback`, registered on this in main.py).
HAND_POINT_CALLBACK_PREFIX = "hp:"
AGAIN_CALLBACK_PREFIX = "again:"
GENERATE_FROM_PROMPT_CALLBACK_PREFIX = "genp:"
STREAM_CANCEL_CALLBACK_DATA = "stream:cancel"
CHARACTER_EDIT_CANCEL_CALLBACK_DATA = "char:edit_cancel"
CHARACTER_RENAME_CANCEL_CALLBACK_DATA = "char:rename_cancel"

#: Hard ceiling on how many images a single "/stream" run can produce, even
#: if nobody sends "/stop" — a safety net against an unattended chat quietly
#: burning GPU time (and Telegram API calls) forever.
STREAM_HARD_LIMIT = 100

CHARACTER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

#: A leading "danbooru:"/"e621:" on a `/tags`/`/tagcheck` query overrides
#: the checkpoint-profile-driven dictionary choice — see `_resolve_tag_sources`.
_TAG_SOURCE_PREFIX_RE = re.compile(r"^(danbooru|e621):\s*", re.IGNORECASE)

#: Hard ceiling on how many comma-separated tokens a single `/tagcheck` call
#: processes — a full paragraph pasted by mistake shouldn't produce a
#: thousand-line reply.
TAGCHECK_TOKEN_LIMIT = 60

CHARACTER_HELP = (
    "Save a reusable character design so you don't have to retype its "
    "description every time:\n"
    "/character save <name> | <positive prompt> [| <negative prompt>]\n"
    "/character delete <name>\n"
    "/characters — list saved characters, activate one, or ✏️ edit/🔤 rename it\n\n"
    "While a character is active, its prompt is folded into every image "
    "you generate until you switch or clear it."
)

# Minimum interval between progress-message edits, to stay well under
# Telegram's per-chat edit rate limit — a 40-step job would otherwise fire
# 40 edits in a few seconds.
PROGRESS_EDIT_INTERVAL = 2.0

# Telegram's hard limit for sendPhoto is exactly 10485760 bytes (10 MiB) — a
# 4x upscale of a 1024px SDXL image routinely lands around 14-15MB as a PNG.
# Stay under it with margin, and fall back to sendDocument (up to 50MB,
# uncompressed) for anything bigger rather than silently failing.
TELEGRAM_PHOTO_SIZE_LIMIT = 10_000_000

# A 4x upscale of an image already at or beyond this size (long edge, in
# pixels) produces a very large, slow render — usually a sign the tapped
# image was already upscaled (its own "🔍 Upscale 4x" button doesn't know
# that), not something intentional. Gated behind an explicit confirmation
# instead of running immediately — see `postprocess_callback`'s
# `UPSCALE_CONFIRM_CALLBACK_KIND` branch.
UPSCALE_CONFIRM_THRESHOLD_PX = 2000

#: Persistent custom keyboard (replaces the device's own keyboard for the
#: whole chat, not an inline button on one message — see `_STREAMING_KEYBOARD`
#: below for why that distinction matters) listing every top-level command.
#: Installed on `/start` and restored by `_finish_stream` once a `/stream`
#: run ends, so the chat always shows *either* this or `_STREAMING_KEYBOARD`,
#: never both and never neither.
_MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [["/model", "/settings"], ["/characters", "/stream"], ["/help"]],
    resize_keyboard=True,
)

#: Swapped in for the duration of a "/stream" run (see `_run_stream`) — while
#: streaming, commands like /model or /settings would just add confusion (or
#: race the stream's own checkpoint/profile resolution), so the keyboard is
#: pared down to the one action that's actually valid: stopping it. Tapping
#: it just sends the text "/stop" as an ordinary message, which
#: `stop_command`'s existing CommandHandler picks up like any other typed
#: "/stop".
_STREAMING_KEYBOARD = ReplyKeyboardMarkup([["/stop"]], resize_keyboard=True)


def _post_process_keyboard(result_id: str) -> InlineKeyboardMarkup:
    """The keyboard attached to a generated/post-processed image: one
    button per `POSTPROCESS_KEYBOARD_LABELS` entry plus Analyze Image (image
    analysis, prompt only, no generation), Analyze Prompt (a `/tagcheck`-
    style tag-health check on the prompt this image was built from — see
    `postprocess_callback`'s `ANALYZE_PROMPT_CALLBACK_KIND` branch), Deep
    Analyze (image analysis via the bigger `analyze_caption_deep` model),
    and Show Prompt (a debugging button that just dumps the exact
    positive/negative prompt this image was generated with, folded
    character and profile prefixes included — see `postprocess_callback`'s
    `SHOW_PROMPT_CALLBACK_KIND` branch), all scoped to `result_id` (see
    `storage.py`'s `pending_result`). "🖐️ Hand Detail" doesn't post-process
    on this tap alone — see `postprocess_callback`'s `"hand"` branch for the
    auto/manual choice it replies with instead."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(label, callback_data=f"pp:{kind}:{result_id}")
                for kind, label in POSTPROCESS_KEYBOARD_LABELS.items()
            ],
            [
                InlineKeyboardButton(
                    "🏷️ Analyze Image", callback_data=f"pp:{ANALYZE_ONLY_CALLBACK_KIND}:{result_id}"
                ),
                InlineKeyboardButton(
                    "🔬 Analyze Prompt",
                    callback_data=f"pp:{ANALYZE_PROMPT_CALLBACK_KIND}:{result_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔎 Deep Analyze", callback_data=f"pp:{DEEP_ANALYZE_CALLBACK_KIND}:{result_id}"
                ),
                InlineKeyboardButton(
                    "🐛 Show Prompt", callback_data=f"pp:{SHOW_PROMPT_CALLBACK_KIND}:{result_id}"
                ),
            ],
        ]
    )


def _again_keyboard(snapshot_id: str) -> InlineKeyboardMarkup:
    """Attached to the "Done" status message of every base generation — lets
    the user crank out another batch of the same prompt with one tap instead
    of retyping it or hunting down a specific image's own Regenerate button.
    Scoped to `snapshot_id` (see storage.py's `generation_snapshot`) so this
    specific message always repeats the generation it was created from, even
    if a newer one has since happened in the same chat."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Generate Again", callback_data=f"{AGAIN_CALLBACK_PREFIX}{snapshot_id}"
                )
            ]
        ]
    )


def _upscale_confirm_keyboard(result_id: str) -> InlineKeyboardMarkup:
    """Attached to the "this image is already large" warning (see
    `postprocess_callback`'s `UPSCALE_CONFIRM_THRESHOLD_PX` gate) — lets the
    user explicitly confirm a 4x upscale of an already-large image instead
    of it running immediately, since that's usually an accidental
    double-upscale rather than something intended."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Upscale anyway",
                    callback_data=f"pp:{UPSCALE_CONFIRM_CALLBACK_KIND}:{result_id}",
                ),
                InlineKeyboardButton(
                    "❌ Cancel", callback_data=f"pp:{UPSCALE_CANCEL_CALLBACK_KIND}:{result_id}"
                ),
            ]
        ]
    )


def _hand_mode_keyboard(result_id: str) -> InlineKeyboardMarkup:
    """Attached to "🖐️ Hand Detail"'s first reply — lets the user pick
    auto-detection (today's YOLO/SAM behavior, `HAND_AUTO_CALLBACK_KIND`) or
    tap a point themselves (`HAND_MANUAL_CALLBACK_KIND`) for when the
    detector can't find the hand at all. See `postprocess_callback`."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🤖 Auto-detect", callback_data=f"pp:{HAND_AUTO_CALLBACK_KIND}:{result_id}"
                ),
                InlineKeyboardButton(
                    "✋ Tap to mark", callback_data=f"pp:{HAND_MANUAL_CALLBACK_KIND}:{result_id}"
                ),
            ]
        ]
    )


def _draw_hand_point_grid(image_bytes: bytes) -> bytes:
    """Overlay a `HAND_POINT_GRID_SIZE`x`HAND_POINT_GRID_SIZE` grid onto a
    copy of `image_bytes`, each cell labeled with the same row-letter/
    column-number text as its matching `_hand_point_keyboard` button (e.g.
    "B3"), so a cell can be identified without having to count grid lines.
    Uses `ImageFont.load_default(size=...)` — Pillow's built-in scalable
    bitmap font (no TTF file to locate/bundle) — with a black outline
    (`stroke_width`/`stroke_fill`) rather than a filled backing box behind
    it, so the label stays legible over any image content without covering
    much of it. Returns PNG bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_color = (255, 0, 0)
    line_width = max(1, min(width, height) // 400)
    cell_width = width / HAND_POINT_GRID_SIZE
    cell_height = height / HAND_POINT_GRID_SIZE
    for i in range(1, HAND_POINT_GRID_SIZE):
        x = round(width * i / HAND_POINT_GRID_SIZE)
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
        y = round(height * i / HAND_POINT_GRID_SIZE)
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    font_size = max(10, round(min(cell_width, cell_height) * 0.12))
    font = ImageFont.load_default(size=font_size)
    padding = max(2, font_size // 3)
    for row in range(HAND_POINT_GRID_SIZE):
        for col in range(HAND_POINT_GRID_SIZE):
            label = f"{chr(ord('A') + row)}{col + 1}"
            text_x = col * cell_width + padding
            text_y = row * cell_height + padding
            draw.text(
                (text_x, text_y),
                label,
                fill=(255, 255, 0),
                font=font,
                stroke_width=max(1, font_size // 8),
                stroke_fill=(0, 0, 0),
            )

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _hand_point_keyboard(result_id: str) -> InlineKeyboardMarkup:
    """One button per grid cell drawn by `_draw_hand_point_grid`, labeled by
    row letter + column number (e.g. "B3") in reading order, callback_data
    `hp:<result_id>:<row>:<col>` (see `hand_point_callback`)."""
    rows = []
    for row in range(HAND_POINT_GRID_SIZE):
        row_letter = chr(ord("A") + row)
        rows.append(
            [
                InlineKeyboardButton(
                    f"{row_letter}{col + 1}",
                    callback_data=f"{HAND_POINT_CALLBACK_PREFIX}{result_id}:{row}:{col}",
                )
                for col in range(HAND_POINT_GRID_SIZE)
            ]
        )
    return InlineKeyboardMarkup(rows)


def _stream_prompt_cancel_keyboard() -> InlineKeyboardMarkup:
    """Attached to the "What should the stream generate?" follow-up prompt
    (see `stream_command`'s promptless branch) — lets someone who tapped the
    bare "/stream" keyboard button by mistake back out without having to
    send a throwaway prompt just to get past it."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Cancel", callback_data=STREAM_CANCEL_CALLBACK_DATA)]]
    )


def _character_edit_cancel_keyboard() -> InlineKeyboardMarkup:
    """Attached to the "Send the new prompt for '<name>'" follow-up (see
    `character_callback`'s "✏️ Edit" branch) — lets a mistaken tap back out
    without having to send a throwaway message just to get past it."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Cancel", callback_data=CHARACTER_EDIT_CANCEL_CALLBACK_DATA)]]
    )


def _character_rename_cancel_keyboard() -> InlineKeyboardMarkup:
    """Attached to the "Send the new name for '<name>'" follow-up (see
    `character_callback`'s "🔤 Rename" branch) — lets a mistaken tap back
    out without having to send a throwaway message just to get past it."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Cancel", callback_data=CHARACTER_RENAME_CANCEL_CALLBACK_DATA)]]
    )


def _generate_from_prompt_keyboard(prompt_id: str, prompt: str) -> InlineKeyboardMarkup:
    """Attached to each of "🏷️ Analyze Image"'s standalone prompt messages: a
    "🎨 Generate" button that generates from *that specific* derived prompt
    (WD14 tags or a Qwen-VL caption) without retyping it, scoped to
    `prompt_id` (see storage.py's `derived_prompt`), plus — only when the
    prompt fits — a "📋 Copy" button (Telegram's native copy-to-clipboard
    `CopyTextButton`, no callback round-trip needed) for pasting the raw
    prompt text elsewhere. Telegram hard-caps `copy_text` at
    `MAX_COPY_TEXT` (256 chars) with no partial-copy fallback, and a WD14
    tag list or a deep caption routinely runs past that — silently copying
    a truncated prefix would hand back a *wrong*, not just shorter, prompt,
    so the button is omitted entirely rather than lie about what got
    copied. The full prompt is still visible in the message text (Telegram
    lets you select/copy that manually) and reusable in full via "🎨
    Generate" (through `derived_prompt`, no length limit)."""
    buttons = [
        InlineKeyboardButton(
            "🎨 Generate", callback_data=f"{GENERATE_FROM_PROMPT_CALLBACK_PREFIX}{prompt_id}"
        )
    ]
    if len(prompt) <= InlineKeyboardButtonLimit.MAX_COPY_TEXT:
        buttons.append(InlineKeyboardButton("📋 Copy", copy_text=CopyTextButton(prompt)))
    return InlineKeyboardMarkup([buttons])


async def _send_result_image(
    message: Message,
    data: bytes,
    filename: str,
    reply_markup: InlineKeyboardMarkup,
) -> Message:
    """Send a generated image, falling back to a document when it's too big
    for Telegram's photo path (see TELEGRAM_PHOTO_SIZE_LIMIT above). Returns
    the sent message — callers need it to read back the file_id Telegram
    assigned, for `_store_pending_result`."""
    if len(data) <= TELEGRAM_PHOTO_SIZE_LIMIT:
        return await message.reply_photo(photo=io.BytesIO(data), reply_markup=reply_markup)
    return await message.reply_document(
        document=io.BytesIO(data),
        filename=filename,
        caption="Sent as a file — too large for Telegram's photo size limit (10MB).",
        reply_markup=reply_markup,
    )


def _extract_file_id(sent_message: Message) -> str:
    """The Telegram `file_id` of whichever attachment `_send_result_image`
    actually sent (a photo, preferring its largest size, or a document for
    oversized images). Raises if `sent_message` has neither."""
    if sent_message.photo:
        return sent_message.photo[-1].file_id  # largest resolution
    if sent_message.document:
        return sent_message.document.file_id
    raise ValueError("Sent message has neither a photo nor a document to read a file_id from")


def _serialize_generation_params(params: GenerationParams) -> dict[str, Any]:
    """Flatten a `GenerationParams` into a plain JSON-able dict for
    `Storage` (deliberately drops `seed` and `filename_prefix` — see
    `test_serialization_omits_seed_so_regenerate_gets_a_fresh_roll`).
    Inverse of `_deserialize_generation_params`."""
    return {
        "checkpoint": params.checkpoint,
        "positive_prompt": params.positive_prompt,
        "negative_prompt": params.negative_prompt,
        "steps": params.steps,
        "cfg": params.cfg,
        "sampler_name": params.sampler_name,
        "scheduler": params.scheduler,
        "width": params.width,
        "height": params.height,
        "batch_size": params.batch_size,
        "clip_skip": params.clip_skip,
        "loras": [
            {
                "name": lora.name,
                "strength_model": lora.strength_model,
                "strength_clip": lora.strength_clip,
            }
            for lora in params.loras
        ],
        "loader": params.loader,
        "clip_name": params.clip_name,
        "clip_type": params.clip_type,
        "vae_name": params.vae_name,
        "model_sampling_shift": params.model_sampling_shift,
    }


def _deserialize_generation_params(data: dict[str, Any]) -> GenerationParams:
    """`.get(..., <field default>)` on everything but checkpoint/prompts lets
    this still read pending_result rows written before the "🔁 Regenerate"
    button existed (when only the post-processing subset of fields was
    stored) — those rows just fall back to GenerationParams' own generic
    defaults for steps/cfg/etc. instead of the exact original values."""
    return GenerationParams(
        checkpoint=data["checkpoint"],
        positive_prompt=data["positive_prompt"],
        negative_prompt=data["negative_prompt"],
        steps=data.get("steps", 30),
        cfg=data.get("cfg", 7.0),
        sampler_name=data.get("sampler_name", "euler"),
        scheduler=data.get("scheduler", "normal"),
        width=data.get("width", 1024),
        height=data.get("height", 1024),
        batch_size=data.get("batch_size", 1),
        clip_skip=data.get("clip_skip", -1),
        loras=[LoraSpec(**lora) for lora in data.get("loras", [])],
        loader=data.get("loader", "checkpoint"),
        clip_name=data.get("clip_name", ""),
        clip_type=data.get("clip_type", "stable_diffusion"),
        vae_name=data.get("vae_name", ""),
        model_sampling_shift=data.get("model_sampling_shift"),
    )


def _make_progress_callback(status_message: Message, label: str = "Generating"):
    """Shared throttled progress-edit closure for `generate_message` and
    `again_callback` — see PROGRESS_EDIT_INTERVAL above. `label` lets
    `_run_stream` prefix each edit with its image count (e.g.
    "Streaming 3/100") instead of the generic "Generating"."""
    last_edit = {"t": 0.0}

    async def on_progress(progress: JobProgress) -> None:
        """Edit `status_message` with the current step/percent, throttled
        to at most once per `PROGRESS_EDIT_INTERVAL`; silently skipped once
        `progress.done` or before the first real progress event arrives."""
        if progress.done or progress.value is None or progress.max is None:
            return
        now = time.monotonic()
        if now - last_edit["t"] < PROGRESS_EDIT_INTERVAL:
            return
        last_edit["t"] = now
        pct = int(100 * progress.value / progress.max) if progress.max else 0
        try:
            await status_message.edit_text(
                f"{label}… {pct}% (step {progress.value}/{progress.max})"
            )
        except Exception:
            logger.debug("Progress edit skipped (rate-limited or unchanged)", exc_info=True)

    return on_progress


async def _run_reporting_errors(
    status_message: Message, label: str, log_label: str, awaitable: Awaitable[T]
) -> T | None:
    """Run `awaitable`, reporting a `ComfyUIError` or any other exception back
    into `status_message` instead of letting it propagate — shared by every
    generation/post-processing entry point below. `label` is the user-facing
    verb ("Generation", "Upscaling", ...); `log_label` is what goes in the
    server log. Returns None on failure (already reported); callers should
    treat that as "stop here"."""
    try:
        return await awaitable
    except ComfyUIError as exc:
        await status_message.edit_text(f"{label} failed: {exc}")
        return None
    except Exception:
        logger.exception("Unexpected error during %s", log_label)
        await status_message.edit_text(f"{label} failed with an unexpected error.")
        return None


async def _send_and_store_result(
    message: Message, chat_id: int, storage: Storage, img: GeneratedImage
) -> None:
    """Send a generated image with its post-processing keyboard, then persist
    what those buttons need (a re-downloadable file_id + the params to build
    the next graph) so they still work after a bot restart — see storage.py."""
    result_id = uuid.uuid4().hex[:12]
    sent = await _send_result_image(
        message, img.data, img.filename, _post_process_keyboard(result_id)
    )
    storage.store_pending_result(
        result_id,
        chat_id,
        _extract_file_id(sent),
        img.filename,
        _serialize_generation_params(img.full_params),
    )


_AnalyzerResult = TypeVar("_AnalyzerResult")


async def _run_analyzer(
    label: str, awaitable: Awaitable[_AnalyzerResult]
) -> _AnalyzerResult | None:
    """Isolate one analyzer's failure (e.g. WD14 files not staged, Ollama
    unreachable) from the other, so a side-by-side comparison still shows
    whichever one actually worked. None means it failed — the caller skips
    the "🎨 Generate" button in that case, since there's no usable prompt to
    generate from. Shared by `postprocess_callback`'s "analyze_only" branch
    and `photo_message`."""
    try:
        return await awaitable
    except Exception:
        logger.exception("%s analyzer failed", label)
        return None


async def _analyze_both(
    source_bytes: bytes, settings: Settings
) -> tuple[str | None, tuple[str, str] | None]:
    """Run both analyzers regardless of any checkpoint's configured
    `prompt_style` — this is for comparing WD14 tags against a Qwen-VL
    caption side by side, so both always run. The caption side is
    `(positive, negative)` — see `analyze_caption` — since this standalone
    display does surface a suggested negative prompt. Used by
    `postprocess_callback`'s "🏷️ Analyze Image"
    branch (for the bot's own generated images, where the quick model is the
    default and `DEEP_ANALYZE_CALLBACK_KIND` is an opt-in extra) —
    `photo_message` uses `_analyze_both_deep` instead, see there for why."""
    tags, caption = await asyncio.gather(
        _run_analyzer("WD14", analyze_tags(source_bytes, settings)),
        _run_analyzer("Qwen-VL", analyze_caption(source_bytes, settings)),
    )
    return tags, caption


async def _analyze_both_deep(
    source_bytes: bytes, settings: Settings
) -> tuple[str | None, tuple[str, str] | None]:
    """Same WD14-tags/Qwen-VL-caption pairing as `_analyze_both` (including
    the caption side's `(positive, negative)` shape), but the caption side
    always uses the bigger `analyze_caption_deep` model. `photo_message`'s
    directly-uploaded photos aren't on the interactive generation critical
    path the way a "🏷️ Analyze Image" tap on a generated image is, so there's no
    reason to default to the cheap model there and make the user ask twice
    for the better one."""
    tags, caption = await asyncio.gather(
        _run_analyzer("WD14", analyze_tags(source_bytes, settings)),
        _run_analyzer("Qwen-VL (deep)", analyze_caption_deep(source_bytes, settings)),
    )
    return tags, caption


async def _send_derived_prompt(
    reply_target: Message,
    storage: Storage,
    chat_id: int,
    checkpoint: str,
    label: str,
    prompt: str | None,
    negative_prompt: str = "",
) -> None:
    """One standalone message per analyzer, each with its own "🎨 Generate"/
    "📋 Copy" buttons scoped to that specific prompt (see storage.py's
    `derived_prompt`) — not a shared button on a combined message, since
    the two prompts are independent and the user may only want to act on
    one of them. `negative_prompt`, when non-empty (a Qwen-VL caption's
    suggested negative — see `analyze_caption` — WD14 tags never have one),
    is shown as its own line and stored alongside so that "🎨 Generate"
    button generates with it as the negative prompt too. Shared by
    `postprocess_callback` and `photo_message`."""
    if prompt is None:
        await reply_target.reply_text(f"{label}: failed — see server log.")
        return
    prompt_id = uuid.uuid4().hex[:12]
    storage.store_derived_prompt(prompt_id, chat_id, checkpoint, prompt, negative_prompt)
    text = f"{label}:\n{prompt}"
    if negative_prompt:
        text += f"\n\n🚫 Suggested negative:\n{negative_prompt}"
    await reply_target.reply_text(
        text, reply_markup=_generate_from_prompt_keyboard(prompt_id, prompt)
    )


async def _deliver_generation_result(
    status_message: Message,
    reply_target: Message,
    chat_id: int,
    storage: Storage,
    images: list[GeneratedImage],
) -> None:
    """Shared tail of `generate_message`/`again_callback` once a batch of
    images has been produced: mint a fresh generation snapshot for the next
    "Generate Again" tap, send + register each image, then report "Done"
    with the "Generate Again" button as a fresh message. `reply_target` is
    the message new image replies attach to (the original prompt message,
    or the callback query's own message).

    The "Done" report is a new message sent *after* the images, not an
    edit of `status_message` in place — an edit can't change a message's
    position in the chat, so editing the progress message would leave
    "Generate Again" sitting above the images it refers to. `status_message`
    is deleted once the images are sent, the same way post-processing
    (`postprocess_callback`) retires its own progress message.

    Images are sent concurrently rather than one at a time — each is an
    independent Telegram call, so a batch shouldn't pay for N sequential
    round-trips. The one tradeoff: for batch_size > 1, the order images
    land in the chat is whatever order their `sendPhoto` calls happen to
    complete in, not necessarily the batch's original order.
    """
    snapshot_id = uuid.uuid4().hex[:12]
    storage.store_generation_snapshot(
        snapshot_id, chat_id, _serialize_generation_params(images[0].full_params)
    )
    await asyncio.gather(
        *(_send_and_store_result(reply_target, chat_id, storage, img) for img in images)
    )
    await status_message.delete()
    await reply_target.reply_text(
        f"Done — {len(images)} image(s).", reply_markup=_again_keyboard(snapshot_id)
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/start` — send the command summary."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return
    await update.effective_message.reply_text(
        "Send me a prompt and I'll generate an image with ComfyUI, or a "
        "photo and I'll analyze it into a prompt. Prefix any word with - to "
        'send it as a negative instead, e.g. "1girl, outdoors, -blurry, '
        '-watermark".\n\n'
        "/model — pick a checkpoint\n"
        "/settings — view or change generation defaults for the current model\n"
        "/character save <name> | <prompt> — save a reusable character design\n"
        "/characters — list saved characters, activate one, or ✏️ edit/🔤 rename it\n"
        f"/stream [prompt] — generate single images back-to-back (up to {STREAM_HARD_LIMIT}) "
        "until /stop, sending each one immediately; omit the prompt and I'll ask for it\n"
        "/stop — stop a running /stream\n"
        "/tags <query> — search danbooru/e621 tags to build a prompt\n"
        "/tagcheck <prompt> — check a prompt's tags against the tag database\n"
        "/help — show this message",
        reply_markup=_MAIN_KEYBOARD,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/help` — alias for `/start`."""
    await start(update, context)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/model` — list checkpoints ComfyUI has installed as an inline
    keyboard (labeled with the matching profile's `display_name` where one
    exists), for `model_callback` to act on."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    client: ComfyClient = context.bot_data["comfy_client"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]

    try:
        checkpoints = await client.list_checkpoints()
    except (ComfyUIError, OSError) as exc:
        logger.exception("Failed to list checkpoints")
        await update.effective_message.reply_text(f"Couldn't reach ComfyUI: {exc}")
        return

    if not checkpoints:
        await update.effective_message.reply_text("ComfyUI reports no checkpoints installed.")
        return

    context.bot_data["available_checkpoints"] = checkpoints

    buttons = []
    for i, ckpt in enumerate(checkpoints):
        profile = resolve_profile(ckpt, profiles)
        label = profile.display_name if profile else ckpt
        buttons.append([InlineKeyboardButton(label, callback_data=f"model:{i}")])

    await update.effective_message.reply_text(
        "Choose a model:", reply_markup=InlineKeyboardMarkup(buttons)
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `model:<index>` tap from `model_command`'s keyboard: persist
    the chosen checkpoint as this chat's selection. The index is stale (and
    rejected) if `context.bot_data["available_checkpoints"]` has since
    changed — e.g. from a newer `/model` call in another chat."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    checkpoints: list[str] = context.bot_data.get("available_checkpoints", [])
    _, index_str = query.data.split(":", 1)
    index = int(index_str)
    if index >= len(checkpoints):
        await query.answer("That model list is stale — run /model again.", show_alert=True)
        return

    checkpoint = checkpoints[index]
    storage: Storage = context.bot_data["storage"]
    storage.set_checkpoint(update.effective_chat.id, checkpoint)

    profiles: list[ModelProfile] = context.bot_data["profiles"]
    profile = resolve_profile(checkpoint, profiles)
    label = profile.display_name if profile else checkpoint

    await query.answer()
    await query.edit_message_text(f"Model set to: {label}")


def _characters_keyboard(
    characters: list[dict[str, Any]], active: str | None
) -> InlineKeyboardMarkup:
    """One row per saved character: an activate button (✅-marked if it's
    `active`) plus "✏️ Edit" (change its prompt in place) and "🔤 Rename"
    buttons, and a "Clear active character" row when one is active."""
    rows = []
    for char in characters:
        label = f"✅ {char['name']}" if char["name"] == active else char["name"]
        rows.append(
            [
                InlineKeyboardButton(label, callback_data=f"char:activate:{char['name']}"),
                InlineKeyboardButton("✏️ Edit", callback_data=f"char:edit:{char['name']}"),
                InlineKeyboardButton("🔤 Rename", callback_data=f"char:rename:{char['name']}"),
            ]
        )
    if active is not None:
        rows.append([InlineKeyboardButton("❌ Clear active character", callback_data="char:clear")])
    return InlineKeyboardMarkup(rows)


async def character_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/character save <name> | <positive> [| <negative>]` or
    `/character delete <name>` — manage this chat's saved characters; any
    other/missing subcommand just shows `CHARACTER_HELP`."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    storage: Storage = context.bot_data["storage"]
    chat_id = update.effective_chat.id

    raw = (message.text or "").split(maxsplit=1)
    rest = raw[1] if len(raw) > 1 else ""

    if rest.startswith("save"):
        body = rest[len("save") :].strip()
        parts = [p.strip() for p in body.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            await message.reply_text(
                "Usage: /character save <name> | <positive prompt> [| <negative prompt>]"
            )
            return
        name = parts[0]
        if not CHARACTER_NAME_RE.match(name):
            await message.reply_text(
                "Character names can only use letters, digits, '-' and '_' (max 32 chars)."
            )
            return
        positive_prompt = parts[1]
        negative_prompt = parts[2] if len(parts) > 2 else ""
        storage.save_character(chat_id, name, positive_prompt, negative_prompt)
        await message.reply_text(f"Saved character '{name}'. Activate it with /characters.")
        return

    if rest.startswith("delete"):
        name = rest[len("delete") :].strip()
        if not name:
            await message.reply_text("Usage: /character delete <name>")
            return
        if storage.get_character(chat_id, name) is None:
            await message.reply_text(f"No saved character named '{name}'.")
            return
        storage.delete_character(chat_id, name)
        await message.reply_text(f"Deleted character '{name}'.")
        return

    await message.reply_text(CHARACTER_HELP)


async def characters_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/characters` — list this chat's saved characters as an inline
    keyboard to activate one, for `character_callback` to act on."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    characters = storage.list_characters(chat_id)
    if not characters:
        await update.effective_message.reply_text("No saved characters yet.\n\n" + CHARACTER_HELP)
        return

    active = storage.get_active_character_name(chat_id)
    header = f"Active character: {active}" if active else "No character active — tap one to use it."
    await update.effective_message.reply_text(
        header, reply_markup=_characters_keyboard(characters, active)
    )


async def character_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `char:clear`, `char:edit_cancel`, `char:rename_cancel`,
    `char:edit:<name>`, `char:rename:<name>` or `char:activate:<name>` tap
    from `characters_command`'s keyboard: clear this chat's active
    character, back out of a pending edit/rename, start one (see
    `_consume_awaiting_character_edit`/`_consume_awaiting_character_rename`
    for how the follow-up message is consumed), or activate a character."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    parts = query.data.split(":", 2)
    action = parts[1]

    if action == "clear":
        storage.clear_active_character(chat_id)
        await query.answer("Active character cleared.")
        await _safe_edit_message(
            query,
            "No character active — tap one to use it.",
            _characters_keyboard(storage.list_characters(chat_id), None),
        )
        return

    if action == "edit_cancel":
        pop_pending(context.chat_data, "awaiting_character_edit", query.message)
        await query.answer("Cancelled.")
        await _safe_edit_message(
            query, "Cancelled — character unchanged.", InlineKeyboardMarkup([])
        )
        return

    if action == "rename_cancel":
        pop_pending(context.chat_data, "awaiting_character_rename", query.message)
        await query.answer("Cancelled.")
        await _safe_edit_message(
            query, "Cancelled — character unchanged.", InlineKeyboardMarkup([])
        )
        return

    name = parts[2]
    character = storage.get_character(chat_id, name)
    if character is None:
        await query.answer(
            "That character no longer exists — refresh with /characters.", show_alert=True
        )
        return

    if action == "edit":
        pop_pending(context.chat_data, "awaiting_character_rename", query.message)
        set_pending(context.chat_data, "awaiting_character_edit", query.message, name)
        await query.answer()
        lines = [
            f"Send the new prompt for '{name}' as:",
            "<positive prompt> [| <negative prompt>]",
            "",
            f"Current positive: {character['positive_prompt']}",
        ]
        if character["negative_prompt"]:
            lines.append(f"Current negative: {character['negative_prompt']}")
        await query.message.reply_text(
            "\n".join(lines), reply_markup=_character_edit_cancel_keyboard()
        )
        return

    if action == "rename":
        pop_pending(context.chat_data, "awaiting_character_edit", query.message)
        set_pending(context.chat_data, "awaiting_character_rename", query.message, name)
        await query.answer()
        await query.message.reply_text(
            f"Send the new name for '{name}' (letters, digits, '-' and '_' only, max 32 chars).",
            reply_markup=_character_rename_cancel_keyboard(),
        )
        return

    storage.set_active_character(chat_id, name)
    await query.answer(f"Activated '{name}'.")
    await _safe_edit_message(
        query,
        f"Active character: {name}",
        _characters_keyboard(storage.list_characters(chat_id), name),
    )


async def _resolve_checkpoint_or_default(
    message: Message,
    chat_id: int,
    storage: Storage,
    client: ComfyClient,
    context: ContextTypes.DEFAULT_TYPE,
) -> str | None:
    """This chat's selected checkpoint, or ComfyUI's first available one
    (persisted as the new selection) if none has been picked yet — shared
    by `generate_message` and `stream_command`. Returns None if it already
    replied with an error (ComfyUI unreachable / no checkpoints installed);
    callers should treat that the same as `_run_reporting_errors`: stop
    here."""
    checkpoint = storage.get_checkpoint(chat_id)
    if checkpoint is not None:
        return checkpoint

    try:
        checkpoints = await client.list_checkpoints()
    except (ComfyUIError, OSError) as exc:
        await message.reply_text(f"Couldn't reach ComfyUI: {exc}")
        return None
    if not checkpoints:
        await message.reply_text("ComfyUI reports no checkpoints installed.")
        return None

    checkpoint = checkpoints[0]
    storage.set_checkpoint(chat_id, checkpoint)
    context.bot_data["available_checkpoints"] = checkpoints
    await message.reply_text(
        f"No model selected yet — defaulting to {checkpoint}. Use /model to change it."
    )
    return checkpoint


#: Matches a "-token" where the "-" sits at the very start of the prompt or
#: right after whitespace/a comma — i.e. a delimiter, not a mid-word hyphen
#: like "well-lit" (the char before "-" there is "l", so the lookbehind
#: fails and it's left alone). The token itself stops at the next comma or
#: whitespace, so both tag-style ("1girl, -watermark, -blurry") and
#: space-separated ("a cat -blurry -watermark") prompts work the same way.
_NEGATIVE_TOKEN_RE = re.compile(r"(?<![^\s,])-([^\s,]+)")

#: A line of 3+ dashes on its own (whitespace either side ignored) splits a
#: prompt message into a positive block above and a negative block below —
#: an alternative to prefixing every single negative tag with "-", which
#: gets tedious for a whole block of them. Requires the dashes to have the
#: line to themselves so it can't misfire on a mid-word hyphen or an
#: em/en-dash used as punctuation.
_NEGATIVE_BLOCK_SEP_RE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)


def _normalize_prompt_block(text: str) -> str:
    """Turn one side of a `_NEGATIVE_BLOCK_SEP_RE`-split prompt — tags spread
    across commas, newlines, or both — into a single comma-joined string."""
    parts = [part.strip() for part in re.split(r"[,\n]", text)]
    return join_nonempty(parts)


def _split_negative_prompt(text: str) -> tuple[str, str]:
    """Split a raw prompt message into `(positive, negative)`. A "---" line
    (see `_NEGATIVE_BLOCK_SEP_RE`) takes priority — everything above it is
    positive, everything below is negative, letting a whole block of
    negative tags be written plainly instead of "-prefixed" one by one.
    Without one, falls back to pulling out individual "-token" negatives
    (see `_NEGATIVE_TOKEN_RE`), so "1girl, outdoors, -blurry, -watermark"
    still generates with "blurry, watermark" appended to the negative
    prompt instead of ending up literally in the positive one. Either way,
    a bare newline between tags is treated the same as a comma (see
    `_normalize_prompt_block`) — for someone who prefers writing one tag
    per line (CRLF included; `message_text` already normalizes it to
    "\\n") over comma-separating them."""
    block_match = _NEGATIVE_BLOCK_SEP_RE.search(text)
    if block_match:
        positive = _normalize_prompt_block(text[: block_match.start()])
        negative = _normalize_prompt_block(text[block_match.end() :])
        return positive, negative

    negatives = [match.group(1) for match in _NEGATIVE_TOKEN_RE.finditer(text)]
    remainder = _NEGATIVE_TOKEN_RE.sub("", text)
    positive = _normalize_prompt_block(remainder)
    positive = re.sub(r"\s{2,}", " ", positive).strip()
    return positive, join_nonempty(negatives)


def _resolve_effective_prompt(
    prompt_text: str, character: dict[str, str] | None
) -> tuple[str, str]:
    """Combine a raw prompt message with the active character (if any) into
    `(effective_prompt, extra_negative_prompt)` for `generate()`: splits the
    message into positive/negative (a "---" block separator, or else
    "-token" negatives — see `_split_negative_prompt`), folds the
    character's own saved positive/negative prompt in around them, and
    leaves profile-level negative defaults for `generate()`/`resolve_generation_params`
    to layer underneath."""
    positive, negative = _split_negative_prompt(prompt_text)
    effective_prompt = (
        join_nonempty([character["positive_prompt"], positive]) if character else positive
    )
    extra_negative = (
        join_nonempty([character["negative_prompt"], negative]) if character else negative
    )
    return effective_prompt, extra_negative


async def generate_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a plain-text message as a generation prompt: resolve this
    chat's checkpoint/profile/override/active-character, run `generate()`
    with live progress, then deliver the result. Defaults to ComfyUI's
    first available checkpoint (and remembers it) if none is selected yet.
    A pending "custom value" `/settings` entry (see
    `handle_custom_value_message`) or an in-progress `/stream` prompt or
    character-edit/-rename entry takes priority over treating the text as a
    prompt."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    if await handle_custom_value_message(update, context):
        return

    if await _consume_awaiting_stream_prompt(update, context):
        return

    if await _consume_awaiting_character_edit(update, context):
        return

    if await _consume_awaiting_character_rename(update, context):
        return

    message = update.effective_message
    prompt_text = (message_text(message) or "").strip()
    if not prompt_text:
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    client: ComfyClient = context.bot_data["comfy_client"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]

    checkpoint = await _resolve_checkpoint_or_default(message, chat_id, storage, client, context)
    if checkpoint is None:
        return

    profile = resolve_profile(checkpoint, profiles)
    override_fields = storage.get_override(chat_id, checkpoint)
    profile = apply_profile_override(profile, checkpoint, override_fields)

    active_character_name = storage.get_active_character_name(chat_id)
    character = (
        storage.get_character(chat_id, active_character_name) if active_character_name else None
    )
    effective_prompt, extra_negative = _resolve_effective_prompt(prompt_text, character)

    status_message = await message.reply_text("Generating… 0%", disable_notification=True)

    images = await _run_reporting_errors(
        status_message,
        "Generation",
        "generation",
        generate(
            client,
            checkpoint,
            effective_prompt,
            profile,
            extra_negative_prompt=extra_negative,
            on_progress=_make_progress_callback(status_message),
        ),
    )
    if images is None:
        return

    await _deliver_generation_result(status_message, message, chat_id, storage, images)


async def photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a directly-uploaded photo — as opposed to one of the bot's own
    generated images, which get analyzed via `_post_process_keyboard`'s
    "🏷️ Analyze Image" button instead — by running it through both analyzers and
    replying with each, including a "🎨 Generate" button on each so the
    derived prompt can be used right away. The caption side always uses the
    bigger `analyze_caption_deep` model (`_analyze_both_deep`, not the
    `_analyze_both`/`ANALYZE_ONLY_CALLBACK_KIND` cheap default that
    `postprocess_callback` uses for generated images) — an uploaded photo
    isn't on any interactive generation critical path, so there's no reason
    to default to the cheap model here. Resolves this chat's checkpoint the
    same way `generate_message` does (falling back to ComfyUI's first
    available one) only to scope that button's eventual profile/style — the
    analysis itself doesn't depend on it."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    client: ComfyClient = context.bot_data["comfy_client"]

    checkpoint = await _resolve_checkpoint_or_default(message, chat_id, storage, client, context)
    if checkpoint is None:
        return

    status_message = await message.reply_text("Analyzing image…", disable_notification=True)

    async def _download_and_analyze_both() -> tuple[str | None, tuple[str, str] | None]:
        tg_file = await context.bot.get_file(message.photo[-1].file_id)
        source_bytes = bytes(await tg_file.download_as_bytearray())
        return await _analyze_both_deep(source_bytes, settings)

    result = await _run_reporting_errors(
        status_message, "Analysis", "uploaded-image analyze", _download_and_analyze_both()
    )
    if result is None:
        return
    tags, caption = result
    caption_positive, caption_negative = caption if caption is not None else (None, "")
    await status_message.delete()

    await _send_derived_prompt(message, storage, chat_id, checkpoint, "🏷️ WD14 tags", tags)
    await _send_derived_prompt(
        message,
        storage,
        chat_id,
        checkpoint,
        "💬 Qwen-VL caption (deep)",
        caption_positive,
        caption_negative,
    )


async def postprocess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `pp:<kind>:<result_id>` tap from `_post_process_keyboard`:
    `kind` is `"analyze_only"` (run *both* analyzers — WD14 tags and
    Qwen-VL caption — and reply with both, no generation, no
    `prompt_style` dispatch; see `ANALYZE_ONLY_CALLBACK_KIND`),
    `"show_prompt"` (reply with the exact positive/negative prompt this
    image was built from, no download or generation and no tag check — see
    `SHOW_PROMPT_CALLBACK_KIND`), `"analyze_prompt"` (a `/tagcheck`-style
    tag-health check on that same prompt instead, if this image's
    checkpoint profile is tag-trained and tag data is imported — see
    `ANALYZE_PROMPT_CALLBACK_KIND`), or
    `"upscale"`/`"face"`/`"hand"` (download the source image and run that
    post-processing stage on it). A fresh `"upscale"` tap on an image
    already at or beyond `UPSCALE_CONFIRM_THRESHOLD_PX` doesn't upscale
    immediately — it downloads just far enough to measure the image, then
    replies with a "this is already large — continue?" prompt
    (`_upscale_confirm_keyboard`) instead, which comes back as either
    `UPSCALE_CONFIRM_CALLBACK_KIND` (proceed) or
    `UPSCALE_CANCEL_CALLBACK_KIND` (abort). A `"face"`/`"hand"` result whose
    detector found nothing to refine (`GeneratedImage.unchanged`, see
    `post_process`) isn't sent or stored at all — it's pixel-identical to
    what's already on screen, so re-posting it would just be noise — a
    "⚠️ No face/hand detected" text reply stands in for it instead. `"hand"`
    itself doesn't post-process immediately — the YOLO bbox detector behind
    it often can't find a hand at all, so it instead replies with
    `_hand_mode_keyboard`'s auto/manual choice: `HAND_AUTO_CALLBACK_KIND`
    (translated back to `"hand"` right before the block above, so it takes
    the same path a direct `"hand"` tap used to) or `HAND_MANUAL_CALLBACK_KIND`
    (downloads the source, overlays a tap grid via `_draw_hand_point_grid`,
    and replies with it plus `_hand_point_keyboard` — a grid-cell tap is
    handled by the separate `hand_point_callback`, not this function, since
    it needs a row/col in its callback_data). Alerts instead if `result_id`
    has expired (see `PENDING_RESULT_TTL_SECONDS`)."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    _, kind, result_id = query.data.split(":", 2)
    storage: Storage = context.bot_data["storage"]
    pending = storage.get_pending_result(result_id)
    if pending is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    client: ComfyClient = context.bot_data["comfy_client"]
    full_params = _deserialize_generation_params(pending["base_params"])

    if kind == SHOW_PROMPT_CALLBACK_KIND:
        negative = full_params.negative_prompt or "(none)"
        await query.message.reply_text(
            f"Positive:\n{full_params.positive_prompt}\n\nNegative:\n{negative}"
        )
        return

    if kind == ANALYZE_PROMPT_CALLBACK_KIND:
        tags_db: TagDatabase = context.bot_data["tags_db"]
        profiles: list[ModelProfile] = context.bot_data["profiles"]
        profile = resolve_profile(full_params.checkpoint, profiles)
        # Tag-health only makes sense for comma-separated booru tags, not a
        # natural-language caption — gated on prompt_style, same as
        # `_analyze_both`'s own tags-vs-caption pairing.
        if profile is None or profile.prompt_style != "tags" or not any(tags_db.stats().values()):
            await query.message.reply_text(
                "No tag check available for this image — its checkpoint isn't "
                "tag-trained, or no tag data has been imported yet."
            )
            return
        sources, _ = _resolve_tag_sources("", full_params.checkpoint, profiles)
        report = ["🔎 Tag check (positive):"]
        report.extend(_tagcheck_lines(full_params.positive_prompt, sources, tags_db, settings))
        if full_params.negative_prompt:
            report.append("")
            report.append("🔎 Tag check (negative):")
            report.extend(_tagcheck_lines(full_params.negative_prompt, sources, tags_db, settings))
        await query.message.reply_text("\n".join(report))
        return

    if kind == ANALYZE_ONLY_CALLBACK_KIND:
        status_message = await query.message.reply_text(
            "Analyzing image…", disable_notification=True
        )
        checkpoint = full_params.checkpoint
        chat_id = pending["chat_id"]

        async def _download_and_analyze_both() -> tuple[str | None, tuple[str, str] | None]:
            tg_file = await context.bot.get_file(pending["file_id"])
            source_bytes = bytes(await tg_file.download_as_bytearray())
            return await _analyze_both(source_bytes, settings)

        result = await _run_reporting_errors(
            status_message, "Analysis", "analyze-only", _download_and_analyze_both()
        )
        if result is None:
            return
        tags, caption = result
        caption_positive, caption_negative = caption if caption is not None else (None, "")
        await status_message.delete()

        await _send_derived_prompt(query.message, storage, chat_id, checkpoint, "🏷️ WD14 tags", tags)
        await _send_derived_prompt(
            query.message,
            storage,
            chat_id,
            checkpoint,
            "💬 Qwen-VL caption",
            caption_positive,
            caption_negative,
        )
        return

    if kind == DEEP_ANALYZE_CALLBACK_KIND:
        status_message = await query.message.reply_text(
            "Deep analyzing image…", disable_notification=True
        )
        checkpoint = full_params.checkpoint
        chat_id = pending["chat_id"]

        async def _download_and_analyze_deep() -> tuple[str, str]:
            tg_file = await context.bot.get_file(pending["file_id"])
            source_bytes = bytes(await tg_file.download_as_bytearray())
            return await analyze_caption_deep(source_bytes, settings)

        caption = await _run_reporting_errors(
            status_message, "Deep analysis", "deep-analyze", _download_and_analyze_deep()
        )
        if caption is None:
            return
        caption_positive, caption_negative = caption
        await status_message.delete()

        await _send_derived_prompt(
            query.message,
            storage,
            chat_id,
            checkpoint,
            "🔎 Deep caption",
            caption_positive,
            caption_negative,
        )
        return

    if kind == UPSCALE_CANCEL_CALLBACK_KIND:
        await _safe_edit_message(query, "Upscale cancelled.", InlineKeyboardMarkup([]))
        return

    if kind == "hand":
        await query.message.reply_text(
            "Auto-detect usually finds it, but you can mark the hand yourself if it keeps missing:",
            reply_markup=_hand_mode_keyboard(result_id),
        )
        return

    if kind == HAND_MANUAL_CALLBACK_KIND:
        tg_file = await context.bot.get_file(pending["file_id"])
        source_bytes = bytes(await tg_file.download_as_bytearray())
        gridded = _draw_hand_point_grid(source_bytes)
        await query.message.reply_photo(
            photo=io.BytesIO(gridded),
            caption="Tap the cell over the hand:",
            reply_markup=_hand_point_keyboard(result_id),
        )
        return

    if kind == HAND_AUTO_CALLBACK_KIND:
        kind = "hand"

    source_bytes: bytes | None = None
    if kind in ("upscale", UPSCALE_CONFIRM_CALLBACK_KIND):
        if kind == UPSCALE_CONFIRM_CALLBACK_KIND:
            await _safe_edit_message(query, "Upscaling anyway…", InlineKeyboardMarkup([]))
        tg_file = await context.bot.get_file(pending["file_id"])
        source_bytes = bytes(await tg_file.download_as_bytearray())
        if kind == "upscale":
            width, height = Image.open(io.BytesIO(source_bytes)).size
            if max(width, height) >= UPSCALE_CONFIRM_THRESHOLD_PX:
                await query.message.reply_text(
                    f"This image is already {width}×{height} — a 4x upscale would "
                    f"produce a {width * 4}×{height * 4} image. Upscale anyway?",
                    reply_markup=_upscale_confirm_keyboard(result_id),
                )
                return
        kind = "upscale"

    label = POSTPROCESS_STATUS_LABELS.get(kind, kind.title())
    status_message = await query.message.reply_text(f"{label}…", disable_notification=True)

    async def _download_and_post_process() -> GeneratedImage:
        """Bundle the file download and the post-process call into one
        awaitable so `_run_reporting_errors` covers both — unless the
        upscale size-check above already downloaded the file, in which
        case that's reused instead of hitting Telegram for it twice."""
        data = source_bytes
        if data is None:
            tg_file = await context.bot.get_file(pending["file_id"])
            data = bytes(await tg_file.download_as_bytearray())
        return await post_process(client, kind, data, pending["filename"], full_params)

    result = await _run_reporting_errors(
        status_message, label, "post-processing", _download_and_post_process()
    )
    if result is None:
        return

    await status_message.delete()
    if result.unchanged:
        subject = "face" if kind == "face" else "hand"
        await query.message.reply_text(f"⚠️ No {subject} detected — image unchanged.")
        return
    await _send_and_store_result(query.message, pending["chat_id"], storage, result)


async def hand_point_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `hp:<result_id>:<row>:<col>` tap from `_hand_point_keyboard`
    (the "✋ Tap to mark" grid `postprocess_callback`'s `HAND_MANUAL_CALLBACK_KIND`
    branch sends) — downloads the source image, turns the tapped cell into
    a fractional (x, y) point at its center, and runs
    `post_process(kind="hand_manual")` on it. Unlike the auto-detect path,
    a manually marked region is never "nothing detected", so there's no
    `unchanged` check here."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    _, result_id, row_str, col_str = query.data.split(":")
    row, col = int(row_str), int(col_str)
    storage: Storage = context.bot_data["storage"]
    pending = storage.get_pending_result(result_id)
    if pending is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    client: ComfyClient = context.bot_data["comfy_client"]
    full_params = _deserialize_generation_params(pending["base_params"])
    point_frac = (
        (col + 0.5) / HAND_POINT_GRID_SIZE,
        (row + 0.5) / HAND_POINT_GRID_SIZE,
    )

    status_message = await query.message.reply_text("Refining hand…", disable_notification=True)

    async def _download_and_post_process() -> GeneratedImage:
        tg_file = await context.bot.get_file(pending["file_id"])
        data = bytes(await tg_file.download_as_bytearray())
        return await post_process(
            client, "hand_manual", data, pending["filename"], full_params, point_frac=point_frac
        )

    result = await _run_reporting_errors(
        status_message, "Refining hand", "post-processing", _download_and_post_process()
    )
    if result is None:
        return

    await status_message.delete()
    await _send_and_store_result(query.message, pending["chat_id"], storage, result)


async def again_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Backs the "🔁 Generate Again" button — re-runs the base generation
    that *this specific message's* button was created from (same resolved
    settings, fresh seed), scoped by the snapshot_id embedded in
    callback_data (see storage.py's `generation_snapshot`) rather than
    "whatever this chat most recently generated", so an older message's
    button can't be hijacked by a newer generation elsewhere in the chat."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    _, snapshot_id = query.data.split(":", 1)
    storage: Storage = context.bot_data["storage"]
    snapshot = storage.get_generation_snapshot(snapshot_id)
    if snapshot is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    chat_id = snapshot["chat_id"]
    client: ComfyClient = context.bot_data["comfy_client"]
    full_params = _deserialize_generation_params(snapshot["params"])

    status_message = await query.message.reply_text("Generating… 0%", disable_notification=True)
    images = await _run_reporting_errors(
        status_message,
        "Generation",
        "repeat generation",
        repeat(client, full_params, on_progress=_make_progress_callback(status_message)),
    )
    if images is None:
        return

    await _deliver_generation_result(status_message, query.message, chat_id, storage, images)


async def generate_from_prompt_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Backs the "🎨 Generate" button attached to one of "🏷️ Analyze Image"'s two
    standalone prompt messages — generates from *that specific* derived
    prompt (WD14 tags or Qwen-VL caption), scoped by the prompt_id embedded
    in callback_data (see storage.py's `derived_prompt`), including that
    prompt's stored negative — empty for WD14 tags, a Qwen-VL caption's
    suggested negative otherwise (see `analyze_caption`). Uses the chat's
    current `/settings` override for the checkpoint the source image was
    generated with, rather than any stale params from that original
    generation."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    _, prompt_id = query.data.split(":", 1)
    storage: Storage = context.bot_data["storage"]
    stored = storage.get_derived_prompt(prompt_id)
    if stored is None:
        await query.answer("That prompt has expired — analyze the image again.", show_alert=True)
        return

    await query.answer()
    chat_id = stored["chat_id"]
    checkpoint = stored["checkpoint"]
    client: ComfyClient = context.bot_data["comfy_client"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]
    profile = resolve_profile(checkpoint, profiles)
    profile = apply_profile_override(profile, checkpoint, storage.get_override(chat_id, checkpoint))

    status_message = await query.message.reply_text("Generating… 0%", disable_notification=True)
    images = await _run_reporting_errors(
        status_message,
        "Generation",
        "generate-from-prompt",
        generate(
            client,
            checkpoint,
            stored["prompt"],
            profile,
            extra_negative_prompt=stored["negative_prompt"],
            on_progress=_make_progress_callback(status_message),
        ),
    )
    if images is None:
        return

    await _deliver_generation_result(status_message, query.message, chat_id, storage, images)


async def stream_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/stream [prompt]` — repeatedly generate a single image (batch_size
    forced to 1 regardless of the checkpoint's own default) from `prompt`
    and send each one immediately as it finishes, until either "/stop"
    cancels it or `STREAM_HARD_LIMIT` is reached. One stream per chat at a
    time; the running task lives in `context.bot_data["active_streams"]`
    keyed by chat_id — in-memory only, since a live asyncio task can't
    survive a bot restart anyway, unlike the sqlite-backed button
    registries in storage.py.

    With no prompt — notably, tapping the bare "/stream" button on
    `_MAIN_KEYBOARD` always sends exactly that, since a reply-keyboard
    button can only ever send fixed text, unlike an inline keyboard's
    `switch_inline_query` — this instead sets `context.chat_data`'s
    `awaiting_stream_prompt` flag, scoped to this message's forum topic via
    `topics.set_pending` (so starting a stream in one topic of a
    topics-enabled group can't be clobbered or wrongly answered by
    unrelated text in another), and asks for the prompt as a follow-up
    message (with a "❌ Cancel" button to back out — see
    `_stream_prompt_cancel_keyboard`/`stream_cancel_callback`), consumed by
    `_consume_awaiting_stream_prompt` (mirrors settings_menu's
    "awaiting_field" custom-value capture)."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    parts = (message.text or "").split(maxsplit=1)
    prompt_text = parts[1].strip() if len(parts) > 1 else ""
    if not prompt_text:
        set_pending(context.chat_data, "awaiting_stream_prompt", message, True)
        await message.reply_text(
            "What should the stream generate? Send the prompt as your next "
            "message (or type /stream <prompt> directly next time).",
            reply_markup=_stream_prompt_cancel_keyboard(),
        )
        return

    await _start_stream(context, update.effective_chat.id, message, prompt_text)


async def _start_stream(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, message: Message, prompt_text: str
) -> None:
    """Reserve `active_streams[chat_id]` and launch `_run_stream` — shared by
    `stream_command` (prompt given inline) and `_consume_awaiting_stream_prompt`
    (prompt given as a follow-up message)."""
    active_streams: dict[int, asyncio.Task] = context.bot_data.setdefault("active_streams", {})
    if chat_id in active_streams and not active_streams[chat_id].done():
        await message.reply_text("A stream is already running in this chat — /stop it first.")
        return

    # No await between the check above and this assignment — reserves the
    # slot synchronously so a second /stream sent in quick succession can't
    # also pass the check before this one claims it.
    active_streams[chat_id] = asyncio.create_task(
        _run_stream(context, chat_id, message, prompt_text)
    )


async def _consume_awaiting_stream_prompt(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """If this chat is mid-"send the /stream prompt" entry (see
    `stream_command`), consume the incoming text as that prompt and start
    the stream, returning True. Otherwise return False so the caller (
    `generate_message`) treats the text as a normal generation prompt
    instead."""
    message = update.effective_message
    if not pop_pending(context.chat_data, "awaiting_stream_prompt", message):
        return False

    prompt_text = (message_text(message) or "").strip()
    if not prompt_text:
        await message.reply_text("Cancelled — no prompt received.")
        return True

    await _start_stream(context, update.effective_chat.id, message, prompt_text)
    return True


async def _consume_awaiting_character_edit(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """If this chat is mid-"send the new prompt for '<name>'" entry (see
    `character_callback`'s "✏️ Edit" branch), consume the incoming text as
    that character's new prompt and save it, returning True. Otherwise
    return False so the caller (`generate_message`) treats the text as a
    normal generation prompt instead. Mirrors
    `_consume_awaiting_stream_prompt`."""
    message = update.effective_message
    name = pop_pending(context.chat_data, "awaiting_character_edit", message)
    if name is None:
        return False

    storage: Storage = context.bot_data["storage"]
    chat_id = update.effective_chat.id

    if storage.get_character(chat_id, name) is None:
        await message.reply_text(f"'{name}' no longer exists — nothing to edit.")
        return True

    body = (message_text(message) or "").strip()
    if not body:
        await message.reply_text("Cancelled — no prompt received.")
        return True

    parts = [p.strip() for p in body.split("|")]
    positive_prompt = parts[0]
    negative_prompt = parts[1] if len(parts) > 1 else ""
    if not positive_prompt:
        await message.reply_text("Positive prompt can't be empty — character unchanged.")
        return True

    storage.save_character(chat_id, name, positive_prompt, negative_prompt)
    await message.reply_text(f"Updated character '{name}'.")
    return True


async def _consume_awaiting_character_rename(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """If this chat is mid-"send the new name for '<name>'" entry (see
    `character_callback`'s "🔤 Rename" branch), consume the incoming text
    as that character's new name and apply it, returning True. Otherwise
    return False so the caller (`generate_message`) treats the text as a
    normal generation prompt instead. Mirrors
    `_consume_awaiting_character_edit`."""
    message = update.effective_message
    old_name = pop_pending(context.chat_data, "awaiting_character_rename", message)
    if old_name is None:
        return False

    storage: Storage = context.bot_data["storage"]
    chat_id = update.effective_chat.id

    if storage.get_character(chat_id, old_name) is None:
        await message.reply_text(f"'{old_name}' no longer exists — nothing to rename.")
        return True

    new_name = (message_text(message) or "").strip()
    if not new_name:
        await message.reply_text("Cancelled — no name received.")
        return True

    if not CHARACTER_NAME_RE.match(new_name):
        await message.reply_text(
            "Character names can only use letters, digits, '-' and '_' (max 32 chars) — "
            "character unchanged."
        )
        return True

    if new_name == old_name:
        await message.reply_text(f"'{old_name}' is already named that.")
        return True

    if storage.get_character(chat_id, new_name) is not None:
        await message.reply_text(
            f"A character named '{new_name}' already exists — pick a different name."
        )
        return True

    storage.rename_character(chat_id, old_name, new_name)
    await message.reply_text(f"Renamed '{old_name}' to '{new_name}'.")
    return True


async def stream_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the "❌ Cancel" tap on the "What should the stream generate?"
    follow-up prompt (see `stream_command`'s promptless branch and
    `_stream_prompt_cancel_keyboard`): clears `awaiting_stream_prompt` so
    the next text message goes back to being treated as a normal
    generation prompt, and edits the button away so a stale tap (the
    prompt was already sent and consumed, or a previous cancel already
    fired) can't be replayed."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    if not pop_pending(context.chat_data, "awaiting_stream_prompt", query.message):
        await query.answer("Nothing to cancel.")
        return

    await query.answer("Cancelled.")
    await _safe_edit_message(query, "Cancelled — no stream started.", InlineKeyboardMarkup([]))


async def _run_stream(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, message: Message, prompt_text: str
) -> None:
    """The full body of a "/stream" run, as a single background task so
    `stream_command` can reserve `active_streams[chat_id]` synchronously
    (see the comment there). Resolves checkpoint/profile/active-character
    exactly like `generate_message`, then generates one image at a time
    (forcing `batch_size=1` via `overrides`) and sends each as soon as it's
    ready, up to `STREAM_HARD_LIMIT` times or until `stop_command` cancels
    this task — which raises `asyncio.CancelledError` at whatever await
    this loop is currently sitting on (typically inside `generate()`'s
    `client.watch` progress loop)."""
    storage: Storage = context.bot_data["storage"]
    client: ComfyClient = context.bot_data["comfy_client"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]

    status_message: Message | None = None
    count = 0
    try:
        checkpoint = await _resolve_checkpoint_or_default(
            message, chat_id, storage, client, context
        )
        if checkpoint is None:
            return

        profile = resolve_profile(checkpoint, profiles)
        profile = apply_profile_override(
            profile, checkpoint, storage.get_override(chat_id, checkpoint)
        )

        active_character_name = storage.get_active_character_name(chat_id)
        character = (
            storage.get_character(chat_id, active_character_name) if active_character_name else None
        )
        effective_prompt, extra_negative = _resolve_effective_prompt(prompt_text, character)

        status_message = await message.reply_text(
            f"🔁 Streaming started (up to {STREAM_HARD_LIMIT} images) — "
            "tap /stop below (or send it) to end early.",
            reply_markup=_STREAMING_KEYBOARD,
            disable_notification=True,
        )

        for i in range(STREAM_HARD_LIMIT):
            count = i + 1
            images = await generate(
                client,
                checkpoint,
                effective_prompt,
                profile,
                extra_negative_prompt=extra_negative,
                overrides={"batch_size": 1},
                on_progress=_make_progress_callback(
                    status_message, label=f"Streaming {count}/{STREAM_HARD_LIMIT}"
                ),
            )
            await _send_and_store_result(message, chat_id, storage, images[0])
        await _finish_stream(
            status_message, f"Stream finished — hit the {STREAM_HARD_LIMIT}-image limit."
        )
    except asyncio.CancelledError:
        await _finish_stream(status_message, f"Stream stopped after {count} image(s).")
        raise
    except ComfyUIError as exc:
        await _finish_stream(
            status_message, f"Stream stopped after {count} image(s) — generation failed: {exc}"
        )
    except Exception:
        logger.exception("Unexpected error during stream")
        await _finish_stream(
            status_message, f"Stream stopped after {count} image(s) — unexpected error."
        )
    finally:
        context.bot_data.get("active_streams", {}).pop(chat_id, None)


async def _finish_stream(status_message: Message | None, text: str) -> None:
    """Report a stream's terminal state and swap `_STREAMING_KEYBOARD` back
    for `_MAIN_KEYBOARD`. Sent as a *new* message replying to
    `status_message` rather than an edit to it, because `editMessageText`
    can only ever set an inline keyboard — changing (or removing) a custom
    reply keyboard requires a newly sent message. A no-op if the stream
    never got far enough to create a status message (e.g. no checkpoint
    available), since `_STREAMING_KEYBOARD` was never installed in that
    case either."""
    if status_message is None:
        return
    await status_message.reply_text(text, reply_markup=_MAIN_KEYBOARD)


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/stop` — cancel this chat's running "/stream", if any (typed
    directly, or sent by tapping the "/stop" button `_STREAMING_KEYBOARD`
    installs for the duration of a stream). The actual "Stream stopped
    after N image(s)" confirmation comes from `_run_stream` catching the
    resulting `CancelledError` and reporting via `_finish_stream`; this just
    acknowledges the request."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    chat_id = update.effective_chat.id
    active_streams: dict[int, asyncio.Task] = context.bot_data.get("active_streams", {})
    task = active_streams.get(chat_id)
    if task is None or task.done():
        await update.effective_message.reply_text("No stream is running in this chat.")
        return

    task.cancel()
    await update.effective_message.reply_text("Stopping the stream…")


def _resolve_tag_sources(
    args_text: str, checkpoint: str | None, profiles: list[ModelProfile]
) -> tuple[list[TagSource], str]:
    """Split a leading "danbooru:"/"e621:" override off `args_text` if
    present (case-insensitive) and use it as-is; otherwise fall back to the
    resolved checkpoint's profile `tag_dictionary` (both sources if there's
    no checkpoint/profile, or the profile leaves it unset). Shared by
    `tags_command` and `tagcheck_command`. Pure function — no I/O, so it's
    unit-testable without a live checkpoint/profile lookup."""
    match = _TAG_SOURCE_PREFIX_RE.match(args_text)
    if match:
        return [TagSource(match.group(1).lower())], args_text[match.end() :]

    profile = resolve_profile(checkpoint, profiles) if checkpoint else None
    if profile is not None and profile.tag_dictionary is not None:
        return [TagSource(profile.tag_dictionary)], args_text
    return [TagSource.DANBOORU, TagSource.E621], args_text


def _tag_result_line(result: TagResult) -> str:
    label = category_label(result.source, result.category)
    line = f"{result.name} ({result.source.value}/{label}) — {result.post_count:,} posts"
    if result.matched_alias:
        line += f' [via alias "{result.matched_alias}"]'
    return line


def _tag_prompt_text(name: str) -> str:
    """Danbooru/e621 tag names are stored with underscores
    (`"blue_eyes"`), but checkpoints are trained on space-separated prompt
    text — pasting the raw underscored form in verbatim is a token
    mismatch the model won't recognize. Swap underscores for spaces for
    the copy-to-clipboard/prompt-insertion text; the display line
    (`_tag_result_line`) still shows the raw stored name, since that's
    what `tags_db.search`/`lookup_exact` actually match against."""
    return name.replace("_", " ")


def _tag_results_keyboard(results: list[TagResult]) -> InlineKeyboardMarkup:
    """One row per hit: a native tap-to-copy button (`CopyTextButton`, same
    pattern `_generate_from_prompt_keyboard` uses) for pasting the tag
    straight into the next prompt message. Tag names are always well under
    Telegram's 256-char `copy_text` cap, so unlike that keyboard, no
    length-gating is needed here."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"📋 {result.name}", copy_text=CopyTextButton(_tag_prompt_text(result.name))
                )
            ]
            for result in results
        ]
    )


def _no_tag_data_message() -> str:
    return "No tag data imported yet — run scripts/update_tag_db.py first (see README)."


async def tags_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/tags <query>` — search the local danbooru/e621 tag database to
    help build a prompt. Scoped to whichever dictionary the current chat's
    checkpoint profile prefers via `tag_dictionary` (both, if none/unset),
    overridable with a leading "danbooru:"/"e621:" on the query itself.
    Results are ranked purely by post_count (`search(..., by_frequency=True)`)
    since this is a "what are the most-used matching tags" listing capped
    to `tag_search_results` — unlike a "did you mean" hint, there's no
    reason for a rare prefix match to outrank a far more common substring
    one here. Each hit gets a native tap-to-copy button for pasting
    straight into the next prompt message. Doesn't require ComfyUI to be
    reachable — this is pure local sqlite lookup, so it works even before a
    model is selected."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    tags_db: TagDatabase = context.bot_data["tags_db"]
    if not any(tags_db.stats().values()):
        await message.reply_text(_no_tag_data_message())
        return

    raw = (message.text or "").split(maxsplit=1)
    args_text = raw[1].strip() if len(raw) > 1 else ""

    storage: Storage = context.bot_data["storage"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]
    checkpoint = storage.get_checkpoint(update.effective_chat.id)
    sources, query = _resolve_tag_sources(args_text, checkpoint, profiles)
    query = query.strip()
    if not query:
        await message.reply_text(
            'Usage: /tags <query> — optionally prefixed with "danbooru:" or "e621:" '
            'to search a specific dictionary, e.g. "/tags e621:fox"'
        )
        return

    results = tags_db.search(query, sources, limit=settings.tag_search_results, by_frequency=True)
    if not results:
        await message.reply_text(f"No tags found for {query!r}.")
        return

    header = f"Tags matching {query!r} ({'/'.join(s.value for s in sources)}):"
    await message.reply_text(
        header + "\n" + "\n".join(_tag_result_line(r) for r in results),
        reply_markup=_tag_results_keyboard(results),
    )


#: ComfyUI/A1111-style emphasis wrapping a whole tag — `(tag)`, `((tag))`,
#: `(tag:1.2)` to raise weight, `[tag]`/`[tag:0.9]` to lower it. Matched
#: separately per bracket kind (rather than one `[(\[]`-class pattern) so
#: mismatched wrapping like `(tag]` is deliberately left alone instead of
#: silently accepted.
_PAREN_WEIGHT_RE = re.compile(r"^\(+(?P<name>.+?)(?::[0-9]*\.?[0-9]+)?\)+$")
_BRACKET_WEIGHT_RE = re.compile(r"^\[+(?P<name>.+?)(?::[0-9]*\.?[0-9]+)?\]+$")


def _strip_prompt_weight(token: str) -> str:
    """Unwrap a comma-split prompt token's emphasis syntax, if any, down to
    the bare tag name — e.g. `"(yellow markings:1.2)"` -> `"yellow
    markings"`. Without this, `_tagcheck_lines` looks up the wrapper
    itself, which is never a real tag, and always reports it as unknown
    regardless of whether the tag inside it is valid."""
    match = _PAREN_WEIGHT_RE.match(token) or _BRACKET_WEIGHT_RE.match(token)
    return match.group("name").strip() if match else token


def _tagcheck_lines(
    prompt_text: str, sources: list[TagSource], tags_db: TagDatabase, settings: Settings
) -> list[str]:
    """Comma-split `prompt_text` (this codebase's usual booru-prompt
    convention) and check each token against `tags_db`: found & common
    (✅), found but rare — i.e. little training data behind it (⚠️), or not
    a known tag at all (❌, with a "did you mean" hint when a close match
    turns up). Emphasis syntax (see `_strip_prompt_weight`) is stripped
    before lookup but the token is still displayed as the user wrote it.
    Shared by `tagcheck_command` and `postprocess_callback`'s
    `SHOW_PROMPT_CALLBACK_KIND` branch."""
    tokens = [token for token in (t.strip() for t in prompt_text.split(",")) if token]
    omitted = max(0, len(tokens) - TAGCHECK_TOKEN_LIMIT)
    tokens = tokens[:TAGCHECK_TOKEN_LIMIT]

    lines = []
    for token in tokens:
        lookup_token = _strip_prompt_weight(token)
        result = tags_db.lookup_exact(lookup_token, sources)
        if result is None:
            hint = ""
            suggestions = tags_db.search(lookup_token, sources, limit=1)
            if suggestions:
                hint = f' — did you mean "{suggestions[0].name}"?'
            lines.append(f"❌ {token} — not a known tag{hint}")
        elif result.post_count < settings.tag_rare_threshold:
            lines.append(f"⚠️ {token} — rare ({result.post_count:,} posts, {result.source.value})")
        else:
            lines.append(f"✅ {token} — {result.post_count:,} posts ({result.source.value})")
    if omitted:
        lines.append(f"…{omitted} more token(s) omitted (limit {TAGCHECK_TOKEN_LIMIT}).")
    return lines


async def tagcheck_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/tagcheck <prompt>` — check each comma-separated tag in `prompt`
    against the tag database (see `_tagcheck_lines`). Same source
    resolution/override as `tags_command`."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    tags_db: TagDatabase = context.bot_data["tags_db"]
    if not any(tags_db.stats().values()):
        await message.reply_text(_no_tag_data_message())
        return

    raw = (message.text or "").split(maxsplit=1)
    args_text = raw[1].strip() if len(raw) > 1 else ""

    storage: Storage = context.bot_data["storage"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]
    checkpoint = storage.get_checkpoint(update.effective_chat.id)
    sources, prompt_text = _resolve_tag_sources(args_text, checkpoint, profiles)
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text(
            "Usage: /tagcheck <prompt> — checks each comma-separated tag against the "
            'tag database. Same "danbooru:"/"e621:" prefix override as /tags.'
        )
        return

    lines = _tagcheck_lines(prompt_text, sources, tags_db, settings)
    await message.reply_text("\n".join(lines))
