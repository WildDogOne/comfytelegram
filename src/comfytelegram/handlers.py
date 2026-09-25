"""Telegram-facing handlers: /start, /help, /model, plain-text generation
requests, and the inline-keyboard callbacks for model selection and
post-processing.

Shared objects (settings, the ComfyUI client, loaded profiles, storage)
live in `context.bot_data`, populated once at startup in `main.py`.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import aiohttp
from PIL import Image, ImageDraw, ImageFont
from telegram import (
    Bot,
    CopyTextButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
    Update,
    WebAppInfo,
)
from telegram.constants import InlineKeyboardButtonLimit
from telegram.error import BadRequest
from telegram.ext import Application, ContextTypes

from comfytelegram.analysis import (
    analyze_caption,
    analyze_caption_deep,
    analyze_tags,
)
from comfytelegram.auth import (
    reject_if_unauthorized,
    reject_if_unauthorized_callback,
    validate_webapp_init_data,
)
from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.generation import GeneratedImage, generate, post_process, repeat
from comfytelegram.message_text import message_text
from comfytelegram.params_serde import (
    deserialize_generation_params,
    serialize_generation_params,
)
from comfytelegram.png_metadata import (
    build_metadata,
    embed_metadata,
    extract_comfy_graph,
    extract_metadata,
    extract_seed,
    is_png,
    summarize_graph,
)
from comfytelegram.profiles import (
    ModelProfile,
    apply_profile_override,
    join_nonempty,
    resolve_profile,
)
from comfytelegram.settings import Settings
from comfytelegram.settings_menu import _safe_edit_message, handle_custom_value_message
from comfytelegram.storage import IMAGE_FORMAT_PNG, Storage
from comfytelegram.tags import TagDatabase, TagResult, TagSource, category_label
from comfytelegram.topics import pop_pending, set_pending
from comfytelegram.workflows import (
    DrawnMaskHandDetailerParams,
    GenerationParams,
    ManualHandDetailerParams,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

POSTPROCESS_KEYBOARD_LABELS = {
    "upscale": "🔍 Upscale 4x",
    "homogenize": "🧵 Homogenize",
    "face": "✨ Face Detail",
    "hand": "🖐️ Hand Detail",
}
POSTPROCESS_STATUS_LABELS = {
    "upscale": "Upscaling",
    "homogenize": "Homogenizing",
    "face": "Refining face",
    "hand": "Refining hand",
}
ANALYZE_ONLY_CALLBACK_KIND = "analyze_only"
DEEP_ANALYZE_CALLBACK_KIND = "deep_analyze"
SHOW_PROMPT_CALLBACK_KIND = "show_prompt"
#: "📥 Download file" — re-sends this image as a Telegram *document*
#: rather than a photo. Telegram re-encodes every `sendPhoto` upload to
#: JPEG and drops all PNG chunks with it, so the copy the user can save
#: out of the chat normally carries none of the metadata
#: `png_metadata.embed_metadata` wrote — only the document path
#: preserves the bytes. That makes this the button that actually enables
#: archiving: download the file it sends, and re-uploading it later
#: (again as a file, not a photo) restores the full post-processing
#: keyboard via `document_message`, no database row required.
ARCHIVE_CALLBACK_KIND = "archive"
ANALYZE_PROMPT_CALLBACK_KIND = "analyze_prompt"
UPSCALE_CONFIRM_CALLBACK_KIND = "upscale_confirmed"
UPSCALE_CANCEL_CALLBACK_KIND = "upscale_cancelled"
#: "🖐️ Hand Detail" doesn't post-process immediately — it asks auto vs.
#: manual first (see `_hand_mode_keyboard`), since the YOLO bbox detector
#: `HAND_AUTO_CALLBACK_KIND` runs often can't find a hand at all.
HAND_AUTO_CALLBACK_KIND = "hand_auto"
HAND_MANUAL_CALLBACK_KIND = "hand_manual"
#: A third `_hand_mode_keyboard` option, shown only when
#: `Settings.inpaint_relay_url` is configured — opens a Telegram WebApp
#: (see inpaint_relay/ at the repo root) for freehand mask drawing instead
#: of a fixed box. See `hand_draw_callback`/`poll_inpaint_jobs`.
HAND_DRAW_CALLBACK_KIND = "hand_draw"
#: Attached to a "🖌️ Draw Mask" result (both the first one, from
#: `_process_one_inpaint_job`, and every subsequent redo) — re-runs the
#: exact same source image and drawn mask through `post_process
#: (kind="hand_drawn")` again for a fresh seed/result, without having to
#: redraw the mask from scratch. See `storage.py`'s `inpaint_redo`.
HAND_REDO_CALLBACK_KIND = "hand_redo"
#: "🔁 x4" next to "🔁 Redo (same mask)" — runs `HAND_REDO_CALLBACK_KIND`'s
#: exact same re-run four times in a row (same source/mask, four fresh
#: seeds) instead of one, for pushing out a batch of quick revisions to
#: pick from without four separate taps. See `postprocess_callback`'s
#: `HAND_REDO4_CALLBACK_KIND`/`FIX_REDO4_CALLBACK_KIND` branch.
HAND_REDO4_CALLBACK_KIND = "hand_redo4"
#: General-purpose counterpart to `HAND_DRAW_CALLBACK_KIND`/
#: `HAND_REDO_CALLBACK_KIND` — "🩹 Fix Artifact" on `_post_process_keyboard`
#: itself rather than behind `_hand_mode_keyboard`'s submenu, since there's
#: no auto-detect/tap-a-point alternative for an arbitrary unwanted region
#: the way there is for a hand. Shares the exact same relay upload/poll/
#: redo machinery as the hand-drawn flow — see `_process_one_inpaint_job`,
#: `_run_drawn_mask_post_process`, `_send_drawn_mask_result_with_redo` —
#: differing only in which `post_process` kind ("fix_drawn" vs
#: "hand_drawn") and `DrawnMaskFixParams`/`DrawnMaskHandDetailerParams`
#: end up used, via `storage.py`'s `inpaint_job.kind` column.
FIX_DRAW_CALLBACK_KIND = "fix_draw"
FIX_REDO_CALLBACK_KIND = "fix_redo"
FIX_REDO4_CALLBACK_KIND = "fix_redo4"
#: "✏️ Detail Prompt" — a third relay-backed drawn-mask flow, alongside
#: "🖌️ Draw Mask"/"🩹 Fix Artifact": the WebApp editor it opens shows the
#: *same* canvas plus, on top of it, editable positive/negative/denoise
#: fields (`Job.mode="mask_prompt"` in inpaint_relay/server.py, vs. the
#: other two's plain `mode="mask"`) — draw the region, describe what should
#: happen there, tap Done once for both. What comes back
#: (`_process_one_inpaint_job`) runs immediately, exactly like the other
#: two — there's no save-for-later step. The prompt is deliberately
#: one-shot: it's specific to *that* mask (there's no way to say "five
#: fingers, no jewellery" about a region that hasn't been marked yet), so
#: unlike the scene-wide `GenerationParams.positive_prompt` nothing here
#: gets remembered on the image for a future, unrelated detailer tap — only
#: "🔁 Redo (same mask)" (`DETAIL_REDO_CALLBACK_KIND`) replays it, since a
#: redo is explicitly "run this exact mask+prompt again", and
#: `storage.py`'s `inpaint_redo` is where it's kept for that (see
#: `store_inpaint_redo`'s `detail_prompt`/`detail_negative_prompt`/
#: `detail_denoise`). Runs through Impact Pack's generic
#: `MaskToSEGS -> DetailerForEach` shape via `post_process(kind=
#: "hand_drawn")` — the same graph/params `DrawnMaskHandDetailerParams`
#: "🖌️ Draw Mask" already uses, reused here for an arbitrary region rather
#: than specifically a hand, since it's already generic (no bbox detector,
#: no hand-specific tuning beyond its default denoise/crop_factor) and its
#: moderate default denoise fits "refine per this description" better than
#: `DrawnMaskFixParams`' removal-oriented one. See
#: `generation.DETAIL_PROMPT_KINDS`.
DETAIL_PROMPT_CALLBACK_KIND = "detail_prompt"
DETAIL_REDO_CALLBACK_KIND = "detail_redo"
DETAIL_REDO4_CALLBACK_KIND = "detail_redo4"

#: What `storage.py`'s `inpaint_job.kind`/a drawn-mask redo tap actually
#: means in terms of `generation.post_process`'s API — looked up by
#: `_process_one_inpaint_job` (from the stored job row) and
#: `postprocess_callback`'s `*_REDO_CALLBACK_KIND`/`*_REDO4_CALLBACK_KIND`
#: branches (from which callback fired) so every end of the drawn-mask flow
#: shares one place that knows "hand" means `post_process(kind=
#: "hand_drawn")` plus a "Refining hand" label and
#: `HAND_REDO_CALLBACK_KIND`/`HAND_REDO4_CALLBACK_KIND`'s redo buttons,
#: "fix" means `"fix_drawn")`/"Fixing artifact"/`FIX_REDO_CALLBACK_KIND`/
#: `FIX_REDO4_CALLBACK_KIND`, and "detail" means `"hand_drawn")` again (see
#: `DETAIL_PROMPT_CALLBACK_KIND`) but labeled "Detailing" with its own
#: `DETAIL_REDO_CALLBACK_KIND`/`DETAIL_REDO4_CALLBACK_KIND` redo buttons.
_DRAWN_MASK_KINDS: dict[str, dict[str, str]] = {
    "hand": {
        "post_process_kind": "hand_drawn",
        "label": "Refining hand",
        "redo_callback_kind": HAND_REDO_CALLBACK_KIND,
        "redo4_callback_kind": HAND_REDO4_CALLBACK_KIND,
    },
    "fix": {
        "post_process_kind": "fix_drawn",
        "label": "Fixing artifact",
        "redo_callback_kind": FIX_REDO_CALLBACK_KIND,
        "redo4_callback_kind": FIX_REDO4_CALLBACK_KIND,
    },
    "detail": {
        "post_process_kind": "hand_drawn",
        "label": "Detailing",
        "redo_callback_kind": DETAIL_REDO_CALLBACK_KIND,
        "redo4_callback_kind": DETAIL_REDO4_CALLBACK_KIND,
    },
}
#: How many revisions "🔁 x4" produces in one tap.
DRAWN_MASK_REDO4_COUNT = 4
#: Timeout for the *small* outbound calls to inpaint_relay (poll, delete) —
#: it's a small, same-purpose-built service the bot fully controls the
#: deployment of, so a slow/unreachable relay should fail fast rather than
#: hang a poller tick or a button tap. These carry no request body worth
#: mentioning, so a short total is exactly right for them.
_INPAINT_RELAY_TIMEOUT = aiohttp.ClientTimeout(total=15)

#: `_relay_create_job`'s own timeout, which the short one above is *not*
#: right for: that POST's body is a full-resolution source PNG, routinely
#: ~20MB after an upscale, going to a relay on a different, public host.
#: `total` covers uploading the body too, so a 15s total demanded ~10
#: Mbit/s sustained and otherwise aborted mid-upload — seen live as
#: `asyncio.CancelledError` here and `starlette.requests.ClientDisconnect`
#: in the relay's log, three taps in a row, each failing at exactly 15.000s.
#: `connect` stays short so a genuinely unreachable relay still fails fast
#: (the property the shared timeout was really there for); `total` is
#: generous because the only thing it now bounds is an upload that has
#: stopped making progress.
_INPAINT_RELAY_UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=15)
#: Default grid resolution for "✋ Tap to mark" (see `_draw_hand_point_grid`/
#: `_hand_point_keyboard`) — coarse enough to keep the keyboard to
#: `HAND_POINT_GRID_SIZE` rows of `HAND_POINT_GRID_SIZE` buttons each. A hand
#: that lands on a cell corner (split across up to 4 tiles) can still miss
#: at this density — `HAND_POINT_GRID_SIZE_FINE`/`_hand_point_density_keyboard`
#: is the escape hatch for that.
HAND_POINT_GRID_SIZE = 4
#: Denser alternative grid, one "🔍 Finer grid" tap away (see
#: `_hand_point_keyboard`/`hand_point_density_callback`) — 8x8 still fits
#: Telegram's 8-buttons-per-row cap exactly, so it needs no extra paging.
HAND_POINT_GRID_SIZE_FINE = 8
#: `ManualHandDetailerParams.box_size_frac` is sized for `HAND_POINT_GRID_SIZE`
#: (deliberately bigger than one cell, for tap-precision margin — see its
#: docstring); scaled by `HAND_POINT_GRID_SIZE / grid_size` so the finer grid
#: marks a correspondingly smaller region instead of still inpainting ~35% of
#: the image regardless of which cell was tapped (see `hand_point_callback`).
HAND_POINT_BOX_SIZE_FRAC_BASE = ManualHandDetailerParams().box_size_frac * HAND_POINT_GRID_SIZE
#: Callback_data prefix for a tapped grid cell
#: (`hp:<result_id>:<grid_size>:<row>:<col>`) — its own namespace, separate
#: from `pp:`, so `postprocess_callback`'s `query.data.split(":", 2)` parsing
#: doesn't need to account for the extra grid_size/row/col fields (see
#: `hand_point_callback`, registered on this in main.py).
HAND_POINT_CALLBACK_PREFIX = "hp:"
#: Callback_data prefix for the "🔍 Finer grid" button
#: (`hpz:<result_id>:<grid_size>`) — re-renders the same source image at a
#: denser grid (see `hand_point_density_callback`, registered on this in
#: main.py). Its own namespace rather than folding into `HAND_POINT_CALLBACK_PREFIX`
#: since it carries no row/col.
HAND_POINT_DENSITY_CALLBACK_PREFIX = "hpz:"
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

# Long-edge cap (pixels) for the "✋ Tap to mark" grid preview (see
# `_draw_hand_point_grid`) — it's only there to pick a cell, not a final
# result, so it's downscaled before the grid/labels are drawn onto it. An
# already-upscaled source (e.g. a 4x-upscale run through Hand Detail again)
# can otherwise render a gridded PNG past TELEGRAM_PHOTO_SIZE_LIMIT, which
# `reply_photo` (unlike `_send_result_image`) has no oversize fallback for.
HAND_POINT_PREVIEW_MAX_DIM = 1280

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


#: `_post_process_keyboard`'s page-1 rows group like with like rather than
#: ranking by frequency: the two whole-frame operations (a 4x upscale
#: scales everything, and "🩹 Fix Artifact" is the general-purpose repair,
#: tied to no particular subject), then the region detailers, then the two
#: buttons that hand something back. The detailer row is the one that runs
#: three wide, because "✏️ Detail Prompt" is closely related to them (all
#: three end up refining one region of the image) and nowhere else fits
#: better; the rest stay at two per row, since a full row of
#: `POSTPROCESS_KEYBOARD_LABELS`-length labels is cramped on a phone
#: screen.
_DETAILER_KINDS = ("face", "hand")

#: Which page of `_post_process_keyboard` a `pp:` tap is asking to show.
#: The page rides in the callback data rather than in `chat_data`, so it
#: survives a restart exactly like every other button here, and two images
#: on screen can sit on different pages without interfering.
MORE_CALLBACK_KIND = "more"
BACK_CALLBACK_KIND = "back"

#: The redo buttons `_send_drawn_mask_result_with_redo` appends below the
#: standard keyboard. `_carried_extra_rows` re-attaches them when the page
#: is flipped, so a drawn-mask result doesn't lose them on the way to
#: page 2 and back — see `postprocess_callback`'s `MORE_CALLBACK_KIND`
#: branch, which has no other way to know a given result has them.
_REDO_CALLBACK_KINDS = frozenset(
    {
        HAND_REDO_CALLBACK_KIND,
        HAND_REDO4_CALLBACK_KIND,
        FIX_REDO_CALLBACK_KIND,
        FIX_REDO4_CALLBACK_KIND,
        DETAIL_REDO_CALLBACK_KIND,
        DETAIL_REDO4_CALLBACK_KIND,
    }
)


def _post_process_keyboard(result_id: str, page: int = 1) -> InlineKeyboardMarkup:
    """The keyboard attached to a generated/post-processed image, in two
    pages toggled in place by "⋯ More"/"‹ Back" (`MORE_CALLBACK_KIND`/
    `BACK_CALLBACK_KIND`), all scoped to `result_id` (see `storage.py`'s
    `pending_result`).

    Page 1 is what gets tapped while actually working an image: 🔍 Upscale
    4x and 🩹 Fix Artifact, then the ✨ Face / 🖐️ Hand detailers with
    ✏️ Detail Prompt beside them (a third, general-purpose region-detail
    action — draw a mask, describe it, run immediately — that belongs with
    the other two more than anywhere else on the keyboard), then
    📥 Download file and 🐛 Show Prompt. Page 2 holds the rest — 🧵 Homogenize
    and the three analyzers. The split is one tap deep and reversible, which is the
    point: ten buttons under every image (and every image *derived* from
    it, stacking down the scrollback) had turned a working keyboard into a
    wall.

    Two things that look like they belong together but don't:
    "🖐️ Hand Detail" doesn't post-process on this tap alone — see
    `postprocess_callback`'s `"hand"` branch for the auto/manual/draw choice
    it replies with instead. And "🩹 Fix Artifact" is shown unconditionally
    even when `Settings.inpaint_relay_url` isn't configured (unlike
    `_hand_mode_keyboard`'s "🖌️ Draw Mask" row, which is), because this
    keyboard has no access to `Settings` — every caller of
    `_send_and_store_result`/`_send_and_store_bot_result` would have to
    thread it through just for that — so `postprocess_callback`'s
    `FIX_DRAW_CALLBACK_KIND` branch checks and reports "not configured" at
    tap time instead."""
    if page == 2:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        POSTPROCESS_KEYBOARD_LABELS["homogenize"],
                        callback_data=f"pp:homogenize:{result_id}",
                    ),
                    InlineKeyboardButton(
                        "🏷️ Analyze Image",
                        callback_data=f"pp:{ANALYZE_ONLY_CALLBACK_KIND}:{result_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🔎 Deep Analyze",
                        callback_data=f"pp:{DEEP_ANALYZE_CALLBACK_KIND}:{result_id}",
                    ),
                    InlineKeyboardButton(
                        "🔬 Analyze Prompt",
                        callback_data=f"pp:{ANALYZE_PROMPT_CALLBACK_KIND}:{result_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "‹ Back", callback_data=f"pp:{BACK_CALLBACK_KIND}:{result_id}"
                    )
                ],
            ]
        )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    POSTPROCESS_KEYBOARD_LABELS["upscale"],
                    callback_data=f"pp:upscale:{result_id}",
                ),
                InlineKeyboardButton(
                    "🩹 Fix Artifact", callback_data=f"pp:{FIX_DRAW_CALLBACK_KIND}:{result_id}"
                ),
            ],
            [
                *(
                    InlineKeyboardButton(
                        POSTPROCESS_KEYBOARD_LABELS[kind], callback_data=f"pp:{kind}:{result_id}"
                    )
                    for kind in _DETAILER_KINDS
                ),
                InlineKeyboardButton(
                    "✏️ Detail Prompt",
                    callback_data=f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:{result_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📥 Download file", callback_data=f"pp:{ARCHIVE_CALLBACK_KIND}:{result_id}"
                ),
                InlineKeyboardButton(
                    "🐛 Show Prompt", callback_data=f"pp:{SHOW_PROMPT_CALLBACK_KIND}:{result_id}"
                ),
            ],
            [InlineKeyboardButton("⋯ More", callback_data=f"pp:{MORE_CALLBACK_KIND}:{result_id}")],
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


def _hand_mode_keyboard(result_id: str, settings: Settings) -> InlineKeyboardMarkup:
    """Attached to "🖐️ Hand Detail"'s first reply — lets the user pick
    auto-detection (today's YOLO/SAM behavior, `HAND_AUTO_CALLBACK_KIND`),
    tap a point themselves (`HAND_MANUAL_CALLBACK_KIND`) for when the
    detector can't find the hand at all, or (only when
    `settings.inpaint_relay_url` is configured) draw a freehand mask in a
    Telegram WebApp (`HAND_DRAW_CALLBACK_KIND`, see `hand_draw_callback`).
    See `postprocess_callback`."""
    row = [
        InlineKeyboardButton(
            "🤖 Auto-detect", callback_data=f"pp:{HAND_AUTO_CALLBACK_KIND}:{result_id}"
        ),
        InlineKeyboardButton(
            "✋ Tap to mark", callback_data=f"pp:{HAND_MANUAL_CALLBACK_KIND}:{result_id}"
        ),
    ]
    if settings.inpaint_relay_url:
        row.append(
            InlineKeyboardButton(
                "🖌️ Draw Mask", callback_data=f"pp:{HAND_DRAW_CALLBACK_KIND}:{result_id}"
            )
        )
    return InlineKeyboardMarkup([row])


def _drawn_mask_redo_button(result_id: str, redo_callback_kind: str) -> InlineKeyboardButton:
    """The extra keyboard row `_send_and_store_result`/`_send_and_store_bot_result`
    attach to a "🖌️ Draw Mask"/"🩹 Fix Artifact" result — `redo_callback_kind`
    is `HAND_REDO_CALLBACK_KIND` or `FIX_REDO_CALLBACK_KIND` (see
    `_DRAWN_MASK_KINDS`), so the button re-runs the same drawn-mask flow it
    came from rather than always assuming hand."""
    return InlineKeyboardButton(
        "🔁 Redo (same mask)", callback_data=f"pp:{redo_callback_kind}:{result_id}"
    )


def _drawn_mask_redo4_button(result_id: str, redo4_callback_kind: str) -> InlineKeyboardButton:
    """Sits next to `_drawn_mask_redo_button` in the same row — runs
    `DRAWN_MASK_REDO4_COUNT` redos in one tap instead of one. `redo4_callback_kind`
    is `HAND_REDO4_CALLBACK_KIND` or `FIX_REDO4_CALLBACK_KIND` (see
    `_DRAWN_MASK_KINDS`)."""
    return InlineKeyboardButton(
        f"🔁 x{DRAWN_MASK_REDO4_COUNT}", callback_data=f"pp:{redo4_callback_kind}:{result_id}"
    )


def _draw_hand_point_grid(image_bytes: bytes, grid_size: int = HAND_POINT_GRID_SIZE) -> bytes:
    """Overlay a `grid_size`x`grid_size` grid onto a copy of `image_bytes`,
    each cell labeled with the same row-letter/column-number text as its
    matching `_hand_point_keyboard` button (e.g. "B3"), so a cell can be
    identified without having to count grid lines. Uses
    `ImageFont.load_default(size=...)` — Pillow's built-in scalable bitmap
    font (no TTF file to locate/bundle) — with a black outline
    (`stroke_width`/`stroke_fill`) rather than a filled backing box behind
    it, so the label stays legible over any image content without covering
    much of it. Downscales first if the source exceeds
    `HAND_POINT_PREVIEW_MAX_DIM` on its long edge — this is only a preview
    for picking a cell, not a final result, and `hand_point_callback`
    re-derives the tapped point's pixel location from the *original*
    downloaded image anyway (see its `point_frac` math), so the preview's
    resolution has no bearing on where the eventual mask ends up. Returns
    PNG bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > HAND_POINT_PREVIEW_MAX_DIM:
        image.thumbnail((HAND_POINT_PREVIEW_MAX_DIM, HAND_POINT_PREVIEW_MAX_DIM), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_color = (255, 0, 0)
    line_width = max(1, min(width, height) // 400)
    cell_width = width / grid_size
    cell_height = height / grid_size
    for i in range(1, grid_size):
        x = round(width * i / grid_size)
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
        y = round(height * i / grid_size)
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    font_size = max(10, round(min(cell_width, cell_height) * 0.12))
    font = ImageFont.load_default(size=font_size)
    padding = max(2, font_size // 3)
    for row in range(grid_size):
        for col in range(grid_size):
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


def _hand_point_keyboard(
    result_id: str, grid_size: int = HAND_POINT_GRID_SIZE
) -> InlineKeyboardMarkup:
    """One button per grid cell drawn by `_draw_hand_point_grid`, labeled by
    row letter + column number (e.g. "B3") in reading order, callback_data
    `hp:<result_id>:<grid_size>:<row>:<col>` (see `hand_point_callback`).
    Below the grid, a "🔍 Finer grid" button (absent once already at
    `HAND_POINT_GRID_SIZE_FINE`) re-renders the same source at that higher
    density — for a hand that lands on a cell corner and is split across
    several tiles at this resolution (see `hand_point_density_callback`)."""
    rows = []
    for row in range(grid_size):
        row_letter = chr(ord("A") + row)
        rows.append(
            [
                InlineKeyboardButton(
                    f"{row_letter}{col + 1}",
                    callback_data=(
                        f"{HAND_POINT_CALLBACK_PREFIX}{result_id}:{grid_size}:{row}:{col}"
                    ),
                )
                for col in range(grid_size)
            ]
        )
    if grid_size < HAND_POINT_GRID_SIZE_FINE:
        rows.append(
            [
                InlineKeyboardButton(
                    f"🔍 Finer grid ({HAND_POINT_GRID_SIZE_FINE}×{HAND_POINT_GRID_SIZE_FINE})",
                    callback_data=(
                        f"{HAND_POINT_DENSITY_CALLBACK_PREFIX}{result_id}:"
                        f"{HAND_POINT_GRID_SIZE_FINE}"
                    ),
                )
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


def _raw_prompt_copy_text(raw_positive: str, raw_negative: str) -> str:
    """The exact text `postprocess_callback`'s `SHOW_PROMPT_CALLBACK_KIND`
    branch hands back for its "as typed" reply — no "Positive:"/
    "Negative:" labels, since this is meant to be 100% reusable as-is,
    whether grabbed via its "📋 Copy" button or selected by hand from the
    message text. `raw_positive` alone when there's no raw negative;
    otherwise `raw_positive` + this bot's own "---" block separator (see
    `_split_negative_prompt`) + `raw_negative`, so pasting the result back
    in as a new prompt message round-trips through that same split."""
    if raw_negative:
        return f"{raw_positive}\n---\n{raw_negative}"
    return raw_positive


async def _download_telegram_file(bot: Bot, file_id: str) -> bytes:
    """Download a Telegram file's bytes by its `file_id` — the `get_file` +
    `download_as_bytearray` two-step every post-processing/import entry
    point below needs before it can hand the source image to ComfyUI or
    Pillow."""
    tg_file = await bot.get_file(file_id)
    return bytes(await tg_file.download_as_bytearray())


#: `_fetch_source_image`'s user-facing heads-up when it has to fall back to
#: Telegram's copy — surfaced so a degraded (possibly JPEG-recompressed)
#: source doesn't look identical to the lossless ComfyUI one it usually
#: gets, in case the visible quality dip is otherwise puzzling.
_SOURCE_FALLBACK_NOTICE = (
    "ℹ️ ComfyUI no longer has the original file for this image — using the "
    "Telegram copy instead, which may already be JPEG-compressed."
)


async def _fetch_source_image(
    client: ComfyClient,
    bot: Bot,
    filename: str,
    file_id: str,
    *,
    on_fallback: Callable[[], Awaitable[None]] | None = None,
) -> bytes:
    """The source-image bytes for a post-processing pass: ComfyUI's own
    untouched output-directory copy (via `filename`, the same `/view`
    lookup `_send_archive_copy` uses) when it's still there, falling back
    to Telegram's `file_id` copy otherwise.

    Tried in that order because `_send_photo_or_document` JPEG-encodes
    every photo-sized send now — `file_id` is a lossy copy for almost every
    result, and a chain of post-processing passes on the same session's
    image (upscale -> face detail -> hand detail, ...) would otherwise
    compound that recompression at every step. The fallback (rather than
    treating a missing ComfyUI file as fatal) is what keeps a cleaned-up
    output directory, or a `document_message`-imported row whose `filename`
    may belong to an entirely different ComfyUI install, from hard-failing
    an otherwise-normal post-processing tap — see `_send_archive_copy`'s
    identical reasoning for why the ComfyUI copy can simply not be there.

    `on_fallback`, when given, is awaited right before that fallback
    download — every caller passes a closure that surfaces
    `_SOURCE_FALLBACK_NOTICE` through whatever reply mechanism it has
    (`query.message.reply_text`, or `bot.send_message` for
    `_process_one_inpaint_job`'s background task), so a quieter-than-usual
    result doesn't look like an unexplained regression."""
    try:
        return await client.get_image_bytes(filename, "", "output")
    except (ComfyUIError, aiohttp.ClientError, TimeoutError):
        if on_fallback is not None:
            await on_fallback()
        return await _download_telegram_file(bot, file_id)


def _source_fallback_notifier(message: Message) -> Callable[[], Awaitable[None]]:
    """`_fetch_source_image`'s `on_fallback` for every caller that has a
    `Message` to reply into (`postprocess_callback`, `hand_point_callback`,
    `hand_point_density_callback` — all callback-query handlers)."""

    async def _notify() -> None:
        await message.reply_text(_SOURCE_FALLBACK_NOTICE, disable_notification=True)

    return _notify


def _source_fallback_notifier_bot(
    bot: Bot, chat_id: int, message_thread_id: int | None
) -> Callable[[], Awaitable[None]]:
    """`_fetch_source_image`'s `on_fallback` for `_process_one_inpaint_job`,
    a background task with no `Message` to reply into —
    `message_thread_id` has to be passed through explicitly for the same
    reason `_send_and_store_bot_result` does."""

    async def _notify() -> None:
        await bot.send_message(
            chat_id,
            _SOURCE_FALLBACK_NOTICE,
            message_thread_id=message_thread_id,
            disable_notification=True,
        )

    return _notify


async def _send_photo_or_document(
    send_photo: Callable[..., Awaitable[Message]],
    send_document: Callable[..., Awaitable[Message]],
    data: bytes,
    filename: str,
    reply_markup: InlineKeyboardMarkup,
    *,
    use_jpeg: bool = True,
) -> Message:
    """Send `data` (raw PNG bytes out of ComfyUI) as a photo, JPEG-encoding
    it first (`_to_display_jpeg`) so the in-chat copy stays small regardless
    of resolution — a 4x upscale is ~19MB as PNG and a few MB as JPEG at
    `_DISPLAY_JPEG_QUALITY`, so this also keeps the overwhelming majority of
    sends, upscales included, under TELEGRAM_PHOTO_SIZE_LIMIT instead of
    falling back to the raw-PNG document path every time. That fallback is
    kept only for the rare case the JPEG re-encode itself still doesn't fit,
    or Pillow can't decode `data` at all (`_to_display_jpeg` returns it
    unchanged then).

    `use_jpeg=False` (the chat's `/settings` "🖼️ Display" toggle —
    `storage.get_image_format`, `IMAGE_FORMAT_PNG` — see
    `_send_and_store_result`/`_send_and_store_bot_result`) skips the
    re-encode entirely and sends the original PNG, still subject to the
    same size-based sendDocument fallback below: for someone running a
    long, multi-day session who'd rather pay the bandwidth than risk any
    recompression across a chain of post-processing passes.

    This is display-only — no metadata rides along, since `png_metadata`'s
    `tEXt` chunk is PNG-specific and this photo may not even be PNG bytes
    any more (irrespective of `use_jpeg`, since Telegram's own `sendPhoto`
    re-encodes to JPEG regardless). "📥 Download file"
    (`_send_archive_copy`) is the metadata-bearing, full-quality copy, and
    it doesn't use this path at all: it re-fetches the original PNG
    straight from ComfyUI instead of whatever was actually sent here.
    Likewise, post-processing sources its next pass's pixels via
    `_fetch_source_image` (ComfyUI first, this photo's `file_id` only as a
    fallback) rather than re-downloading what this function sent, so a
    chain of edits doesn't compound this JPEG recompression at every step
    even when `use_jpeg` is True.

    Shared by `_send_result_image` (`message.reply_*`) and
    `_send_and_store_bot_result` (`bot.send_*` against a bare chat_id, for
    code with no `Message` to reply into) so the size threshold and
    oversized-file caption live in one place. Returns the sent message —
    callers need it to read back the file_id Telegram assigned, for
    `_store_pending_result`."""
    if use_jpeg:
        jpeg = await asyncio.to_thread(_to_display_jpeg, data, _DISPLAY_JPEG_QUALITY)
        if jpeg is not data and len(jpeg) <= TELEGRAM_PHOTO_SIZE_LIMIT:
            return await send_photo(photo=io.BytesIO(jpeg), reply_markup=reply_markup)
    if len(data) <= TELEGRAM_PHOTO_SIZE_LIMIT:
        return await send_photo(photo=io.BytesIO(data), reply_markup=reply_markup)
    return await send_document(
        document=io.BytesIO(data),
        filename=filename,
        caption="Sent as a file — too large for Telegram's photo size limit (10MB).",
        reply_markup=reply_markup,
    )


async def _send_result_image(
    message: Message,
    data: bytes,
    filename: str,
    reply_markup: InlineKeyboardMarkup,
    *,
    use_jpeg: bool = True,
) -> Message:
    """`_send_photo_or_document` bound to `message.reply_photo`/
    `reply_document`."""
    return await _send_photo_or_document(
        lambda **kw: message.reply_photo(**kw),
        lambda **kw: message.reply_document(**kw),
        data,
        filename,
        reply_markup,
        use_jpeg=use_jpeg,
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


#: Moved to `params_serde` so `generation.py` can embed the same dict into
#: each image's PNG metadata chunk without importing `handlers` (which
#: imports *it*). Kept under the old private names here because this
#: module's call sites — and the tests pinning their behaviour — reference
#: them by those names.
_serialize_generation_params = serialize_generation_params
_deserialize_generation_params = deserialize_generation_params


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


def _post_process_keyboard_with_extra_rows(
    result_id: str,
    extra_keyboard_rows: list[list[InlineKeyboardButton]] | None,
    page: int = 1,
) -> InlineKeyboardMarkup:
    """`_post_process_keyboard(result_id, page)` plus any
    `extra_keyboard_rows` appended below its own — shared by
    `_send_and_store_result`/`_send_and_store_bot_result` so both send the
    same keyboard shape for a given `result_id`/`extra_keyboard_rows` pair,
    and by the page toggle, which re-appends the same rows to the other
    page (see `_carried_extra_rows`)."""
    keyboard = _post_process_keyboard(result_id, page)
    if not extra_keyboard_rows:
        return keyboard
    return InlineKeyboardMarkup(list(keyboard.inline_keyboard) + extra_keyboard_rows)


def _carried_extra_rows(
    markup: InlineKeyboardMarkup | None,
) -> list[list[InlineKeyboardButton]]:
    """The extra rows on an existing post-processing keyboard that have to
    survive a page flip — today just the "🔁 Redo (same mask)"/"🔁 x4" pair
    a drawn-mask result carries (`_REDO_CALLBACK_KINDS`).

    They're recovered by reading the message's current keyboard rather than
    looked up, because nothing else knows they're there: `inpaint_redo`
    records the mask but not which of the two drawn-mask kinds produced it,
    so the buttons can't be rebuilt from storage alone. A row counts only
    if *every* button in it is a redo button, so a future extra row can't
    be half-carried."""
    if markup is None:
        return []
    return [
        list(row) for row in markup.inline_keyboard if row and all(_is_redo_button(b) for b in row)
    ]


def _is_redo_button(button: InlineKeyboardButton) -> bool:
    """True for a `pp:<redo kind>:<result_id>` button — see
    `_REDO_CALLBACK_KINDS`. A `CopyTextButton`/`WebAppInfo` button has no
    `callback_data` at all, hence the None check."""
    data = button.callback_data
    if not data:
        return False
    parts = data.split(":")
    return len(parts) > 1 and parts[1] in _REDO_CALLBACK_KINDS


async def _send_and_store_result(
    message: Message,
    chat_id: int,
    storage: Storage,
    img: GeneratedImage,
    *,
    result_id: str | None = None,
    extra_keyboard_rows: list[list[InlineKeyboardButton]] | None = None,
) -> str:
    """Send a generated image with its post-processing keyboard, then persist
    what those buttons need (a re-downloadable file_id + the params to build
    the next graph) so they still work after a bot restart — see storage.py.
    `result_id` defaults to a fresh uuid like every button here, but a
    caller that needs to know it ahead of time to embed in its own extra
    button (`extra_keyboard_rows`, appended below the standard ones — see
    `postprocess_callback`'s `HAND_REDO_CALLBACK_KIND` branch) can pass one
    in instead. Returns whichever id was actually used."""
    result_id = result_id or uuid.uuid4().hex[:12]
    reply_markup = _post_process_keyboard_with_extra_rows(result_id, extra_keyboard_rows)
    use_jpeg = storage.get_image_format(chat_id) != IMAGE_FORMAT_PNG
    sent = await _send_result_image(
        message, img.data, img.filename, reply_markup, use_jpeg=use_jpeg
    )
    storage.store_pending_result(
        result_id,
        chat_id,
        _extract_file_id(sent),
        img.filename,
        _serialize_generation_params(img.full_params),
    )
    return result_id


async def _send_and_store_bot_result(
    bot: Bot,
    chat_id: int,
    storage: Storage,
    img: GeneratedImage,
    *,
    message_thread_id: int | None = None,
    result_id: str | None = None,
    extra_keyboard_rows: list[list[InlineKeyboardButton]] | None = None,
) -> str:
    """`_send_and_store_result`'s counterpart for code with no `Message` to
    reply into — `poll_inpaint_jobs` runs as a background task, not in
    response to an update, so it sends straight at `chat_id` via `bot.
    send_photo`/`send_document` instead of `message.reply_photo`/
    `reply_document`. `message_thread_id` must be passed through explicitly
    for the same reason: unlike `message.reply_*` (which Telegram routes
    into the replied-to message's own forum topic automatically), a bare
    `chat_id` send has no topic context of its own and defaults to the
    chat's General topic otherwise. `result_id`/`extra_keyboard_rows` mirror
    `_send_and_store_result`'s — see there."""
    result_id = result_id or uuid.uuid4().hex[:12]
    reply_markup = _post_process_keyboard_with_extra_rows(result_id, extra_keyboard_rows)
    use_jpeg = storage.get_image_format(chat_id) != IMAGE_FORMAT_PNG
    sent = await _send_photo_or_document(
        lambda **kw: bot.send_photo(chat_id, message_thread_id=message_thread_id, **kw),
        lambda **kw: bot.send_document(chat_id, message_thread_id=message_thread_id, **kw),
        img.data,
        img.filename,
        reply_markup,
        use_jpeg=use_jpeg,
    )
    storage.store_pending_result(
        result_id,
        chat_id,
        _extract_file_id(sent),
        img.filename,
        _serialize_generation_params(img.full_params),
    )
    return result_id


#: JPEG quality for the mask editor's reference copy — see
#: `_to_display_jpeg`. High enough that the user is painting over something
#: that looks like the real image, low enough to be worth the round trip.
_RELAY_DISPLAY_JPEG_QUALITY = 85

#: JPEG quality for the in-chat display copy of a generated/post-processed
#: image (`_send_photo_or_document`) — higher than the relay's reference
#: copy above, since this is the finished artwork the user is actually
#: judging, not a rough backdrop to paint over.
_DISPLAY_JPEG_QUALITY = 92


def _to_display_jpeg(source: bytes, quality: int = _RELAY_DISPLAY_JPEG_QUALITY) -> bytes:
    """Re-encode a source image as a JPEG at its *original* pixel
    dimensions, at the given `quality` — used both for `_relay_create_job`'s
    upload to the mask editor and for `_send_photo_or_document`'s in-chat
    display copy.

    Dimensions are deliberately untouched: for the relay, its mask editor
    sizes its mask canvas off `image.naturalWidth/Height`, and
    `post_process` scales that mask to the real image, so resizing here
    would silently change the geometry the mask comes back in — no such
    constraint applies to the chat-display use, but there's no reason to
    treat it differently. Either way this copy is never fed back into a
    ComfyUI graph: the relay mask is composited against the full-quality
    original re-downloaded from `pending_result`'s `file_id`, and
    post-processing sources its next pass from `_fetch_source_image`
    (ComfyUI's own copy first), never from what this function returns.

    Worth doing because the difference is not marginal: a 4096x4096 upscale
    is ~19MB as PNG and a low-single-digit number of MB as JPEG at either
    quality level here. Returns `source` unchanged if Pillow can't decode
    it — callers treat that as "couldn't re-encode, send the original
    instead" rather than a hard failure.
    """
    try:
        with Image.open(io.BytesIO(source)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=quality)
    except Exception:
        logger.warning("Couldn't re-encode source as JPEG; using it as-is", exc_info=True)
        return source
    return buf.getvalue()


async def _relay_create_job(
    settings: Settings, image_bytes: bytes, *, meta: dict[str, Any] | None = None
) -> str:
    """POST the source image to inpaint_relay's `POST /jobs`, returning the
    job token it assigns. Raises on any HTTP/network failure — the caller
    (`postprocess_callback`'s `HAND_DRAW_CALLBACK_KIND`/`FIX_DRAW_CALLBACK_KIND`/
    `DETAIL_PROMPT_CALLBACK_KIND` branches) reports that back to the chat
    like any other post-processing error.

    `meta`, when given, becomes the relay's `X-Job-Meta` header — base64'd
    JSON rather than raw text, so it can't run into HTTP header encoding
    rules (headers are meant to stay ASCII/latin-1). Omitted entirely for
    "🖌️ Draw Mask"/"🩹 Fix Artifact" (the relay defaults an unset header to
    `mode="mask"`, a canvas with no prompt fields);
    `DETAIL_PROMPT_CALLBACK_KIND` is the one caller that passes one,
    carrying `mode="mask_prompt"` (the same canvas plus editable
    positive/negative/denoise fields, submitted together with the mask —
    see `_relay_poll_result`) and read-only reference settings to display
    (`_detail_prompt_readonly_info`).

    The image is compressed here (`_to_display_jpeg`) rather than on the
    relay, which is where this used to happen: the relay re-encoded on
    arrival, so the full PNG crossed the internet only to be thrown away at
    the far end. Encoding first makes the upload ~8x smaller for the same
    bytes served to the editor. `Content-Type` tells the relay what it
    actually got, since it no longer re-encodes and has to serve this copy
    back verbatim. The decode+encode itself runs off the event loop
    (`asyncio.to_thread`) — a full-resolution JPEG re-encode is real CPU
    work, and `concurrent_updates(True)` only actually gets other updates
    running concurrently if nothing blocks the one event loop thread."""
    payload = await asyncio.to_thread(_to_display_jpeg, image_bytes)
    content_type = "image/jpeg" if payload is not image_bytes else "image/png"
    headers = {
        "Authorization": f"Bearer {settings.inpaint_relay_shared_secret}",
        "Content-Type": content_type,
    }
    if meta is not None:
        headers["X-Job-Meta"] = base64.b64encode(json.dumps(meta).encode()).decode("ascii")
    async with (
        aiohttp.ClientSession(timeout=_INPAINT_RELAY_UPLOAD_TIMEOUT) as session,
        session.post(
            f"{settings.inpaint_relay_url}/jobs",
            data=payload,
            headers=headers,
        ) as resp,
    ):
        resp.raise_for_status()
        payload = await resp.json()
    return payload["token"]


async def _relay_poll_result(settings: Settings, token: str) -> dict[str, Any] | None:
    """GET a job's status from inpaint_relay's `GET /jobs/{token}/result`.
    Returns None if it's still waiting on a submission (`status="pending"`)
    or the relay no longer knows about it at all (404 — e.g. a relay
    restart lost its in-memory job store, see inpaint_relay's own docs), or
    `{"mask": <bytes>, "positive": <str|None>, "negative": <str|None>,
    "denoise": <float|None>, "init_data": <str>}` once submitted — every
    job submits a mask (both "mask" and "mask_prompt" jobs draw one);
    `positive`/`negative`/`denoise` are only ever non-None for a
    `DETAIL_PROMPT_CALLBACK_KIND` job's `mode="mask_prompt"` submission, so
    `_process_one_inpaint_job` can pass all three straight into
    `post_process` unconditionally — they're simply always `None` for
    "🖌️ Draw Mask"/"🩹 Fix Artifact"."""
    async with (
        aiohttp.ClientSession(timeout=_INPAINT_RELAY_TIMEOUT) as session,
        session.get(
            f"{settings.inpaint_relay_url}/jobs/{token}/result",
            headers={"Authorization": f"Bearer {settings.inpaint_relay_shared_secret}"},
        ) as resp,
    ):
        if resp.status == 404:
            return None
        resp.raise_for_status()
        payload = await resp.json()
    if payload.get("status") != "submitted":
        return None
    return {
        "mask": base64.b64decode(payload["mask_base64"]),
        "positive": payload.get("positive"),
        "negative": payload.get("negative"),
        "denoise": payload.get("denoise"),
        "init_data": payload["init_data"],
    }


async def _relay_delete_job(settings: Settings, token: str) -> None:
    """Best-effort cleanup of a consumed job on inpaint_relay — a failure
    here just means the relay's own TTL sweep clears it out later instead,
    so it's logged rather than raised."""
    try:
        async with aiohttp.ClientSession(timeout=_INPAINT_RELAY_TIMEOUT) as session:
            await session.delete(
                f"{settings.inpaint_relay_url}/jobs/{token}",
                headers={"Authorization": f"Bearer {settings.inpaint_relay_shared_secret}"},
            )
    except Exception:
        logger.warning("Failed to delete inpaint_relay job %s", token, exc_info=True)


async def _run_drawn_mask_post_process(
    client: ComfyClient,
    status_message: Message,
    source_bytes: bytes,
    source_filename: str,
    full_params: GenerationParams,
    mask_bytes: bytes,
    *,
    post_process_kind: Literal["hand_drawn", "fix_drawn"],
    label: str,
    profiles: list[ModelProfile],
    detail_prompt: str | None = None,
    detail_negative_prompt: str | None = None,
    detail_denoise: float | None = None,
) -> GeneratedImage | None:
    """Shared by `_process_one_inpaint_job` (a freshly submitted mask) and
    `postprocess_callback`'s `*_REDO_CALLBACK_KIND` branch (re-running a
    previously drawn one against a fresh seed — see storage.py's
    `inpaint_redo` — for when a detailer result comes out badly and
    redrawing the whole mask from scratch would be overkill) — both need
    the exact same `post_process` call, status-message error reporting, and
    status-message cleanup around it. `post_process_kind`/`label` come from
    `_DRAWN_MASK_KINDS` ("hand"/"fix"/"detail"). `profiles` only matters for
    `post_process_kind="fix_drawn"` — see `generation.
    _fix_artifact_override_base` — passed through unconditionally since
    "hand_drawn" ignores it. `detail_prompt`/`detail_negative_prompt`/
    `detail_denoise` are "✏️ Detail Prompt"'s one-shot mask+prompt override
    (see `DETAIL_PROMPT_CALLBACK_KIND`) — `None`/`None`/`None` for
    "🖌️ Draw Mask"/"🩹 Fix Artifact", which never collect a prompt at all.
    Returns None on failure (already reported into `status_message` by
    `_run_reporting_errors`); callers should treat that as "stop here",
    same as `_run_reporting_errors` itself."""
    generated = await _run_reporting_errors(
        status_message,
        label,
        "drawn-mask post-processing",
        post_process(
            client,
            post_process_kind,
            source_bytes,
            source_filename,
            full_params,
            mask_bytes=mask_bytes,
            detail_prompt=detail_prompt,
            detail_negative_prompt=detail_negative_prompt,
            denoise=detail_denoise,
            profiles=profiles,
        ),
    )
    if generated is not None:
        await status_message.delete()
    return generated


async def _send_drawn_mask_result_with_redo(
    send: Callable[..., Awaitable[str]],
    storage: Storage,
    source_file_id: str,
    source_filename: str,
    mask_bytes: bytes,
    generated: GeneratedImage,
    *,
    redo_callback_kind: str,
    redo4_callback_kind: str,
    detail_prompt: str | None = None,
    detail_negative_prompt: str | None = None,
    detail_denoise: float | None = None,
) -> None:
    """Send a `post_process(kind="hand_drawn"/"fix_drawn")` result with
    "🔁 Redo (same mask)"/"🔁 x4" buttons attached (one row), and persist what
    they need to run again (`storage.py`'s `inpaint_redo`) — shared by
    `_process_one_inpaint_job` and `postprocess_callback`'s
    `*_REDO_CALLBACK_KIND`/`*_REDO4_CALLBACK_KIND` branches, which differ
    only in *how* they send (`send` is `_send_and_store_bot_result` or
    `_send_and_store_result`, already bound to everything but `img`,
    `result_id` and `extra_keyboard_rows`) and which redo buttons
    (`redo_callback_kind`/`redo4_callback_kind`, from `_DRAWN_MASK_KINDS`)
    the result should carry. Every result carries both, including each of
    the `DRAWN_MASK_REDO4_COUNT` results a "🔁 x4" tap itself produces — the
    chain never drops back to single-redo-only. `detail_prompt`/
    `detail_negative_prompt`/`detail_denoise` are "✏️ Detail Prompt"'s
    one-shot override (`None` for "🖌️ Draw Mask"/"🩹 Fix Artifact") —
    stored alongside the mask in `inpaint_redo` rather than passed to
    `send`, since it's tied to *this specific mask*, not to the image in
    general (a redo replays both together; a fresh detail/draw-mask/
    fix-artifact tap always starts from nothing, whatever prompt a sibling
    result's redo happens to carry)."""
    new_result_id = uuid.uuid4().hex[:12]
    await send(
        generated,
        result_id=new_result_id,
        extra_keyboard_rows=[
            [
                _drawn_mask_redo_button(new_result_id, redo_callback_kind),
                _drawn_mask_redo4_button(new_result_id, redo4_callback_kind),
            ]
        ],
    )
    storage.store_inpaint_redo(
        new_result_id,
        source_file_id,
        source_filename,
        mask_bytes,
        detail_prompt=detail_prompt,
        detail_negative_prompt=detail_negative_prompt,
        detail_denoise=detail_denoise,
    )


async def _run_one_drawn_mask_redo(
    reply_message: Message,
    client: ComfyClient,
    storage: Storage,
    chat_id: int,
    full_params: GenerationParams,
    redo: dict[str, Any],
    drawn_mask_kind: dict[str, str],
    profiles: list[ModelProfile],
    source_bytes: bytes,
) -> bool:
    """One "🔁 Redo (same mask)"/"🔁 x4" iteration: its own status message,
    `post_process` call, and result send (with fresh redo buttons of its
    own, so the chain keeps going indefinitely either way) — shared by
    `postprocess_callback`'s `*_REDO_CALLBACK_KIND`/`*_REDO4_CALLBACK_KIND`
    branch, which calls this once or `DRAWN_MASK_REDO4_COUNT` times
    depending on which button was tapped. `source_bytes` is downloaded once
    by the caller and reused across every iteration, rather than re-fetched
    per call. The one-shot `detail_prompt`/`detail_negative_prompt`/
    `detail_denoise` a "✏️ Detail Prompt" redo replays come from `redo`
    itself (`storage.py`'s `inpaint_redo`, stored alongside the mask at
    submission time — see `_send_drawn_mask_result_with_redo`), not a
    caller-supplied override; they're simply absent for a "🖌️ Draw Mask"/
    "🩹 Fix Artifact" redo. Returns True on success, False on failure
    (already reported into that iteration's own status message by
    `_run_reporting_errors`) — the caller stops the loop on the first False
    rather than continuing to burn ComfyUI time on a source/mask
    combination that just failed."""
    detail_prompt = redo.get("detail_prompt")
    detail_negative_prompt = redo.get("detail_negative_prompt")
    detail_denoise = redo.get("detail_denoise")
    status_message = await reply_message.reply_text(
        f"{drawn_mask_kind['label']} (drawn mask)…", disable_notification=True
    )
    generated = await _run_drawn_mask_post_process(
        client,
        status_message,
        source_bytes,
        redo["source_filename"],
        full_params,
        redo["mask_png"],
        post_process_kind=drawn_mask_kind["post_process_kind"],
        label=drawn_mask_kind["label"],
        profiles=profiles,
        detail_prompt=detail_prompt,
        detail_negative_prompt=detail_negative_prompt,
        detail_denoise=detail_denoise,
    )
    if generated is None:
        return False
    await _send_drawn_mask_result_with_redo(
        lambda img, **kw: _send_and_store_result(reply_message, chat_id, storage, img, **kw),
        storage,
        redo["source_file_id"],
        redo["source_filename"],
        redo["mask_png"],
        generated,
        redo_callback_kind=drawn_mask_kind["redo_callback_kind"],
        redo4_callback_kind=drawn_mask_kind["redo4_callback_kind"],
        detail_prompt=detail_prompt,
        detail_negative_prompt=detail_negative_prompt,
        detail_denoise=detail_denoise,
    )
    return True


async def _process_one_inpaint_job(
    application: Application,
    settings: Settings,
    storage: Storage,
    client: ComfyClient,
    job: dict[str, Any],
) -> None:
    """One `poll_inpaint_jobs` tick's worth of work for a single
    outstanding job: check the relay, and if a mask has been drawn,
    validate it actually came from Telegram (see `auth.
    validate_webapp_init_data` — the relay itself can't check this, since
    it never holds the bot token) before running it through `post_process`
    and posting the result — `job["kind"]` ("hand"/"fix"/"detail", see
    `storage.py`'s `inpaint_job.kind`) resolves via `_DRAWN_MASK_KINDS` to
    which `post_process` kind, status label, and redo button apply; one
    unified path handles all three, since a "detail" job is just a mask
    job whose result also carries a one-shot `positive`/`negative`/
    `denoise` (`None` for the other two — see `_relay_poll_result`), passed
    straight into `post_process` alongside it. Every path past that
    validation step (including failure) sends *something* into the chat —
    this runs unattended, so silently dropping a job on error would leave
    someone staring at the "Draw over the area..."/"Uploading image..."
    message forever with no idea whether it worked."""
    token = job["token"]
    chat_id = job["chat_id"]
    message_thread_id = job["message_thread_id"]
    drawn_mask_kind = _DRAWN_MASK_KINDS[job["kind"]]
    result = await _relay_poll_result(settings, token)
    if result is None:
        return  # still pending, or the relay lost track of it — try again next tick

    # Either way past this point, comfytelegram is done with this job —
    # clear it locally and tell the relay it can forget it too.
    storage.delete_inpaint_job(token)
    await _relay_delete_job(settings, token)

    verified = validate_webapp_init_data(result["init_data"], settings.telegram_bot_token)
    if verified is None:
        logger.warning("Dropping inpaint_relay job %s: invalid initData signature", token)
        await application.bot.send_message(
            chat_id,
            "⚠️ Couldn't verify the editor submission — please try again.",
            message_thread_id=message_thread_id,
        )
        return

    pending = storage.get_pending_result(job["result_id"])
    if pending is None:
        logger.warning(
            "Dropping inpaint_relay job %s: its source pending_result has expired", token
        )
        await application.bot.send_message(
            chat_id,
            "⚠️ That image has expired — generate a new one before drawing a mask.",
            message_thread_id=message_thread_id,
        )
        return

    status_message = await application.bot.send_message(
        chat_id,
        f"{drawn_mask_kind['label']} (drawn mask)…",
        disable_notification=True,
        message_thread_id=message_thread_id,
    )
    full_params = _deserialize_generation_params(pending["base_params"])
    detail_prompt = result.get("positive")
    detail_negative_prompt = result.get("negative")
    detail_denoise = result.get("denoise")
    source_bytes = await _fetch_source_image(
        client,
        application.bot,
        pending["filename"],
        pending["file_id"],
        on_fallback=_source_fallback_notifier_bot(application.bot, chat_id, message_thread_id),
    )

    generated = await _run_drawn_mask_post_process(
        client,
        status_message,
        source_bytes,
        pending["filename"],
        full_params,
        result["mask"],
        post_process_kind=drawn_mask_kind["post_process_kind"],
        label=drawn_mask_kind["label"],
        profiles=application.bot_data["profiles"],
        detail_prompt=detail_prompt,
        detail_negative_prompt=detail_negative_prompt,
        detail_denoise=detail_denoise,
    )
    if generated is None:
        return
    await _send_drawn_mask_result_with_redo(
        lambda img, **kw: _send_and_store_bot_result(
            application.bot,
            chat_id,
            storage,
            img,
            message_thread_id=message_thread_id,
            **kw,
        ),
        storage,
        pending["file_id"],
        pending["filename"],
        result["mask"],
        generated,
        redo_callback_kind=drawn_mask_kind["redo_callback_kind"],
        redo4_callback_kind=drawn_mask_kind["redo4_callback_kind"],
        detail_prompt=detail_prompt,
        detail_negative_prompt=detail_negative_prompt,
        detail_denoise=detail_denoise,
    )


async def poll_inpaint_jobs(application: Application) -> None:
    """Background task (started in main.py's `_post_init`, cancelled in
    `_post_shutdown` — same pattern as the tag-db refresh task): every
    `settings.inpaint_poll_interval_seconds`, checks inpaint_relay for each
    outstanding "🖌️ Draw Mask" job (see storage.py's `inpaint_job` table).
    The relay can't push to comfytelegram directly — the home box isn't
    reachable from the public internet, only the relay's public host is —
    so this is outbound polling, the same posture `run_polling()` already
    uses against Telegram itself. Runs until cancelled; each job's failures
    are isolated so one bad job (an unreachable relay, an expired
    pending_result) doesn't stop the rest from being checked."""
    settings: Settings = application.bot_data["settings"]
    storage: Storage = application.bot_data["storage"]
    while True:
        await asyncio.sleep(settings.inpaint_poll_interval_seconds)
        client: ComfyClient = application.bot_data["comfy_client"]
        for job in storage.list_inpaint_jobs():
            try:
                await _process_one_inpaint_job(application, settings, storage, client, job)
            except Exception:
                logger.exception("Error polling inpaint_relay job %s", job["token"])


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


async def _send_tag_and_caption_prompts(
    reply_target: Message,
    storage: Storage,
    chat_id: int,
    checkpoint: str,
    caption_label: str,
    tags: str | None,
    caption: tuple[str, str] | None,
) -> None:
    """Send the WD14-tags and Qwen-VL-caption derived prompts as a pair via
    `_send_derived_prompt` — every side-by-side analysis call site
    (`photo_message`, `_import_without_metadata`, `postprocess_callback`'s
    `ANALYZE_ONLY_CALLBACK_KIND`) sends this same pair once its `(tags,
    caption)` result comes back; only `caption_label` differs between them
    (quick vs. deep vs. imported-image)."""
    caption_positive, caption_negative = caption if caption is not None else (None, "")
    await _send_derived_prompt(reply_target, storage, chat_id, checkpoint, "🏷️ WD14 tags", tags)
    await _send_derived_prompt(
        reply_target,
        storage,
        chat_id,
        checkpoint,
        caption_label,
        caption_positive,
        caption_negative,
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


def _checkpoint_labels(checkpoints: list[str], profiles: list[ModelProfile]) -> list[str]:
    """Resolve each checkpoint to its matching profile's `display_name`
    (falling back to the raw filename when none matches), then disambiguate
    any labels that collide — e.g. several version variants of one model
    (`FurryToonMix_v2.safetensors`, `FurryToonMix_v3.safetensors`) all
    matching the same profile's glob and thus getting the same static
    `display_name` — by appending each colliding checkpoint's filename stem,
    the one piece of information that actually varies between them."""
    raw_labels = []
    for ckpt in checkpoints:
        profile = resolve_profile(ckpt, profiles)
        raw_labels.append(profile.display_name if profile else ckpt)
    counts = Counter(raw_labels)
    return [
        f"{label} ({Path(ckpt).stem})" if counts[label] > 1 else label
        for ckpt, label in zip(checkpoints, raw_labels)
    ]


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/model` — list checkpoints ComfyUI has installed as an inline
    keyboard (labeled with the matching profile's `display_name` where one
    exists, disambiguated by filename when several checkpoints share a
    label — see `_checkpoint_labels`), for `model_callback` to act on."""
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

    labels = _checkpoint_labels(checkpoints, profiles)
    buttons = [
        [InlineKeyboardButton(label, callback_data=f"model:{i}")] for i, label in enumerate(labels)
    ]

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
    label = _checkpoint_labels(checkpoints, profiles)[index]

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

    if action in ("edit_cancel", "rename_cancel"):
        pending_key = (
            "awaiting_character_edit" if action == "edit_cancel" else "awaiting_character_rename"
        )
        pop_pending(context.chat_data, pending_key, query.message)
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
) -> tuple[str, str, str, str]:
    """Combine a raw prompt message with the active character (if any) into
    `(effective_prompt, extra_negative_prompt, raw_positive, raw_negative)`
    for `generate()`: splits the message into positive/negative (a "---"
    block separator, or else "-token" negatives — see
    `_split_negative_prompt`), folds the character's own saved
    positive/negative prompt in around them, and leaves profile-level
    negative defaults for `generate()`/`resolve_generation_params` to layer
    underneath. `raw_positive`/`raw_negative` are that same split, *before*
    the character folding — exactly what the user typed, for "🐛 Show
    Prompt"'s second output (see `GenerationParams.raw_positive_prompt`)."""
    raw_positive, raw_negative = _split_negative_prompt(prompt_text)
    effective_prompt = (
        join_nonempty([character["positive_prompt"], raw_positive]) if character else raw_positive
    )
    extra_negative = (
        join_nonempty([character["negative_prompt"], raw_negative]) if character else raw_negative
    )
    return effective_prompt, extra_negative, raw_positive, raw_negative


def _resolve_profile_and_prompt(
    chat_id: int,
    checkpoint: str,
    prompt_text: str,
    storage: Storage,
    profiles: list[ModelProfile],
) -> tuple[ModelProfile, str, str, str | None, str | None]:
    """This chat's profile (with its `/settings` override applied) plus the
    active character folded into `prompt_text` via `_resolve_effective_prompt`
    — the shared setup `generate_message` and `_run_stream` both need before
    calling `generate()`. Returns `(profile, effective_prompt, extra_negative,
    raw_positive, raw_negative)`."""
    profile = resolve_profile(checkpoint, profiles)
    profile = apply_profile_override(profile, checkpoint, storage.get_override(chat_id, checkpoint))

    active_character_name = storage.get_active_character_name(chat_id)
    character = (
        storage.get_character(chat_id, active_character_name) if active_character_name else None
    )
    effective_prompt, extra_negative, raw_positive, raw_negative = _resolve_effective_prompt(
        prompt_text, character
    )
    return profile, effective_prompt, extra_negative, raw_positive, raw_negative


async def generate_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a plain-text message as a generation prompt: resolve this
    chat's checkpoint/profile/override/active-character, run `generate()`
    with live progress, then deliver the result. Defaults to ComfyUI's
    first available checkpoint (and remembers it) if none is selected yet.
    A pending "custom value" `/settings` entry (see
    `handle_custom_value_message`) or an in-progress `/stream` prompt or
    character-edit/-rename entry takes priority over treating the text as a
    prompt. "✏️ Detail Prompt" has no entry here to take priority over —
    it's a webapp editor now, not a chat follow-up (see
    `DETAIL_PROMPT_CALLBACK_KIND`)."""
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

    profile, effective_prompt, extra_negative, raw_positive, raw_negative = (
        _resolve_profile_and_prompt(chat_id, checkpoint, prompt_text, storage, profiles)
    )

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
            raw_positive_prompt=raw_positive,
            raw_negative_prompt=raw_negative,
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
        source_bytes = await _download_telegram_file(context.bot, message.photo[-1].file_id)
        return await _analyze_both_deep(source_bytes, settings)

    result = await _run_reporting_errors(
        status_message, "Analysis", "uploaded-image analyze", _download_and_analyze_both()
    )
    if result is None:
        return
    tags, caption = result
    await status_message.delete()
    await _send_tag_and_caption_prompts(
        message, storage, chat_id, checkpoint, "💬 Qwen-VL caption (deep)", tags, caption
    )


#: Telegram's Bot API can't hand a bot any file over 20MB — `getFile`
#: refuses outright, so an import of, say, a 4x-upscaled PNG simply isn't
#: possible through the bot API no matter how it's sent. Checked up front
#: against `Document.file_size` so the user gets a clear explanation
#: instead of an opaque `getFile` failure.
IMPORT_MAX_FILE_BYTES = 20 * 1024 * 1024

#: How much of a prompt an import summary prints before eliding. The full
#: text is always recoverable from the restored keyboard's "🐛 Show Prompt";
#: this is only about not spending a 4096-character Telegram message on two
#: prompts before any of the actual settings are visible.
_IMPORT_PROMPT_PREVIEW = 400


def _elide(text: str, limit: int = _IMPORT_PROMPT_PREVIEW) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _import_summary(metadata: dict[str, Any], params: GenerationParams) -> str:
    """The human-readable "here's what was in that file" reply for a
    successful import. Deliberately prints the settings rather than just
    "imported ✅" — the whole point of the round trip is that the file
    carries them, so showing them is the confirmation that it worked."""
    lines = ["📥 Imported — this image's own settings are restored.", ""]
    lines.append(f"Model: {params.checkpoint}")
    if params.loader == "split":
        lines.append(f"CLIP/VAE: {params.clip_name or '—'} / {params.vae_name or '—'}")
    lines.append(
        f"{params.steps} steps · cfg {params.cfg} · {params.sampler_name}/{params.scheduler}"
    )
    lines.append(f"Size: {params.width}×{params.height} · clip skip {params.clip_skip}")
    if metadata.get("seed") is not None:
        lines.append(f"Seed: {metadata['seed']}")
    if params.loras:
        lines.append(
            "LoRAs: " + ", ".join(f"{lora.name} @ {lora.strength_model}" for lora in params.loras)
        )
    origin = metadata.get("kind")
    created = metadata.get("created_at")
    if origin or created:
        lines.append(f"Origin: {origin or 'unknown'}{f' · {created}' if created else ''}")
    lines.append("")
    lines.append(f"Positive:\n{_elide(params.positive_prompt)}")
    if params.negative_prompt:
        lines.append(f"\nNegative:\n{_elide(params.negative_prompt)}")
    return "\n".join(lines)


def _foreign_image_summary(summary: dict[str, Any]) -> str:
    """The reply for a PNG that carries ComfyUI's `prompt` chunk but none of
    ours — an export from Krita AI Diffusion, a raw ComfyUI run, another
    bot. Its settings can be read well enough to show and to seed a new
    generation from, but not well enough to restore post-processing
    buttons, since those need a full `GenerationParams` this graph can't
    reconstruct (see `png_metadata`'s module docstring)."""
    lines = ["🔍 No comfytelegram metadata — but this image has a ComfyUI workflow in it.", ""]
    if "checkpoint" in summary:
        lines.append(f"Model: {summary['checkpoint']}")
    detail = " · ".join(
        str(part)
        for part in (
            f"{summary['steps']} steps" if "steps" in summary else "",
            f"cfg {summary['cfg']}" if "cfg" in summary else "",
            summary.get("sampler_name", ""),
            f"seed {summary['seed']}" if "seed" in summary else "",
        )
        if part
    )
    if detail:
        lines.append(detail)
    return "\n".join(lines)


async def document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Import an image the user uploaded as a *file*, restoring the
    post-processing keyboard from the metadata embedded in it.

    This is the read side of `png_metadata`: download an image the bot may
    never have seen (or generated months ago, on a database since wiped),
    read its `comfytelegram` chunk, write a fresh `pending_result` row from
    the `params` dict it carries, and reply with the standard
    `_post_process_keyboard`. Every button then works — upscale, the
    detailers, "🩹 Fix Artifact", "🔁 Generate Again" — against an image
    with no prior row of its own. That makes "download for archiving,
    re-upload when more work is needed" a complete round trip.

    Registered on `filters.Document.IMAGE`, not `filters.PHOTO`, and that
    distinction is the whole feature: Telegram re-encodes every photo-type
    upload to JPEG, which strips PNG chunks in both directions. A photo
    therefore *cannot* carry metadata and is handled by `photo_message`
    (which analyzes its pixels instead); only the document path preserves
    the bytes. An image sent as a file with no metadata of ours falls back
    to `_foreign_image_summary` plus the same tag/caption analysis
    `photo_message` does, so uploading any image as a file still does
    something useful.
    """
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    document = message.document
    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    client: ComfyClient = context.bot_data["comfy_client"]

    if document.file_size and document.file_size > IMPORT_MAX_FILE_BYTES:
        await message.reply_text(
            f"That file is {document.file_size / 1_048_576:.1f}MB — Telegram won't let "
            "a bot download anything over 20MB, so I can't read it. (This is a Bot API "
            "limit, not a setting.)"
        )
        return

    status_message = await message.reply_text("Reading image…", disable_notification=True)

    async def _download() -> bytes:
        return await _download_telegram_file(context.bot, document.file_id)

    data = await _run_reporting_errors(status_message, "Import", "document import", _download())
    if data is None:
        return

    if not is_png(data):
        await status_message.edit_text(
            "That file isn't a PNG, so it can't carry generation metadata. "
            "Send it as a photo instead if you want it analyzed."
        )
        return

    metadata = extract_metadata(data)
    if metadata is None:
        await _import_without_metadata(message, status_message, context, data, chat_id)
        return

    params = _deserialize_generation_params(metadata["params"])
    result_id = uuid.uuid4().hex[:12]
    storage.store_pending_result(
        result_id,
        chat_id,
        document.file_id,
        metadata.get("filename") or document.file_name or "imported.png",
        metadata["params"],
    )
    await status_message.delete()

    text = _import_summary(metadata, params)
    missing = await _checkpoint_missing_note(client, params.checkpoint)
    if missing:
        text += f"\n\n{missing}"
    await message.reply_text(text, reply_markup=_post_process_keyboard(result_id))


async def _checkpoint_missing_note(client: ComfyClient, checkpoint: str) -> str:
    """A warning line if the imported image's checkpoint isn't installed on
    this ComfyUI — worth saying up front, since the buttons would otherwise
    all look live and then fail at submit time on a model that isn't there.
    Silent (returns "") if ComfyUI can't be reached at all: that's a
    separate problem the buttons will report in their own way, and guessing
    "model missing" from it would be wrong."""
    try:
        available = await client.list_checkpoints()
    except (ComfyUIError, aiohttp.ClientError, TimeoutError, OSError):
        return ""
    if checkpoint in available:
        return ""
    return (
        f"⚠️ This ComfyUI doesn't have “{checkpoint}” installed — generation "
        "buttons will fail until it is."
    )


async def _import_without_metadata(
    message: Message,
    status_message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    data: bytes,
    chat_id: int,
) -> None:
    """A PNG file with no `comfytelegram` chunk: report whatever ComfyUI's
    own `prompt` chunk yields (see `png_metadata.summarize_graph`) and
    otherwise treat it the way `photo_message` treats an uploaded photo —
    run both analyzers and offer their prompts. Post-processing buttons
    are deliberately *not* offered: without a full `GenerationParams` there
    is nothing to rebuild a graph from."""
    settings: Settings = context.bot_data["settings"]
    storage: Storage = context.bot_data["storage"]
    client: ComfyClient = context.bot_data["comfy_client"]

    graph = extract_comfy_graph(data)
    if graph is not None:
        summary = summarize_graph(graph)
        await message.reply_text(_foreign_image_summary(summary))
        positive = summary.get("positive_prompt")
        if positive:
            checkpoint = await _resolve_checkpoint_or_default(
                message, chat_id, storage, client, context
            )
            if checkpoint is not None:
                await _send_derived_prompt(
                    message,
                    storage,
                    chat_id,
                    checkpoint,
                    "🧩 Prompt from the embedded workflow",
                    positive,
                    summary.get("negative_prompt", ""),
                )
            await status_message.delete()
            return

    checkpoint = await _resolve_checkpoint_or_default(message, chat_id, storage, client, context)
    if checkpoint is None:
        return
    await status_message.edit_text("Analyzing the image…")

    result = await _run_reporting_errors(
        status_message, "Analysis", "imported-image analyze", _analyze_both_deep(data, settings)
    )
    if result is None:
        return
    tags, caption = result
    await status_message.delete()
    await _send_tag_and_caption_prompts(
        message, storage, chat_id, checkpoint, "💬 Qwen-VL caption (deep)", tags, caption
    )


#: How long a metadata-bearing archive copy can take to pull back out of
#: ComfyUI and push to Telegram before we give up — an upscaled PNG can be
#: tens of megabytes, but an unbounded wait would leave the user staring at
#: a status message forever if the ComfyUI host has gone away.
ARCHIVE_SEND_TIMEOUT_SECONDS = 300


async def _send_archive_copy(
    message: Message,
    client: ComfyClient,
    pending: dict[str, Any],
    full_params: GenerationParams,
) -> None:
    """Re-send a previously generated image as an uncompressed document, with
    its generation metadata embedded (see `png_metadata`).

    The copy has to come back out of *ComfyUI*, not from the stored Telegram
    `file_id`: that file_id points at whatever `_send_result_image` actually
    sent, which for anything under `TELEGRAM_PHOTO_SIZE_LIMIT` is the
    JPEG Telegram re-encoded the photo into — the PNG chunks are already
    gone from it. `pending["filename"]` is the name `SaveImage` wrote in
    ComfyUI's output directory, so `/view` still has the original bytes as
    long as that directory hasn't been cleaned out.

    Metadata gets re-embedded here rather than reused, because the file on
    ComfyUI's disk never had our chunk — `generation._tag_images` stamps
    the copy in memory on its way to Telegram, leaving the server's own
    file untouched. The seed is recovered from ComfyUI's `prompt` chunk on
    the fetched file, which is the only place it survives (see
    `png_metadata.extract_seed`).
    """
    status_message = await message.reply_text(
        "Fetching the original file…", disable_notification=True
    )

    async def _fetch_and_send() -> None:
        data = await client.get_image_bytes(pending["filename"], "", "output")
        graph = extract_comfy_graph(data)
        stamped = embed_metadata(
            data,
            build_metadata(
                params=pending["base_params"],
                kind="archive",
                filename=pending["filename"],
                seed=extract_seed(graph) if graph else None,
                checkpoint=full_params.checkpoint,
            ),
        )
        await message.reply_document(
            document=io.BytesIO(stamped),
            filename=pending["filename"],
            caption=(
                "Full-quality PNG — this is the copy to save. Its generation "
                "settings are stored inside the file, so sending it back to me "
                "as a file (not a photo) restores all its buttons."
            ),
        )

    try:
        await asyncio.wait_for(_fetch_and_send(), timeout=ARCHIVE_SEND_TIMEOUT_SECONDS)
    except ComfyUIError:
        logger.warning("Archive copy unavailable for %s", pending["filename"], exc_info=True)
        await status_message.edit_text(
            "ComfyUI no longer has the original file for this image — its output "
            "directory has probably been cleaned since it was generated."
        )
        return
    except (TimeoutError, aiohttp.ClientError):
        logger.warning("Archive copy failed for %s", pending["filename"], exc_info=True)
        await status_message.edit_text(
            "Couldn't fetch the original file from ComfyUI — see the logs."
        )
        return
    await status_message.delete()


def _detail_prompt_readonly_info() -> dict[str, Any]:
    """Read-only reference info sent to the "✏️ Detail Prompt" webapp
    alongside the editable positive/negative/denoise fields — a single flat
    dict, since unlike the old per-image override this replaced, the run
    that follows is always `post_process(kind="hand_drawn")`'s
    `DrawnMaskHandDetailerParams` (see `DETAIL_PROMPT_CALLBACK_KIND`), not
    a choice made later. Built from that dataclass's own class-level
    defaults: static, no image/profile dependency, so this needs no
    arguments and could be computed once, but stays a function since it's
    only ever called from one place."""
    params = DrawnMaskHandDetailerParams()
    return {
        "steps": params.steps,
        "cfg": params.cfg,
        "sampler_name": params.sampler_name,
        "scheduler": params.scheduler,
        "denoise": params.denoise,
    }


async def postprocess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `pp:<kind>:<result_id>` tap from `_post_process_keyboard`:
    `kind` is `"analyze_only"` (run *both* analyzers — WD14 tags and
    Qwen-VL caption — and reply with both, no generation, no
    `prompt_style` dispatch; see `ANALYZE_ONLY_CALLBACK_KIND`),
    `"show_prompt"` (reply with the exact positive/negative prompt this
    image was built from, no download or generation and no tag check — see
    `SHOW_PROMPT_CALLBACK_KIND` — followed by a second reply with just what
    the user actually typed, stripped of the profile's prompt prefixes and
    any active character's saved prompt — see
    `GenerationParams.raw_positive_prompt`/`_resolve_effective_prompt` —
    or a "not available" note for a pending_result row stored before that
    field existed. That second reply is plain, label-free text — see
    `_raw_prompt_copy_text` — plus, when it's short enough for Telegram's
    `MAX_COPY_TEXT` cap, a "📋 Copy" button (same gating as
    `_generate_from_prompt_keyboard`'s), so it's reusable as a whole
    without hand-editing out "Positive:"/"Negative:" labels first),
    `"analyze_prompt"` (a `/tagcheck`-style
    tag-health check on that same prompt instead, if this image's
    checkpoint profile is tag-trained and tag data is imported — see
    `ANALYZE_PROMPT_CALLBACK_KIND`), or
    `"upscale"`/`"homogenize"`/`"face"`/`"hand"` (download the source image
    and run that post-processing stage on it). A fresh `"upscale"` tap on an
    image already at or beyond `UPSCALE_CONFIRM_THRESHOLD_PX` doesn't upscale
    immediately — it downloads just far enough to measure the image, then
    replies with a "this is already large — continue?" prompt
    (`_upscale_confirm_keyboard`) instead, which comes back as either
    `UPSCALE_CONFIRM_CALLBACK_KIND` (proceed) or
    `UPSCALE_CANCEL_CALLBACK_KIND` (abort) — `"homogenize"` has no such gate,
    since it never changes the image's pixel dimensions (see
    `generation.post_process`'s `TiledRefineParams` branch). A `"face"`/`"hand"` result whose
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
    it needs a row/col in its callback_data). `HAND_REDO_CALLBACK_KIND`
    ("🔁 Redo (same mask)", attached to every hand-drawn-mask result) skips
    the WebApp editor entirely and re-runs `post_process(kind="hand_drawn")`
    against the exact same source image and mask stored in `storage.py`'s
    `inpaint_redo` — a fresh call gets a fresh seed automatically (see
    `DrawnMaskHandDetailerParams.seed`'s default), so a bad detailer result
    doesn't require redrawing the whole mask just to try again. Alerts
    instead if `result_id` has expired (see `PENDING_RESULT_TTL_SECONDS`).
    `FIX_DRAW_CALLBACK_KIND`/`FIX_REDO_CALLBACK_KIND` ("🩹 Fix Artifact")
    mirror `HAND_DRAW_CALLBACK_KIND`/`HAND_REDO_CALLBACK_KIND` exactly, just
    for `post_process(kind="fix_drawn")` (general-purpose artifact removal)
    instead of a hand — see `_DRAWN_MASK_KINDS`."""
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

    if kind in (MORE_CALLBACK_KIND, BACK_CALLBACK_KIND):
        await _show_keyboard_page(query, result_id, 2 if kind == MORE_CALLBACK_KIND else 1)
        return

    if kind == ARCHIVE_CALLBACK_KIND:
        await _send_archive_copy(query.message, client, pending, full_params)
        return

    if kind == SHOW_PROMPT_CALLBACK_KIND:
        negative = full_params.negative_prompt or "(none)"
        await query.message.reply_text(
            f"Positive:\n{full_params.positive_prompt}\n\nNegative:\n{negative}"
        )
        if full_params.raw_positive_prompt or full_params.raw_negative_prompt:
            copy_text = _raw_prompt_copy_text(
                full_params.raw_positive_prompt, full_params.raw_negative_prompt
            )
            keyboard = None
            if len(copy_text) <= InlineKeyboardButtonLimit.MAX_COPY_TEXT:
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("📋 Copy", copy_text=CopyTextButton(copy_text))]]
                )
            await query.message.reply_text(f"As typed:\n\n{copy_text}", reply_markup=keyboard)
        else:
            await query.message.reply_text(
                "As typed (no profile/character prompt): not available — this image "
                "was generated before this existed."
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
            source_bytes = await _fetch_source_image(
                client,
                context.bot,
                pending["filename"],
                pending["file_id"],
                on_fallback=_source_fallback_notifier(query.message),
            )
            return await _analyze_both(source_bytes, settings)

        result = await _run_reporting_errors(
            status_message, "Analysis", "analyze-only", _download_and_analyze_both()
        )
        if result is None:
            return
        tags, caption = result
        await status_message.delete()
        await _send_tag_and_caption_prompts(
            query.message, storage, chat_id, checkpoint, "💬 Qwen-VL caption", tags, caption
        )
        return

    if kind == DEEP_ANALYZE_CALLBACK_KIND:
        status_message = await query.message.reply_text(
            "Deep analyzing image…", disable_notification=True
        )
        checkpoint = full_params.checkpoint
        chat_id = pending["chat_id"]

        async def _download_and_analyze_deep() -> tuple[str, str]:
            source_bytes = await _fetch_source_image(
                client,
                context.bot,
                pending["filename"],
                pending["file_id"],
                on_fallback=_source_fallback_notifier(query.message),
            )
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
            reply_markup=_hand_mode_keyboard(result_id, settings),
        )
        return

    if kind == HAND_MANUAL_CALLBACK_KIND:
        source_bytes = await _fetch_source_image(
            client,
            context.bot,
            pending["filename"],
            pending["file_id"],
            on_fallback=_source_fallback_notifier(query.message),
        )
        gridded = await asyncio.to_thread(_draw_hand_point_grid, source_bytes)
        await query.message.reply_photo(
            photo=io.BytesIO(gridded),
            caption="Tap the cell over the hand:",
            reply_markup=_hand_point_keyboard(result_id),
        )
        return

    if kind in (HAND_DRAW_CALLBACK_KIND, FIX_DRAW_CALLBACK_KIND, DETAIL_PROMPT_CALLBACK_KIND):
        if not settings.inpaint_relay_url:
            await query.message.reply_text("Mask drawing isn't configured on this bot.")
            return
        if kind == HAND_DRAW_CALLBACK_KIND:
            job_kind = "hand"
        elif kind == FIX_DRAW_CALLBACK_KIND:
            job_kind = "fix"
        else:
            job_kind = "detail"
        # DETAIL_PROMPT_CALLBACK_KIND is the one caller that wants the
        # editor's prompt fields too (`Job.mode="mask_prompt"` — see
        # `_relay_create_job`); the other two just draw a plain mask
        # (`meta=None`, which the relay defaults to `mode="mask"`). The
        # fields pre-fill with this image's own current prompt — the usual
        # reason to open this is to cut most of a scene-wide prompt down to
        # what the marked region actually needs, not type one from scratch
        # — but nothing about them is saved anywhere once the editor closes
        # (see `DETAIL_PROMPT_CALLBACK_KIND`'s docstring).
        meta = (
            {
                "mode": "mask_prompt",
                "positive": full_params.positive_prompt,
                "negative": full_params.negative_prompt,
                "readonly": _detail_prompt_readonly_info(),
            }
            if job_kind == "detail"
            else None
        )
        # Uploading a large source image to a remote relay host can take a
        # few seconds, with nothing on screen to show for it in the
        # meantime — long enough that a user unsure whether their tap
        # registered taps "Draw Mask" again. `uploads_in_progress` (checked
        # and set synchronously, no `await` in between, so two concurrently
        # scheduled taps on the same result_id can't both pass the check)
        # turns a repeat tap into a no-op reply instead of a second relay
        # job/editor button; `status_message`, sent before the upload even
        # starts, is the actual "yes, this registered" feedback.
        uploads_in_progress: set[str] = context.bot_data.setdefault(
            "inpaint_uploads_in_progress", set()
        )
        if result_id in uploads_in_progress:
            await query.message.reply_text(
                "Still uploading that image to the mask editor — hang tight."
            )
            return
        uploads_in_progress.add(result_id)
        try:
            status_message = await query.message.reply_text(
                "Uploading image to the mask editor…", disable_notification=True
            )
            source_bytes = await _fetch_source_image(
                client,
                context.bot,
                pending["filename"],
                pending["file_id"],
                on_fallback=_source_fallback_notifier(query.message),
            )
            try:
                token = await _relay_create_job(settings, source_bytes, meta=meta)
            except Exception:
                logger.exception("Failed to create inpaint_relay job")
                await status_message.edit_text(
                    "Couldn't reach the mask editor server — try again later."
                )
                return
            storage.store_inpaint_job(
                token,
                result_id,
                pending["chat_id"],
                query.message.message_thread_id,
                kind=job_kind,
            )
            editor_prompt = (
                "Draw over the region to detail, describe what should be there, then "
                "tap Done in the editor:"
                if job_kind == "detail"
                else "Draw over the area to fix, then tap Done in the editor:"
            )
            await status_message.edit_text(
                editor_prompt,
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "🎨 Open mask editor",
                                web_app=WebAppInfo(
                                    url=f"{settings.inpaint_relay_url}/jobs/{token}"
                                ),
                            )
                        ]
                    ]
                ),
            )
        finally:
            uploads_in_progress.discard(result_id)
        return

    if kind in (
        HAND_REDO_CALLBACK_KIND,
        FIX_REDO_CALLBACK_KIND,
        DETAIL_REDO_CALLBACK_KIND,
        HAND_REDO4_CALLBACK_KIND,
        FIX_REDO4_CALLBACK_KIND,
        DETAIL_REDO4_CALLBACK_KIND,
    ):
        is_redo4 = kind in (
            HAND_REDO4_CALLBACK_KIND,
            FIX_REDO4_CALLBACK_KIND,
            DETAIL_REDO4_CALLBACK_KIND,
        )
        if kind in (HAND_REDO_CALLBACK_KIND, HAND_REDO4_CALLBACK_KIND):
            redo_kind, redo_button_label = "hand", "🖌️ Draw Mask"
        elif kind in (FIX_REDO_CALLBACK_KIND, FIX_REDO4_CALLBACK_KIND):
            redo_kind, redo_button_label = "fix", "🩹 Fix Artifact"
        else:
            redo_kind, redo_button_label = "detail", "✏️ Detail Prompt"
        drawn_mask_kind = _DRAWN_MASK_KINDS[redo_kind]
        redo = storage.get_inpaint_redo(result_id)
        if redo is None:
            await query.message.reply_text(
                f"That mask has expired — draw a new one with {redo_button_label}."
            )
            return
        source_bytes = await _fetch_source_image(
            client,
            context.bot,
            redo["source_filename"],
            redo["source_file_id"],
            on_fallback=_source_fallback_notifier(query.message),
        )
        # "🔁 x4" runs the same iteration DRAWN_MASK_REDO4_COUNT times instead
        # of once, reusing the one source download above across all of them.
        # Stops at the first failure rather than continuing to spend ComfyUI
        # time on a source/mask combination that just failed.
        for _ in range(DRAWN_MASK_REDO4_COUNT if is_redo4 else 1):
            ok = await _run_one_drawn_mask_redo(
                query.message,
                client,
                storage,
                pending["chat_id"],
                full_params,
                redo,
                drawn_mask_kind,
                context.bot_data["profiles"],
                source_bytes,
            )
            if not ok:
                break
        return

    if kind == HAND_AUTO_CALLBACK_KIND:
        kind = "hand"

    source_bytes: bytes | None = None
    if kind in ("upscale", UPSCALE_CONFIRM_CALLBACK_KIND):
        if kind == UPSCALE_CONFIRM_CALLBACK_KIND:
            await _safe_edit_message(query, "Upscaling anyway…", InlineKeyboardMarkup([]))
        source_bytes = await _fetch_source_image(
            client,
            context.bot,
            pending["filename"],
            pending["file_id"],
            on_fallback=_source_fallback_notifier(query.message),
        )
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
            data = await _fetch_source_image(
                client,
                context.bot,
                pending["filename"],
                pending["file_id"],
                on_fallback=_source_fallback_notifier(query.message),
            )
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


async def _show_keyboard_page(query, result_id: str, page: int) -> None:
    """Swap the tapped image's keyboard to `page` in place — only the reply
    markup is edited, never the message itself, since most of these are
    photo/document messages whose text can't be edited at all (which is why
    `settings_menu._safe_edit_message`, an `edit_message_text` wrapper,
    isn't what's used here). Any redo rows the message carries are
    re-appended to the new page (see `_carried_extra_rows`).

    A "message is not modified" BadRequest is swallowed: it means the
    keyboard already shows that page, which happens on a double-tap or on
    a tap replayed against a message that was already flipped, and is not
    worth surfacing as an error."""
    markup = _post_process_keyboard_with_extra_rows(
        result_id, _carried_extra_rows(query.message.reply_markup), page
    )
    try:
        await query.edit_message_reply_markup(reply_markup=markup)
    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise


async def hand_point_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `hp:<result_id>:<grid_size>:<row>:<col>` tap from
    `_hand_point_keyboard` (the "✋ Tap to mark" grid
    `postprocess_callback`'s `HAND_MANUAL_CALLBACK_KIND` branch sends, or
    `hand_point_density_callback`'s finer re-render) — downloads the source
    image, turns the tapped cell into a fractional (x, y) point at its
    center, and runs `post_process(kind="hand_manual")` on it. The marked
    box is scaled down for a denser `grid_size` (see
    `HAND_POINT_BOX_SIZE_FRAC_BASE`) so it stays roughly cell-sized instead
    of covering the same fraction of the image regardless of density.
    Unlike the auto-detect path, a manually marked region is never "nothing
    detected", so there's no `unchanged` check here."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    _, result_id, grid_size_str, row_str, col_str = query.data.split(":")
    grid_size, row, col = int(grid_size_str), int(row_str), int(col_str)
    storage: Storage = context.bot_data["storage"]
    pending = storage.get_pending_result(result_id)
    if pending is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    client: ComfyClient = context.bot_data["comfy_client"]
    full_params = _deserialize_generation_params(pending["base_params"])
    point_frac = (
        (col + 0.5) / grid_size,
        (row + 0.5) / grid_size,
    )
    box_size_frac = HAND_POINT_BOX_SIZE_FRAC_BASE / grid_size

    status_message = await query.message.reply_text("Refining hand…", disable_notification=True)

    async def _download_and_post_process() -> GeneratedImage:
        data = await _fetch_source_image(
            client,
            context.bot,
            pending["filename"],
            pending["file_id"],
            on_fallback=_source_fallback_notifier(query.message),
        )
        return await post_process(
            client,
            "hand_manual",
            data,
            pending["filename"],
            full_params,
            point_frac=point_frac,
            box_size_frac=box_size_frac,
        )

    result = await _run_reporting_errors(
        status_message, "Refining hand", "post-processing", _download_and_post_process()
    )
    if result is None:
        return

    await status_message.delete()
    await _send_and_store_result(query.message, pending["chat_id"], storage, result)


async def hand_point_density_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a `hpz:<result_id>:<grid_size>` tap from `_hand_point_keyboard`'s
    "🔍 Finer grid" button — re-downloads the same source image and replies
    with a fresh gridded photo/keyboard at the denser `grid_size`, the same
    way `postprocess_callback`'s `HAND_MANUAL_CALLBACK_KIND` branch renders
    the initial grid."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    _, result_id, grid_size_str = query.data.split(":")
    grid_size = int(grid_size_str)
    storage: Storage = context.bot_data["storage"]
    pending = storage.get_pending_result(result_id)
    if pending is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    client: ComfyClient = context.bot_data["comfy_client"]
    source_bytes = await _fetch_source_image(
        client,
        context.bot,
        pending["filename"],
        pending["file_id"],
        on_fallback=_source_fallback_notifier(query.message),
    )
    gridded = await asyncio.to_thread(_draw_hand_point_grid, source_bytes, grid_size)
    await query.message.reply_photo(
        photo=io.BytesIO(gridded),
        caption="Tap the cell over the hand:",
        reply_markup=_hand_point_keyboard(result_id, grid_size),
    )


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
            raw_positive_prompt=stored["prompt"],
            raw_negative_prompt=stored["negative_prompt"],
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


def _make_cancel_callback(
    pending_key: str, cancelled_text: str
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    """Build a "❌ Cancel" callback for a one-shot `awaiting_*` follow-up
    entry (the same pattern `settings_menu.py`'s custom-value capture and
    this file's character-edit/-rename/stream-prompt entries all use):
    clears `pending_key` so the next text message goes back to being treated
    as a normal generation prompt, and edits the button away so a stale tap
    can't be replayed."""

    async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        settings: Settings = context.bot_data["settings"]
        user_id = update.effective_user.id if update.effective_user else None
        if await reject_if_unauthorized_callback(query, user_id, settings):
            return

        if not pop_pending(context.chat_data, pending_key, query.message):
            await query.answer("Nothing to cancel.")
            return

        await query.answer("Cancelled.")
        await _safe_edit_message(query, cancelled_text, InlineKeyboardMarkup([]))

    return callback


#: "❌ Cancel" on the "What should the stream generate?" follow-up — see
#: `stream_command`'s promptless branch and `_stream_prompt_cancel_keyboard`.
stream_cancel_callback = _make_cancel_callback(
    "awaiting_stream_prompt", "Cancelled — no stream started."
)


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

        profile, effective_prompt, extra_negative, raw_positive, raw_negative = (
            _resolve_profile_and_prompt(chat_id, checkpoint, prompt_text, storage, profiles)
        )

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
                raw_positive_prompt=raw_positive,
                raw_negative_prompt=raw_negative,
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


async def _resolve_tag_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE, usage_text: str
) -> tuple[TagDatabase, Settings, list[TagSource], str] | None:
    """Shared `/tags`/`/tagcheck` setup: auth, the "no tag data imported"
    guard, argument parsing, and `_resolve_tag_sources` resolution — bailing
    out with `usage_text` if the query/prompt ends up empty. Returns None
    after already replying, the same "stop here" convention
    `_run_reporting_errors`/`_resolve_checkpoint_or_default` use elsewhere
    in this file. Returns `(tags_db, settings, sources, query)` on success."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return None

    message = update.effective_message
    tags_db: TagDatabase = context.bot_data["tags_db"]
    if not any(tags_db.stats().values()):
        await message.reply_text(_no_tag_data_message())
        return None

    raw = (message.text or "").split(maxsplit=1)
    args_text = raw[1].strip() if len(raw) > 1 else ""

    storage: Storage = context.bot_data["storage"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]
    checkpoint = storage.get_checkpoint(update.effective_chat.id)
    sources, query = _resolve_tag_sources(args_text, checkpoint, profiles)
    query = query.strip()
    if not query:
        await message.reply_text(usage_text)
        return None
    return tags_db, settings, sources, query


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
    resolved = await _resolve_tag_query(
        update,
        context,
        'Usage: /tags <query> — optionally prefixed with "danbooru:" or "e621:" '
        'to search a specific dictionary, e.g. "/tags e621:fox"',
    )
    if resolved is None:
        return
    tags_db, settings, sources, query = resolved
    message = update.effective_message

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
    resolved = await _resolve_tag_query(
        update,
        context,
        "Usage: /tagcheck <prompt> — checks each comma-separated tag against the "
        'tag database. Same "danbooru:"/"e621:" prefix override as /tags.',
    )
    if resolved is None:
        return
    tags_db, settings, sources, prompt_text = resolved
    message = update.effective_message

    lines = _tagcheck_lines(prompt_text, sources, tags_db, settings)
    await message.reply_text("\n".join(lines))
