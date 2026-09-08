"""Telegram-facing handlers: /start, /help, /model, plain-text generation
requests, and the inline-keyboard callbacks for model selection and
post-processing.

Shared objects (settings, the ComfyUI client, loaded profiles, storage)
live in `context.bot_data`, populated once at startup in `main.py`.
"""

from __future__ import annotations

import io
import logging
import re
import time
import uuid
from collections.abc import Awaitable
from typing import Any, TypeVar

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import ContextTypes

from comfytelegram.auth import reject_if_unauthorized, reject_if_unauthorized_callback
from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.generation import GeneratedImage, generate, post_process, regenerate, repeat
from comfytelegram.profiles import (
    ModelProfile,
    apply_profile_override,
    join_nonempty,
    resolve_profile,
)
from comfytelegram.settings import Settings
from comfytelegram.settings_menu import _safe_edit_message, handle_custom_value_message
from comfytelegram.storage import Storage
from comfytelegram.workflows import GenerationParams, LoraSpec

logger = logging.getLogger(__name__)

T = TypeVar("T")

POSTPROCESS_KEYBOARD_LABELS = {"upscale": "🔍 Upscale 4x", "face": "✨ Face Detail"}
REGENERATE_CALLBACK_KIND = "regen"
AGAIN_CALLBACK_PREFIX = "again:"

CHARACTER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

CHARACTER_HELP = (
    "Save a reusable character design so you don't have to retype its "
    "description every time:\n"
    "/character save <name> | <positive prompt> [| <negative prompt>]\n"
    "/character delete <name>\n"
    "/characters — list saved characters and activate one\n\n"
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


def _post_process_keyboard(result_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(label, callback_data=f"pp:{kind}:{result_id}")
                for kind, label in POSTPROCESS_KEYBOARD_LABELS.items()
            ],
            [InlineKeyboardButton("🔁 Regenerate", callback_data=f"pp:{REGENERATE_CALLBACK_KIND}:{result_id}")],
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
        [[InlineKeyboardButton("🔁 Generate Again", callback_data=f"{AGAIN_CALLBACK_PREFIX}{snapshot_id}")]]
    )


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
    if sent_message.photo:
        return sent_message.photo[-1].file_id  # largest resolution
    if sent_message.document:
        return sent_message.document.file_id
    raise ValueError("Sent message has neither a photo nor a document to read a file_id from")


def _serialize_generation_params(params: GenerationParams) -> dict[str, Any]:
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
            {"name": lora.name, "strength_model": lora.strength_model, "strength_clip": lora.strength_clip}
            for lora in params.loras
        ],
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
    )


def _make_progress_callback(status_message: Message):
    """Shared throttled progress-edit closure for `generate_message` and
    `again_callback` — see PROGRESS_EDIT_INTERVAL above."""
    last_edit = {"t": 0.0}

    async def on_progress(progress: JobProgress) -> None:
        if progress.done or progress.value is None or progress.max is None:
            return
        now = time.monotonic()
        if now - last_edit["t"] < PROGRESS_EDIT_INTERVAL:
            return
        last_edit["t"] = now
        pct = int(100 * progress.value / progress.max) if progress.max else 0
        try:
            await status_message.edit_text(f"Generating… {pct}% (step {progress.value}/{progress.max})")
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
    sent = await _send_result_image(message, img.data, img.filename, _post_process_keyboard(result_id))
    storage.store_pending_result(
        result_id, chat_id, _extract_file_id(sent), img.filename, _serialize_generation_params(img.full_params)
    )


async def _deliver_generation_result(
    status_message: Message, reply_target: Message, chat_id: int, storage: Storage, images: list[GeneratedImage]
) -> None:
    """Shared tail of `generate_message`/`again_callback` once a batch of
    images has been produced: mint a fresh generation snapshot for the next
    "Generate Again" tap, update the status message, then send + register
    each image. `reply_target` is the message new image replies attach to
    (the original prompt message, or the callback query's own message)."""
    snapshot_id = uuid.uuid4().hex[:12]
    storage.store_generation_snapshot(snapshot_id, chat_id, _serialize_generation_params(images[0].full_params))
    await status_message.edit_text(f"Done — {len(images)} image(s).", reply_markup=_again_keyboard(snapshot_id))
    for img in images:
        await _send_and_store_result(reply_target, chat_id, storage, img)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return
    await update.effective_message.reply_text(
        "Send me a prompt and I'll generate an image with ComfyUI.\n\n"
        "/model — pick a checkpoint\n"
        "/settings — view or change generation defaults for the current model\n"
        "/character save <name> | <prompt> — save a reusable character design\n"
        "/characters — list saved characters and activate one\n"
        "/help — show this message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


def _characters_keyboard(characters: list[dict[str, Any]], active: str | None) -> InlineKeyboardMarkup:
    rows = []
    for char in characters:
        label = f"✅ {char['name']}" if char["name"] == active else char["name"]
        rows.append([InlineKeyboardButton(label, callback_data=f"char:activate:{char['name']}")])
    if active is not None:
        rows.append([InlineKeyboardButton("❌ Clear active character", callback_data="char:clear")])
    return InlineKeyboardMarkup(rows)


async def character_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    await update.effective_message.reply_text(header, reply_markup=_characters_keyboard(characters, active))


async def character_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    name = parts[2]
    if storage.get_character(chat_id, name) is None:
        await query.answer("That character no longer exists — refresh with /characters.", show_alert=True)
        return

    storage.set_active_character(chat_id, name)
    await query.answer(f"Activated '{name}'.")
    await _safe_edit_message(
        query, f"Active character: {name}", _characters_keyboard(storage.list_characters(chat_id), name)
    )


async def generate_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    if await handle_custom_value_message(update, context):
        return

    message = update.effective_message
    prompt_text = (message.text or "").strip()
    if not prompt_text:
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    client: ComfyClient = context.bot_data["comfy_client"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]

    checkpoint = storage.get_checkpoint(chat_id)
    if checkpoint is None:
        try:
            checkpoints = await client.list_checkpoints()
        except (ComfyUIError, OSError) as exc:
            await message.reply_text(f"Couldn't reach ComfyUI: {exc}")
            return
        if not checkpoints:
            await message.reply_text("ComfyUI reports no checkpoints installed.")
            return
        checkpoint = checkpoints[0]
        storage.set_checkpoint(chat_id, checkpoint)
        context.bot_data["available_checkpoints"] = checkpoints
        await message.reply_text(
            f"No model selected yet — defaulting to {checkpoint}. Use /model to change it."
        )

    profile = resolve_profile(checkpoint, profiles)
    override_fields = storage.get_override(chat_id, checkpoint)
    profile = apply_profile_override(profile, checkpoint, override_fields)

    active_character_name = storage.get_active_character_name(chat_id)
    character = storage.get_character(chat_id, active_character_name) if active_character_name else None
    effective_prompt = join_nonempty([character["positive_prompt"], prompt_text]) if character else prompt_text
    extra_negative = character["negative_prompt"] if character else ""

    status_message = await message.reply_text("Generating… 0%")

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


async def postprocess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    if kind == REGENERATE_CALLBACK_KIND:
        status_message = await query.message.reply_text("Regenerating…")
        result = await _run_reporting_errors(
            status_message, "Regeneration", "regeneration", regenerate(client, full_params)
        )
        if result is None:
            return
        await status_message.delete()
        await _send_and_store_result(query.message, pending["chat_id"], storage, result)
        return

    label = "Upscaling" if kind == "upscale" else "Refining face"
    status_message = await query.message.reply_text(f"{label}…")

    async def _download_and_post_process() -> GeneratedImage:
        tg_file = await context.bot.get_file(pending["file_id"])
        source_bytes = bytes(await tg_file.download_as_bytearray())
        return await post_process(client, kind, source_bytes, pending["filename"], full_params)

    result = await _run_reporting_errors(status_message, label, "post-processing", _download_and_post_process())
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

    status_message = await query.message.reply_text("Generating… 0%")
    images = await _run_reporting_errors(
        status_message,
        "Generation",
        "repeat generation",
        repeat(client, full_params, on_progress=_make_progress_callback(status_message)),
    )
    if images is None:
        return

    await _deliver_generation_result(status_message, query.message, chat_id, storage, images)
