"""Telegram-facing handlers: /start, /help, /model, plain-text generation
requests, and the inline-keyboard callbacks for model selection and
post-processing.

Shared objects (settings, the ComfyUI client, loaded profiles, storage)
live in `context.bot_data`, populated once at startup in `main.py`.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import ContextTypes

from comfytelegram.auth import is_authorized, reject_if_unauthorized
from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.generation import GeneratedImage, generate, post_process
from comfytelegram.profiles import ModelProfile, apply_profile_override, resolve_profile
from comfytelegram.settings import Settings
from comfytelegram.settings_menu import handle_custom_value_message
from comfytelegram.storage import Storage
from comfytelegram.workflows import LoraSpec, PostProcessBaseParams

logger = logging.getLogger(__name__)

POSTPROCESS_KEYBOARD_LABELS = {"upscale": "🔍 Upscale 4x", "face": "✨ Face Detail"}

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
            ]
        ]
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


def _serialize_base_params(params: PostProcessBaseParams) -> dict[str, Any]:
    return {
        "checkpoint": params.checkpoint,
        "positive_prompt": params.positive_prompt,
        "negative_prompt": params.negative_prompt,
        "clip_skip": params.clip_skip,
        "loras": [
            {"name": lora.name, "strength_model": lora.strength_model, "strength_clip": lora.strength_clip}
            for lora in params.loras
        ],
    }


def _deserialize_base_params(data: dict[str, Any]) -> PostProcessBaseParams:
    return PostProcessBaseParams(
        checkpoint=data["checkpoint"],
        positive_prompt=data["positive_prompt"],
        negative_prompt=data["negative_prompt"],
        clip_skip=data["clip_skip"],
        loras=[LoraSpec(**lora) for lora in data.get("loras", [])],
    )


async def _send_and_store_result(
    message: Message, chat_id: int, storage: Storage, img: GeneratedImage
) -> None:
    """Send a generated image with its post-processing keyboard, then persist
    what those buttons need (a re-downloadable file_id + the params to build
    the next graph) so they still work after a bot restart — see storage.py."""
    result_id = uuid.uuid4().hex[:12]
    sent = await _send_result_image(message, img.data, img.filename, _post_process_keyboard(result_id))
    storage.store_pending_result(
        result_id, chat_id, _extract_file_id(sent), img.filename, _serialize_base_params(img.base_params)
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return
    await update.effective_message.reply_text(
        "Send me a prompt and I'll generate an image with ComfyUI.\n\n"
        "/model — pick a checkpoint\n"
        "/settings — view or change generation defaults for the current model\n"
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
    if update.effective_user is None or not is_authorized(settings, update.effective_user.id):
        await query.answer("Not authorized.", show_alert=True)
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
    status_message = await message.reply_text("Generating… 0%")

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

    try:
        images = await generate(client, checkpoint, prompt_text, profile, on_progress=on_progress)
    except ComfyUIError as exc:
        await status_message.edit_text(f"Generation failed: {exc}")
        return
    except Exception:
        logger.exception("Unexpected error during generation")
        await status_message.edit_text("Generation failed with an unexpected error.")
        return

    await status_message.edit_text(f"Done — {len(images)} image(s).")

    for img in images:
        await _send_and_store_result(message, chat_id, storage, img)


async def postprocess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    if update.effective_user is None or not is_authorized(settings, update.effective_user.id):
        await query.answer("Not authorized.", show_alert=True)
        return

    _, kind, result_id = query.data.split(":", 2)
    storage: Storage = context.bot_data["storage"]
    pending = storage.get_pending_result(result_id)
    if pending is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    client: ComfyClient = context.bot_data["comfy_client"]
    label = "Upscaling" if kind == "upscale" else "Refining face"
    status_message = await query.message.reply_text(f"{label}…")

    try:
        tg_file = await context.bot.get_file(pending["file_id"])
        source_bytes = bytes(await tg_file.download_as_bytearray())
        base_params = _deserialize_base_params(pending["base_params"])
        result = await post_process(client, kind, source_bytes, pending["filename"], base_params)
    except ComfyUIError as exc:
        await status_message.edit_text(f"{label} failed: {exc}")
        return
    except Exception:
        logger.exception("Unexpected error during post-processing")
        await status_message.edit_text(f"{label} failed with an unexpected error.")
        return

    await status_message.delete()
    await _send_and_store_result(query.message, pending["chat_id"], storage, result)
