"""Telegram-facing handlers: /start, /help, /model, plain-text generation
requests, and the inline-keyboard callbacks for model selection and
post-processing.

Shared objects (settings, the ComfyUI client, loaded profiles, bot state)
live in `context.bot_data`, populated once at startup in `main.py`.
"""

from __future__ import annotations

import io
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import ContextTypes

from comfytelegram.comfy_client import ComfyClient, ComfyUIError, JobProgress
from comfytelegram.generation import generate, post_process
from comfytelegram.profiles import ModelProfile, resolve_profile
from comfytelegram.settings import Settings
from comfytelegram.state import BotState, PendingResult

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


def _is_authorized(settings: Settings, user_id: int | None) -> bool:
    if not settings.allowed_user_ids:
        return True
    return user_id in settings.allowed_user_ids


async def _reject_if_unauthorized(update: Update, settings: Settings) -> bool:
    user = update.effective_user
    if user is not None and _is_authorized(settings, user.id):
        return False
    if update.effective_message is not None:
        await update.effective_message.reply_text("You're not authorized to use this bot.")
    return True


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
) -> None:
    """Send a generated image, falling back to a document when it's too big
    for Telegram's photo path (see TELEGRAM_PHOTO_SIZE_LIMIT above)."""
    if len(data) <= TELEGRAM_PHOTO_SIZE_LIMIT:
        await message.reply_photo(photo=io.BytesIO(data), reply_markup=reply_markup)
        return
    await message.reply_document(
        document=io.BytesIO(data),
        filename=filename,
        caption="Sent as a file — too large for Telegram's photo size limit (10MB).",
        reply_markup=reply_markup,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await _reject_if_unauthorized(update, settings):
        return
    await update.effective_message.reply_text(
        "Send me a prompt and I'll generate an image with ComfyUI.\n\n"
        "/model — pick a checkpoint\n"
        "/help — show this message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await _reject_if_unauthorized(update, settings):
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
    if update.effective_user is None or not _is_authorized(settings, update.effective_user.id):
        await query.answer("Not authorized.", show_alert=True)
        return

    checkpoints: list[str] = context.bot_data.get("available_checkpoints", [])
    _, index_str = query.data.split(":", 1)
    index = int(index_str)
    if index >= len(checkpoints):
        await query.answer("That model list is stale — run /model again.", show_alert=True)
        return

    checkpoint = checkpoints[index]
    state: BotState = context.bot_data["state"]
    state.set_checkpoint(update.effective_chat.id, checkpoint)

    profiles: list[ModelProfile] = context.bot_data["profiles"]
    profile = resolve_profile(checkpoint, profiles)
    label = profile.display_name if profile else checkpoint

    await query.answer()
    await query.edit_message_text(f"Model set to: {label}")


async def generate_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await _reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    prompt_text = (message.text or "").strip()
    if not prompt_text:
        return

    chat_id = update.effective_chat.id
    state: BotState = context.bot_data["state"]
    client: ComfyClient = context.bot_data["comfy_client"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]

    checkpoint = state.get_checkpoint(chat_id)
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
        state.set_checkpoint(chat_id, checkpoint)
        context.bot_data["available_checkpoints"] = checkpoints
        await message.reply_text(
            f"No model selected yet — defaulting to {checkpoint}. Use /model to change it."
        )

    profile = resolve_profile(checkpoint, profiles)
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
        result_id = state.store_result(
            PendingResult(image_bytes=img.data, filename=img.filename, base_params=img.base_params)
        )
        await _send_result_image(message, img.data, img.filename, _post_process_keyboard(result_id))


async def postprocess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    if update.effective_user is None or not _is_authorized(settings, update.effective_user.id):
        await query.answer("Not authorized.", show_alert=True)
        return

    _, kind, result_id = query.data.split(":", 2)
    state: BotState = context.bot_data["state"]
    pending = state.get_result(result_id)
    if pending is None:
        await query.answer("That result has expired — generate a new image.", show_alert=True)
        return

    await query.answer()
    client: ComfyClient = context.bot_data["comfy_client"]
    label = "Upscaling" if kind == "upscale" else "Refining face"
    status_message = await query.message.reply_text(f"{label}…")

    try:
        result = await post_process(client, kind, pending.image_bytes, pending.filename, pending.base_params)
    except ComfyUIError as exc:
        await status_message.edit_text(f"{label} failed: {exc}")
        return
    except Exception:
        logger.exception("Unexpected error during post-processing")
        await status_message.edit_text(f"{label} failed with an unexpected error.")
        return

    await status_message.delete()
    new_result_id = state.store_result(
        PendingResult(image_bytes=result.data, filename=result.filename, base_params=result.base_params)
    )
    await _send_result_image(
        query.message, result.data, result.filename, _post_process_keyboard(new_result_id)
    )
