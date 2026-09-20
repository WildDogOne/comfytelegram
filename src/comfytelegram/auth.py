"""Shared authorization check, used by both handlers.py and settings_menu.py."""

from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qsl

from telegram import CallbackQuery, Update

from comfytelegram.settings import Settings


def is_authorized(settings: Settings, user_id: int | None) -> bool:
    """True if `user_id` may use the bot — always true when `ALLOWED_USER_IDS`
    is unset (no allowlist configured means anyone can use it)."""
    if not settings.allowed_user_ids:
        return True
    return user_id in settings.allowed_user_ids


async def reject_if_unauthorized(update: Update, settings: Settings) -> bool:
    """For message-based handlers: if `update`'s sender isn't authorized,
    reply with a rejection message and return True (caller should stop).
    Returns False, with no message sent, if they're authorized."""
    user = update.effective_user
    if user is not None and is_authorized(settings, user.id):
        return False
    if update.effective_message is not None:
        await update.effective_message.reply_text("You're not authorized to use this bot.")
    return True


async def reject_if_unauthorized_callback(
    query: CallbackQuery, user_id: int | None, settings: Settings
) -> bool:
    """`reject_if_unauthorized`'s counterpart for callback-query handlers —
    there's no message to reply into, so the rejection surfaces as an
    alert toast on the tapped button instead."""
    if is_authorized(settings, user_id):
        return False
    await query.answer("Not authorized.", show_alert=True)
    return True


def validate_webapp_init_data(init_data: str, bot_token: str) -> dict[str, str] | None:
    """Verify a Telegram WebApp's `Telegram.WebApp.initData` string per
    Telegram's documented algorithm
    (https://core.telegram.org/bots/webapps#validating-data-received-via-the-web-app),
    returning the parsed fields if it's genuinely signed by this bot's
    token, or None if the signature is missing/invalid.

    This is the only thing standing between inpaint_relay (a public,
    unauthenticated-by-design host — see `handlers.py`'s
    `poll_inpaint_jobs`) and running an arbitrary mask through ComfyUI: the
    relay itself never sees the bot token and can't check this, so
    comfytelegram must, after pulling a submitted mask back from the relay
    and before acting on it."""
    try:
        parsed = dict(parse_qsl(init_data, strict_parsing=True))
    except ValueError:
        return None
    received_hash = parsed.pop("hash", None)
    if not received_hash:
        return None
    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(parsed.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_hash, received_hash):
        return None
    return parsed
