"""Shared authorization check, used by both handlers.py and settings_menu.py."""

from __future__ import annotations

from telegram import CallbackQuery, Update

from comfytelegram.settings import Settings


def is_authorized(settings: Settings, user_id: int | None) -> bool:
    if not settings.allowed_user_ids:
        return True
    return user_id in settings.allowed_user_ids


async def reject_if_unauthorized(update: Update, settings: Settings) -> bool:
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
