"""Bot process entrypoint: wires settings, the ComfyUI client, and Telegram
handlers together, then runs polling."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from comfytelegram.comfy_client import ComfyClient
from comfytelegram.handlers import (
    generate_message,
    help_command,
    model_callback,
    model_command,
    postprocess_callback,
    start,
)
from comfytelegram.profiles import load_profiles
from comfytelegram.settings import Settings, load_settings
from comfytelegram.settings_menu import settings_callback, settings_command
from comfytelegram.state import BotState
from comfytelegram.storage import Storage

logger = logging.getLogger(__name__)


async def _post_init(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    client = ComfyClient(settings.comfyui_http_base, settings.comfyui_ws_base)
    await client.__aenter__()
    application.bot_data["comfy_client"] = client
    logger.info("Connected to ComfyUI at %s", settings.comfyui_http_base)


async def _post_shutdown(application: Application) -> None:
    client: ComfyClient | None = application.bot_data.get("comfy_client")
    if client is not None:
        await client.__aexit__(None, None, None)
    storage: Storage | None = application.bot_data.get("storage")
    if storage is not None:
        storage.close()


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Without this, python-telegram-bot just logs unhandled exceptions
    server-side and the user sees nothing — a job can fail silently mid-flow.
    (This is exactly how a real bug slipped through the first live test:
    Telegram's sendPhoto 10MB limit rejected a 4x-upscaled image, and there
    was no error handler to surface it back to the chat.)"""
    logger.error("Unhandled exception while processing an update", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat is not None:
        try:
            await context.bot.send_message(
                update.effective_chat.id, "Something went wrong handling that — check the bot's logs."
            )
        except Exception:
            logger.exception("Failed to notify the chat about the error")


def build_application(settings: Settings) -> Application:
    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    application.bot_data["settings"] = settings
    application.bot_data["profiles"] = load_profiles(settings.model_profiles_dir)
    application.bot_data["state"] = BotState()
    application.bot_data["storage"] = Storage(settings.state_db_path)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))
    application.add_handler(CallbackQueryHandler(postprocess_callback, pattern=r"^pp:"))
    application.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^st:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_message))
    application.add_error_handler(_error_handler)

    return application


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    application = build_application(settings)
    logger.info("Starting comfytelegram bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
