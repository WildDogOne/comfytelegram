"""Bot process entrypoint: wires settings, the ComfyUI client, and Telegram
handlers together, then runs polling."""

from __future__ import annotations

import asyncio
import logging

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from comfytelegram.comfy_client import ComfyClient
from comfytelegram.handlers import (
    AGAIN_CALLBACK_PREFIX,
    GENERATE_FROM_PROMPT_CALLBACK_PREFIX,
    HAND_POINT_CALLBACK_PREFIX,
    STREAM_CANCEL_CALLBACK_DATA,
    again_callback,
    character_callback,
    character_command,
    characters_command,
    generate_from_prompt_callback,
    generate_message,
    hand_point_callback,
    help_command,
    model_callback,
    model_command,
    photo_message,
    postprocess_callback,
    start,
    stop_command,
    stream_cancel_callback,
    stream_command,
    tagcheck_command,
    tags_command,
)
from comfytelegram.message_text import TEXT_CONTENT, message_text
from comfytelegram.profiles import load_profiles
from comfytelegram.settings import Settings, load_settings
from comfytelegram.settings_menu import settings_callback, settings_command
from comfytelegram.storage import Storage
from comfytelegram.tags import TagDatabase, refresh_if_stale

logger = logging.getLogger(__name__)

#: Every `/command` the bot answers, as (name, callback, menu description).
#: Single source of truth: `build_application` registers a `CommandHandler`
#: per entry, `_UNHANDLED_FILTER` below is built from the same names, and
#: `_post_init` pushes the descriptions to Telegram's own "/" command menu
#: (`Bot.set_my_commands` — the same registry BotFather's `/setcommands`
#: edits, settable directly instead) — so adding a command here can't leave
#: any of the three disagreeing about what's known.
_COMMANDS = (
    ("start", start, "Show the welcome message and command keyboard"),
    ("help", help_command, "Show the command list"),
    ("model", model_command, "Pick a checkpoint"),
    ("settings", settings_command, "View or change generation defaults for this model"),
    ("character", character_command, "Save or delete a reusable character design"),
    ("characters", characters_command, "List, activate, edit, or rename saved characters"),
    ("stream", stream_command, "Generate images back-to-back until /stop"),
    ("stop", stop_command, "Stop a running /stream"),
    ("tags", tags_command, "Search danbooru/e621 tags to build a prompt"),
    ("tagcheck", tagcheck_command, "Check a prompt's tags against the tag database"),
)

#: Matches exactly the messages none of the real handlers can claim: a
#: `/command` that isn't one of `_COMMANDS` (`filters.COMMAND` keeps it out
#: of `generate_message`, and no `CommandHandler` wants it), or a message
#: that's neither text nor a photo (a sticker, a voice note, a document).
#: Both used to be dropped in total silence — no reply, no log line, the
#: bot simply appearing to ignore you. Pairs with `TEXT_CONTENT` rather
#: than `filters.TEXT` so it stays the exact complement of the real
#: handlers' filters, and doesn't claim rich messages back off them.
_UNHANDLED_FILTER = (
    filters.COMMAND
    & ~filters.Regex(rf"^/({'|'.join(name for name, _, _ in _COMMANDS)})(@\w+)?(\s|$)")
) | ~(TEXT_CONTENT | filters.PHOTO)


async def _log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Record every update the bot receives, before dispatch. Registered in
    its own group (-1) so it observes without consuming. Exists because
    "I sent a prompt and nothing happened" is otherwise indistinguishable
    between "the update never arrived" and "it arrived but no handler
    matched it" — with this line in the log, the two look different."""
    text = message_text(update.effective_message)
    logger.info(
        "update %s chat=%s %s",
        update.update_id,
        update.effective_chat.id if update.effective_chat else None,
        f"text={text!r}" if text is not None else update,
    )


async def _unhandled_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer anything `_UNHANDLED_FILTER` catches instead of ignoring it.
    Runs in group 1, i.e. after the real handlers, and its filter is the
    complement of theirs, so it can't double-reply to a message they
    already took."""
    message = update.effective_message
    logger.warning("No handler matched update %s: %r", update.update_id, message_text(message))
    await message.reply_text(
        "I didn't understand that. Send a plain text prompt to generate, a "
        "photo to analyze it, or /help for the command list.\n\n(A message "
        'starting with "/" is read as a command — drop the slash to use it '
        "as a prompt.)"
    )


async def _post_init(application: Application) -> None:
    """python-telegram-bot startup hook: open the `ComfyClient` (needs a
    running event loop, so it can't be built at `build_application` time),
    stash it in `bot_data` alongside the other shared singletons, push
    `_COMMANDS`' descriptions to Telegram's "/" command menu — the same
    registry BotFather's `/setcommands` edits, so this keeps the menu in
    sync with the bot's actual commands on every startup instead of
    needing a manual BotFather edit whenever one is added/removed/
    reworded — and, unless disabled, fire the tag database's staleness
    check as a background task (see `_start_tag_db_refresh`)."""
    settings: Settings = application.bot_data["settings"]
    client = ComfyClient(settings.comfyui_http_base, settings.comfyui_ws_base)
    await client.__aenter__()
    application.bot_data["comfy_client"] = client
    logger.info("Connected to ComfyUI at %s", settings.comfyui_http_base)

    commands = [BotCommand(name, description) for name, _, description in _COMMANDS]
    await application.bot.set_my_commands(commands)
    logger.info(
        "Registered %d bot commands with Telegram: %s",
        len(commands),
        ", ".join(c.command for c in commands),
    )

    _start_tag_db_refresh(application, settings)


def _start_tag_db_refresh(application: Application, settings: Settings) -> None:
    """Fire `refresh_if_stale` as an unawaited background task, so a fresh
    checkout self-populates `tags_db` and an existing one stays within
    `tag_db_max_age_days` without blocking bot startup on a multi-MB GitHub
    download (or failing it outright if there's no network right now — a
    failed refresh just logs a warning and leaves /tags/tagcheck reporting
    "no data yet" or serving whatever was already imported). Kept as its
    own function so `build_application`'s test coverage can leave
    `tag_db_auto_update` off without needing a running event loop. The task
    is stashed in `bot_data` purely to keep a strong reference to it (an
    unreferenced asyncio task can be garbage-collected mid-flight) and so
    `_post_shutdown` can cancel it if it's still running."""
    if not settings.tag_db_auto_update:
        return
    tags_db: TagDatabase = application.bot_data["tags_db"]
    application.bot_data["tag_db_refresh_task"] = asyncio.create_task(
        refresh_if_stale(tags_db, settings.tag_db_max_age_days)
    )


async def _post_shutdown(application: Application) -> None:
    """python-telegram-bot shutdown hook: cancel any still-running "/stream"
    tasks and the tag database refresh task (so they don't linger as
    unawaited tasks after the event loop they belong to closes — though for
    the refresh task, cancellation only stops it between sources, since its
    actual network I/O runs in a worker thread `asyncio.to_thread` doesn't
    interrupt), then close the ComfyUI HTTP session and the sqlite
    connections cleanly."""
    for task in application.bot_data.get("active_streams", {}).values():
        task.cancel()
    tag_db_refresh_task: asyncio.Task | None = application.bot_data.get("tag_db_refresh_task")
    if tag_db_refresh_task is not None:
        tag_db_refresh_task.cancel()
    client: ComfyClient | None = application.bot_data.get("comfy_client")
    if client is not None:
        await client.__aexit__(None, None, None)
    storage: Storage | None = application.bot_data.get("storage")
    if storage is not None:
        storage.close()
    tags_db: TagDatabase | None = application.bot_data.get("tags_db")
    if tags_db is not None:
        tags_db.close()


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
                update.effective_chat.id,
                "Something went wrong handling that — check the bot's logs.",
            )
        except Exception:
            logger.exception("Failed to notify the chat about the error")


def build_application(settings: Settings) -> Application:
    """Construct the python-telegram-bot `Application`: load profiles, open
    `Storage`, register every command/callback/message handler, but don't
    start polling (see `run()`)."""
    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .concurrent_updates(True)
        .build()
    )
    application.bot_data["settings"] = settings
    application.bot_data["profiles"] = load_profiles(settings.model_profiles_dir)
    application.bot_data["storage"] = Storage(settings.state_db_path)
    application.bot_data["tags_db"] = TagDatabase(settings.tags_db_path)

    for name, callback, _ in _COMMANDS:
        application.add_handler(CommandHandler(name, callback))
    application.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))
    application.add_handler(CallbackQueryHandler(postprocess_callback, pattern=r"^pp:"))
    application.add_handler(
        CallbackQueryHandler(hand_point_callback, pattern=rf"^{HAND_POINT_CALLBACK_PREFIX}")
    )
    application.add_handler(
        CallbackQueryHandler(again_callback, pattern=rf"^{AGAIN_CALLBACK_PREFIX}")
    )
    application.add_handler(
        CallbackQueryHandler(
            generate_from_prompt_callback, pattern=rf"^{GENERATE_FROM_PROMPT_CALLBACK_PREFIX}"
        )
    )
    application.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^st:"))
    application.add_handler(CallbackQueryHandler(character_callback, pattern=r"^char:"))
    application.add_handler(
        CallbackQueryHandler(stream_cancel_callback, pattern=rf"^{STREAM_CANCEL_CALLBACK_DATA}$")
    )
    application.add_handler(MessageHandler(filters.PHOTO, photo_message))
    application.add_handler(MessageHandler(TEXT_CONTENT & ~filters.COMMAND, generate_message))

    # Group -1 runs before the real handlers and, being its own group,
    # never consumes the update — it only records that the bot saw it.
    application.add_handler(TypeHandler(Update, _log_update), group=-1)
    application.add_handler(MessageHandler(_UNHANDLED_FILTER, _unhandled_message), group=1)

    application.add_error_handler(_error_handler)

    return application


def run() -> None:
    """Entry point (`comfytelegram` console script): load settings, build
    the application, and poll Telegram for updates until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx (python-telegram-bot's HTTP backend) logs every request's full
    # URL at INFO — including the bot token, since Telegram's API embeds it
    # in the path (https://api.telegram.org/bot<TOKEN>/<method>). Left at
    # the basicConfig default, that's the token in plaintext in every log
    # line, forever. WARNING still surfaces real HTTP failures.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings()
    application = build_application(settings)
    logger.info("Starting comfytelegram bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
