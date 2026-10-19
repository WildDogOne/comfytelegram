"""In-progress status for long-running jobs: a Telegram message draft in a
private chat, an edited status message everywhere else.

`sendMessageDraft` (Bot API 9.3/9.5) shows an animated, temporary preview
that is never stored in the chat — the natural home for "Generating… 40%".
Its behaviour, measured live against the real API (the docs say little):

- Only one draft shows per chat. Two `draft_id`s updated side by side
  flicker between each other, so every job running in a chat shares one
  `DraftBoard` that lists them all, one line each.
- A draft stays where it first appeared. Updating the same `draft_id`
  after the bot has sent a message keeps it *above* that message; a new
  `draft_id` shows below it (and replaces the old one). So the board takes
  a fresh id whenever a job's result has gone out (`Status.message_sent`,
  and every time a job leaves the board).
- A draft outlives at least 45s of silence, and nothing clears it early:
  sending a message, emptying it, or setting it to the final text first
  all left it up ~20s more. So when the last job leaves, the board sets
  the draft to `DRAFT_DONE_TEXT`, which is at least true while it lingers.
- Updates every 0.5s were fine; every 0.25s hit "retry after 3-4s" after
  ~20 updates. Jobs update at most every `PROGRESS_EDIT_INTERVAL` (2s) on
  their own, and the board never sends more often than
  `DRAFT_MIN_INTERVAL` however many jobs are running.

`Status` stands in for the status `Message` the handlers used to create
and edit directly. Its `edit_text`/`delete` keep a `Message`'s meaning —
`edit_text` sets the job's *lasting* text (an error, "No faces found") —
while `update` is the new, transient channel for progress. In draft mode a
lasting text is sent as a real message, since a draft would vanish; with no
`DraftBoards` in `bot_data`, outside a private chat, or once a draft call
fails, `Status` is just the old edited message.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import warnings
from typing import Any

from telegram import Bot, Message
from telegram.constants import ChatType
from telegram.error import RetryAfter, TelegramError
from telegram.warnings import PTBDeprecationWarning

logger = logging.getLogger(__name__)

#: `bot_data` key of the application's `DraftBoards`.
DRAFT_BOARDS_KEY = "draft_boards"

#: Re-send an unchanged draft this often. It was measured lasting 45s+,
#: so this leaves plenty of margin for queue waits and model loads, which
#: report no progress.
DRAFT_KEEPALIVE_SECONDS = 20.0

#: Minimum gap between two draft sends for one chat, whatever the number
#: of jobs updating it.
DRAFT_MIN_INTERVAL = 1.0

#: A job line still on the board after this long is dropped: a safety net
#: for a job whose handler died without closing its `Status`, which would
#: otherwise keep the draft alive forever via the keep-alive.
DRAFT_MAX_AGE_SECONDS = 30 * 60.0

#: What the draft says once its last job has left the board.
DRAFT_DONE_TEXT = "✅ Done"


class DraftBoard:
    """The one draft of one private chat (or private-chat topic), listing
    every job currently running there. A background worker task does all
    the sending, so `set`/`remove` never block the job reporting them; it
    exits once the board is empty and its last text is out."""

    def __init__(self, bot: Bot, chat_id: int, message_thread_id: int | None) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_thread_id = message_thread_id
        self._lines: dict[int, str] = {}
        self._added: dict[int, float] = {}
        self._tasks: dict[int, asyncio.Task | None] = {}
        self._next_key = 0
        self._draft_id = random.randint(1, 2**30)
        self._sent_with_id = False
        self._last_sent = 0.0
        self._dirty = asyncio.Event()
        self._worker: asyncio.Task | None = None
        #: Set once a draft call fails for any reason other than a rate
        #: limit; every `Status` on this board then falls back to a message.
        self.failed = False

    def add(self, text: str) -> int:
        """Put a new job line on the board; returns its key."""
        key = self._next_key
        self._next_key += 1
        self._lines[key] = text
        self._added[key] = time.monotonic()
        self._tasks[key] = asyncio.current_task()
        self._changed()
        return key

    def set(self, key: int, text: str) -> None:
        if key in self._lines and self._lines[key] != text:
            self._lines[key] = text
            self._changed()

    def remove(self, key: int) -> None:
        """Take a job off the board. The draft moves below whatever the job
        just sent (a fresh `draft_id`), or turns into `DRAFT_DONE_TEXT` if
        it was the last one."""
        if self._lines.pop(key, None) is None:
            return
        self._added.pop(key, None)
        self._tasks.pop(key, None)
        if self._lines:
            self.new_draft_id()
        self._changed()

    def new_draft_id(self) -> None:
        """Show the draft below the chat's newest message from here on."""
        self._draft_id += 1
        self._sent_with_id = False
        self._changed()

    def _changed(self) -> None:
        if self.failed:
            return
        self._dirty.set()
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    def _render(self) -> str:
        # Telegram animates its own "..." after a draft's text, so the
        # trailing "…" that status messages end in would show twice.
        return "\n".join(self._lines.values()).removesuffix("…")

    def _prune(self) -> None:
        """Drop lines of jobs that can't be running any more: their task is
        done (the handler returned or crashed without closing its
        `Status`), or they've been up longer than `DRAFT_MAX_AGE_SECONDS`."""
        now = time.monotonic()
        for key in list(self._lines):
            task = self._tasks.get(key)
            if (task is not None and task.done()) or now - self._added[key] > (
                DRAFT_MAX_AGE_SECONDS
            ):
                self._lines.pop(key)
                self._added.pop(key)
                self._tasks.pop(key)

    async def _run(self) -> None:
        try:
            while True:
                try:
                    await asyncio.wait_for(self._dirty.wait(), timeout=DRAFT_KEEPALIVE_SECONDS)
                except TimeoutError:
                    pass
                self._dirty.clear()
                wait = self._last_sent + DRAFT_MIN_INTERVAL - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._prune()
                if not self._lines:
                    if self._sent_with_id:
                        await self._send(DRAFT_DONE_TEXT)
                        self._draft_id += 1
                        self._sent_with_id = False
                    if not self._lines and not self._dirty.is_set():
                        return
                    continue
                if not await self._send(self._render()):
                    return
        finally:
            self._worker = None

    async def _send(self, text: str) -> bool:
        """Send `text` as the draft; False once the board has failed."""
        try:
            await self._bot.send_message_draft(
                self._chat_id, self._draft_id, text, message_thread_id=self._message_thread_id
            )
        except RetryAfter as exc:
            with warnings.catch_warnings():
                # PTB 22 warns that `retry_after` will become a timedelta.
                warnings.simplefilter("ignore", PTBDeprecationWarning)
                delay = exc.retry_after
            seconds = delay.total_seconds() if hasattr(delay, "total_seconds") else float(delay)
            logger.debug("Draft rate-limited for %ss", seconds)
            self._last_sent = time.monotonic() + seconds
            self._dirty.set()
            return True
        except TelegramError:
            logger.warning(
                "Message drafts failed in chat %s; using status messages instead",
                self._chat_id,
                exc_info=True,
            )
            self.failed = True
            return False
        self._last_sent = time.monotonic()
        self._sent_with_id = True
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()


class DraftBoards:
    """Every chat's `DraftBoard`, kept in `bot_data[DRAFT_BOARDS_KEY]`."""

    def __init__(self) -> None:
        self._boards: dict[tuple[int, int | None], DraftBoard] = {}

    def get(self, bot: Bot, chat_id: int, message_thread_id: int | None) -> DraftBoard:
        key = (chat_id, message_thread_id)
        board = self._boards.get(key)
        if board is None:
            board = self._boards[key] = DraftBoard(bot, chat_id, message_thread_id)
        return board

    def cancel_all(self) -> None:
        for board in self._boards.values():
            board.cancel()


class Status:
    """One job's in-progress status. Create with `start_reply_status` or
    `start_chat_status`, report progress with `update`, and end it with
    `edit_text` (a lasting outcome), `delete` (the result went out on its
    own) or `close`."""

    def __init__(
        self,
        bot: Bot | None,
        chat_id: int,
        message_thread_id: int | None,
        *,
        reply_to: Message | None,
        board: DraftBoard | None,
        message: Message | None = None,
        owns_message: bool = True,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_thread_id = message_thread_id
        self._reply_to = reply_to
        self._board = board
        self._key: int | None = None
        self._message = message
        self._owns_message = owns_message

    @property
    def uses_draft(self) -> bool:
        return self._key is not None and self._board is not None and not self._board.failed

    async def _start(self, text: str | None) -> None:
        if self._board is not None and not self._board.failed:
            self._key = self._board.add(text or "…")
        elif self._message is None and text is not None:
            self._message = await self._send(text, disable_notification=True)

    async def _send(self, text: str, **kwargs: Any) -> Message:
        if self._reply_to is not None:
            return await self._reply_to.reply_text(text, **kwargs)
        return await self._bot.send_message(
            self._chat_id, text, message_thread_id=self._message_thread_id, **kwargs
        )

    def _leave_board(self) -> None:
        if self._board is not None and self._key is not None:
            self._board.remove(self._key)
        self._key = None

    async def update(self, text: str) -> None:
        """Transient progress text. Never raises: a progress report that
        doesn't get through (rate limit, unchanged text) just gets skipped."""
        if self.uses_draft:
            self._board.set(self._key, text)
            return
        self._key = None
        try:
            if self._message is None:
                self._message = await self._send(text, disable_notification=True)
            else:
                await self._message.edit_text(text)
        except Exception:
            logger.debug("Progress update skipped (rate-limited or unchanged)", exc_info=True)

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        """The job's lasting text, like `Message.edit_text` on the old status
        message. In draft mode it's sent as a message (once; later calls
        edit that message), since a draft would disappear."""
        self._leave_board()
        if self._message is None:
            self._message = await self._send(text, **kwargs)
        else:
            await self._message.edit_text(text, **kwargs)

    async def delete(self) -> None:
        """End the job with no lasting text of its own (its result went out
        as separate messages)."""
        self._leave_board()
        if self._message is not None and self._owns_message:
            await self._message.delete()
            self._message = None

    async def close(self) -> None:
        """End the job, leaving any message it sent in place."""
        self._leave_board()

    def message_sent(self) -> None:
        """The job just sent a message (a `/stream` image) and keeps going:
        show the draft below it from now on."""
        if self.uses_draft:
            self._board.new_draft_id()


def _boards(bot_data: Any) -> DraftBoards | None:
    try:
        boards = bot_data.get(DRAFT_BOARDS_KEY)
    except AttributeError:
        return None
    return boards if isinstance(boards, DraftBoards) else None


async def start_reply_status(
    bot_data: Any, message: Message, text: str | None, *, existing: Message | None = None
) -> Status:
    """A `Status` for a job started by `message` (lasting texts reply to
    it). In edit mode `text` is posted right away, as a reply, unless
    `existing` is given: then that message is edited by `update` instead
    (and never deleted), and left as it is until the first one. `text` is
    the draft's first line in draft mode; None starts it at "…"."""
    chat = message.chat
    thread_id = message.message_thread_id if message.is_topic_message else None
    boards = _boards(bot_data)
    bot = None
    board = None
    if boards is not None and chat.type == ChatType.PRIVATE:
        bot = message.get_bot()
        board = boards.get(bot, chat.id, thread_id)
    status = Status(
        bot,
        chat.id,
        thread_id,
        reply_to=message,
        board=board,
        message=existing,
        owns_message=existing is None,
    )
    await status._start(text)
    return status


async def start_chat_status(
    bot_data: Any, bot: Bot, chat_id: int, text: str, *, message_thread_id: int | None = None
) -> Status:
    """A `Status` for a job with no message to reply to (a background
    task). Private chats are told apart by their positive id."""
    boards = _boards(bot_data)
    board = None
    if boards is not None and chat_id > 0:
        board = boards.get(bot, chat_id, message_thread_id)
    status = Status(bot, chat_id, message_thread_id, reply_to=None, board=board)
    await status._start(text)
    return status
