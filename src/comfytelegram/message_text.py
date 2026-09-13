"""Reading a message's text across both forms Telegram sends it in.

Newer Telegram clients send some messages — multi-paragraph ones in
particular — as a "rich message" carrying `{"blocks": [...]}` instead of
filling the plain `text` field. python-telegram-bot 22.8 has no model for
that field, so it lands in `Message.api_kwargs` untouched and
`Message.text` is None. `filters.TEXT` therefore doesn't match, which
before this module meant such a message matched no handler at all and was
dropped without a reply or a log line — the bot silently ignoring a
perfectly ordinary prompt, reproducibly but only for messages composed
with more than one paragraph.

`message_text` reads whichever form arrived, and `TEXT_CONTENT` is the
filter counterpart to use in place of `filters.TEXT`.
"""

from __future__ import annotations

from telegram import Message
from telegram.ext import filters


def message_text(message: Message | None) -> str | None:
    """This message's text, from `Message.text` or, failing that, a
    reconstruction of the `rich_message` blocks in `api_kwargs`. Blocks are
    joined with newlines, which restores the paragraph breaks the sender
    typed — the "---" negative-prompt separator (see
    `handlers._split_negative_prompt`) depends on those surviving. Returns
    None when the message carries no text at all in either form (a sticker,
    a voice note), so callers can keep treating None as "not a text
    message"."""
    if message is None:
        return None
    if message.text is not None:
        return message.text

    rich_message = message.api_kwargs.get("rich_message")
    if not isinstance(rich_message, dict):
        return None
    blocks = rich_message.get("blocks")
    if not isinstance(blocks, list):
        return None

    texts = [block.get("text") for block in blocks if isinstance(block, dict)]
    if not any(isinstance(text, str) and text for text in texts):
        return None
    return "\n".join(text for text in texts if isinstance(text, str))


class _TextContent(filters.MessageFilter):
    """`filters.TEXT`, but also matching the `rich_message` form it misses."""

    __slots__ = ()

    def filter(self, message: Message) -> bool:
        return message_text(message) is not None


#: Drop-in replacement for `filters.TEXT` that doesn't miss rich messages.
TEXT_CONTENT = _TextContent(name="TEXT_CONTENT")
