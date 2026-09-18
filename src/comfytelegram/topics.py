"""Scoping "awaiting free-text reply" `chat_data` flags to Telegram forum
topics.

`context.chat_data` is scoped per chat, not per topic. That's fine for a
normal chat, but a group with Telegram's Topics feature enabled effectively
runs several independent conversations side by side in one chat — without
this, a flag like `awaiting_field` (set while editing settings in one
topic) is either silently clobbered by an unrelated edit started in another
topic, or wrongly consumes text typed in that other topic as the pending
answer, since both look identical to a chat-scoped flag. Filing every such
flag under `topic_id()` keeps them independent per topic; a chat with no
topics (the common case) just files everything under `NO_TOPIC`, so
behavior there is unchanged.
"""

from __future__ import annotations

from typing import Any

from telegram import Message

#: Key for messages outside any forum topic — a regular group/DM, or a
#: forum group's General topic, which Telegram sends with no
#: `message_thread_id` at all.
NO_TOPIC = 0


def topic_id(message: Message | None) -> int:
    """`message`'s forum topic id, or `NO_TOPIC` if it isn't inside one."""
    if message is None:
        return NO_TOPIC
    return message.message_thread_id or NO_TOPIC


def set_pending(chat_data: dict[str, Any], key: str, message: Message | None, value: Any) -> None:
    """Set `key`'s pending free-text-reply value, scoped to `message`'s
    topic."""
    chat_data.setdefault(key, {})[topic_id(message)] = value


def pop_pending(chat_data: dict[str, Any], key: str, message: Message | None) -> Any:
    """Pop and return `key`'s pending value for `message`'s topic, or None
    if nothing is pending there. Drops `key` entirely once its last topic
    entry is gone, so an idle chat's `chat_data` doesn't accumulate empty
    per-topic dicts."""
    topics = chat_data.get(key)
    if not topics:
        return None
    value = topics.pop(topic_id(message), None)
    if not topics:
        chat_data.pop(key, None)
    return value
