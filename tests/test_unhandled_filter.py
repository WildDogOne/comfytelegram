import datetime
from unittest.mock import MagicMock

import pytest
from telegram import Chat, Message, MessageEntity, User

from comfytelegram.main import _UNHANDLED_FILTER


def _message(text: str | None = None, *, photo=None, sticker=None, **api_kwargs) -> Message:
    entities = []
    if text is not None and text.startswith("/"):
        entities = [MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
    message = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=Chat(1, "private"),
        from_user=User(1, "u", False),
        text=text,
        entities=entities,
        photo=photo or (),
        sticker=sticker,
        api_kwargs=api_kwargs or None,
    )
    message.set_bot(MagicMock())
    return message


def _caught(message: Message) -> bool:
    update = MagicMock(
        effective_message=message,
        message=message,
        channel_post=None,
        edited_message=None,
        edited_channel_post=None,
    )
    return bool(_UNHANDLED_FILTER.check_update(update))


@pytest.mark.parametrize(
    "text",
    [
        "a red fox in a forest",
        "a fox\n---\nblurry",  # the "---" negative-prompt block separator
        "/help",
        "/stream a fox",
        "/characters",  # must not be mistaken for unknown via the /character prefix
    ],
)
def test_messages_the_real_handlers_claim_are_left_alone(text):
    """The fallback's filter is the complement of the real handlers' — if it
    also matched these, every prompt would get a spurious second reply."""
    assert _caught(_message(text)) is False


def test_photos_are_left_alone():
    assert _caught(_message(photo=(MagicMock(),))) is False


@pytest.mark.parametrize("text", ["/genrate a fox", "/summon a dragon"])
def test_unknown_commands_are_caught(text):
    """`filters.COMMAND` keeps these out of `generate_message` and no
    `CommandHandler` claims them, so without the fallback they vanish in
    total silence — the "I sent something and nothing happened" symptom."""
    assert _caught(_message(text)) is True


def test_non_text_non_photo_messages_are_caught():
    assert _caught(_message(sticker=MagicMock())) is True


def test_rich_message_prompts_are_left_to_the_real_handler():
    """A multi-paragraph "rich message" has `text=None`, so the old
    `filters.TEXT`-based complement claimed it and the bot answered "I
    didn't understand that" to an ordinary prompt."""
    message = _message(None, rich_message={"blocks": [{"type": "paragraph", "text": "a red fox"}]})
    assert _caught(message) is False
