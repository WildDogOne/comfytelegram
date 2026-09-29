import datetime
from unittest.mock import MagicMock

import pytest
from telegram import (
    Chat,
    Document,
    ForumTopicClosed,
    ForumTopicCreated,
    ForumTopicEdited,
    ForumTopicReopened,
    Message,
    MessageEntity,
    User,
)

from comfytelegram.main import _UNHANDLED_FILTER


def _message(
    text: str | None = None,
    *,
    photo=None,
    sticker=None,
    document=None,
    forum_topic_edited=None,
    **api_kwargs,
) -> Message:
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
        document=document,
        forum_topic_edited=forum_topic_edited,
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


def test_an_image_sent_as_a_file_is_claimed_by_the_import_handler():
    """`document_message` is bound to `filters.Document.IMAGE` — an image
    sent as a file is the archive round trip's whole input path, so it must
    not fall through to `_unhandled_message`."""
    png = Document(file_id="f", file_unique_id="u", file_name="x.png", mime_type="image/png")
    assert not _caught(_message(document=png))


def test_a_non_image_file_is_still_unhandled():
    """Only images are claimed — a PDF or a zip has nothing to import."""
    pdf = Document(file_id="f", file_unique_id="u", file_name="x.pdf", mime_type="application/pdf")
    assert _caught(_message(document=pdf))


def test_renaming_a_forum_topic_is_left_alone():
    """Renaming a thread sends a service `Message` with `forum_topic_edited`
    set and no text — before `filters.StatusUpdate.ALL` was carved out of
    `_UNHANDLED_FILTER`, that matched the same "neither text, photo, nor
    image" branch a sticker does, so every thread rename got a spurious
    "I didn't understand that" reply for an action nobody typed at all."""
    message = _message(forum_topic_edited=ForumTopicEdited(name="New Topic Name"))
    assert _caught(message) is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("forum_topic_created", ForumTopicCreated(name="general", icon_color=0)),
        ("forum_topic_closed", ForumTopicClosed()),
        ("forum_topic_reopened", ForumTopicReopened()),
        ("pinned_message", "sentinel"),  # StatusUpdate.PINNED_MESSAGE only checks truthiness
    ],
)
def test_other_service_messages_are_left_alone_too(field, value):
    """Same fix, same reasoning, for the other common group/topic
    housekeeping actions that arrive as a service message rather than
    something a user typed. Built directly rather than through `_message`,
    since that helper only forwards a fixed set of named fields and routes
    anything else into `api_kwargs` (for genuinely non-standard fields like
    the rich-message test above) rather than setting a real attribute."""
    message = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=Chat(1, "private"),
        from_user=User(1, "u", False),
        **{field: value},
    )
    message.set_bot(MagicMock())
    assert _caught(message) is False
