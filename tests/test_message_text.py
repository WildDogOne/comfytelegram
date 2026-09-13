import datetime
from unittest.mock import MagicMock

from telegram import Chat, Message, User

from comfytelegram.message_text import message_text


def _message(text: str | None = None, **api_kwargs) -> Message:
    message = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.UTC),
        chat=Chat(1, "private"),
        from_user=User(1, "u", False),
        text=text,
        api_kwargs=api_kwargs or None,
    )
    message.set_bot(MagicMock())
    return message


def _rich(*paragraphs: str) -> dict:
    return {"rich_message": {"blocks": [{"type": "paragraph", "text": p} for p in paragraphs]}}


def test_plain_text_is_returned_as_is():
    assert message_text(_message("a red fox")) == "a red fox"


def test_rich_message_blocks_are_reassembled():
    """Newer clients send multi-paragraph messages this way, leaving
    `Message.text` None — the case that used to make the bot silently
    ignore a perfectly ordinary prompt."""
    message = _message(None, **_rich("a red fox", "", "---", "blurry, watermark"))
    assert message.text is None
    assert message_text(message) == "a red fox\n\n---\nblurry, watermark"


def test_rich_message_preserves_the_negative_prompt_separator():
    """The paragraph breaks have to survive reassembly or the "---" block
    separator stops being on a line of its own and no longer splits."""
    from comfytelegram.handlers import _split_negative_prompt

    recovered = message_text(_message(None, **_rich("1girl, outdoors", "---", "blurry")))
    assert _split_negative_prompt(recovered) == ("1girl, outdoors", "blurry")


def test_message_without_text_in_either_form_is_none():
    assert message_text(_message(None)) is None
    assert message_text(None) is None


def test_malformed_rich_message_is_treated_as_no_text():
    assert message_text(_message(None, rich_message="nonsense")) is None
    assert message_text(_message(None, rich_message={"blocks": "nonsense"})) is None
    assert message_text(_message(None, **_rich("", ""))) is None
