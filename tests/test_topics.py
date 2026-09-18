from unittest.mock import MagicMock

from comfytelegram.topics import NO_TOPIC, pop_pending, set_pending, topic_id


def test_topic_id_returns_no_topic_for_a_message_outside_any_topic():
    message = MagicMock(message_thread_id=None)
    assert topic_id(message) == NO_TOPIC


def test_topic_id_returns_no_topic_for_a_missing_message():
    assert topic_id(None) == NO_TOPIC


def test_topic_id_returns_the_message_thread_id_when_set():
    message = MagicMock(message_thread_id=99)
    assert topic_id(message) == 99


def test_set_then_pop_pending_round_trips_within_the_same_topic():
    chat_data: dict = {}
    message = MagicMock(message_thread_id=5)

    set_pending(chat_data, "awaiting_field", message, "steps")

    assert pop_pending(chat_data, "awaiting_field", message) == "steps"
    assert "awaiting_field" not in chat_data


def test_pending_state_does_not_leak_across_topics():
    chat_data: dict = {}
    topic_a = MagicMock(message_thread_id=1)
    topic_b = MagicMock(message_thread_id=2)

    set_pending(chat_data, "awaiting_character_edit", topic_a, "fox")

    assert pop_pending(chat_data, "awaiting_character_edit", topic_b) is None
    assert pop_pending(chat_data, "awaiting_character_edit", topic_a) == "fox"


def test_pop_pending_returns_none_when_nothing_is_pending():
    chat_data: dict = {}
    message = MagicMock(message_thread_id=None)
    assert pop_pending(chat_data, "awaiting_field", message) is None
