from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from comfytelegram.fav_menu import (
    FAV_HELP,
    _category_pick_keyboard,
    _entry_keyboard,
    _parse_save,
    _tag_info_lines,
    fav_callback,
    fav_command,
    favs_command,
    handle_fav_edit_message,
)
from comfytelegram.storage import Storage
from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.schema import TagRow, TagSource


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


@pytest.fixture
def tags_db(tmp_path: Path) -> TagDatabase:
    db = TagDatabase(tmp_path / "tags.sqlite3")
    db.replace_source(TagSource.DANBOORU, [TagRow("wlop", 1, 4213, ())])
    yield db
    db.close()


def _context(storage: Storage, tags_db: TagDatabase | None = None) -> MagicMock:
    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "tags_db": tags_db,
    }
    context.chat_data = {}
    return context


def _command_update(text: str, user_id: int = 7) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_message.text = text
    update.effective_message.message_thread_id = None
    update.effective_message.api_kwargs = {}
    update.effective_message.reply_text = AsyncMock()
    return update


def _callback_update(data: str, user_id: int = 7) -> tuple[MagicMock, AsyncMock]:
    query = AsyncMock()
    query.data = data
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_user.id = user_id
    return update, query


def _buttons(markup) -> list:
    return [b for row in markup.inline_keyboard for b in row]


# --- storage ---------------------------------------------------------------


def test_favorites_are_per_user(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "by wlop")
    storage.save_favorite(2, "dyn", "pose", "dynamic pose")
    assert [f["name"] for f in storage.list_favorites(1)] == ["wlop"]
    assert storage.get_favorite(2, "wlop") is None


def test_save_favorite_overwrites_same_name(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "by wlop", "first")
    storage.save_favorite(1, "wlop", "style", "wlop style", "second")
    assert storage.get_favorite(1, "wlop") == {
        "name": "wlop",
        "category": "style",
        "text": "wlop style",
        "note": "second",
    }
    assert len(storage.list_favorites(1)) == 1


def test_list_favorite_categories_counts_per_category(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "by wlop")
    storage.save_favorite(1, "sakimi", "artist", "sakimichan")
    storage.save_favorite(1, "dyn", "pose", "dynamic pose")
    assert storage.list_favorite_categories(1) == [("artist", 2), ("pose", 1)]
    assert [f["name"] for f in storage.list_favorites(1, "artist")] == ["sakimi", "wlop"]


def test_delete_favorite_reports_whether_it_existed(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "by wlop")
    assert storage.delete_favorite(1, "wlop") is True
    assert storage.delete_favorite(1, "wlop") is False


# --- parsing ---------------------------------------------------------------


def test_parse_save_with_note():
    assert _parse_save("Artist WLOP | by wlop | painterly") == (
        "artist",
        "wlop",
        "by wlop",
        "painterly",
    )


def test_parse_save_without_note_keeps_commas_in_text():
    assert _parse_save("pose dyn | dynamic pose, foreshortening") == (
        "pose",
        "dyn",
        "dynamic pose, foreshortening",
        "",
    )


@pytest.mark.parametrize(
    "body",
    [
        "artist wlop",  # no text
        "wlop | by wlop",  # no category
        "artist my wlop | by wlop",  # name with a space
        "artist wl$op | by wlop",  # bad name chars
        "art!st wlop | by wlop",  # bad category chars
        "artist wlop |   ",  # empty text
        "artist wlop | a | b | c",  # too many fields
    ],
)
def test_parse_save_rejects_bad_input(body: str):
    assert isinstance(_parse_save(body), str)


# --- tag lookup ------------------------------------------------------------


def test_tag_info_lines_reports_known_tag(tags_db: TagDatabase):
    lines = _tag_info_lines(tags_db, "wlop")
    assert len(lines) == 1
    assert "danbooru" in lines[0] and "artist" in lines[0] and "4,213" in lines[0]


def test_tag_info_lines_skips_phrases_and_missing_db(tags_db: TagDatabase):
    assert _tag_info_lines(tags_db, "wlop, dynamic pose") == []
    assert _tag_info_lines(None, "wlop") == []


# --- /fav ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fav_command_saves_and_reports_known_tag(storage: Storage, tags_db: TagDatabase):
    update = _command_update("/fav artist wlop | wlop | painterly")
    await fav_command(update, _context(storage, tags_db))

    assert storage.get_favorite(7, "wlop")["note"] == "painterly"
    reply = update.effective_message.reply_text.await_args.args[0]
    assert reply.startswith("Saved favorite 'wlop'")
    assert "Known tag" in reply


@pytest.mark.asyncio
async def test_fav_command_says_updated_when_overwriting(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    update = _command_update("/fav artist wlop | wlop")
    await fav_command(update, _context(storage))
    assert update.effective_message.reply_text.await_args.args[0].startswith("Updated")


@pytest.mark.asyncio
async def test_fav_command_without_args_shows_help(storage: Storage):
    update = _command_update("/fav")
    await fav_command(update, _context(storage))
    assert update.effective_message.reply_text.await_args.args[0] == FAV_HELP


@pytest.mark.asyncio
async def test_fav_delete(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    update = _command_update("/fav delete WLOP")
    await fav_command(update, _context(storage))
    assert storage.get_favorite(7, "wlop") is None
    assert "Deleted" in update.effective_message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_fav_delete_with_a_pipe_is_a_save_into_a_delete_category(storage: Storage):
    update = _command_update("/fav delete x | some text")
    await fav_command(update, _context(storage))
    assert storage.get_favorite(7, "x")["category"] == "delete"


# --- /favs browser ---------------------------------------------------------


@pytest.mark.asyncio
async def test_favs_command_lists_categories(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    storage.save_favorite(7, "dyn", "pose", "dynamic pose")
    update = _command_update("/favs")
    await favs_command(update, _context(storage))

    markup = update.effective_message.reply_text.await_args.kwargs["reply_markup"]
    data = [b.callback_data for b in _buttons(markup)]
    assert data == ["fv:c:artist", "fv:c:pose", "fv:close"]


@pytest.mark.asyncio
async def test_favs_command_with_nothing_saved_shows_help(storage: Storage):
    update = _command_update("/favs")
    await favs_command(update, _context(storage))
    assert FAV_HELP in update.effective_message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_category_callback_lists_favorites(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    update, query = _callback_update("fv:c:artist")
    await fav_callback(update, _context(storage))

    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    data = [b.callback_data for b in _buttons(markup)]
    assert data == ["fv:e:wlop", "fv:home"]


@pytest.mark.asyncio
async def test_entry_callback_shows_text_and_note(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop", "painterly")
    update, query = _callback_update("fv:e:wlop")
    await fav_callback(update, _context(storage))

    text = query.edit_message_text.await_args.args[0]
    assert "by wlop" in text and "painterly" in text


@pytest.mark.asyncio
async def test_callbacks_only_see_the_tapping_users_favorites(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "secret text")
    update, query = _callback_update("fv:e:wlop", user_id=2)
    await fav_callback(update, _context(storage))

    query.answer.assert_awaited_once_with("That favorite no longer exists.")
    assert "secret text" not in query.edit_message_text.await_args.args[0]


@pytest.mark.asyncio
async def test_delete_needs_confirmation(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    storage.save_favorite(7, "sakimi", "artist", "sakimichan")

    update, query = _callback_update("fv:d:wlop")
    await fav_callback(update, _context(storage))
    assert storage.get_favorite(7, "wlop") is not None

    update, query = _callback_update("fv:dy:wlop")
    await fav_callback(update, _context(storage))
    assert storage.get_favorite(7, "wlop") is None
    # back on the (still non-empty) category view
    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert "fv:e:sakimi" in [b.callback_data for b in _buttons(markup)]


@pytest.mark.asyncio
async def test_deleting_last_in_category_returns_home(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    update, query = _callback_update("fv:dy:wlop")
    await fav_callback(update, _context(storage))
    assert "no favorites yet" in query.edit_message_text.await_args.args[0]


def test_entry_keyboard_omits_copy_for_long_text():
    short = {"name": "a", "category": "c", "text": "x", "note": ""}
    long = {**short, "text": "x" * 300}
    assert any(b.copy_text for b in _buttons(_entry_keyboard(short)))
    assert not any(b.copy_text for b in _buttons(_entry_keyboard(long)))


# --- storage: edit ---------------------------------------------------------


def test_update_favorite_changes_only_given_fields(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "by wlop", "note")
    storage.update_favorite(1, "wlop", text="wlop style")
    assert storage.get_favorite(1, "wlop") == {
        "name": "wlop",
        "category": "artist",
        "text": "wlop style",
        "note": "note",
    }


def test_rename_favorite_refuses_a_taken_name(storage: Storage):
    storage.save_favorite(1, "wlop", "artist", "by wlop")
    storage.save_favorite(1, "dyn", "pose", "dynamic pose")
    assert storage.rename_favorite(1, "wlop", "dyn") is False
    assert storage.rename_favorite(1, "wlop", "w") is True
    assert storage.get_favorite(1, "w")["text"] == "by wlop"
    assert storage.get_favorite(1, "wlop") is None


# --- edit flow -------------------------------------------------------------


async def _start_edit(storage: Storage, context, data: str, user_id: int = 7) -> AsyncMock:
    update, query = _callback_update(data, user_id=user_id)
    await fav_callback(update, context)
    return query


@pytest.mark.asyncio
async def test_entry_keyboard_has_edit_button(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    fav = storage.get_favorite(7, "wlop")
    assert "fv:ed:wlop" in [b.callback_data for b in _buttons(_entry_keyboard(fav))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "typed", "expected"),
    [
        ("t", "wlop style, painterly", {"text": "wlop style, painterly"}),
        ("n", "great on illustrious", {"note": "great on illustrious"}),
        ("n", "-", {"note": ""}),
        ("c", "Style", {"category": "style"}),
    ],
)
async def test_typed_edit_updates_the_field(storage: Storage, code, typed, expected):
    storage.save_favorite(7, "wlop", "artist", "by wlop", "old note")
    context = _context(storage)
    await _start_edit(storage, context, f"fv:ef:{code}:wlop")

    update = _command_update(typed)
    assert await handle_fav_edit_message(update, context) is True

    fav = storage.get_favorite(7, "wlop")
    for field, value in expected.items():
        assert fav[field] == value
    # replies with the updated entry so editing can continue
    assert "reply_markup" in update.effective_message.reply_text.await_args.kwargs


@pytest.mark.asyncio
async def test_typed_rename(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    context = _context(storage)
    await _start_edit(storage, context, "fv:ef:r:wlop")

    await handle_fav_edit_message(_command_update("WL"), context)
    assert storage.get_favorite(7, "wlop") is None
    assert storage.get_favorite(7, "wl")["text"] == "by wlop"


@pytest.mark.asyncio
async def test_invalid_typed_edit_changes_nothing(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    storage.save_favorite(7, "dyn", "pose", "dynamic pose")
    context = _context(storage)
    await _start_edit(storage, context, "fv:ef:r:wlop")

    update = _command_update("dyn")
    await handle_fav_edit_message(update, context)
    assert "already have" in update.effective_message.reply_text.await_args.args[0]
    assert storage.get_favorite(7, "wlop") is not None


@pytest.mark.asyncio
async def test_edit_is_scoped_to_the_editing_user(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    context = _context(storage)
    await _start_edit(storage, context, "fv:ef:t:wlop", user_id=7)

    # another user's message in the same chat is not consumed as the edit
    assert await handle_fav_edit_message(_command_update("hijack", user_id=8), context) is False
    assert storage.get_favorite(7, "wlop")["text"] == "by wlop"


@pytest.mark.asyncio
async def test_cancel_clears_the_pending_edit(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    context = _context(storage)
    await _start_edit(storage, context, "fv:ef:t:wlop")
    query = await _start_edit(storage, context, "fv:ex:wlop")

    assert "by wlop" in query.edit_message_text.await_args.args[0]
    assert await handle_fav_edit_message(_command_update("a prompt"), context) is False


@pytest.mark.asyncio
async def test_category_edit_offers_other_existing_categories(storage: Storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    storage.save_favorite(7, "dyn", "pose", "dynamic pose")
    context = _context(storage)
    query = await _start_edit(storage, context, "fv:ef:c:wlop")

    data = [
        b.callback_data for b in _buttons(query.edit_message_text.await_args.kwargs["reply_markup"])
    ]
    assert data == ["fv:ec:pose:wlop", "fv:ex:wlop"]

    await _start_edit(storage, context, "fv:ec:pose:wlop")
    assert storage.get_favorite(7, "wlop")["category"] == "pose"
    # picking a button also ends the typed-entry wait
    assert await handle_fav_edit_message(_command_update("a prompt"), context) is False


def test_edit_callback_data_fits_telegram_limit():
    fav = {"name": "n" * 32, "category": "c" * 24, "text": "x", "note": ""}
    storage = MagicMock()
    storage.list_favorite_categories.return_value = [("d" * 24, 1)]
    for markup in (_entry_keyboard(fav), _category_pick_keyboard(storage, 1, fav)):
        for b in _buttons(markup):
            if b.callback_data:
                assert len(b.callback_data.encode()) <= 64
