"""`/fav` and `/favs` — a personal library of reusable prompt pieces (single
tags like "dynamic pose", artist names, short phrases like "dramatic
lighting, rim light") saved for later.

Unlike characters (`handlers.character_command`), which are whole prompt
snippets activated per chat, favorites are small, composable, and keyed per
Telegram *user* (`storage.py`'s `favorite` table) — they're a personal
collection that follows the user into every chat and forum topic, and
nobody else sees them.

Each favorite has a short `name` handle restricted to `FAVORITE_NAME_RE`,
deliberately typeable rather than a free-text title, because the planned
next step is expanding `$name` (and `$category?` as a random pick from a
category) inside a prompt. The free-text description lives in `note`
instead. Names and categories are lowercased on save so that future
expansion doesn't have to care about case.

`/favs` is an edit-in-place inline-keyboard browser in the same style as
`lora_menu.py` (its own `fv:<action>` callback namespace): categories →
favorites in one category → one favorite's detail with copy/edit/delete. Every
lookup is by the *tapping* user's id, so a `/favs` message in a group
can't be used to browse someone else's library.

"✏️ Edit" changes one field at a time. Text, note and name are typed as a
follow-up message (`handle_fav_edit_message`, checked from
`handlers.generate_message` the same way `/lora`'s custom strength entry
is); category can also be picked from the user's existing categories as
buttons. The pending-edit flag is keyed per user as well as per topic
(`_pending_key`), so in a group one user's next message can't be taken as
another user's edit.
"""

from __future__ import annotations

import re

from telegram import CopyTextButton, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import InlineKeyboardButtonLimit
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from comfytelegram.auth import reject_if_unauthorized, reject_if_unauthorized_callback
from comfytelegram.message_text import message_text
from comfytelegram.settings import Settings
from comfytelegram.storage import Storage
from comfytelegram.tags import TagDatabase, TagSource, category_label
from comfytelegram.topics import pop_pending, set_pending

#: A favorite's handle — short and typeable, since it's meant to be written
#: as `$name` inside a prompt later. Also keeps `fv:<action>:<name>`
#: comfortably under Telegram's 64-byte callback_data limit.
FAVORITE_NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")

#: Same shape as a name, a bit shorter — categories become wildcard pools
#: (`$artist?`) later, so the same "typeable" reasoning applies.
FAVORITE_CATEGORY_RE = re.compile(r"^[a-z0-9_-]{1,24}$")

FAV_HELP = (
    "Save tags, artist names or short prompt phrases to reuse later:\n"
    "/fav <category> <name> | <text> [| <note>]\n"
    "/fav delete <name>\n"
    "/favs — browse your favorites by category\n\n"
    "Examples:\n"
    "/fav artist wlop | by wlop | painterly, great on Illustrious\n"
    "/fav pose dyn | dynamic pose, foreshortening\n\n"
    "Names and categories use letters, digits, '-' and '_' only. Saving under "
    "an existing name replaces it. Favorites are personal — only you see yours."
)

_NO_FAVORITES_TEXT = "You have no favorites yet.\n\n" + FAV_HELP

#: `fv:ef:<code>:<name>` field codes for "✏️ Edit" — short codes keep the
#: callback_data under Telegram's 64-byte cap alongside a 32-char name.
_EDIT_FIELDS = {"t": "text", "n": "note", "c": "category", "r": "name"}

#: Sent instead of a note to clear it — an empty message can't be sent.
_CLEAR_NOTE = "-"


def _pending_key(user_id: int) -> str:
    return f"awaiting_fav_edit:{user_id}"


def _truncate(s: str, n: int = 40) -> str:
    """Shorten `s` to at most `n` characters, replacing the tail with `…`."""
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_save(body: str) -> tuple[str, str, str, str] | str:
    """Parse `<category> <name> | <text> [| <note>]` into
    `(category, name, text, note)`, or return an error message to reply
    with instead."""
    parts = [p.strip() for p in body.split("|")]
    if len(parts) < 2 or len(parts) > 3:
        return "Usage: /fav <category> <name> | <text> [| <note>]"
    head = parts[0].split()
    if len(head) != 2:
        return "Usage: /fav <category> <name> | <text> [| <note>] — category and name are one word each."
    category, name = head[0].lower(), head[1].lower()
    text = parts[1]
    note = parts[2] if len(parts) > 2 else ""
    if not FAVORITE_CATEGORY_RE.match(category):
        return "Categories can only use letters, digits, '-' and '_' (max 24 chars)."
    if not FAVORITE_NAME_RE.match(name):
        return "Names can only use letters, digits, '-' and '_' (max 32 chars)."
    if not text:
        return "The text to save can't be empty."
    return category, name, text, note


def _tag_info_lines(tags_db: TagDatabase | None, text: str) -> list[str]:
    """One line per tag source that knows `text` as an exact tag, e.g.
    "danbooru: artist, 4,213 posts" — a free typo check on save. Skipped
    for anything comma-separated (a phrase, not a single tag) or when no
    tag data has been imported."""
    if tags_db is None or "," in text:
        return []
    lines = []
    for source in TagSource:
        hit = tags_db.lookup_exact(text, [source])
        if hit is not None:
            lines.append(
                f"{source.value}: {hit.name} ({category_label(source, hit.category)}, "
                f"{hit.post_count:,} posts)"
            )
    return lines


async def fav_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/fav <category> <name> | <text> [| <note>]` or `/fav delete <name>`;
    anything else shows `FAV_HELP`."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    message = update.effective_message
    storage: Storage = context.bot_data["storage"]
    user_id = update.effective_user.id

    raw = (message.text or "").split(maxsplit=1)
    rest = raw[1].strip() if len(raw) > 1 else ""

    # "/fav delete x | y" is a save into a category called "delete", not a
    # delete — only a pipe-less "delete <name>" is the delete subcommand.
    if rest.split(maxsplit=1)[:1] == ["delete"] and "|" not in rest:
        name = rest[len("delete") :].strip().lower()
        if not name:
            await message.reply_text("Usage: /fav delete <name>")
            return
        if storage.delete_favorite(user_id, name):
            await message.reply_text(f"Deleted favorite '{name}'.")
        else:
            await message.reply_text(f"No favorite named '{name}'.")
        return

    if not rest:
        await message.reply_text(FAV_HELP)
        return

    parsed = _parse_save(rest)
    if isinstance(parsed, str):
        await message.reply_text(parsed)
        return
    category, name, text, note = parsed

    existed = storage.get_favorite(user_id, name) is not None
    storage.save_favorite(user_id, name, category, text, note)

    reply = f"{'Updated' if existed else 'Saved'} favorite '{name}' in {category}."
    tag_lines = _tag_info_lines(context.bot_data.get("tags_db"), text)
    if tag_lines:
        reply += "\nKnown tag — " + "; ".join(tag_lines)
    await message.reply_text(reply + "\nBrowse with /favs.")


def _home_view(storage: Storage, user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    categories = storage.list_favorite_categories(user_id)
    if not categories:
        return _NO_FAVORITES_TEXT, InlineKeyboardMarkup(
            [[InlineKeyboardButton("✖ Close", callback_data="fv:close")]]
        )
    total = sum(count for _, count in categories)
    rows = [
        [InlineKeyboardButton(f"{category} ({count})", callback_data=f"fv:c:{category}")]
        for category, count in categories
    ]
    rows.append([InlineKeyboardButton("✖ Close", callback_data="fv:close")])
    return f"⭐ Favorites · {total} saved", InlineKeyboardMarkup(rows)


def _category_view(
    storage: Storage, user_id: int, category: str
) -> tuple[str, InlineKeyboardMarkup] | None:
    """None if the category has no favorites (any more)."""
    favorites = storage.list_favorites(user_id, category)
    if not favorites:
        return None
    rows = [
        [
            InlineKeyboardButton(
                _truncate(f"{fav['name']} · {fav['text']}"), callback_data=f"fv:e:{fav['name']}"
            )
        ]
        for fav in favorites
    ]
    rows.append([InlineKeyboardButton("‹ Back", callback_data="fv:home")])
    return f"⭐ Favorites › {category}", InlineKeyboardMarkup(rows)


def _entry_text(fav: dict) -> str:
    text = f"⭐ {fav['category']} › {fav['name']}\n\n{fav['text']}"
    if fav["note"]:
        text += f"\n\n📝 {fav['note']}"
    return text


def _entry_keyboard(fav: dict) -> InlineKeyboardMarkup:
    """📋 Copy only when the text fits Telegram's copy-button cap — a
    truncated copy would silently hand back the wrong text (same gating as
    `handlers._generate_from_prompt_keyboard`)."""
    action_row = []
    if len(fav["text"]) <= InlineKeyboardButtonLimit.MAX_COPY_TEXT:
        action_row.append(InlineKeyboardButton("📋 Copy", copy_text=CopyTextButton(fav["text"])))
    action_row.append(InlineKeyboardButton("✏️ Edit", callback_data=f"fv:ed:{fav['name']}"))
    action_row.append(InlineKeyboardButton("🗑 Delete", callback_data=f"fv:d:{fav['name']}"))
    return InlineKeyboardMarkup(
        [
            action_row,
            [InlineKeyboardButton("‹ Back", callback_data=f"fv:c:{fav['category']}")],
        ]
    )


def _delete_confirm_keyboard(fav: dict) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🗑 Yes, delete", callback_data=f"fv:dy:{fav['name']}"),
                InlineKeyboardButton("Cancel", callback_data=f"fv:e:{fav['name']}"),
            ]
        ]
    )


def _edit_keyboard(fav: dict) -> InlineKeyboardMarkup:
    name = fav["name"]
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Text", callback_data=f"fv:ef:t:{name}"),
                InlineKeyboardButton("Note", callback_data=f"fv:ef:n:{name}"),
            ],
            [
                InlineKeyboardButton("Category", callback_data=f"fv:ef:c:{name}"),
                InlineKeyboardButton("Name", callback_data=f"fv:ef:r:{name}"),
            ],
            [InlineKeyboardButton("‹ Back", callback_data=f"fv:e:{name}")],
        ]
    )


def _category_pick_keyboard(storage: Storage, user_id: int, fav: dict) -> InlineKeyboardMarkup:
    """Every other existing category as a one-tap button; a brand-new one
    is typed as a follow-up, same as the other fields."""
    name = fav["name"]
    rows = [
        [InlineKeyboardButton(category, callback_data=f"fv:ec:{category}:{name}")]
        for category, _ in storage.list_favorite_categories(user_id)
        if category != fav["category"]
    ]
    rows.append([InlineKeyboardButton("‹ Cancel", callback_data=f"fv:ex:{name}")])
    return InlineKeyboardMarkup(rows)


def _edit_prompt_text(fav: dict, field: str) -> str:
    current = fav[field] or "(empty)"
    prompt = {
        "text": "Send the new text.",
        "note": f"Send the new note, or {_CLEAR_NOTE} to remove it.",
        "category": "Tap an existing category, or send a new one.",
        "name": "Send the new name (letters, digits, '-' and '_').",
    }[field]
    return f"✏️ {fav['category']} › {fav['name']}\n\nCurrent {field}: {current}\n\n{prompt}"


def _apply_edit(storage: Storage, user_id: int, fav: dict, field: str, raw: str) -> str | None:
    """Validate and apply one typed edit. Returns an error message to reply
    with (nothing changed), or None on success."""
    value = raw.strip()
    if field == "text":
        if not value:
            return "The text can't be empty — favorite unchanged."
        storage.update_favorite(user_id, fav["name"], text=value)
    elif field == "note":
        storage.update_favorite(user_id, fav["name"], note="" if value == _CLEAR_NOTE else value)
    elif field == "category":
        value = value.lower()
        if not FAVORITE_CATEGORY_RE.match(value):
            return "Categories can only use letters, digits, '-' and '_' (max 24 chars)."
        storage.update_favorite(user_id, fav["name"], category=value)
    else:
        value = value.lower()
        if not FAVORITE_NAME_RE.match(value):
            return "Names can only use letters, digits, '-' and '_' (max 32 chars)."
        if value != fav["name"] and not storage.rename_favorite(user_id, fav["name"], value):
            return f"You already have a favorite named '{value}' — pick a different name."
    return None


async def _safe_edit_message(query, text: str, reply_markup: InlineKeyboardMarkup) -> None:
    """See `settings_menu._safe_edit_message` — swallow "message is not
    modified" from a double-tap."""
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise


async def favs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/favs` — open the favorites browser on its category list."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return
    storage: Storage = context.bot_data["storage"]
    text, keyboard = _home_view(storage, update.effective_user.id)
    await update.effective_message.reply_text(text, reply_markup=keyboard)


async def fav_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch a `fv:<action>[:<arg>]` callback: open a category (`c`), a
    favorite (`e`), ask to delete one (`d`) or confirm it (`dy`), go back to
    the category list (`home`), or close the menu."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    storage: Storage = context.bot_data["storage"]
    _, action, *rest = query.data.split(":", 2)
    arg = rest[0] if rest else ""

    if action == "close":
        await query.answer()
        await query.message.delete()
        return

    if action == "home":
        await query.answer()
        await _safe_edit_message(query, *_home_view(storage, user_id))
        return

    if action == "c":
        view = _category_view(storage, user_id, arg)
        if view is None:
            await query.answer("That category is empty now.")
            await _safe_edit_message(query, *_home_view(storage, user_id))
            return
        await query.answer()
        await _safe_edit_message(query, *view)
        return

    if action in ("ef", "ec"):
        # "fv:ef:<code>:<name>" / "fv:ec:<category>:<name>" — names and
        # categories can't contain ':' (FAVORITE_*_RE), so this split is safe.
        sub, _, arg = arg.partition(":")
    if action == "ex":
        pop_pending(context.chat_data, _pending_key(user_id), query.message)
        action = "e"

    if action in ("e", "d", "dy", "ed", "ef", "ec"):
        fav = storage.get_favorite(user_id, arg)
        if fav is None:
            await query.answer("That favorite no longer exists.")
            await _safe_edit_message(query, *_home_view(storage, user_id))
            return

        if action == "e":
            await query.answer()
            await _safe_edit_message(query, _entry_text(fav), _entry_keyboard(fav))
            return

        if action == "ed":
            await query.answer()
            await _safe_edit_message(
                query, _entry_text(fav) + "\n\nWhat do you want to change?", _edit_keyboard(fav)
            )
            return

        if action == "ef":
            field = _EDIT_FIELDS.get(sub)
            if field is None:
                await query.answer("Unknown field.", show_alert=True)
                return
            set_pending(
                context.chat_data, _pending_key(user_id), query.message, (fav["name"], field)
            )
            await query.answer()
            if field == "category":
                keyboard = _category_pick_keyboard(storage, user_id, fav)
            else:
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("‹ Cancel", callback_data=f"fv:ex:{fav['name']}")]]
                )
            await _safe_edit_message(query, _edit_prompt_text(fav, field), keyboard)
            return

        if action == "ec":
            pop_pending(context.chat_data, _pending_key(user_id), query.message)
            storage.update_favorite(user_id, fav["name"], category=sub)
            await query.answer(f"Moved to {sub}.")
            fav = storage.get_favorite(user_id, fav["name"])
            await _safe_edit_message(query, _entry_text(fav), _entry_keyboard(fav))
            return

        if action == "d":
            await query.answer()
            await _safe_edit_message(
                query, _entry_text(fav) + "\n\nDelete this favorite?", _delete_confirm_keyboard(fav)
            )
            return

        storage.delete_favorite(user_id, fav["name"])
        await query.answer(f"Deleted '{fav['name']}'.")
        view = _category_view(storage, user_id, fav["category"])
        await _safe_edit_message(query, *(view or _home_view(storage, user_id)))
        return

    await query.answer("Unknown action.", show_alert=True)


async def handle_fav_edit_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """If this user is mid "✏️ Edit" on a favorite, consume the incoming
    text as the new value and return True; otherwise return False so the
    caller (`handlers.generate_message`) treats it as a normal prompt.
    Replies with the updated favorite and its keyboard, so editing can
    carry on from there."""
    if update.effective_user is None:
        return False
    user_id = update.effective_user.id
    message = update.effective_message
    pending = pop_pending(context.chat_data, _pending_key(user_id), message)
    if pending is None:
        return False
    name, field = pending

    storage: Storage = context.bot_data["storage"]
    fav = storage.get_favorite(user_id, name)
    if fav is None:
        await message.reply_text(f"'{name}' no longer exists — nothing to edit.")
        return True

    error = _apply_edit(storage, user_id, fav, field, message_text(message) or "")
    if error is not None:
        await message.reply_text(error)
        return True

    new_name = (message_text(message) or "").strip().lower() if field == "name" else name
    fav = storage.get_favorite(user_id, new_name)
    await message.reply_text(_entry_text(fav), reply_markup=_entry_keyboard(fav))
    return True
