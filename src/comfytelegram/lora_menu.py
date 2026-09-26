"""The `/lora` menu — an in-place inline-keyboard toggle for which of a
checkpoint's model-profile LoRAs actually apply to this chat's own
generations.

`model_profiles/*.json`'s `loras` list is the only source of "which LoRAs
are safe to use with this checkpoint" — ComfyUI's `LoraLoader` node lists
every `.safetensors` under `models/loras` with no architecture metadata at
all, so there's no way to auto-detect which of them fit, say, an SDXL vs.
an Anima checkpoint without a human curating that list once, in the
profile's JSON. This menu doesn't touch that curation — it just replaces
"flip `default_enabled` in the JSON and restart the bot" with a live
per-chat toggle (`storage.py`'s `lora_override` table, applied via
`profiles.apply_lora_overrides`), the same override pattern `/settings`
already uses for numeric/prompt fields. It only affects fresh generations
(`generate()`) — post-processing stays locked to whatever LoRAs actually
made the original image, same as checkpoint/prompt (see
`generation._refresh_tunable_defaults`'s docstring for why that split
exists).

The optional "ℹ️ Info" button per LoRA (shown only when
`Settings.comfyui_loras_dir` is configured) answers the two things that
curated list still can't: what trigger words actually activate the LoRA,
and what base model it was trained against — via `comfytelegram.civitai`,
which hashes the file locally and looks it up on CivitAI. See that
module's docstring for how that lookup works and why it needs filesystem
access to the model file at all.

The "⚙️" button per LoRA opens a per-LoRA screen for adjusting
`strength_model`/`strength_clip` live, per chat — the other two knobs a
profile's `loras` entry carries besides on/off. Same override pattern as
the enabled toggle, just its own `storage.py` table
(`lora_strength_override`, applied via `profiles.
apply_lora_strength_overrides`) and edited from a separate screen, since
the two are independent concerns. A custom exact value is entered as a
follow-up chat message, the same one-shot `context.chat_data` flag pattern
`settings_menu.py`'s "✏️ Custom value" uses (see
`handle_lora_custom_value_message`, checked from `handlers.generate_message`
right alongside `settings_menu.handle_custom_value_message`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from comfytelegram.auth import reject_if_unauthorized, reject_if_unauthorized_callback
from comfytelegram.civitai import (
    CivitaiLookupError,
    fetch_civitai_info,
    hash_lora_file,
    resolve_lora_path,
)
from comfytelegram.message_text import message_text
from comfytelegram.profiles import (
    LoraDefault,
    ModelProfile,
    apply_lora_overrides,
    apply_lora_strength_overrides,
    apply_profile_override,
    resolve_profile,
)
from comfytelegram.settings import Settings
from comfytelegram.storage import Storage
from comfytelegram.topics import pop_pending, set_pending

logger = logging.getLogger(__name__)

#: `lr:d:<code>:<index>:<delta>`/`lr:c:<code>:<index>` callback codes for
#: the two strength fields a per-LoRA screen edits — short codes rather
#: than the field names themselves, same reasoning as using a list index
#: instead of a LoRA's own filename: shorter callback_data.
_STRENGTH_FIELDS = {"m": "strength_model", "c": "strength_clip"}
_STRENGTH_FIELD_LABELS = {"m": "Model strength", "c": "Clip strength"}

#: Stepper increment for a per-LoRA strength field. No preset-value row
#: (unlike `settings_menu.py`'s numeric field submenus) — see
#: `_lora_field_keyboard`.
_STRENGTH_STEP = 0.05

#: Shown (with just a Close button) when the resolved checkpoint has no
#: profile, or a profile with an empty `loras` list — there's nothing to
#: toggle either way.
_NO_LORAS_TEXT = (
    "This model has no LoRAs configured — add them to its entry in model_profiles/*.json first."
)


def _truncate(s: str, n: int = 40) -> str:
    """Shorten `s` to at most `n` characters, replacing the tail with `…`
    if it was longer — LoRA filenames run much longer than /settings'
    field labels, hence the wider cap than `settings_menu._truncate`."""
    return s if len(s) <= n else s[: n - 1] + "…"


def _display_name(name: str) -> str:
    """The basename of a LoRA's `loras[].name` — that field is a path
    relative to `comfyui_loras_dir`, forward-slash-joined even for a
    subdirectory entry (see `civitai.py`'s `resolve_lora_path`/
    `lora_discovery.py`'s `_scan_lora_files`, and the shipped
    `furrytoonmix_illustrious.json` LoRAs, all under an `illustrious/...`
    prefix) — useful for keying storage/callback lookups and resolving the
    actual file, but the directory part is noise in anything a user reads,
    so every user-facing string here shows just the filename. `lora.name`
    itself is untouched; only display strings go through this."""
    return name.rsplit("/", 1)[-1]


def _format_strength(value: float) -> str:
    """Whole floats without a trailing `.0`, same convention as
    `settings_menu._format_value`."""
    return str(int(value)) if value == int(value) else str(value)


def _resolve_effective_profile(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, checkpoint: str
) -> ModelProfile | None:
    """The shipped profile for `checkpoint` with this chat's `/settings`
    override and `/lora` toggle/strength overrides all layered on top, so
    each entry in `.loras` already carries this chat's actual current
    `default_enabled`/`strength_model`/`strength_clip` state — nothing else
    in this module needs to look at the raw override dicts directly."""
    storage: Storage = context.bot_data["storage"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]
    profile = resolve_profile(checkpoint, profiles)
    profile = apply_profile_override(profile, checkpoint, storage.get_override(chat_id, checkpoint))
    profile = apply_lora_overrides(profile, storage.get_lora_overrides(chat_id, checkpoint))
    return apply_lora_strength_overrides(
        profile, storage.get_lora_strength_overrides(chat_id, checkpoint)
    )


def _home_text(checkpoint: str, profile: ModelProfile) -> str:
    return f"🎛 LoRAs · {profile.display_name}"


def _home_keyboard(profile: ModelProfile, show_info: bool) -> InlineKeyboardMarkup:
    """Two rows per configured LoRA: a toggle button (✅/◻️, name, and its
    configured `strength_model`) alone on its own row, then a second,
    narrower row with "⚙️ Strength" (opens that LoRA's strength-editing
    screen) and, when `show_info` (i.e. `Settings.comfyui_loras_dir` is
    configured), "ℹ️ Info". Telegram splits a row's width evenly across
    however many buttons are in it, with no way to weight one wider — a
    single earlier row of [toggle, ⚙️, ℹ️] gave the name column only about
    a third of the message width regardless of how short the icon buttons'
    own text was, truncating even fairly short filenames to a handful of
    characters (confirmed from an actual screenshot). Splitting the toggle
    onto its own row gives it the full width; doubling the row count per
    LoRA is the trade-off. Then Reset-all and Close. `profile.loras`'
    order is stable across a render/tap pair — loaded once from disk at
    startup into `bot_data['profiles']`, never reloaded mid-request — so a
    button's list index is a safe, short stand-in for the LoRA's own
    (possibly long, possibly colon-containing) filename in callback_data,
    the same reasoning `handlers.py`'s `hand_point_callback` uses a row/col
    pair instead of embedding text."""
    rows = []
    for index, lora in enumerate(profile.loras):
        rows.append(
            [
                InlineKeyboardButton(
                    f"{'✅' if lora.default_enabled else '◻️'} "
                    f"{_truncate(_display_name(lora.name))} ({_format_strength(lora.strength_model)})",
                    callback_data=f"lr:t:{index}",
                )
            ]
        )
        secondary_row = [InlineKeyboardButton("⚙️ Strength", callback_data=f"lr:f:{index}")]
        if show_info:
            secondary_row.append(InlineKeyboardButton("ℹ️ Info", callback_data=f"lr:i:{index}"))
        rows.append(secondary_row)
    rows.append([InlineKeyboardButton("🔄 Reset all to model defaults", callback_data="lr:ra")])
    rows.append([InlineKeyboardButton("✖ Close", callback_data="lr:close")])
    return InlineKeyboardMarkup(rows)


def _lora_field_text(lora: LoraDefault) -> str:
    return (
        f"🎛 LoRAs › {_display_name(lora.name)}\n\n"
        f"Model strength: {_format_strength(lora.strength_model)}\n"
        f"Clip strength: {_format_strength(lora.strength_clip)}"
    )


def _lora_field_keyboard(index: int, lora: LoraDefault) -> InlineKeyboardMarkup:
    """A single screen editing both of one LoRA's strength fields — a
    stepper plus custom-value entry per field, combined here onto one
    screen since the two are tightly coupled to this one LoRA and always
    worth seeing together. No preset-value row (unlike `settings_menu.py`'s
    numeric field submenus) — a fixed set of quick values made this screen
    look busier than it needed to be for a field that's really just "nudge
    up/down or type an exact number"."""
    rows: list[list[InlineKeyboardButton]] = []
    for code, value in (("m", lora.strength_model), ("c", lora.strength_clip)):
        rows.append(
            [InlineKeyboardButton(f"── {_STRENGTH_FIELD_LABELS[code]} ──", callback_data="lr:noop")]
        )
        rows.append(
            [
                InlineKeyboardButton("➖", callback_data=f"lr:d:{code}:{index}:{-_STRENGTH_STEP}"),
                InlineKeyboardButton(_format_strength(value), callback_data="lr:noop"),
                InlineKeyboardButton("➕", callback_data=f"lr:d:{code}:{index}:{_STRENGTH_STEP}"),
            ]
        )
        rows.append([InlineKeyboardButton("✏️ Custom value", callback_data=f"lr:c:{code}:{index}")])
    rows.append(
        [
            InlineKeyboardButton("🔄 Reset", callback_data=f"lr:pr:{index}"),
            InlineKeyboardButton("↩ Back", callback_data="lr:home"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _render(
    checkpoint: str, profile: ModelProfile | None, show_info: bool
) -> tuple[str, InlineKeyboardMarkup]:
    """The `(text, keyboard)` pair for the current state — shared by the
    initial `/lora` send and every in-place callback edit."""
    if profile is None or not profile.loras:
        return _NO_LORAS_TEXT, InlineKeyboardMarkup(
            [[InlineKeyboardButton("✖ Close", callback_data="lr:close")]]
        )
    return _home_text(checkpoint, profile), _home_keyboard(profile, show_info)


async def _safe_edit_message(query, text: str, reply_markup: InlineKeyboardMarkup) -> None:
    """See `settings_menu._safe_edit_message` — same "message is not
    modified" BadRequest swallowed for the same reason: the callback's own
    `answer()` toast is the real confirmation the user sees, so a
    byte-identical re-render (e.g. double-tapping Reset with nothing
    overridden) shouldn't trip the bot-wide error handler."""
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise


async def _show_home(
    query, context: ContextTypes.DEFAULT_TYPE, chat_id: int, checkpoint: str
) -> None:
    settings: Settings = context.bot_data["settings"]
    profile = _resolve_effective_profile(context, chat_id, checkpoint)
    text, keyboard = _render(checkpoint, profile, show_info=bool(settings.comfyui_loras_dir))
    await _safe_edit_message(query, text, keyboard)


async def _show_lora_field(
    query, context: ContextTypes.DEFAULT_TYPE, chat_id: int, checkpoint: str, index: int
) -> None:
    """Edit `query`'s message in place to show `index`'s strength-editing
    screen, or fall back to the home render if the LoRA list changed out
    from under it (e.g. a profile reload from `/lora`'s auto-discovery job
    landed between two taps)."""
    profile = _resolve_effective_profile(context, chat_id, checkpoint)
    if profile is None or index >= len(profile.loras):
        settings: Settings = context.bot_data["settings"]
        text, keyboard = _render(checkpoint, profile, show_info=bool(settings.comfyui_loras_dir))
        await _safe_edit_message(query, text, keyboard)
        return
    lora = profile.loras[index]
    await _safe_edit_message(query, _lora_field_text(lora), _lora_field_keyboard(index, lora))


def _format_civitai_text(lora_name: str, cached: dict) -> str:
    label = _display_name(lora_name)
    if not cached["found"]:
        return (
            f"ℹ️ {label}\n\n"
            "No CivitAI match for this file's hash — it may be privately trained, "
            "hosted elsewhere, or renamed/modified since it was downloaded."
        )
    words = ", ".join(cached["trained_words"]) if cached["trained_words"] else "(none listed)"
    return (
        f"ℹ️ {label}\n\n"
        f"{cached['model_name']} — base model: {cached['base_model']}\n"
        f"Trigger words: {words}\n"
        f"{cached['civitai_url']}"
    )


async def _civitai_info_text(
    storage: Storage, loras_dir: Path, lora: LoraDefault, *, force_refresh: bool
) -> str:
    """The `/lora` "ℹ️ Info" reply text for `lora` — a cache hit if one
    exists and `force_refresh` wasn't requested (see
    `Storage.get_lora_civitai_cache`), otherwise hash the file on disk and
    query CivitAI fresh, caching whatever comes back either way (even "not
    found", so a future ordinary tap doesn't repeat the hash+query just to
    learn the same answer again — only "🔄 Refresh" ever does). A
    `CivitaiLookupError` (network failure, not a clean "no match") is
    deliberately *not* cached — see that exception's docstring — so the
    next tap (the "🔄 Refresh" button the caller always attaches) tries
    again instead of being stuck on a wrong permanent answer."""
    if not force_refresh:
        cached = storage.get_lora_civitai_cache(lora.name)
        if cached is not None:
            return _format_civitai_text(lora.name, cached)

    path = resolve_lora_path(loras_dir, lora.name)
    if path is None:
        return f"Couldn't find {lora.name} under COMFYUI_LORAS_DIR — check the mount/path."

    sha256 = await hash_lora_file(path)
    try:
        info = await fetch_civitai_info(sha256)
    except CivitaiLookupError as exc:
        logger.warning("CivitAI lookup failed for %s: %s", lora.name, exc)
        return (
            f"ℹ️ {_display_name(lora.name)}\n\nCivitAI lookup failed — try 🔄 Refresh in a moment."
        )
    storage.set_lora_civitai_cache(lora.name, sha256, info)
    cached = storage.get_lora_civitai_cache(lora.name)
    return _format_civitai_text(lora.name, cached)


async def lora_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/lora` — show this chat's LoRA toggles for its currently-selected
    checkpoint. Tells the user to `/model` first if none is selected."""
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    checkpoint = storage.get_checkpoint(chat_id)
    if checkpoint is None:
        await update.effective_message.reply_text("No model selected yet — use /model first.")
        return

    profile = _resolve_effective_profile(context, chat_id, checkpoint)
    text, keyboard = _render(checkpoint, profile, show_info=bool(settings.comfyui_loras_dir))
    await update.effective_message.reply_text(text, reply_markup=keyboard)


async def lora_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch a `lr:<action>[:<index>]` callback: toggle one LoRA by its
    position in `profile.loras` (`t`), look up or re-look-up its CivitAI
    info (`i`/`ir` — only reachable when `Settings.comfyui_loras_dir` is
    set, since `_home_keyboard` omits the Info button otherwise), open a
    LoRA's strength-editing screen (`f`) or step/start-custom-entry-for one
    of its two strength fields there (`d`/`c`, keyed by `_STRENGTH_FIELDS`)
    or reset just that LoRA's strengths (`pr`), go back
    to the home screen (`home`), reset every stored toggle/strength
    override for this checkpoint back to the profile's own defaults (`ra`),
    no-op on a disabled label button (`noop`), or close the menu."""
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if await reject_if_unauthorized_callback(query, user_id, settings):
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    checkpoint = storage.get_checkpoint(chat_id)
    if checkpoint is None:
        await query.answer("No model selected — use /model first.", show_alert=True)
        return

    parts = query.data.split(":")
    action = parts[1]

    if action == "noop":
        await query.answer()
        return

    if action == "close":
        await query.answer()
        await query.message.delete()
        return

    if action == "home":
        await query.answer()
        await _show_home(query, context, chat_id, checkpoint)
        return

    if action == "ra":
        storage.clear_lora_overrides(chat_id, checkpoint)
        storage.clear_lora_strength_overrides(chat_id, checkpoint)
        await query.answer("Reset to model defaults.")
        await _show_home(query, context, chat_id, checkpoint)
        return

    if action == "t":
        profile = _resolve_effective_profile(context, chat_id, checkpoint)
        index = int(parts[2])
        if profile is None or index >= len(profile.loras):
            await query.answer("That LoRA list changed — reopen /lora.", show_alert=True)
            return
        lora = profile.loras[index]
        new_enabled = not lora.default_enabled
        storage.set_lora_override(chat_id, checkpoint, lora.name, new_enabled)
        await query.answer(f"{_display_name(lora.name)}: {'on' if new_enabled else 'off'}")
        await _show_home(query, context, chat_id, checkpoint)
        return

    if action == "f":
        index = int(parts[2])
        profile = _resolve_effective_profile(context, chat_id, checkpoint)
        if profile is None or index >= len(profile.loras):
            await query.answer("That LoRA list changed — reopen /lora.", show_alert=True)
            return
        await query.answer()
        await _show_lora_field(query, context, chat_id, checkpoint, index)
        return

    if action == "pr":
        index = int(parts[2])
        profile = _resolve_effective_profile(context, chat_id, checkpoint)
        if profile is None or index >= len(profile.loras):
            await query.answer("That LoRA list changed — reopen /lora.", show_alert=True)
            return
        storage.clear_lora_strength_override(chat_id, checkpoint, profile.loras[index].name)
        await query.answer("Reset to model defaults.")
        await _show_lora_field(query, context, chat_id, checkpoint, index)
        return

    if action in ("d", "c"):
        code = parts[2]
        field = _STRENGTH_FIELDS.get(code)
        if field is None:
            await query.answer("Unknown field.", show_alert=True)
            return
        index = int(parts[3])
        profile = _resolve_effective_profile(context, chat_id, checkpoint)
        if profile is None or index >= len(profile.loras):
            await query.answer("That LoRA list changed — reopen /lora.", show_alert=True)
            return
        lora = profile.loras[index]

        if action == "c":
            set_pending(
                context.chat_data,
                "awaiting_lora_strength",
                query.message,
                (checkpoint, lora.name, field, index),
            )
            await query.answer()
            await _safe_edit_message(
                query,
                f"🎛 LoRAs › {_display_name(lora.name)}\n\n"
                f"Send the new {_STRENGTH_FIELD_LABELS[code]} as a number.",
                InlineKeyboardMarkup(
                    [[InlineKeyboardButton("↩ Cancel", callback_data=f"lr:f:{index}")]]
                ),
            )
            return

        new_value = round(getattr(lora, field) + float(parts[4]), 2)
        storage.set_lora_strength_override(chat_id, checkpoint, lora.name, field, new_value)
        await query.answer(f"{_STRENGTH_FIELD_LABELS[code]}: {_format_strength(new_value)}")
        await _show_lora_field(query, context, chat_id, checkpoint, index)
        return

    if action in ("i", "ir"):
        profile = _resolve_effective_profile(context, chat_id, checkpoint)
        index = int(parts[2])
        if profile is None or index >= len(profile.loras):
            await query.answer("That LoRA list changed — reopen /lora.", show_alert=True)
            return
        loras_dir = settings.comfyui_loras_dir
        if not loras_dir:
            await query.answer(
                "CivitAI info isn't set up — set COMFYUI_LORAS_DIR.", show_alert=True
            )
            return
        lora = profile.loras[index]
        await query.answer()
        status = await query.message.reply_text(
            f"🔍 Looking up {_display_name(lora.name)}…", disable_notification=True
        )
        text = await _civitai_info_text(storage, loras_dir, lora, force_refresh=action == "ir")
        refresh_row = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"lr:ir:{index}")]]
        await status.edit_text(text, reply_markup=InlineKeyboardMarkup(refresh_row))
        return

    await query.answer("Unknown action.", show_alert=True)


async def handle_lora_custom_value_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """If this chat is mid custom-strength entry (see the "✏️ Custom value"
    button on a LoRA's strength-editing screen), consume the incoming text
    as that value and return True. Otherwise return False so the caller
    treats it as a normal generation prompt. Mirrors
    `settings_menu.handle_custom_value_message`'s one-shot `chat_data` flag
    pattern — checked from `handlers.generate_message` right alongside it."""
    message = update.effective_message
    pending = pop_pending(context.chat_data, "awaiting_lora_strength", message)
    if pending is None:
        return False
    checkpoint, lora_name, field, index = pending

    raw_value = (message_text(message) or "").strip()
    try:
        value = float(raw_value)
    except ValueError:
        await message.reply_text("Invalid value — send a plain number, e.g. 0.8.")
        return True

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    storage.set_lora_strength_override(chat_id, checkpoint, lora_name, field, value)

    profile = _resolve_effective_profile(context, chat_id, checkpoint)
    if profile is not None and index < len(profile.loras):
        lora = profile.loras[index]
        await message.reply_text(
            _lora_field_text(lora), reply_markup=_lora_field_keyboard(index, lora)
        )
    else:
        await message.reply_text("Updated.")
    return True
