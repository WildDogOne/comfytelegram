"""The /settings menu — an in-place, edited inline-keyboard UI for viewing
and changing per-chat generation defaults. Replaces the earlier
`/override <field>=<value>` command, which worked but wasn't discoverable
or friendly.

Follows the standard Telegram settings-menu pattern: one message gets
edited as the user navigates (never spammed as new messages), a breadcrumb
in the header text, and a way back from every screen. Numeric fields get
stepper (+/-) buttons plus quick presets; enum fields (sampler, scheduler)
get a button grid built from ComfyUI's own live `/object_info` choices, so
the menu can never offer a value the server would reject. Both kinds fall
back to a "custom value" free-text prompt for anything the quick buttons
don't cover.

Free-text capture is a one-field flag in `context.chat_data` rather than a
full ConversationHandler — `handle_custom_value_message()` is checked at
the top of `handlers.generate_message` before that text is treated as a
generation prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

from pydantic import ValidationError
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from comfytelegram.auth import is_authorized, reject_if_unauthorized
from comfytelegram.comfy_client import ComfyClient, ComfyUIError
from comfytelegram.profiles import (
    PROMPT_OVERRIDE_FIELDS,
    ModelProfile,
    ProfileDefaults,
    apply_profile_override,
    resolve_generation_params,
    resolve_profile,
)
from comfytelegram.settings import Settings
from comfytelegram.storage import Storage


@dataclass(frozen=True)
class NumericFieldMeta:
    key: str
    label: str
    step: float
    presets: tuple[float, ...]
    is_int: bool = False
    min_value: float | None = None
    kind: Literal["numeric"] = "numeric"


@dataclass(frozen=True)
class EnumFieldMeta:
    key: str
    label: str
    preferred: tuple[str, ...] = dataclass_field(default_factory=tuple)
    kind: Literal["enum"] = "enum"


@dataclass(frozen=True)
class TextFieldMeta:
    """Free-text fields that live on the `ModelProfile` itself rather than
    `GenerationParams` (`positive_prompt_prefix`/`negative_prompt_prefix`)
    — no stepper/presets/enum grid make sense here, just edit-in-place and
    reset. See `PROMPT_OVERRIDE_FIELDS` in `profiles/loader.py` for how
    these get routed onto the profile instead of `.defaults` on override."""

    key: str
    label: str
    kind: Literal["text"] = "text"


FieldMeta = NumericFieldMeta | EnumFieldMeta | TextFieldMeta

FIELDS: list[FieldMeta] = [
    NumericFieldMeta("cfg", "CFG", step=0.5, presets=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0)),
    NumericFieldMeta("steps", "Steps", step=5, presets=(15, 20, 25, 30, 40, 50), is_int=True, min_value=1),
    EnumFieldMeta(
        "sampler_name",
        "Sampler",
        preferred=(
            "euler",
            "euler_ancestral",
            "heun",
            "dpm_2",
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_3m_sde",
            "ddim",
            "uni_pc",
            "lcm",
        ),
    ),
    EnumFieldMeta("scheduler", "Scheduler"),
    NumericFieldMeta("clip_skip", "Clip Skip", step=1, presets=(-1, -2, -3), is_int=True),
    NumericFieldMeta("width", "Width", step=64, presets=(512, 768, 1024, 1280), is_int=True, min_value=64),
    NumericFieldMeta("height", "Height", step=64, presets=(512, 768, 1024, 1280), is_int=True, min_value=64),
    NumericFieldMeta("batch_size", "Batch", step=1, presets=(1, 2, 3, 4), is_int=True, min_value=1),
    TextFieldMeta("positive_prompt_prefix", "Default Positive"),
    TextFieldMeta("negative_prompt_prefix", "Default Negative"),
]
FIELDS_BY_KEY: dict[str, FieldMeta] = {f.key: f for f in FIELDS}


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value if value else "(none)"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _truncate(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _effective_params(checkpoint: str, profile: ModelProfile | None):
    """Same resolution path generation actually uses — read the numeric
    fields off the result instead of duplicating the merge/fallback logic."""
    return resolve_generation_params(checkpoint, "", profile)


def _field_value(meta: FieldMeta, checkpoint: str, profile: ModelProfile | None) -> Any:
    """`positive_prompt_prefix`/`negative_prompt_prefix` aren't
    `GenerationParams` fields (see `TextFieldMeta`'s docstring) — their
    current value has to come straight off the profile, not off
    `_effective_params()`'s resolved output."""
    if isinstance(meta, TextFieldMeta):
        return getattr(profile, meta.key) if profile is not None else ""
    return getattr(_effective_params(checkpoint, profile), meta.key)


def _coerce_field_value(field: str, raw_value: str) -> dict[str, Any]:
    """Turn a user-submitted string into a validated override dict for one
    field. Numeric/enum fields go through `ProfileDefaults` for type
    coercion and validation; prompt-prefix fields are free text, so any
    string (including empty, to blank out a prefix) is accepted as-is."""
    if field in PROMPT_OVERRIDE_FIELDS:
        return {field: raw_value.strip()}
    return ProfileDefaults.model_validate({field: raw_value}).model_dump(exclude_none=True)


def _curate(choices: list[str], preferred: tuple[str, ...], max_items: int = 12) -> list[str]:
    if not preferred:
        return choices[:max_items]
    curated = [c for c in preferred if c in choices]
    if len(curated) < 4:  # curation missed most of what the server actually has — show the raw list instead
        return choices[:max_items]
    return curated[:max_items]


def _home_text(checkpoint: str, profile: ModelProfile | None) -> str:
    label = profile.display_name if profile else checkpoint
    return f"⚙️ Settings · {label}"


def _home_keyboard(
    checkpoint: str, profile: ModelProfile | None, override_fields: dict[str, Any]
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for meta in FIELDS:
        value = _field_value(meta, checkpoint, profile)
        marker = "★ " if meta.key in override_fields else ""
        label = f"{marker}{meta.label}: {_truncate(_format_value(value))}"
        row.append(InlineKeyboardButton(label, callback_data=f"st:f:{meta.key}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("🔄 Reset all to model defaults", callback_data="st:ra")])
    rows.append([InlineKeyboardButton("✖ Close", callback_data="st:close")])
    return InlineKeyboardMarkup(rows)


def _numeric_submenu_keyboard(meta: NumericFieldMeta, value: Any) -> InlineKeyboardMarkup:
    step = int(meta.step) if meta.is_int else meta.step
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("➖", callback_data=f"st:d:{meta.key}:{-step}"),
            InlineKeyboardButton(_format_value(value), callback_data="st:noop"),
            InlineKeyboardButton("➕", callback_data=f"st:d:{meta.key}:{step}"),
        ]
    ]
    preset_row: list[InlineKeyboardButton] = []
    for preset in meta.presets:
        preset_row.append(InlineKeyboardButton(_format_value(preset), callback_data=f"st:v:{meta.key}:{preset}"))
        if len(preset_row) == 3:
            rows.append(preset_row)
            preset_row = []
    if preset_row:
        rows.append(preset_row)
    rows.append([InlineKeyboardButton("✏️ Custom value", callback_data=f"st:c:{meta.key}")])
    rows.append(
        [
            InlineKeyboardButton("↩ Back", callback_data="st:home"),
            InlineKeyboardButton("🔄 Reset", callback_data=f"st:r:{meta.key}"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _enum_submenu_keyboard(meta: EnumFieldMeta, current: Any, choices: list[str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for choice in _curate(choices, meta.preferred):
        marker = "• " if choice == current else ""
        row.append(InlineKeyboardButton(f"{marker}{choice}", callback_data=f"st:v:{meta.key}:{choice}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("✏️ Custom value", callback_data=f"st:c:{meta.key}")])
    rows.append(
        [
            InlineKeyboardButton("↩ Back", callback_data="st:home"),
            InlineKeyboardButton("🔄 Reset", callback_data=f"st:r:{meta.key}"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _text_submenu_keyboard(meta: TextFieldMeta) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✏️ Edit", callback_data=f"st:c:{meta.key}")],
            [
                InlineKeyboardButton("↩ Back", callback_data="st:home"),
                InlineKeyboardButton("🔄 Reset", callback_data=f"st:r:{meta.key}"),
            ],
        ]
    )


async def _enum_choices(client: ComfyClient, meta: EnumFieldMeta, fallback: Any) -> list[str]:
    try:
        if meta.key == "sampler_name":
            return await client.list_samplers()
        return await client.list_schedulers()
    except (ComfyUIError, OSError):
        return [fallback] if fallback else []


async def _safe_edit_message(query, text: str, reply_markup: InlineKeyboardMarkup) -> None:
    """Tapping a preset that happens to match the value already shown (e.g. a
    batch-size preset of 1 when nothing's been overridden yet) produces a
    byte-identical message — Telegram's edit_message_text then raises "message
    is not modified" as a BadRequest. That's not a real failure (the value did
    get saved; the callback's own `answer()` toast is the actual confirmation
    the user sees), so it's swallowed here rather than tripping the bot-wide
    error handler and showing a scary "something went wrong" message for a
    successful save.
    """
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise


async def _render_field_submenu(
    query, context: ContextTypes.DEFAULT_TYPE, checkpoint: str, profile: ModelProfile | None, meta: FieldMeta
) -> None:
    value = _field_value(meta, checkpoint, profile)
    text = f"⚙️ Settings › {meta.label}"
    if isinstance(meta, NumericFieldMeta):
        keyboard = _numeric_submenu_keyboard(meta, value)
    elif isinstance(meta, EnumFieldMeta):
        client: ComfyClient = context.bot_data["comfy_client"]
        choices = await _enum_choices(client, meta, value)
        keyboard = _enum_submenu_keyboard(meta, value, choices)
    else:
        text = f"⚙️ Settings › {meta.label}\n\nCurrent: {_format_value(value)}"
        keyboard = _text_submenu_keyboard(meta)
    await _safe_edit_message(query, text, keyboard)


def _resolve_effective_profile(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, checkpoint: str
) -> tuple[ModelProfile | None, dict[str, Any]]:
    storage: Storage = context.bot_data["storage"]
    profiles: list[ModelProfile] = context.bot_data["profiles"]
    base_profile = resolve_profile(checkpoint, profiles)
    override_fields = storage.get_override(chat_id, checkpoint)
    return apply_profile_override(base_profile, checkpoint, override_fields), override_fields


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if await reject_if_unauthorized(update, settings):
        return

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    checkpoint = storage.get_checkpoint(chat_id)
    if checkpoint is None:
        await update.effective_message.reply_text("No model selected yet — use /model first.")
        return

    profile, override_fields = _resolve_effective_profile(context, chat_id, checkpoint)
    await update.effective_message.reply_text(
        _home_text(checkpoint, profile), reply_markup=_home_keyboard(checkpoint, profile, override_fields)
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    settings: Settings = context.bot_data["settings"]
    if update.effective_user is None or not is_authorized(settings, update.effective_user.id):
        await query.answer("Not authorized.", show_alert=True)
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
        profile, override_fields = _resolve_effective_profile(context, chat_id, checkpoint)
        await _safe_edit_message(
            query, _home_text(checkpoint, profile), _home_keyboard(checkpoint, profile, override_fields)
        )
        return

    if action == "ra":
        storage.clear_override(chat_id, checkpoint)
        await query.answer("Reset to model defaults.")
        profile, override_fields = _resolve_effective_profile(context, chat_id, checkpoint)
        await _safe_edit_message(
            query, _home_text(checkpoint, profile), _home_keyboard(checkpoint, profile, override_fields)
        )
        return

    field = parts[2]
    meta = FIELDS_BY_KEY.get(field)
    if meta is None:
        await query.answer("Unknown field.", show_alert=True)
        return

    if action == "f":
        await query.answer()
        profile, _ = _resolve_effective_profile(context, chat_id, checkpoint)
        await _render_field_submenu(query, context, checkpoint, profile, meta)
        return

    if action == "r":
        storage.clear_override_field(chat_id, checkpoint, field)
        await query.answer(f"Reset {meta.label}.")
        profile, _ = _resolve_effective_profile(context, chat_id, checkpoint)
        await _render_field_submenu(query, context, checkpoint, profile, meta)
        return

    if action == "c":
        context.chat_data["awaiting_field"] = field
        await query.answer()
        await _safe_edit_message(
            query,
            f"⚙️ Settings › {meta.label}\n\nSend the new value as a message.",
            InlineKeyboardMarkup([[InlineKeyboardButton("↩ Cancel", callback_data=f"st:f:{field}")]]),
        )
        return

    if action == "d":
        assert isinstance(meta, NumericFieldMeta)
        profile, _ = _resolve_effective_profile(context, chat_id, checkpoint)
        current = getattr(_effective_params(checkpoint, profile), field)
        new_value = current + float(parts[3])
        if meta.is_int:
            new_value = round(new_value)
        if meta.min_value is not None:
            new_value = max(meta.min_value, new_value)
        storage.set_override_fields(chat_id, checkpoint, {field: new_value})
        await query.answer(f"{meta.label} set to {_format_value(new_value)}")
        profile, _ = _resolve_effective_profile(context, chat_id, checkpoint)
        await _render_field_submenu(query, context, checkpoint, profile, meta)
        return

    if action == "v":
        raw_value = parts[3]
        try:
            coerced = _coerce_field_value(field, raw_value)
        except ValidationError:
            await query.answer("Invalid value.", show_alert=True)
            return
        storage.set_override_fields(chat_id, checkpoint, coerced)
        await query.answer(f"{meta.label} set to {_format_value(coerced[field])}")
        profile, _ = _resolve_effective_profile(context, chat_id, checkpoint)
        await _render_field_submenu(query, context, checkpoint, profile, meta)
        return

    await query.answer("Unknown action.", show_alert=True)


async def handle_custom_value_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """If this chat is mid-"custom value" entry, consume the incoming text
    message as that value and return True. Otherwise return False so the
    caller treats it as a normal generation prompt."""
    field = context.chat_data.get("awaiting_field")
    if field is None:
        return False
    del context.chat_data["awaiting_field"]

    message = update.effective_message
    raw_value = (message.text or "").strip()
    meta = FIELDS_BY_KEY.get(field)
    if meta is None or not raw_value:
        await message.reply_text("Cancelled.")
        return True

    try:
        coerced = _coerce_field_value(field, raw_value)
    except ValidationError as exc:
        await message.reply_text(f"Invalid value: {exc.errors()[0]['msg']}")
        return True

    chat_id = update.effective_chat.id
    storage: Storage = context.bot_data["storage"]
    checkpoint = storage.get_checkpoint(chat_id)
    if checkpoint is None:
        return True

    storage.set_override_fields(chat_id, checkpoint, coerced)
    profile, override_fields = _resolve_effective_profile(context, chat_id, checkpoint)
    await message.reply_text(
        f"Updated {meta.label}.",
        reply_markup=None,
    )
    await message.reply_text(
        _home_text(checkpoint, profile), reply_markup=_home_keyboard(checkpoint, profile, override_fields)
    )
    return True
