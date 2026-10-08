"""`/rp` regional prompt syntax: parsing a message into a global prompt plus
per-region prompts, laying the regions out as fractional boxes, and the
help/template text `handlers.py` shows for it.

A message looks like::

    2girls, standing side by side, park
    [left] blonde hair, cat ears, black hoodie
    [right] black hair, fox ears, white kimono
    ---
    lowres, bad anatomy

Any plain prompt message with such a tag line is regional (`is_regional_prompt`);
`/rp` itself only explains the format and offers the template.
Lines before the first `[position]` tag are the global prompt (whole image,
and prepended to every region — see `resolve_generation_params`);
each tag's text runs until the next tag or the `---` negative separator.
Positions split the image into equal columns (`left`/`center`/`right`) or
rows (`top`/`middle`/`bottom`) — two slots, or three when the centre one is
used. That's deliberately all there is: the bot is driven from a phone
keyboard, and free-form box coordinates would be one more thing to get
wrong for very little gain over halves and thirds.

Telegram-independent, like `generation.py` — `handlers.py` owns the chat
flow, this module only turns text into `RegionSpec`s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from comfytelegram.prompt_refs import ReferenceExpander, UnknownReferenceError
from comfytelegram.workflows import RegionSpec

#: Same rule as `handlers._NEGATIVE_BLOCK_SEP_RE`: a line of 3+ dashes on
#: its own splits positive from negative.
_NEGATIVE_BLOCK_SEP_RE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)

#: A `[position]` tag at the start of a line, plus whatever follows it on
#: that same line.
_REGION_TAG_RE = re.compile(r"^[ \t]*\[([^\]\n]*)\][ \t]*(.*)$")

_HORIZONTAL = ("left", "right")
_VERTICAL = ("top", "bottom")
#: The centre slot — which name it ends up displayed under depends on the
#: axis the other tags pick (see `_layout`).
_CENTRE_ALIASES = {"center": "center", "centre": "center", "middle": "center"}

#: A line opening with one of the position tags — what makes a plain prompt
#: message regional without `/rp` (see `is_regional_prompt`). Only the known
#: tags count, so a prompt that happens to start a line with some other
#: bracketed text stays an ordinary prompt.
_KNOWN_TAG_LINE_RE = re.compile(
    r"^[ \t]*\[[ \t]*(?:left|right|top|bottom|center|centre|middle)[ \t]*\]",
    re.IGNORECASE | re.MULTILINE,
)
_VALID_TAGS = "[left] [center] [right] or [top] [middle] [bottom]"

REGIONAL_TEMPLATE = (
    "masterpiece, best quality, 2girls, standing side by side, park\n"
    "[left] blonde hair, green eyes, cat ears, black hoodie\n"
    "[right] black hair, purple eyes, fox ears, white kimono\n"
    "---\n"
    "lowres, bad anatomy"
)

REGIONAL_HELP = (
    "🧩 Regional prompt — give each part of the image its own prompt, so "
    "two characters stop swapping hair, ears and outfits.\n\n"
    "Send your next message in this format (📋 copies a template). You don't "
    "need /rp for it — any prompt with these [position] lines is regional:\n\n"
    f"{REGIONAL_TEMPLATE}\n\n"
    "• First lines: the global prompt — scene, style, how many characters. "
    "It's also added to every region, so keep it to what they share and put "
    "anything about one character in its region.\n"
    f"• Then one line per region: {_VALID_TAGS}. Use left/right (side by "
    "side) or top/bottom (stacked), not both; add center/middle for three "
    "regions.\n"
    "• $name inserts a saved character (/characters) or favorite (/favs), "
    "e.g. [left] $alice, waving. A character's negative prompt is added to "
    "the negatives. The active character isn't applied here — place "
    "characters with $name instead.\n"
    "• Optional: a --- line, then negative tags.\n\n"
    "Regions don't move people: top/bottom only helps if the scene really "
    "stacks them, so for characters standing together use left/right with "
    "a landscape size (/settings)."
)


class RegionalPromptError(ValueError):
    """A `/rp` message that can't be turned into regions — the message is
    shown to the user as-is, so it should say how to fix it."""


@dataclass(frozen=True)
class RegionalPrompt:
    """A parsed `/rp` message. `global_prompt`/`negative` are left as typed
    (the caller runs them through the normal prompt handling — `-token`
    negatives, active character); region prompts are already normalized to
    one comma-joined string each."""

    global_prompt: str
    regions: list[RegionSpec]
    negative: str


def _normalize(text: str) -> str:
    parts = [part.strip() for part in re.split(r"[,\n]", text)]
    return ", ".join(part for part in parts if part)


def is_regional_prompt(text: str) -> bool:
    """Whether a plain prompt message uses the regional syntax — any line
    starting with a known position tag. Lets `handlers.generate_message`
    route it to the regional flow without `/rp`, which stays the way to get
    the format explained and a template to copy."""
    return _KNOWN_TAG_LINE_RE.search(text) is not None


def parse_regional_prompt(text: str) -> RegionalPrompt:
    """Split a `/rp` message into global prompt, laid-out regions and
    negative block. Raises `RegionalPromptError` for anything the layout
    can't represent: no regions, an unknown or repeated position, an empty
    region, or mixing the column and row axes."""
    negative = ""
    separator = _NEGATIVE_BLOCK_SEP_RE.search(text)
    if separator:
        negative = text[separator.end() :].strip()
        text = text[: separator.start()]

    global_lines: list[str] = []
    bodies: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        tag = _REGION_TAG_RE.match(line)
        if tag is None:
            (global_lines if current is None else current).append(line)
            continue
        raw_label = tag.group(1).strip().lower()
        label = _CENTRE_ALIASES.get(raw_label, raw_label)
        if label not in (*_HORIZONTAL, *_VERTICAL, "center"):
            raise RegionalPromptError(f"Unknown region [{tag.group(1)}] — use {_VALID_TAGS}.")
        if label in bodies:
            raise RegionalPromptError(f"[{raw_label}] appears twice — give each region one tag.")
        current = bodies[label] = [tag.group(2)]

    if not bodies:
        raise RegionalPromptError(
            f"No regions found — start a line with {_VALID_TAGS} to give that part of "
            "the image its own prompt."
        )
    prompts = {label: _normalize("\n".join(lines)) for label, lines in bodies.items()}
    for label, prompt in prompts.items():
        if not prompt:
            raise RegionalPromptError(f"[{label}] has no prompt — write its tags after the tag.")

    return RegionalPrompt(
        global_prompt="\n".join(global_lines).strip(),
        regions=_layout(prompts),
        negative=negative,
    )


def _layout(prompts: dict[str, str]) -> list[RegionSpec]:
    """Turn `{label: prompt}` into boxes, ordered left-to-right/top-to-bottom.
    Equal slots along one axis: two, or three when the centre slot is used.
    A lone centre tag counts as a column."""
    horizontal = any(label in _HORIZONTAL for label in prompts)
    vertical = any(label in _VERTICAL for label in prompts)
    if horizontal and vertical:
        raise RegionalPromptError(
            "Mixing left/right with top/bottom isn't supported — split the image "
            "either side by side or stacked."
        )
    order = ("top", "center", "bottom") if vertical else ("left", "center", "right")
    slots = 3 if "center" in prompts else 2
    slot_index = {order[0]: 0, "center": 1, order[2]: slots - 1}
    size = round(1 / slots, 3)

    regions = []
    for label in sorted(prompts, key=slot_index.__getitem__):
        start = round(slot_index[label] / slots, 3)
        extent = round(1 - start, 3) if slot_index[label] == slots - 1 else size
        display = "middle" if vertical and label == "center" else label
        if vertical:
            box = {"x": 0.0, "y": start, "w": 1.0, "h": extent}
        else:
            box = {"x": start, "y": 0.0, "w": extent, "h": 1.0}
        regions.append(RegionSpec(label=display, prompt=prompts[label], **box))
    return regions


def expand_references(parsed: RegionalPrompt, references: ReferenceExpander) -> RegionalPrompt:
    """Replace every `$name` in the global prompt and the regions with that
    saved character's or favorite's text (see `prompt_refs`). The
    referenced characters' negative prompts are left on `references.negatives`
    for the caller — the negative is global under Attention Couple, so
    there's nowhere more specific to put them. Raises `RegionalPromptError`
    listing what's saved for an unknown `$name`, rather than leaving a
    literal `$name` in the prompt for the model to puzzle over."""
    try:
        return RegionalPrompt(
            global_prompt=references.expand(parsed.global_prompt),
            regions=[
                replace(region, prompt=_normalize(references.expand(region.prompt)))
                for region in parsed.regions
            ],
            negative=references.expand(parsed.negative),
        )
    except UnknownReferenceError as exc:
        raise RegionalPromptError(str(exc)) from exc


def format_regional_prompt(global_prompt: str, regions: list[RegionSpec]) -> str:
    """The `/rp` syntax for `global_prompt` plus `regions` — the inverse of
    `parse_regional_prompt`'s positive half, used to record what was typed
    (`GenerationParams.raw_positive_prompt`) so "🐛 Show Prompt"'s copy
    button hands back something `/rp` accepts again."""
    lines = [global_prompt] if global_prompt else []
    lines += [f"[{region.label}] {region.prompt}" for region in regions]
    return "\n".join(lines)
