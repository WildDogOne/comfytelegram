"""`$name` references inside a prompt: a saved character (`/character`,
per chat) or a favorite (`/fav`, per user) swapped in for its text.

Used everywhere a prompt is typed — a plain message and `/stream`
(`handlers._resolve_effective_prompt`), each region of a regional prompt
(`regional.expand_references`) and the "✏️ Detail Prompt" editor's fields
(`handlers._process_one_inpaint_job`).

A name that is both a character and a favorite resolves to the character:
characters are shared by the whole chat and carry a negative prompt too, so
they're the more deliberate of the two. Names match exactly first, then
case-insensitively (`$Alice` for a character saved as `alice`).

Telegram-independent — callers hand in the already-loaded lists.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: `$name` — same character set as `handlers.CHARACTER_NAME_RE`, which is a
#: superset of `fav_menu.FAVORITE_NAME_RE`, so every saved name can be
#: referenced. Not preceded by a word character or another `$`, so `US$5`
#: and `$$` aren't references.
REFERENCE_RE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]{1,32})")

#: How many favorite names an unknown-reference error lists before pointing
#: at `/favs` instead — the list can run to dozens.
_MAX_LISTED_FAVORITES = 15

#: `reference_suggestions` caps each preview at this many characters — it's
#: a reminder of what a name inserts, not the full text.
_SUGGESTION_PREVIEW_CHARS = 60

#: Rough JSON-size budget for `reference_suggestions`. They ride to
#: inpaint_relay in the base64'd `X-Job-Meta` header, next to the image's
#: prompt, and uvicorn's h11 parser rejects a header block over 16KB.
SUGGESTION_BUDGET_BYTES = 6000


class UnknownReferenceError(ValueError):
    """A `$name` that's neither a saved character nor a favorite. The
    message is shown to the user as-is and lists what is saved."""


class ReferenceExpander:
    """Expands `$name`s across one or more texts of the same prompt
    (`expand`), remembering which characters were used so their negative
    prompts can be added once each (`negatives`).

    `strict=False` drops an unknown `$name` instead of raising, recording it
    in `unknown` — for "✏️ Detail Prompt", whose submission comes with a
    hand-drawn mask that a rejected prompt would throw away."""

    def __init__(
        self,
        characters: list[dict[str, Any]],
        favorites: list[dict[str, Any]],
        *,
        strict: bool = True,
    ) -> None:
        self._characters = characters
        self._favorites = favorites
        self._strict = strict
        self._used_characters: list[dict[str, Any]] = []
        self.unknown: list[str] = []

    def _lookup(self, name: str) -> tuple[str, dict[str, Any]] | None:
        folded = name.casefold()
        for kind, entries in (("character", self._characters), ("favorite", self._favorites)):
            match = next((e for e in entries if e["name"] == name), None) or next(
                (e for e in entries if e["name"].casefold() == folded), None
            )
            if match is not None:
                return kind, match
        return None

    def _unknown_message(self, name: str) -> str:
        lines = [f"Nothing saved under ${name}."]
        if self._characters:
            lines.append("Characters: " + ", ".join(f"${c['name']}" for c in self._characters))
        if self._favorites:
            names = [f"${f['name']}" for f in self._favorites[:_MAX_LISTED_FAVORITES]]
            more = " …see /favs" if len(self._favorites) > _MAX_LISTED_FAVORITES else ""
            lines.append("Favorites: " + ", ".join(names) + more)
        if len(lines) == 1:
            lines.append("No characters or favorites are saved yet (/characters, /fav).")
        return "\n".join(lines)

    def _substitute(self, match: re.Match[str]) -> str:
        name = match.group(1)
        found = self._lookup(name)
        if found is None:
            if self._strict:
                raise UnknownReferenceError(self._unknown_message(name))
            if name not in self.unknown:
                self.unknown.append(name)
            return ""
        kind, entry = found
        if kind == "favorite":
            return entry["text"]
        if entry not in self._used_characters:
            self._used_characters.append(entry)
        return entry["positive_prompt"]

    def expand(self, text: str) -> str:
        """`text` with every `$name` replaced. Tidies the commas an empty
        expansion (or a dropped unknown name) leaves behind."""
        if "$" not in text:
            return text
        expanded = REFERENCE_RE.sub(self._substitute, text)
        expanded = re.sub(r"[ \t]*,(?:[ \t]*,)+", ",", expanded)
        return re.sub(r"^[ \t]*,[ \t]*|[ \t]*,[ \t]*$", "", expanded, flags=re.MULTILINE)

    @property
    def negatives(self) -> str:
        """The referenced characters' negative prompts, in order of first
        use, each once."""
        return ", ".join(
            c["negative_prompt"] for c in self._used_characters if c["negative_prompt"]
        )


def has_references(text: str | None) -> bool:
    return bool(text) and REFERENCE_RE.search(text) is not None


def reference_suggestions(
    characters: list[dict[str, Any]],
    favorites: list[dict[str, Any]],
    *,
    budget: int = SUGGESTION_BUDGET_BYTES,
) -> list[dict[str, str]]:
    """What `$` can complete to in the "✏️ Detail Prompt" editor:
    `{"name", "kind": "character"|"favorite", "preview"}`, characters
    first. A favorite shadowed by a same-named character is left out, since
    `$name` would insert the character. Keeps the JSON within `budget`
    bytes (see `SUGGESTION_BUDGET_BYTES`): once previews no longer fit,
    later entries go without one, and past that they're left out."""
    character_names = {c["name"].casefold() for c in characters}
    entries = [("character", c["name"], c["positive_prompt"]) for c in characters] + [
        ("favorite", f["name"], f["text"])
        for f in favorites
        if f["name"].casefold() not in character_names
    ]
    suggestions: list[dict[str, str]] = []
    used = 2  # the enclosing []
    for kind, name, text in entries:
        preview = " ".join(text.split())
        if len(preview) > _SUGGESTION_PREVIEW_CHARS:
            preview = preview[: _SUGGESTION_PREVIEW_CHARS - 1].rstrip() + "…"
        suggestion = {"name": name, "kind": kind, "preview": preview}
        size = len(json.dumps(suggestion)) + 2
        if used + size > budget:
            # Out of room for previews: a bare name still completes.
            suggestion["preview"] = ""
            size = len(json.dumps(suggestion)) + 2
            if used + size > budget:
                break
        used += size
        suggestions.append(suggestion)
    return suggestions
