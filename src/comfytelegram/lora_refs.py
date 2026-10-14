"""`<lora:name>` / `<lora:name:strength>` / `<lora:name:model:clip>` tags
inside a prompt — A1111's syntax, switching on a LoRA (or changing its
strength) for one generation without going through `/lora`.

Only LoRAs listed in the checkpoint's model profile (`ModelProfile.loras`)
can be named: that curation is the only signal for which LoRAs suit the
checkpoint's architecture at all (see `lora_menu.py`), so a typed tag gets
the same guard rail as a `/lora` toggle. `name` matches the LoRA's path as
ComfyUI lists it (`chars/alice_v2.safetensors`), with or without the
extension, or just its file name when that's unambiguous — always
case-insensitively, and with `\\` and `/` treated alike, since ComfyUI on
Windows lists subfolders with backslashes.

A tag is applied as a profile layer (`apply_prompt_loras`, like
`profiles.apply_lora_overrides`), so `resolve_generation_params` picks up
the LoRA and its `trigger_words` exactly as if it were default-enabled.
Strength `0` switches a default-enabled LoRA off. Tags are read after
`$name` expansion, so a saved character or favorite can carry its own
LoRA.

Used for typed prompts (`handlers._resolve_profile_and_prompt`) and for
"✏️ Detail Prompt" submissions (`generation.post_process`, via
`resolve_detail_loras`), whose editor autocompletes them from
`lora_suggestions`. Not inside a `/rp` region: every region shares one
model, so a region can't have a LoRA of its own.

Telegram-independent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from comfytelegram.profiles.schema import LoraDefault, ModelProfile
from comfytelegram.prompt_refs import UnknownReferenceError, shorten_preview, tidy_commas

_NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)"

#: `<lora:name>`, `<lora:name:0.8>` or `<lora:name:0.8:0.5>` (model, then
#: clip strength). `name` runs up to the first `:`/`>` — LoRA filenames
#: carry dots, spaces and subfolders (`/`), but not colons, which Windows
#: doesn't allow in a filename anyway.
LORA_TAG_RE = re.compile(
    rf"<lora:\s*([^:<>]+?)\s*(?::\s*({_NUMBER})\s*)?(?::\s*({_NUMBER})\s*)?>",
    re.IGNORECASE,
)

#: How many available LoRAs an unknown-name error lists.
_MAX_LISTED_LORAS = 20

#: `lora_suggestions` caps each trigger-word preview at this many characters.
_SUGGESTION_PREVIEW_CHARS = 30


class UnknownLoraError(UnknownReferenceError):
    """A `<lora:…>` tag naming nothing in the checkpoint's profile (or more
    than one thing). A subclass of `UnknownReferenceError` so every place
    that reports a bad `$name` back to the user reports this the same way."""


@dataclass(frozen=True)
class LoraTag:
    """One parsed tag. A strength left out is `None` — keep the profile's
    own (or `/lora`'s override of it); a lone strength sets both."""

    name: str
    strength_model: float | None = None
    strength_clip: float | None = None


def has_lora_tags(text: str | None) -> bool:
    return bool(text) and LORA_TAG_RE.search(text) is not None


def extract_lora_tags(text: str) -> tuple[str, list[LoraTag]]:
    """`text` with every tag removed (and the commas they leave behind
    tidied), plus the tags in order."""
    tags: list[LoraTag] = []

    def _take(match: re.Match[str]) -> str:
        model = float(match.group(2)) if match.group(2) is not None else None
        clip = float(match.group(3)) if match.group(3) is not None else model
        tags.append(LoraTag(match.group(1), model, clip))
        return ""

    if "<" not in text:
        return text, tags
    cleaned = LORA_TAG_RE.sub(_take, text)
    if not tags:
        return text, tags
    return tidy_commas(cleaned), tags


def _normalize(path: str) -> str:
    return path.replace("\\", "/").strip().casefold()


#: ComfyUI core's `folder_paths.supported_pt_extensions` — same set as
#: `lora_discovery.LORA_EXTENSIONS`. Only these count as an extension, so a
#: dot inside a name ("style_v1.5") isn't mistaken for one.
_EXTENSIONS = {"ckpt", "pt", "pt2", "bin", "pth", "safetensors", "pkl", "sft"}


def _strip_extension(path: str) -> str:
    stem, dot, extension = path.rpartition(".")
    return stem if dot and extension.casefold() in _EXTENSIONS else path


def match_lora(name: str, loras: list[LoraDefault]) -> LoraDefault | list[LoraDefault] | None:
    """The profile LoRA `name` refers to: its full path (with or without
    the extension) first, then its file name alone. A list when the file
    name alone matches more than one LoRA (same name in two folders), None
    when nothing matches."""
    wanted = _normalize(name)
    for lora in loras:
        full = _normalize(lora.name)
        if wanted in (full, _strip_extension(full)):
            return lora
    candidates = [
        lora
        for lora in loras
        if wanted
        in (
            PurePosixPath(_normalize(lora.name)).name,
            _strip_extension(PurePosixPath(_normalize(lora.name)).name),
        )
    ]
    if len(candidates) == 1:
        return candidates[0]
    return candidates or None


def _display(lora: LoraDefault) -> str:
    """`lora`'s path as it's typed in a tag: forward slashes, no extension."""
    return _strip_extension(lora.name.replace("\\", "/"))


def _unknown_message(
    name: str, found: list[LoraDefault] | None, profile: ModelProfile | None
) -> str:
    if isinstance(found, list):
        options = ", ".join(f"<lora:{_display(lora)}>" for lora in found)
        return f"<lora:{name}> matches more than one LoRA — write its folder too: {options}"
    if profile is None or not profile.loras:
        return (
            f"No LoRA called {name} — this checkpoint's profile doesn't list any LoRAs, "
            "so there's nothing to choose from."
        )
    names = [_display(lora) for lora in profile.loras[:_MAX_LISTED_LORAS]]
    more = " …see /lora" if len(profile.loras) > _MAX_LISTED_LORAS else ""
    return f"No LoRA called {name} for this checkpoint.\nAvailable: " + ", ".join(names) + more


def _resolve(tag: LoraTag, profile: ModelProfile | None, *, strict: bool) -> LoraDefault | None:
    found = match_lora(tag.name, profile.loras) if profile is not None else None
    if isinstance(found, LoraDefault):
        return found
    if strict:
        raise UnknownLoraError(_unknown_message(tag.name, found, profile))
    return None


def _with_strengths(lora: LoraDefault, tag: LoraTag) -> LoraDefault:
    update: dict[str, object] = {}
    if tag.strength_model is not None:
        update["strength_model"] = tag.strength_model
    if tag.strength_clip is not None:
        update["strength_clip"] = tag.strength_clip
    resolved = lora.model_copy(update=update)
    return resolved.model_copy(
        update={"default_enabled": bool(resolved.strength_model or resolved.strength_clip)}
    )


def apply_prompt_loras(profile: ModelProfile | None, tags: list[LoraTag]) -> ModelProfile | None:
    """Layer typed tags on top of `profile` (already carrying `/settings`
    and `/lora` overrides): each named LoRA is switched on, at the tag's
    strength if it gave one — or off, at strength 0. A later tag for the
    same LoRA wins. Raises `UnknownLoraError` for a name the profile
    doesn't list."""
    if not tags:
        return profile
    by_name: dict[str, LoraDefault] = {}
    for tag in tags:
        lora = _resolve(tag, profile, strict=True)
        assert lora is not None
        by_name[lora.name] = _with_strengths(by_name.get(lora.name, lora), tag)
    assert profile is not None  # _resolve raised otherwise
    return profile.model_copy(
        update={"loras": [by_name.get(lora.name, lora) for lora in profile.loras]}
    )


def resolve_detail_loras(
    text: str | None, profile: ModelProfile | None
) -> tuple[str | None, list[LoraDefault], list[str]]:
    """For a "✏️ Detail Prompt" field: `(text without tags, the LoRAs it
    names at their resolved strengths, names that matched nothing)`. Never
    raises — an editor submission comes with a drawn mask that rejecting it
    would throw away, so unknown names are reported back instead. A
    resolved LoRA at strength 0 is still returned (`default_enabled`
    False): there it removes one the image was generated with."""
    if not text:
        return text, [], []
    cleaned, tags = extract_lora_tags(text)
    by_name: dict[str, LoraDefault] = {}
    unknown: list[str] = []
    for tag in tags:
        lora = _resolve(tag, profile, strict=False)
        if lora is None:
            if tag.name not in unknown:
                unknown.append(tag.name)
            continue
        by_name[lora.name] = _with_strengths(by_name.get(lora.name, lora), tag)
    return cleaned, list(by_name.values()), unknown


def strip_lora_tags(text: str | None, names: list[str]) -> str | None:
    """`text` without the tags naming any of `names` (as typed) — how a
    "✏️ Detail Prompt" submission drops the ones `resolve_detail_loras`
    couldn't match before it's stored for redo."""
    if not text or not names:
        return text
    dropped = {name.casefold() for name in names}
    cleaned = LORA_TAG_RE.sub(
        lambda m: "" if m.group(1).casefold() in dropped else m.group(0), text
    )
    return tidy_commas(cleaned)


def _strength(lora: LoraDefault) -> str:
    """`lora`'s configured strength as written after the name in a tag:
    `0.9`, or `1:0.5` when model and clip differ."""
    if lora.strength_model == lora.strength_clip:
        return f"{lora.strength_model:g}"
    return f"{lora.strength_model:g}:{lora.strength_clip:g}"


def lora_tag(lora: LoraDefault) -> str:
    """The tag that inserts `lora` at its configured strength — what the
    editor's autocomplete types (built page-side from `lora_suggestions`'
    `name`/`strength`, the same way)."""
    return f"<lora:{_display(lora)}:{_strength(lora)}>"


def lora_suggestions(profile: ModelProfile | None) -> list[dict[str, str]]:
    """What `<` can complete to in the "✏️ Detail Prompt" editor: one
    `{"name", "strength", "preview"}` per profile LoRA — every one of them
    (see `prompt_refs.reference_suggestions` for why there's no size
    budget) — `name` being the path without its extension, `strength` as
    `_strength` writes it and `preview` its trigger words."""
    if profile is None:
        return []
    return [
        {
            "name": _display(lora),
            "strength": _strength(lora),
            "preview": shorten_preview(lora.trigger_words, _SUGGESTION_PREVIEW_CHARS),
        }
        for lora in profile.loras
    ]
