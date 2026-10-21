"""`<ar:2:3>` tags inside a prompt — pick the image's aspect ratio for one
generation without going through `/settings`. Also spelled
`<aspect:…>`/`<aspectratio:…>`, with `:`, `x` or `/` between the two sides
(`<ar:16x9>`), or as a single number (`<ar:1.5>`, width over height).

The tag only says the *shape*: the size keeps the pixel count of the
profile's (or `/settings`') width x height, snapped to 64 — the same thing
`/pose` does with a skeleton's shape (`pose.size_for_pose`), so a
1024x1024 SDXL profile stays near 1MP (`<ar:2:3>` -> 832x1280) and a model
trained at another resolution keeps its own. A typed tag wins
over the pose's shape (the caller pins the size via `overrides`, which
`generation.generate` leaves alone).

Read after `$name` expansion, like `<lora:…>`, so a favorite can carry
one. Where the size is already fixed — post-processing, "✏️ Detail
Prompt", "🎨 Paint Regions" re-running an existing image — the tag is just
taken out of the text (`strip_aspect_tags`).

Telegram-independent."""

from __future__ import annotations

import re
from dataclasses import dataclass

from comfytelegram.prompt_refs import UnknownReferenceError, tidy_commas

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"

ASPECT_TAG_RE = re.compile(
    rf"<(?:ar|aspect|aspect_?ratio):\s*({_NUMBER})\s*(?:[:x/]\s*({_NUMBER})\s*)?>",
    re.IGNORECASE,
)

#: The widest (or tallest) shape a tag may ask for. SDXL-family models fall
#: apart well before this; it's here to catch a typo like `<ar:23>`.
MAX_ASPECT = 4.0

#: Generation sizes are snapped to this (`/pose` sizes go through here too).
SIZE_MULTIPLE = 64


class InvalidAspectRatioError(UnknownReferenceError):
    """An `<ar:…>` tag with a zero side or a shape past `MAX_ASPECT`. A
    subclass of `UnknownReferenceError` so every place that reports a bad
    `$name` back to the user reports this the same way."""


@dataclass(frozen=True)
class AspectRatio:
    width: float
    height: float

    @property
    def label(self) -> str:
        return f"{self.width:g}:{self.height:g}"


def has_aspect_tags(text: str | None) -> bool:
    return bool(text) and ASPECT_TAG_RE.search(text) is not None


def extract_aspect_tag(text: str) -> tuple[str, AspectRatio | None]:
    """`text` with every aspect tag removed (and the commas they leave
    tidied), plus the last one's ratio — the last wins, so a tag typed after
    a favorite that carries its own overrides it. Raises
    `InvalidAspectRatioError` for a ratio that can't be generated."""
    found: list[AspectRatio] = []

    def _take(match: re.Match[str]) -> str:
        width = float(match.group(1))
        height = float(match.group(2)) if match.group(2) is not None else 1.0
        if width <= 0 or height <= 0:
            raise InvalidAspectRatioError(
                f"{match.group(0)} has a zero side — write it like <ar:2:3>."
            )
        ratio = AspectRatio(width, height)
        if max(width / height, height / width) > MAX_ASPECT:
            raise InvalidAspectRatioError(
                f"{match.group(0)} is too stretched — keep it within "
                f"{MAX_ASPECT:g}:1 either way, e.g. <ar:2:3> or <ar:16:9>."
            )
        found.append(ratio)
        return ""

    if "<" not in text:
        return text, None
    cleaned = ASPECT_TAG_RE.sub(_take, text)
    if not found:
        return text, None
    return tidy_commas(cleaned).strip(), found[-1]


def strip_aspect_tags(text: str | None) -> str | None:
    """`text` without any aspect tag, valid or not — for a prompt whose
    image already has its size."""
    if not has_aspect_tags(text):
        return text
    return tidy_commas(ASPECT_TAG_RE.sub("", text)).strip()


def size_for_aspect(
    aspect_width: float, aspect_height: float, width: int, height: int
) -> tuple[int, int]:
    """A generation size of shape `aspect_width`:`aspect_height` with about
    the pixel count of `width`x`height`, each side snapped to
    `SIZE_MULTIPLE`."""
    scale = (width * height / (aspect_width * aspect_height)) ** 0.5
    return (
        max(SIZE_MULTIPLE, round(aspect_width * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE),
        max(SIZE_MULTIPLE, round(aspect_height * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE),
    )
