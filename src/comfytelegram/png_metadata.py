"""Read and write comfytelegram's own metadata chunk inside a PNG.

Every image this bot sends carries, in a PNG `tEXt` chunk keyed
`comfytelegram`, the same resolved `GenerationParams` dict that
`storage.py`'s `pending_result` row holds — plus the seed, the kind of run
that produced it and a timestamp. That makes a downloaded PNG a
self-contained archive: sent back to the bot later it can be re-registered
as a `pending_result` (see `handlers.py`'s `document_message`) and every
post-processing button works again, on an image the bot has no database row
for at all.

Why our own chunk rather than ComfyUI's:
ComfyUI's `SaveImage` already writes the executed API-format graph into a
`prompt` chunk, and that *is* read here (`extract_comfy_graph`) as a
best-effort fallback for foreign images. But going from a graph back to
`GenerationParams` is lossy in the direction that matters. It needs a
separate walker per graph shape this codebase builds (txt2img vs upscale vs
`_build_detailer` vs `DetailerForEach` vs split-loader), each one breaking
whenever `workflows/builder.py` rewires a node; and some fields were never
nodes in the first place — `GenerationParams.raw_positive_prompt` (what the
user actually typed, before profile prefixes and the active character's
prompt got folded in, which "🐛 Show Prompt" and "🔁 Generate Again" both
need) only ever existed in Python, since the graph carries the single
concatenated string that reached `CLIPTextEncode`. Writing the dict we
already serialize for SQLite sidesteps all of that: the reader is
`params_serde.deserialize_generation_params`, which existed anyway.

Everything here is byte-level splicing — a chunk is inserted before `IEND`
and the CRC recomputed, with `IHDR`/`IDAT` never touched. No decode, no
re-encode, so embedding cannot alter a single pixel (and costs ~200 bytes
on an 18MB image). PNG defines `tEXt` as ancillary, so ComfyUI, Pillow,
browsers and Telegram's thumbnailer all pass an unrecognised keyword
through untouched.

Two things this does *not* survive, both by design of what's upstream:
ComfyUI writes a brand-new PNG for every post-processing pass, so the chunk
has to be re-embedded on each send rather than once at generation; and
Telegram's `sendPhoto` re-encodes to JPEG, dropping every chunk — only the
`sendDocument` path preserves the bytes, which is why "📥 Download file"
exists.
"""

from __future__ import annotations

import json
import logging
import struct
import time
import zlib
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Our own chunk's keyword. PNG limits a keyword to 1-79 latin-1 characters.
METADATA_KEYWORD = "comfytelegram"

#: ComfyUI `SaveImage`'s own chunk, holding the executed API-format graph.
COMFY_PROMPT_KEYWORD = "prompt"

#: Bumped if the chunk's *shape* changes incompatibly. Readers should treat
#: an unknown (higher) version as unreadable rather than guessing, but the
#: `params` sub-dict is separately tolerant of missing keys — see
#: `params_serde.deserialize_generation_params`.
SCHEMA_VERSION = 1


def is_png(data: bytes) -> bool:
    """True if `data` starts with the PNG signature. Telegram hands back
    whatever the user uploaded, so this is the guard before any of the
    chunk walking below (a JPEG would otherwise be read as garbage
    lengths)."""
    return data[:8] == PNG_SIGNATURE


def _iter_chunks(data: bytes) -> Iterator[tuple[int, bytes, bytes]]:
    """Walk the file's chunks, yielding `(offset, type, payload)` for each —
    `offset` being the start of that chunk's 4-byte length field, so a
    caller can splice around it. Stops at `IEND` or at the first structurally
    impossible chunk rather than raising, since these bytes come from
    whatever a user chose to upload."""
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        payload_end = offset + 8 + length
        if payload_end + 4 > len(data):
            return  # truncated/corrupt — stop, don't raise
        yield offset, chunk_type, data[offset + 8 : payload_end]
        if chunk_type == b"IEND":
            return
        offset = payload_end + 4


def _decode_text_payload(chunk_type: bytes, payload: bytes) -> tuple[str, str] | None:
    """One text chunk's `(keyword, value)`, or None if it can't be decoded.

    All three of PNG's text chunk types are handled because we don't get to
    choose which one *ComfyUI* used: Pillow's `PngInfo.add_text` (what
    `SaveImage` goes through) falls back from `tEXt` to `iTXt` whenever the
    value isn't latin-1 encodable, which a prompt containing CJK or emoji
    is not. Our own chunk is always plain `tEXt` — `json.dumps` escapes
    non-ASCII to `\\uXXXX` by default, so the value is pure ASCII and the
    simplest chunk type always suffices."""
    try:
        if chunk_type == b"tEXt":
            keyword, value = payload.split(b"\x00", 1)
            return keyword.decode("latin-1"), value.decode("latin-1")
        if chunk_type == b"zTXt":
            keyword, rest = payload.split(b"\x00", 1)
            # rest[0] is the compression method; only 0 (zlib) is defined.
            return keyword.decode("latin-1"), zlib.decompress(rest[1:]).decode("latin-1")
        if chunk_type == b"iTXt":
            keyword, rest = payload.split(b"\x00", 1)
            compressed = rest[0]
            # rest[1] is the compression method, then two NUL-terminated
            # fields we don't use (language tag, translated keyword).
            _language, rest = rest[2:].split(b"\x00", 1)
            _translated, text = rest.split(b"\x00", 1)
            if compressed:
                text = zlib.decompress(text)
            return keyword.decode("latin-1"), text.decode("utf-8")
    except (ValueError, IndexError, zlib.error, UnicodeDecodeError):
        return None
    return None


def read_text_chunks(data: bytes) -> dict[str, str]:
    """Every decodable text chunk in `data`, keyed by keyword. Empty for a
    non-PNG or a file with no text chunks."""
    if not is_png(data):
        return {}
    found: dict[str, str] = {}
    for _offset, chunk_type, payload in _iter_chunks(data):
        decoded = _decode_text_payload(chunk_type, payload)
        if decoded is not None:
            found.setdefault(*decoded)
    return found


def _build_text_chunk(keyword: str, text: str) -> bytes:
    """A complete `tEXt` chunk (length + type + payload + CRC) ready to
    splice into a file."""
    payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    crc = zlib.crc32(b"tEXt" + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + b"tEXt" + payload + struct.pack(">I", crc)


def _without_keyword(data: bytes, keyword: str) -> bytes:
    """`data` with any existing text chunk for `keyword` removed, so
    re-embedding replaces rather than accumulates. Relevant because a
    post-processing pass loads a previously-embedded image back into
    ComfyUI, and while `SaveImage` writes a fresh file today, an upstream
    change that copied source metadata forward would otherwise leave two
    conflicting chunks."""
    keep = bytearray(data[: len(PNG_SIGNATURE)])
    for offset, chunk_type, payload in _iter_chunks(data):
        decoded = _decode_text_payload(chunk_type, payload)
        if decoded is not None and decoded[0] == keyword:
            continue
        keep += data[offset : offset + 12 + len(payload)]
    return bytes(keep)


def insert_text_chunk(data: bytes, keyword: str, text: str) -> bytes:
    """`data` with a `tEXt` chunk holding `keyword`/`text` spliced in just
    before `IEND`, replacing any existing chunk with that keyword. Raises
    `ValueError` if `data` isn't a PNG with a locatable `IEND`."""
    if not is_png(data):
        raise ValueError("Not a PNG file")
    stripped = _without_keyword(data, keyword)
    for offset, chunk_type, _payload in _iter_chunks(stripped):
        if chunk_type == b"IEND":
            return stripped[:offset] + _build_text_chunk(keyword, text) + stripped[offset:]
    raise ValueError("PNG has no IEND chunk")


def build_metadata(
    *,
    params: dict[str, Any],
    kind: str,
    filename: str,
    seed: int | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Assemble the dict `embed_metadata` writes.

    `params` is `params_serde.serialize_generation_params` output verbatim —
    the *same* dict `storage.store_pending_result` persists — so an import
    can hand it straight back to `store_pending_result` with no translation.
    That deliberately excludes the seed (so "🔁 Generate Again" rerolls
    rather than reproducing the same image — see
    `test_serialization_omits_seed_so_regenerate_gets_a_fresh_roll`), which
    is why `seed` rides *alongside* it as its own top-level key: recorded
    for the archive and displayable on import, without feeding regeneration.
    `kind` records which run produced the file ("txt2img", "repeat", or a
    `generation.post_process` kind), which no field of `params` implies.
    `checkpoint` is denormalised out of `params` purely so a human reading
    the raw chunk sees the model without parsing the nested dict."""
    return {
        "app": "comfytelegram",
        "schema": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,
        "filename": filename,
        "seed": seed,
        "checkpoint": checkpoint if checkpoint is not None else params.get("checkpoint"),
        "params": params,
    }


def embed_metadata(data: bytes, metadata: dict[str, Any]) -> bytes:
    """`data` with `metadata` written into its `comfytelegram` chunk.

    Returns `data` unchanged if it isn't a PNG or the splice fails for any
    reason — metadata is a convenience, and an image the user is waiting
    for must never be lost to a chunk-writing bug."""
    try:
        return insert_text_chunk(data, METADATA_KEYWORD, json.dumps(metadata))
    except (ValueError, UnicodeEncodeError, TypeError):
        logger.warning("Could not embed metadata into image, sending it unmodified", exc_info=True)
        return data


def extract_metadata(data: bytes) -> dict[str, Any] | None:
    """The `comfytelegram` chunk's decoded dict, or None if `data` has no
    such chunk, it isn't valid JSON, or it was written by a schema version
    newer than this build understands."""
    raw = read_text_chunks(data).get(METADATA_KEYWORD)
    if raw is None:
        return None
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Image has a %s chunk that isn't valid JSON", METADATA_KEYWORD)
        return None
    if not isinstance(metadata, dict) or not isinstance(metadata.get("params"), dict):
        return None
    if metadata.get("schema", 0) > SCHEMA_VERSION:
        logger.warning(
            "Image metadata schema %s is newer than this build's %s",
            metadata.get("schema"),
            SCHEMA_VERSION,
        )
        return None
    return metadata


def extract_comfy_graph(data: bytes) -> dict[str, Any] | None:
    """ComfyUI's own `prompt` chunk — the executed API-format graph — or
    None. Present on anything `SaveImage` wrote, including images from
    other front-ends entirely (a Krita AI Diffusion export carries one too),
    which is what makes it a usable fallback when `extract_metadata` finds
    nothing of ours."""
    raw = read_text_chunks(data).get(COMFY_PROMPT_KEYWORD)
    if raw is None:
        return None
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return graph if isinstance(graph, dict) else None


def extract_seed(graph: dict[str, Any]) -> int | None:
    """The sampling seed out of an API-format graph, for recording in our
    own chunk. Read from the graph rather than from `GenerationParams`
    because `params.seed` is normally None: the builders roll the actual
    value at graph-build time (`workflows/builder.py`'s `_resolve_seed`), so
    the submitted graph is the only place the seed a given image was
    produced with exists. Prefers `KSampler`-family nodes over the
    `RandomNoise`/inpaint-model seeds a post-processing graph also carries,
    and returns the lowest node id's seed when several qualify, so a
    multi-sampler graph picks its first pass deterministically."""
    candidates: list[tuple[int, int, int]] = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key in ("seed", "noise_seed"):
            value = inputs.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                sampler_first = 0 if "sampler" in str(node.get("class_type", "")).lower() else 1
                try:
                    order = int(node_id)
                except (TypeError, ValueError):
                    order = 1 << 30
                candidates.append((sampler_first, order, value))
                break
    if not candidates:
        return None
    return min(candidates)[2]


#: Node inputs a conditioning chain is threaded through, in the order
#: `_follow_to_text` tries them. A prompt rarely reaches the sampler
#: straight from `CLIPTextEncode` — a ControlNet apply, a style model, a
#: conditioning combine or an Anima control node usually sits in between,
#: each passing the conditioning along under one of these names.
_CONDITIONING_INPUTS = ("conditioning", "positive", "negative", "conditioning_1", "cond")

#: How many nodes deep `_follow_to_text` will chase a conditioning chain
#: before giving up. Purely a guard against a cyclic or pathological graph;
#: real chains are a handful of nodes at most.
_FOLLOW_MAX_DEPTH = 12


def _follow_to_text(graph: dict[str, Any], ref: Any, depth: int = 0) -> str | None:
    """Chase a `[node_id, output_index]` conditioning link back to the
    `CLIPTextEncode` that produced it, returning its `text`."""
    if depth > _FOLLOW_MAX_DEPTH or not isinstance(ref, list) or not ref:
        return None
    node = graph.get(str(ref[0]))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    text = inputs.get("text")
    if isinstance(text, str):
        return text
    for key in _CONDITIONING_INPUTS:
        if key in inputs:
            found = _follow_to_text(graph, inputs[key], depth + 1)
            if found is not None:
                return found
    return None


def summarize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Best-effort settings read straight out of an API-format graph, for an
    image that has ComfyUI's `prompt` chunk but not ours.

    This is deliberately *not* how an image generated by this bot is
    imported — see this module's docstring for why a graph can't
    reconstruct a full `GenerationParams`. It exists so a PNG from some
    other front-end entirely (a Krita AI Diffusion export, a raw ComfyUI
    run, someone else's bot) still yields its prompt and headline settings
    to show the user, rather than the bot saying only "no metadata".

    Every key may be absent: a graph shape this doesn't recognise just
    produces a smaller dict rather than raising.
    """
    summary: dict[str, Any] = {}

    sampler = None
    for node_id in sorted(graph, key=lambda n: (len(n), n)):
        node = graph.get(node_id)
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        inputs = node["inputs"]
        # Identified by shape, not by name: `KSampler`, `CFGGuider` and
        # every custom sampler wrapper all take linked `positive`/
        # `negative` conditioning, while name-matching on "sampler"
        # picks up `KSamplerSelect` — which carries neither.
        if sampler is None and all(
            isinstance(inputs.get(side), list) for side in ("positive", "negative")
        ):
            sampler = inputs
        for key, field in (
            ("ckpt_name", "checkpoint"),
            ("unet_name", "checkpoint"),
            ("sampler_name", "sampler_name"),
            ("scheduler", "scheduler"),
        ):
            if key in inputs and field not in summary:
                summary[field] = inputs[key]
        for key in ("steps", "cfg", "width", "height", "denoise"):
            if key in inputs and key not in summary and not isinstance(inputs[key], list):
                summary[key] = inputs[key]

    seed = extract_seed(graph)
    if seed is not None:
        summary["seed"] = seed
    if sampler is not None:
        for field, key in (("positive_prompt", "positive"), ("negative_prompt", "negative")):
            text = _follow_to_text(graph, sampler.get(key))
            if text is not None:
                summary[field] = text
    return summary
