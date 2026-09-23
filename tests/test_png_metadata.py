"""Chunk-level round-trip tests for `png_metadata`.

The images here are built with Pillow rather than fixtures so the tests
stay independent of a live ComfyUI (and of any checked-in binary), and so
a "pixels are untouched" assertion can compare against a known original.
"""

import io
import json
import struct
import zlib

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfytelegram.params_serde import deserialize_generation_params, serialize_generation_params
from comfytelegram.png_metadata import (
    METADATA_KEYWORD,
    build_metadata,
    embed_metadata,
    extract_comfy_graph,
    extract_metadata,
    extract_seed,
    insert_text_chunk,
    is_png,
    read_text_chunks,
)
from comfytelegram.workflows import GenerationParams, LoraSpec


def make_png(text: dict[str, str] | None = None, size=(8, 8)) -> bytes:
    """A small PNG, optionally carrying text chunks written the way ComfyUI's
    `SaveImage` writes them (through Pillow's `PngInfo`)."""
    info = PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    buf = io.BytesIO()
    Image.new("RGB", size, (20, 120, 200)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def idat_bytes(png: bytes) -> bytes:
    """Every `IDAT` payload concatenated — the actual compressed pixels,
    for asserting that embedding metadata didn't touch them."""
    out, offset = b"", 8
    while offset + 8 <= len(png):
        (length,) = struct.unpack(">I", png[offset : offset + 4])
        if png[offset + 4 : offset + 8] == b"IDAT":
            out += png[offset + 8 : offset + 8 + length]
        offset += 12 + length
    return out


def sample_params() -> GenerationParams:
    return GenerationParams(
        checkpoint="anima-base-v1.0.safetensors",
        positive_prompt="masterpiece, a cat",
        negative_prompt="worst quality",
        steps=24,
        cfg=4.5,
        sampler_name="euler_ancestral",
        scheduler="normal",
        width=832,
        height=1216,
        clip_skip=-2,
        loras=[LoraSpec(name="detail.safetensors", strength_model=0.7, strength_clip=0.6)],
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        clip_type="omnigen2",
        vae_name="qwen_image_vae.safetensors",
        raw_positive_prompt="a cat",
    )


def test_embed_then_extract_round_trips_every_generation_param():
    """The whole point: an image carries enough to rebuild the exact
    `GenerationParams` it was made with, with no database row involved."""
    params = sample_params()
    metadata = build_metadata(
        params=serialize_generation_params(params),
        kind="txt2img",
        filename="comfytelegram_00001_.png",
        seed=12345,
    )
    png = embed_metadata(make_png(), metadata)

    restored = deserialize_generation_params(extract_metadata(png)["params"])
    # seed/filename_prefix are deliberately not serialized (see
    # params_serde), so compare everything else field by field.
    assert restored == params


def test_metadata_records_seed_and_kind_outside_the_params_blob():
    """`params` has to stay byte-identical to what `store_pending_result`
    persists, so the seed rides alongside it rather than inside it — an
    import can display the seed without "🔁 Generate Again" reproducing it
    instead of rerolling."""
    metadata = build_metadata(
        params=serialize_generation_params(sample_params()),
        kind="upscale",
        filename="comfytelegram_upscale_00002_.png",
        seed=999,
    )
    png = embed_metadata(make_png(), metadata)

    extracted = extract_metadata(png)
    assert extracted["seed"] == 999
    assert extracted["kind"] == "upscale"
    assert extracted["checkpoint"] == "anima-base-v1.0.safetensors"
    assert "seed" not in extracted["params"]


def test_embedding_does_not_touch_the_pixels():
    """Metadata is spliced in as a chunk, never re-encoded — so an archived
    image is bit-for-bit the one ComfyUI produced."""
    original = make_png(size=(64, 64))
    stamped = embed_metadata(original, build_metadata(params={}, kind="txt2img", filename="x.png"))

    assert idat_bytes(stamped) == idat_bytes(original)
    assert len(stamped) > len(original)


def test_comfyui_prompt_chunk_survives_our_embedding():
    """We add a chunk beside ComfyUI's, never in place of it — the executed
    graph stays readable for anything that wants it."""
    graph = {"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}}}
    png = embed_metadata(
        make_png({"prompt": json.dumps(graph)}),
        build_metadata(params={}, kind="txt2img", filename="x.png"),
    )

    assert extract_comfy_graph(png) == graph
    assert extract_metadata(png) is not None


def test_re_embedding_replaces_rather_than_accumulates():
    png = embed_metadata(make_png(), build_metadata(params={}, kind="txt2img", filename="a.png"))
    png = embed_metadata(png, build_metadata(params={}, kind="upscale", filename="b.png"))

    assert extract_metadata(png)["filename"] == "b.png"
    assert extract_metadata(png)["kind"] == "upscale"
    # exactly one chunk with our keyword survived
    assert sum(1 for k in read_text_chunks(png) if k == METADATA_KEYWORD) == 1


def test_non_ascii_prompt_survives_a_latin1_only_text_chunk():
    """`tEXt` payloads are latin-1, but `json.dumps` escapes non-ASCII to
    `\\uXXXX` by default — so a CJK/emoji prompt round-trips without
    needing the more complex `iTXt` chunk type."""
    params = serialize_generation_params(
        GenerationParams(
            checkpoint="x.safetensors",
            positive_prompt="猫, 高品質 🐈",
            negative_prompt="低品質",
        )
    )
    png = embed_metadata(
        make_png(), build_metadata(params=params, kind="txt2img", filename="x.png")
    )

    assert extract_metadata(png)["params"]["positive_prompt"] == "猫, 高品質 🐈"


def test_reads_itxt_chunks_that_pillow_writes_for_non_latin1_values():
    """Pillow's `add_text` silently upgrades to `iTXt` when a value isn't
    latin-1 encodable — which is exactly what ComfyUI's own `prompt` chunk
    does for a non-ASCII prompt, so the fallback reader has to handle it."""
    graph = {"3": {"class_type": "KSampler", "inputs": {"text": "猫", "seed": 7}}}
    png = make_png({"prompt": json.dumps(graph, ensure_ascii=False)})

    assert extract_comfy_graph(png) == graph


def test_extract_metadata_ignores_a_foreign_or_broken_chunk():
    assert extract_metadata(make_png()) is None
    assert extract_metadata(make_png({METADATA_KEYWORD: "not json"})) is None
    assert extract_metadata(make_png({METADATA_KEYWORD: '{"params": "wrong type"}'})) is None
    assert extract_metadata(b"\xff\xd8\xff\xe0 this is a jpeg") is None


def test_extract_metadata_refuses_a_newer_schema_rather_than_guessing():
    png = make_png({METADATA_KEYWORD: json.dumps({"schema": 99, "params": {"checkpoint": "x"}})})
    assert extract_metadata(png) is None


def test_embed_metadata_returns_the_image_unchanged_when_it_cannot_write():
    """An image the user is waiting for must never be lost to a metadata
    bug — a non-PNG (or an unserializable payload) just passes through."""
    jpeg = b"\xff\xd8\xff\xe0 not a png"
    assert embed_metadata(jpeg, {"params": {}}) == jpeg

    png = make_png()
    assert embed_metadata(png, {"params": {"bad": {1, 2}}}) == png


def test_extract_seed_prefers_the_sampler_over_other_seeded_nodes():
    """A post-processing graph seeds several nodes (an inpaint model, a
    noise source); the sampler's is the one that identifies the image."""
    graph = {
        "17": {"class_type": "INPAINT_InpaintWithModel", "inputs": {"seed": 111}},
        "24": {"class_type": "RandomNoise", "inputs": {"noise_seed": 222}},
        "26": {"class_type": "SamplerCustomAdvanced", "inputs": {"seed": 333}},
    }
    assert extract_seed(graph) == 333
    assert extract_seed({"9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}}) is None


def test_extract_seed_ignores_booleans_that_json_would_read_as_ints():
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": True, "noise_seed": 5}}}
    assert extract_seed(graph) == 5


def test_insert_text_chunk_rejects_a_non_png():
    with pytest.raises(ValueError):
        insert_text_chunk(b"\xff\xd8\xff\xe0", "k", "v")


def test_chunk_walk_survives_a_truncated_file():
    """These bytes come from whatever a user chose to upload, so a corrupt
    file has to read as "no metadata", not raise."""
    truncated = make_png()[: len(make_png()) // 2]
    assert is_png(truncated)
    assert extract_metadata(truncated) is None
    assert read_text_chunks(truncated) == {}


def test_written_chunk_has_a_valid_crc():
    """Otherwise strict decoders would reject the file outright."""
    png = embed_metadata(make_png(), build_metadata(params={}, kind="txt2img", filename="x.png"))
    offset = 8
    while offset + 8 <= len(png):
        (length,) = struct.unpack(">I", png[offset : offset + 4])
        chunk = png[offset + 4 : offset + 8 + length]
        stored = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        assert zlib.crc32(chunk) & 0xFFFFFFFF == stored
        if png[offset + 4 : offset + 8] == b"IEND":
            break
        offset += 12 + length
    # and Pillow still opens it
    Image.open(io.BytesIO(png)).load()
