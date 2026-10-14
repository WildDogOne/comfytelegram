import io
import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from comfytelegram.generation import generate, repeat
from comfytelegram.handlers import (
    RP_CANCEL_CALLBACK_DATA,
    _consume_awaiting_rp_prompt,
    generate_message,
    rp_cancel_callback,
    rp_command,
)
from comfytelegram.params_serde import (
    deserialize_generation_params,
    serialize_generation_params,
)
from comfytelegram.png_metadata import extract_metadata
from comfytelegram.profiles import resolve_generation_params
from comfytelegram.profiles.schema import LoraDefault, ModelProfile, ProfileDefaults
from comfytelegram.prompt_refs import ReferenceExpander
from comfytelegram.regional import (
    REGIONAL_HELP,
    REGIONAL_TEMPLATE,
    RegionalPromptError,
    expand_references,
    format_regional_prompt,
    is_regional_prompt,
    normalize_region_mask,
    painted_regions,
    parse_regional_prompt,
)
from comfytelegram.topics import NO_TOPIC
from comfytelegram.workflows import (
    GenerationParams,
    RegionSpec,
    build_txt2img,
    region_mask_filename,
)


def _boxes(regions: list[RegionSpec]) -> list[tuple]:
    return [(r.label, r.x, r.y, r.w, r.h) for r in regions]


# --- parsing ---------------------------------------------------------------


def test_parse_splits_global_regions_and_negative():
    parsed = parse_regional_prompt(
        "2girls, park\n[left] blonde hair\n[right] black hair\n---\nlowres, bad anatomy"
    )

    assert parsed.global_prompt == "2girls, park"
    assert [(r.label, r.prompt) for r in parsed.regions] == [
        ("left", "blonde hair"),
        ("right", "black hair"),
    ]
    assert parsed.negative == "lowres, bad anatomy"


def test_parse_lays_out_two_columns_as_halves():
    parsed = parse_regional_prompt("[right] b\n[left] a")

    assert _boxes(parsed.regions) == [
        ("left", 0.0, 0.0, 0.5, 1.0),
        ("right", 0.5, 0.0, 0.5, 1.0),
    ]


def test_parse_lays_out_three_columns_as_thirds_ending_at_the_edge():
    parsed = parse_regional_prompt("[left] a\n[center] b\n[right] c")

    assert _boxes(parsed.regions) == [
        ("left", 0.0, 0.0, 0.333, 1.0),
        ("center", 0.333, 0.0, 0.333, 1.0),
        ("right", 0.667, 0.0, 0.333, 1.0),
    ]


def test_parse_lays_out_rows_and_names_the_centre_middle():
    parsed = parse_regional_prompt("[top] a\n[centre] b\n[bottom] c")

    assert _boxes(parsed.regions) == [
        ("top", 0.0, 0.0, 1.0, 0.333),
        ("middle", 0.0, 0.333, 1.0, 0.333),
        ("bottom", 0.0, 0.667, 1.0, 0.333),
    ]


def test_parse_region_body_spans_lines_and_tags_are_case_insensitive():
    parsed = parse_regional_prompt("scene\n[LEFT] blonde hair\ncat ears,\n[Right] fox ears")

    assert parsed.regions[0].prompt == "blonde hair, cat ears"
    assert parsed.regions[1].prompt == "fox ears"


def test_parse_allows_an_empty_global_prompt():
    assert parse_regional_prompt("[left] a\n[right] b").global_prompt == ""


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("just a prompt", "No regions found"),
        ("[up] a", "Unknown region [up]"),
        ("[left] a\n[left] b", "appears twice"),
        ("[left]\n[right] b", "[left] has no prompt"),
        ("[left] a\n[top] b", "Mixing left/right with top/bottom"),
    ],
)
def test_parse_rejects_with_a_fixable_message(text, fragment):
    with pytest.raises(RegionalPromptError, match=fragment.replace("[", r"\[")):
        parse_regional_prompt(text)


def test_template_parses_and_fits_a_copy_button():
    parsed = parse_regional_prompt(REGIONAL_TEMPLATE)

    assert [r.label for r in parsed.regions] == ["left", "right"]
    assert len(REGIONAL_TEMPLATE) <= 256


def test_format_round_trips_through_parse():
    parsed = parse_regional_prompt("scene\n[top] a, b\n[bottom] c")

    text = format_regional_prompt(parsed.global_prompt, parsed.regions)

    assert text == "scene\n[top] a, b\n[bottom] c"
    assert parse_regional_prompt(text).regions == parsed.regions


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("scene\n[left] a\n[right] b", True),
        ("  [ Top ] a", True),
        ("[centre] a", True),
        ("1girl, [left] hand raised", False),
        ("[artist name] style, 1girl", False),
        ("just a prompt", False),
    ],
)
def test_is_regional_prompt_needs_a_line_starting_with_a_known_tag(text, expected):
    assert is_regional_prompt(text) is expected


# --- $name characters ---------------------------------------------------------

_CHARACTERS = [
    {"name": "alice", "positive_prompt": "blonde hair,\ncat ears", "negative_prompt": "tail"},
    {"name": "Bob", "positive_prompt": "fox ears", "negative_prompt": ""},
]


_FAVORITES = [{"name": "wlop", "category": "artist", "text": "by wlop", "note": ""}]


def test_expand_references_substitutes_in_regions_global_and_negative():
    parsed = parse_regional_prompt(
        "2girls, $Bob nearby, $wlop\n[left] 2girls, $alice\n[right] $Bob\n---\nlowres, $wlop"
    )
    references = ReferenceExpander(_CHARACTERS, _FAVORITES)

    expanded = expand_references(parsed, references)

    assert expanded.global_prompt == "2girls, fox ears nearby, by wlop"
    assert [r.prompt for r in expanded.regions] == ["2girls, blonde hair, cat ears", "fox ears"]
    assert expanded.negative == "lowres, by wlop"
    assert references.negatives == "tail"
    # The parsed prompt itself keeps its $names for "as typed".
    assert parsed.regions[0].prompt == "2girls, $alice"


def test_expand_references_matches_names_case_insensitively_and_dedupes_negatives():
    parsed = parse_regional_prompt("[left] $ALICE\n[right] $alice")
    references = ReferenceExpander(_CHARACTERS, [])

    expanded = expand_references(parsed, references)

    assert expanded.regions[0].prompt == expanded.regions[1].prompt == "blonde hair, cat ears"
    assert references.negatives == "tail"


def test_expand_references_rejects_an_unknown_name_listing_saved_ones():
    parsed = parse_regional_prompt("[left] $carol\n[right] b")

    with pytest.raises(RegionalPromptError, match=r"Nothing saved under \$carol\.") as exc:
        expand_references(parsed, ReferenceExpander(_CHARACTERS, _FAVORITES))
    assert "Characters: $alice, $Bob" in str(exc.value)
    assert "Favorites: $wlop" in str(exc.value)
    with pytest.raises(RegionalPromptError, match="No characters or favorites are saved yet"):
        expand_references(parsed, ReferenceExpander([], []))


# --- graph -----------------------------------------------------------------


def test_build_txt2img_without_regions_has_no_couple_node():
    prompt, _ = build_txt2img(
        GenerationParams(checkpoint="c", positive_prompt="p", negative_prompt="")
    )

    assert not any(n["class_type"] == "AttentionCouplePPM" for n in prompt.values())


def test_build_txt2img_with_regions_patches_the_sampler_model():
    params = GenerationParams(
        checkpoint="c",
        positive_prompt="global",
        negative_prompt="neg",
        regions=parse_regional_prompt("[left] a\n[right] b").regions,
        regional_base_weight=0.2,
        regional_region_weight=0.8,
    )

    prompt, _ = build_txt2img(params)

    couple_id, couple = next(
        (k, n) for k, n in prompt.items() if n["class_type"] == "AttentionCouplePPM"
    )
    sampler = next(n for n in prompt.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["model"] == [couple_id, 0]
    # Sampler's own positive stays the global prompt — the node's base_cond.
    assert sampler["inputs"]["positive"] == couple["inputs"]["base_cond"]

    def mask(ref):
        return prompt[ref[0]]["inputs"]

    base = mask(couple["inputs"]["base_mask"])
    assert (base["x"], base["y"], base["w"], base["h"], base["value"]) == (0.0, 0.0, 1.0, 1.0, 0.2)
    right = mask(couple["inputs"]["mask_2"])
    assert (right["x"], right["w"], right["value"]) == (0.5, 0.5, 0.8)
    assert prompt[couple["inputs"]["cond_1"][0]]["inputs"]["text"] == "a"
    assert prompt[couple["inputs"]["cond_2"][0]]["inputs"]["text"] == "b"
    assert "cond_3" not in couple["inputs"]


def _painting(width=64, height=64, *, painted_cols=slice(0, 24)) -> bytes:
    """A soft-edged grayscale painting the size of the generation: white
    over `painted_cols`, a grey brush fringe beside it."""
    pixels = np.zeros((height, width), dtype=np.uint8)
    pixels[:, painted_cols] = 255
    pixels[:, painted_cols.stop] = 200  # fringe that rounds up
    pixels[:, painted_cols.stop + 1] = 60  # fringe that rounds down
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, "PNG")
    return out.getvalue()


def _pixels(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"))


def test_build_txt2img_with_painted_regions_weights_the_mask_and_its_inverse():
    regions = painted_regions(_painting(), "a", "b", 1020, 1024)
    params = GenerationParams(
        checkpoint="c",
        positive_prompt="global",
        negative_prompt="",
        width=1020,
        height=1024,
        regions=regions,
        regional_region_weight=0.8,
    )

    prompt, _ = build_txt2img(params)

    couple = next(n for n in prompt.values() if n["class_type"] == "AttentionCouplePPM")

    def weighted(ref):
        composite = prompt[ref[0]]
        assert composite["class_type"] == "MaskComposite"
        assert composite["inputs"]["operation"] == "multiply"
        solid = prompt[composite["inputs"]["destination"][0]]["inputs"]
        # Sized like the latent (width floored to a multiple of 8), so it
        # stacks with the LatentToMaskBB base mask.
        assert (solid["value"], solid["width"], solid["height"]) == (0.8, 1016, 1024)
        return prompt[composite["inputs"]["source"][0]]

    filename = region_mask_filename(regions[0].mask_png)
    painted = weighted(couple["inputs"]["mask_1"])
    assert painted["class_type"] == "LoadImageMask"
    assert painted["inputs"] == {"image": filename, "channel": "red"}
    inverted = weighted(couple["inputs"]["mask_2"])
    assert inverted["class_type"] == "InvertMask"
    assert prompt[inverted["inputs"]["mask"][0]]["inputs"]["image"] == filename
    # The global mask stays a full-canvas box.
    assert prompt[couple["inputs"]["base_mask"][0]]["class_type"] == "LatentToMaskBB"


def test_batch_offset_skips_to_that_noise_draw():
    plain, _ = build_txt2img(GenerationParams("c", "p", "", batch_size=4))
    assert not any(n["class_type"] == "LatentFromBatch" for n in plain.values())

    prompt, _ = build_txt2img(GenerationParams("c", "p", "", batch_size=1, batch_offset=2))

    sampler = next(n for n in prompt.values() if n["class_type"] == "KSampler")
    picked = prompt[sampler["inputs"]["latent_image"][0]]
    assert picked["class_type"] == "LatentFromBatch"
    assert (picked["inputs"]["batch_index"], picked["inputs"]["length"]) == (2, 1)
    empty = prompt[picked["inputs"]["samples"][0]]
    assert empty["class_type"] == "EmptyLatentImage"
    assert empty["inputs"]["batch_size"] == 3


@pytest.mark.asyncio
async def test_batch_images_record_their_noise_index_and_repeat_starts_over():
    client = MagicMock(upload_image=AsyncMock())
    raw = [(b"a", "a.png"), (b"b", "b.png")]
    with patch("comfytelegram.generation._run_graph", new=AsyncMock(return_value=(raw, {}))):
        fresh = await generate(client, "c", "scene", _profile(), overrides={"batch_offset": 3})
        again = await repeat(client, fresh[0].full_params)

    assert [image.noise_index for image in fresh] == [3, 4]
    assert [image.noise_index for image in again] == [0, 1]
    assert again[0].full_params.batch_offset == 0


@pytest.mark.asyncio
async def test_each_batch_image_carries_its_noise_index_in_its_png():
    def png() -> bytes:
        out = io.BytesIO()
        Image.new("RGB", (4, 4)).save(out, "PNG")
        return out.getvalue()

    raw = [(png(), "a.png"), (png(), "b.png")]
    with patch("comfytelegram.generation._run_graph", new=AsyncMock(return_value=(raw, {}))):
        images = await generate(
            MagicMock(), "c", "scene", _profile(), overrides={"batch_offset": 1}
        )

    assert [extract_metadata(image.data)["noise_index"] for image in images] == [1, 2]


def test_normalize_region_mask_resizes_to_the_latent_size_and_thresholds():
    mask = normalize_region_mask(_painting(), 85, 40)

    pixels = _pixels(mask)
    assert pixels.shape == (40, 80)
    assert set(np.unique(pixels)) == {0, 255}
    assert Image.open(io.BytesIO(mask)).mode == "1"


def test_painted_regions_are_the_painting_and_its_inverse_with_boxes_and_labels():
    painted, rest = painted_regions(_painting(), "a", "b", 64, 64)

    assert (painted.label, painted.prompt, painted.invert_mask) == ("left", "a", False)
    assert (rest.label, rest.prompt, rest.invert_mask) == ("right", "b", True)
    assert painted.mask_png == rest.mask_png
    # The fringe pixel at column 24 (200) counts as painted, 25 (60) doesn't.
    assert (painted.x, painted.w) == (0.0, 25 / 64)
    assert (rest.x, rest.w) == (25 / 64, 39 / 64)
    assert (painted.y, painted.h) == (0.0, 1.0)


def test_painted_regions_label_the_painted_side_by_where_it_is():
    painted, rest = painted_regions(_painting(painted_cols=slice(40, 62)), "a", "b", 64, 64)
    assert (painted.label, rest.label) == ("right", "left")

    pixels = np.zeros((64, 64), dtype=np.uint8)
    pixels[40:, :] = 255
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, "PNG")
    painted, rest = painted_regions(out.getvalue(), "a", "b", 64, 64)
    assert (painted.label, rest.label) == ("bottom", "top")


@pytest.mark.parametrize("value", [0, 255])
def test_painted_regions_reject_an_empty_or_full_painting(value):
    out = io.BytesIO()
    Image.new("L", (64, 64), value).save(out, "PNG")

    with pytest.raises(RegionalPromptError):
        painted_regions(out.getvalue(), "a", "b", 64, 64)


def test_painted_region_survives_serialization():
    regions = painted_regions(_painting(), "a", "b", 64, 64)
    params = GenerationParams(
        checkpoint="c", positive_prompt="", negative_prompt="", regions=regions
    )

    data = serialize_generation_params(params)
    json.dumps(data)  # what storage and the PNG metadata both do with it
    restored = deserialize_generation_params(data)

    assert restored.regions == regions


@pytest.mark.asyncio
async def test_generate_and_repeat_upload_the_painted_mask_once_before_running():
    regions = painted_regions(_painting(), "a", "b", 64, 64)
    client = MagicMock()
    client.upload_image = AsyncMock()
    order = []
    client.upload_image.side_effect = lambda *a, **kw: order.append("upload")

    async def run_graph(*args, **kwargs):
        order.append("run")
        return [], {}

    with patch("comfytelegram.generation._run_graph", new=run_graph):
        await generate(client, "c", "scene", _profile(), regions=regions)
        await repeat(client, replace(GenerationParams("c", "p", ""), regions=regions))

    assert order == ["upload", "run", "upload", "run"]
    client.upload_image.assert_awaited_with(
        regions[0].mask_png, filename=region_mask_filename(regions[0].mask_png), overwrite=True
    )


# --- resolution / serialization ---------------------------------------------


def _profile(**defaults) -> ModelProfile:
    return ModelProfile(
        match=["*"],
        display_name="p",
        positive_prompt_prefix="masterpiece",
        loras=[
            LoraDefault(name="on.safetensors", default_enabled=True, trigger_words="trig"),
            LoraDefault(name="off.safetensors", default_enabled=False, trigger_words="nope"),
        ],
        defaults=ProfileDefaults(**defaults),
    )


def test_resolve_prepends_the_whole_global_prompt_to_each_region():
    regions = parse_regional_prompt("[left] a\n[right] b").regions

    params = resolve_generation_params("c", "scene", _profile(), regions=regions)

    assert params.positive_prompt == "masterpiece, trig, scene"
    assert [r.prompt for r in params.regions] == [
        "masterpiece, trig, scene, a",
        "masterpiece, trig, scene, b",
    ]
    # The caller's own RegionSpecs aren't mutated.
    assert regions[0].prompt == "a"


def test_resolve_takes_regional_weights_from_the_profile():
    params = resolve_generation_params(
        "c", "scene", _profile(regional_base_weight=0.2, regional_region_weight=0.8)
    )

    assert (params.regional_base_weight, params.regional_region_weight) == (0.2, 0.8)
    assert params.regions == []


def test_resolve_keeps_generic_weights_without_a_profile_setting():
    params = resolve_generation_params("c", "scene", _profile())

    assert (params.regional_base_weight, params.regional_region_weight) == (0.1, 0.9)


def test_serialization_round_trips_regions():
    params = GenerationParams(
        checkpoint="c",
        positive_prompt="p",
        negative_prompt="n",
        regions=parse_regional_prompt("[top] a\n[bottom] b").regions,
        regional_base_weight=0.3,
        regional_region_weight=0.7,
    )

    restored = deserialize_generation_params(serialize_generation_params(params))

    assert restored.regions == params.regions
    assert (restored.regional_base_weight, restored.regional_region_weight) == (0.3, 0.7)


def test_deserialization_of_old_rows_has_no_regions():
    restored = deserialize_generation_params(
        {"checkpoint": "c", "positive_prompt": "p", "negative_prompt": "n"}
    )

    assert restored.regions == []


# --- handlers ----------------------------------------------------------------


def _update_mock() -> tuple[MagicMock, AsyncMock]:
    message = AsyncMock()
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    update.effective_user.id = 1
    return update, message


def _context(chat_data: dict | None = None) -> MagicMock:
    context = MagicMock()
    storage = MagicMock()
    storage.list_characters.return_value = []
    storage.list_favorites.return_value = []
    context.bot_data = {"settings": MagicMock(allowed_user_ids=None), "storage": storage}
    context.chat_data = {} if chat_data is None else chat_data
    return context


@pytest.mark.asyncio
async def test_rp_command_explains_the_format_with_a_copy_template_button():
    update, message = _update_mock()
    message.text = "/rp"
    context = _context()

    await rp_command(update, context)

    assert message.reply_text.await_args.args[0] == REGIONAL_HELP
    assert context.chat_data["awaiting_rp_prompt"][NO_TOPIC] is True
    ((copy_button, cancel_button),) = message.reply_text.await_args.kwargs[
        "reply_markup"
    ].inline_keyboard
    assert copy_button.copy_text.text == REGIONAL_TEMPLATE
    assert cancel_button.callback_data == RP_CANCEL_CALLBACK_DATA


@pytest.mark.asyncio
async def test_rp_command_with_inline_prompt_generates_right_away():
    update, message = _update_mock()
    message.text = "/rp scene\n[left] a\n[right] b"
    context = _context()

    with patch("comfytelegram.handlers._generate_regional", new=AsyncMock()) as generate:
        await rp_command(update, context)

    generate.assert_awaited_once_with(context, 1, message, "scene\n[left] a\n[right] b")
    assert "awaiting_rp_prompt" not in context.chat_data


@pytest.mark.asyncio
async def test_consume_awaiting_rp_prompt_ignores_text_when_not_waiting():
    update, _message = _update_mock()

    assert await _consume_awaiting_rp_prompt(update, _context()) is False


@pytest.mark.asyncio
async def test_consume_awaiting_rp_prompt_re_arms_on_a_parse_error():
    update, message = _update_mock()
    message.text = "no tags here"
    context = _context({"awaiting_rp_prompt": {NO_TOPIC: True}})

    assert await _consume_awaiting_rp_prompt(update, context) is True

    assert context.chat_data["awaiting_rp_prompt"][NO_TOPIC] is True
    reply = message.reply_text.await_args.args[0]
    assert reply.startswith("⚠️ No regions found")


@pytest.mark.asyncio
async def test_consume_awaiting_rp_prompt_generates_with_regions():
    update, message = _update_mock()
    message.text = "scene\n[left] $alice\n[right] b\n---\nlowres"
    context = _context({"awaiting_rp_prompt": {NO_TOPIC: True}})
    storage = MagicMock()
    storage.list_characters.return_value = _CHARACTERS
    storage.get_checkpoint.return_value = "ckpt.safetensors"
    storage.get_override.return_value = None
    storage.get_lora_overrides.return_value = {}
    storage.get_lora_strength_overrides.return_value = {}
    storage.get_active_character_name.return_value = "Bob"
    context.bot_data.update({"storage": storage, "comfy_client": AsyncMock(), "profiles": []})

    with (
        patch("comfytelegram.handlers.generate", new=AsyncMock(return_value=["img"])) as generate,
        patch("comfytelegram.handlers._deliver_generation_result", new=AsyncMock()) as deliver,
    ):
        assert await _consume_awaiting_rp_prompt(update, context) is True

    args, kwargs = generate.call_args
    assert args[1:3] == ("ckpt.safetensors", "scene")
    # Bob is active but /rp ignores him; alice's negative joins the block.
    assert args[2] == "scene"
    assert kwargs["extra_negative_prompt"] == "lowres, tail"
    assert [r.prompt for r in kwargs["regions"]] == ["blonde hair, cat ears", "b"]
    assert kwargs["raw_positive_prompt"] == "scene\n[left] $alice\n[right] b"
    storage.get_active_character_name.assert_not_called()
    deliver.assert_awaited_once()


@pytest.mark.asyncio
async def test_rp_cancel_callback_clears_the_flag():
    query = AsyncMock()
    query.message.message_thread_id = None
    update = MagicMock()
    update.callback_query = query
    update.effective_chat.id = 1
    update.effective_user.id = 1
    context = _context({"awaiting_rp_prompt": {NO_TOPIC: True}})

    await rp_cancel_callback(update, context)

    assert "awaiting_rp_prompt" not in context.chat_data
    assert query.edit_message_text.await_args.args[0] == "Cancelled — no regional prompt started."


@pytest.mark.asyncio
async def test_consume_awaiting_rp_prompt_re_arms_on_an_unknown_character():
    update, message = _update_mock()
    message.text = "[left] $nobody\n[right] b"
    context = _context({"awaiting_rp_prompt": {NO_TOPIC: True}})

    assert await _consume_awaiting_rp_prompt(update, context) is True

    assert context.chat_data["awaiting_rp_prompt"][NO_TOPIC] is True
    assert message.reply_text.await_args.args[0].startswith("⚠️ Nothing saved under $nobody")


@pytest.mark.asyncio
async def test_generate_message_routes_regional_syntax_without_re_arming():
    update, message = _update_mock()
    message.text = "scene\n[left] a\n[right] b"
    context = _context()

    with patch("comfytelegram.handlers._generate_regional", new=AsyncMock()) as regional:
        await generate_message(update, context)

    regional.assert_awaited_once_with(
        context, 1, message, "scene\n[left] a\n[right] b", rearm_on_error=False
    )


@pytest.mark.asyncio
async def test_auto_detected_regional_error_does_not_arm_the_rp_follow_up():
    update, message = _update_mock()
    message.text = "[left] a\n[top] b"
    context = _context()

    await generate_message(update, context)

    assert "awaiting_rp_prompt" not in context.chat_data
    assert message.reply_text.await_args.args[0].startswith("⚠️ Mixing left/right")
    (row,) = message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard
    assert [b.text for b in row] == ["📋 Copy template"]
