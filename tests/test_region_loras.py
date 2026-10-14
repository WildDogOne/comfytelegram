"""Per-region LoRAs in `/rp`: parsing (`lora_refs.resolve_region_loras`),
carrying them on `RegionSpec`, the mask a region's pass repaints
(`generation.region_lora_mask`), the pass itself (`apply_region_lora(s)`),
and the bot side — "🎭 Region LoRAs" button, region chooser,
`region_lora_callback`, the `/settings` toggle."""

import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from test_detail_prompt import _callback_update, _context
from test_relay_upload import _noisy_png, _session_patch

from comfytelegram.generation import (
    REGION_LORA_DEFAULT_DENOISE,
    GeneratedImage,
    _apply_detail_loras,
    apply_region_loras,
    generate,
    has_region_loras,
    mask_to_png,
    region_lora_mask,
)
from comfytelegram.handlers import (
    REGION_LORA_CALLBACK_PREFIX,
    REGION_LORAS_CALLBACK_KIND,
    _carried_extra_rows,
    _consume_awaiting_rp_prompt,
    _region_lora_keyboard,
    _relay_create_job,
    _send_and_store_result,
    _serialize_generation_params,
    postprocess_callback,
    region_lora_callback,
)
from comfytelegram.lora_refs import UnknownLoraError, lora_tag_for_spec, resolve_region_loras
from comfytelegram.params_serde import (
    deserialize_generation_params,
    serialize_generation_params,
)
from comfytelegram.profiles import LoraDefault, ModelProfile, apply_checkpoint_switch
from comfytelegram.regional import painted_regions
from comfytelegram.storage import Storage
from comfytelegram.topics import NO_TOPIC
from comfytelegram.workflows import (
    GenerationParams,
    LoraSpec,
    PostProcessBaseParams,
    RegionSpec,
)

_MARLOW = LoraDefault(
    name="illustrious/marlowV3/marlowV3.safetensors",
    default_enabled=False,
    trigger_words="marlow",
    strength_model=0.9,
    strength_clip=0.9,
)
_INK = LoraDefault(name="ink.safetensors", default_enabled=True)
_PROFILE = ModelProfile(match=["*"], display_name="p", loras=[_MARLOW, _INK])
_MARLOW_SPEC = LoraSpec("illustrious/marlowV3/marlowV3.safetensors", 0.9, 0.9)


def _regions(*, left_loras=(_MARLOW_SPEC,)) -> list[RegionSpec]:
    return [
        RegionSpec("left", "scene, marlow, red hoodie", 0.0, 0.0, 0.5, 1.0, list(left_loras)),
        RegionSpec("right", "scene, bunny", 0.5, 0.0, 0.5, 1.0),
    ]


def _params(**kw) -> GenerationParams:
    fields = {
        "checkpoint": "ckpt.safetensors",
        "positive_prompt": "scene",
        "negative_prompt": "lowres",
        "loras": [LoraSpec("ink.safetensors")],
        "regions": _regions(),
    }
    fields.update(kw)
    return GenerationParams(**fields)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


# --- parsing / params ------------------------------------------------------------


def test_region_loras_come_out_of_the_prompt_with_trigger_words_in_front():
    prompt, loras = resolve_region_loras("<lora:marlowV3:0.7>, red hoodie", _PROFILE)

    assert prompt == "marlow, red hoodie"
    assert loras == [LoraSpec("illustrious/marlowV3/marlowV3.safetensors", 0.7, 0.7)]


def test_region_lora_at_zero_is_dropped_and_unknown_raises():
    assert resolve_region_loras("<lora:marlowV3:0>, x", _PROFILE) == ("x", [])
    with pytest.raises(UnknownLoraError):
        resolve_region_loras("<lora:nobody>, x", _PROFILE)


def test_lora_tag_for_spec_uses_the_full_path():
    assert lora_tag_for_spec(_MARLOW_SPEC) == "<lora:illustrious/marlowV3/marlowV3:0.9>"
    assert lora_tag_for_spec(LoraSpec("a\\b.safetensors", 1.0, 0.5)) == "<lora:a/b:1:0.5>"


def test_region_loras_round_trip_and_old_rows_have_none():
    params = _params(regional_lora_denoise=0.7)

    restored = deserialize_generation_params(serialize_generation_params(params))

    assert restored.regions == params.regions
    assert restored.regional_lora_denoise == 0.7
    old = serialize_generation_params(params)
    for region in old["regions"]:
        del region["loras"]
    assert all(r.loras == [] for r in deserialize_generation_params(old).regions)


def test_checkpoint_switch_drops_region_loras():
    switched = apply_checkpoint_switch(_params(), "other.safetensors", None)

    assert [r.loras for r in switched.regions] == [[], []]
    assert switched.regions[0].prompt == "scene, marlow, red hoodie"


def test_has_region_loras():
    assert has_region_loras(_params())
    assert not has_region_loras(_params(regions=_regions(left_loras=())))


# --- the mask ---------------------------------------------------------------------


def test_region_mask_is_the_people_inside_the_box_grown_a_little():
    people = np.zeros((200, 200), dtype=bool)
    people[50:150, 40:160] = True  # one blob straddling both halves
    left = _regions()[0]

    mask, from_people = region_lora_mask(people, left)

    assert from_people
    assert mask[100, 60] and not mask[10, 10]  # person yes, sky no
    assert mask[100, 101]  # grown slightly past the box edge
    assert not mask[100, 140]  # but not into the other character
    assert mask[49, 60]  # and past the outline


def test_region_mask_falls_back_to_the_box_without_people_in_it():
    people = np.zeros((200, 200), dtype=bool)
    people[50:150, 150:190] = True  # only on the right

    mask, from_people = region_lora_mask(people, _regions()[0])

    assert not from_people
    assert mask[:, :100].all() and not mask[:, 100:].any()


def test_region_mask_follows_a_painted_side_instead_of_a_box():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[:, :30] = 255
    pixels[60:, 30:70] = 255  # the painted side reaches under the other character
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    _painted, rest = painted_regions(buffer.getvalue(), "a", "b", 100, 100)
    people = np.zeros((200, 200), dtype=bool)  # image upscaled 2x since
    people[20:200, 20:180] = True

    rest_mask, from_people = region_lora_mask(people, rest)

    assert from_people
    assert rest_mask[50, 150] and not rest_mask[50, 30]
    assert not rest_mask[180, 100]  # painted side, though inside rest's box


def test_region_mask_falls_back_to_the_painted_side_without_people_in_it():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[:, :30] = 255
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    painted, _rest = painted_regions(buffer.getvalue(), "a", "b", 100, 100)

    mask, from_people = region_lora_mask(np.zeros((100, 100), dtype=bool), painted)

    assert not from_people
    assert mask[:, :30].all() and not mask[:, 30:].any()


def test_mask_png_is_white_on_black_grayscale():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 2] = True

    image = Image.open(io.BytesIO(mask_to_png(mask)))

    assert (image.mode, image.size) == ("L", (6, 4))
    assert image.getpixel((2, 1)) == 255 and image.getpixel((0, 0)) == 0


# --- the passes -------------------------------------------------------------------


def test_extra_loras_add_to_the_image_loras_or_replace_them_when_disabled():
    base = PostProcessBaseParams(
        checkpoint="ckpt",
        positive_prompt="p",
        negative_prompt="n",
        loras=[LoraSpec("ink.safetensors")],
    )

    added = _apply_detail_loras(base, [_PROFILE], [_MARLOW_SPEC])
    replaced = _apply_detail_loras(
        replace(base, detailer_disable_lora=True), [_PROFILE], [_MARLOW_SPEC]
    )

    assert added.loras == [LoraSpec("ink.safetensors"), _MARLOW_SPEC]
    assert replaced.loras == [_MARLOW_SPEC]
    assert replaced.detailer_disable_lora is False


@pytest.mark.asyncio
async def test_apply_region_loras_runs_each_lora_region_in_turn_on_the_last_result():
    regions = _regions()
    regions[1] = replace(regions[1], loras=[LoraSpec("ink.safetensors", 0.5, 0.5)])
    params = _params(regions=regions, regional_lora_denoise=0.7)
    image = GeneratedImage(data=b"base", filename="base.png", full_params=params)
    people = np.ones((100, 100), dtype=bool)
    outputs = [
        GeneratedImage(data=b"after-left", filename="l.png", full_params=params),
        GeneratedImage(data=b"after-right", filename="r.png", full_params=params),
    ]
    stages = []

    async def on_stage(text):
        stages.append(text)

    with (
        patch("comfytelegram.generation.detect_people", new=AsyncMock(return_value=people)) as det,
        patch("comfytelegram.generation.post_process", new=AsyncMock(side_effect=outputs)) as pp,
    ):
        result = await apply_region_loras(MagicMock(), image, on_stage=on_stage)

    assert result is outputs[1]
    det.assert_awaited_once()
    first, second = pp.await_args_list
    assert first.args[1:4] == ("hand_drawn", b"base", "base.png")
    assert second.args[1:4] == ("hand_drawn", b"after-left", "l.png")
    assert first.kwargs["detail_prompt"] == "scene, marlow, red hoodie"
    assert first.kwargs["extra_loras"] == [_MARLOW_SPEC]
    assert first.kwargs["denoise"] == 0.7
    assert first.kwargs["is_detail_prompt"] is True
    left_mask = np.asarray(Image.open(io.BytesIO(first.kwargs["mask_bytes"])))
    assert left_mask[:, 10].all() and not left_mask[:, 90].any()
    assert stages == ["Region LoRAs 1/2 (left)", "Region LoRAs 2/2 (right)"]


@pytest.mark.asyncio
async def test_apply_region_loras_uses_the_default_denoise():
    image = GeneratedImage(data=b"base", filename="base.png", full_params=_params())
    with (
        patch(
            "comfytelegram.generation.detect_people",
            new=AsyncMock(return_value=np.ones((10, 10), dtype=bool)),
        ),
        patch("comfytelegram.generation.post_process", new=AsyncMock(return_value=image)) as pp,
    ):
        await apply_region_loras(MagicMock(), image)

    assert pp.await_args.kwargs["denoise"] == REGION_LORA_DEFAULT_DENOISE


@pytest.mark.asyncio
async def test_generate_applies_region_loras_only_when_asked():
    raw = [(b"img", "a.png")]
    with (
        patch("comfytelegram.generation._run_graph", new=AsyncMock(return_value=(raw, {}))),
        patch(
            "comfytelegram.generation.apply_region_loras",
            new=AsyncMock(side_effect=lambda client, image, **kw: replace(image, data=b"done")),
        ) as apply,
    ):
        plain = await generate(MagicMock(), "ckpt", "scene", _PROFILE, regions=_regions())
        auto = await generate(
            MagicMock(), "ckpt", "scene", _PROFILE, regions=_regions(), auto_region_loras=True
        )

    assert plain[0].data != b"done"
    assert auto[0].data == b"done"
    apply.assert_awaited_once()


# --- bot side ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rp_with_region_loras_generates_with_them(storage):
    storage.set_region_lora_auto(1, True)
    message = AsyncMock()
    message.message_thread_id = None
    message.text = "scene\n[left] <lora:marlowV3:0.7>, red hoodie\n[right] bunny"
    message.from_user.id = 1
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    context = MagicMock()
    context.chat_data = {"awaiting_rp_prompt": {NO_TOPIC: True}}
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [_PROFILE],
    }
    storage.set_checkpoint(1, "ckpt.safetensors")

    with (
        patch("comfytelegram.handlers.generate", new=AsyncMock(return_value=["img"])) as gen,
        patch("comfytelegram.handlers._deliver_generation_result", new=AsyncMock()),
    ):
        assert await _consume_awaiting_rp_prompt(update, context) is True

    kwargs = gen.await_args.kwargs
    left, right = kwargs["regions"]
    assert left.prompt == "marlow, red hoodie"
    assert left.loras == [LoraSpec("illustrious/marlowV3/marlowV3.safetensors", 0.7, 0.7)]
    assert right.loras == []
    assert kwargs["auto_region_loras"] is True
    # As typed keeps the tag.
    assert "<lora:marlowV3:0.7>" in kwargs["raw_positive_prompt"]


@pytest.mark.asyncio
async def test_results_with_region_loras_get_the_button_and_it_survives_a_page_flip(storage):
    message = AsyncMock()
    message.reply_photo.return_value = MagicMock(photo=[MagicMock(file_id="f")])
    image = GeneratedImage(data=_noisy_png(), filename="a.png", full_params=_params())

    await _send_and_store_result(message, 1, storage, image, result_id="abc")

    markup = message.reply_photo.await_args.kwargs["reply_markup"]
    region_rows = [
        row
        for row in markup.inline_keyboard
        if row[0].callback_data == f"pp:{REGION_LORAS_CALLBACK_KIND}:abc"
    ]
    assert len(region_rows) == 1
    assert _carried_extra_rows(markup) == [list(row) for row in region_rows]

    plain = GeneratedImage(
        data=_noisy_png(), filename="a.png", full_params=_params(regions=_regions(left_loras=()))
    )
    await _send_and_store_result(message, 1, storage, plain, result_id="def")
    markup = message.reply_photo.await_args.kwargs["reply_markup"]
    assert not any(
        "region_loras" in (b.callback_data or "") for r in markup.inline_keyboard for b in r
    )


def test_region_chooser_lists_lora_regions_then_all():
    keyboard = _region_lora_keyboard("abc", _params(), editor_available=True)

    assert [[b.text for b in row] for row in keyboard.inline_keyboard] == [
        ["🎭 left: marlowV3"],
        ["🤖 All, automatically"],
    ]
    assert keyboard.inline_keyboard[0][0].callback_data == f"{REGION_LORA_CALLBACK_PREFIX}abc:0"
    assert keyboard.inline_keyboard[1][0].callback_data == f"{REGION_LORA_CALLBACK_PREFIX}abc:all"
    no_editor = _region_lora_keyboard("abc", _params(), editor_available=False)
    assert [[b.text for b in row] for row in no_editor.inline_keyboard] == [
        ["🤖 All, automatically"]
    ]


@pytest.mark.asyncio
async def test_region_loras_button_replies_with_the_chooser(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, query = _callback_update(f"pp:{REGION_LORAS_CALLBACK_KIND}:abc")
    context = _context(storage)
    context.bot_data["settings"] = MagicMock(allowed_user_ids=None, inpaint_relay_url="https://r")

    await postprocess_callback(update, context)

    markup = query.message.reply_text.await_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].text == "🎭 left: marlowV3"


@pytest.mark.asyncio
async def test_picking_a_region_opens_the_editor_with_mask_prompt_and_denoise(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, _query = _callback_update(f"{REGION_LORA_CALLBACK_PREFIX}abc:0")
    context = _context(storage)
    context.bot_data["settings"] = MagicMock(allowed_user_ids=None, inpaint_relay_url="https://r")
    people = np.zeros((10, 10), dtype=bool)
    people[2:8, 1:9] = True

    with (
        patch("comfytelegram.handlers._open_mask_editor", new=AsyncMock()) as open_editor,
        patch("comfytelegram.handlers.detect_people", new=AsyncMock(return_value=people)),
    ):
        await region_lora_callback(update, context)
        kwargs = open_editor.await_args.kwargs
        mask_png = await kwargs["initial_mask_for"](b"src")

    assert kwargs["job_kind"] == "detail"
    assert kwargs["meta_overrides"] == {
        "positive": "<lora:illustrious/marlowV3/marlowV3:0.9> scene, marlow, red hoodie",
        "denoise_default": REGION_LORA_DEFAULT_DENOISE,
    }
    mask = np.asarray(Image.open(io.BytesIO(mask_png)))
    assert mask[5, 2] == 255 and mask[5, 8] == 0


@pytest.mark.asyncio
async def test_open_editor_uploads_the_initial_mask_and_denoise_default(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, query = _callback_update(f"{REGION_LORA_CALLBACK_PREFIX}abc:0")
    context = _context(storage)
    context.bot_data["settings"] = MagicMock(
        allowed_user_ids=None, inpaint_relay_url="https://r", sam_encoder_path=Path("/nope")
    )
    query.message.reply_text.return_value = AsyncMock(message_id=5)

    with (
        patch(
            "comfytelegram.handlers.detect_people",
            new=AsyncMock(return_value=np.ones((10, 10), dtype=bool)),
        ),
        patch(
            "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok")
        ) as create,
    ):
        await region_lora_callback(update, context)

    kwargs = create.await_args.kwargs
    assert kwargs["meta"]["mode"] == "mask_prompt"
    assert kwargs["meta"]["positive"].startswith("<lora:illustrious/marlowV3/marlowV3:0.9>")
    assert kwargs["meta"]["denoise_default"] == REGION_LORA_DEFAULT_DENOISE
    assert Image.open(io.BytesIO(kwargs["initial_mask"])).size == (10, 10)
    assert storage.list_inpaint_jobs()[0]["kind"] == "detail"


@pytest.mark.asyncio
async def test_all_automatically_runs_every_region_and_sends_the_result(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, query = _callback_update(f"{REGION_LORA_CALLBACK_PREFIX}abc:all")
    context = _context(storage)
    query.message.reply_text.return_value = AsyncMock()
    done = GeneratedImage(data=b"done", filename="d.png", full_params=_params())

    with (
        patch(
            "comfytelegram.handlers.apply_region_loras", new=AsyncMock(return_value=done)
        ) as apply,
        patch("comfytelegram.handlers._send_and_store_result", new=AsyncMock()) as send,
    ):
        await region_lora_callback(update, context)

    image = apply.await_args.args[1]
    assert (image.data, image.filename) == (b"orig", "a.png")
    assert send.await_args.args[3] is done


@pytest.mark.asyncio
async def test_a_region_without_loras_is_refused(storage):
    storage.store_pending_result("abc", 1, "file", "a.png", _serialize_generation_params(_params()))
    update, query = _callback_update(f"{REGION_LORA_CALLBACK_PREFIX}abc:1")

    await region_lora_callback(update, _context(storage))

    assert query.answer.await_args.kwargs["show_alert"] is True


@pytest.mark.asyncio
async def test_relay_upload_sends_the_initial_mask_part():
    settings = MagicMock(inpaint_relay_url="https://relay.example", inpaint_relay_shared_secret="s")
    captured: dict = {}

    with patch("comfytelegram.handlers.aiohttp.ClientSession", _session_patch(captured)):
        await _relay_create_job(
            settings, _noisy_png(), meta={"mode": "mask_prompt"}, initial_mask=b"M"
        )

    assert captured["parts"]["initial_mask"] == (b"M", "image/png", "initial_mask.png")
    assert json.loads(captured["parts"]["meta"][0]) == {"mode": "mask_prompt"}


def test_region_lora_auto_setting_defaults_off(storage):
    assert storage.get_region_lora_auto(1) is False
    storage.set_region_lora_auto(1, True)
    assert storage.get_region_lora_auto(1) is True
    assert storage.get_region_lora_auto(2) is False
