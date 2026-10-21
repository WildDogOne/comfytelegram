"""`<ar:…>` aspect-ratio tags in prompts (`aspect_refs`): parsing, sizing,
typed prompts (`_resolve_profile_and_prompt`, `generate_message`, `/rp`)
winning over a `/pose` shape, and being stripped where the size is fixed."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from comfytelegram.aspect_refs import (
    AspectRatio,
    InvalidAspectRatioError,
    extract_aspect_tag,
    size_for_aspect,
    strip_aspect_tags,
)
from comfytelegram.handlers import (
    _expand_detail_prompt_references,
    _resolve_profile_and_prompt,
    generate_message,
)
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.prompt_refs import UnknownReferenceError
from comfytelegram.storage import Storage
from comfytelegram.workflows import GenerationParams


def _profile(**kwargs) -> ModelProfile:
    return ModelProfile(
        match=["*"],
        display_name="p",
        defaults=ProfileDefaults(width=1024, height=1024),
        pose_controlnet="openpose.safetensors",
        **kwargs,
    )


def _skeleton(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "black").save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


# --- parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ratio"),
    [
        ("<ar:2:3>", AspectRatio(2, 3)),
        ("<AR: 16x9 >", AspectRatio(16, 9)),
        ("<aspectratio:3/2>", AspectRatio(3, 2)),
        ("<aspect_ratio:1.5>", AspectRatio(1.5, 1)),
        ("<aspect:1:1>", AspectRatio(1, 1)),
    ],
)
def test_spellings(text, ratio):
    cleaned, parsed = extract_aspect_tag(f"1girl, {text}, forest")

    assert parsed == ratio
    assert cleaned == "1girl, forest"


def test_last_tag_wins_and_text_without_one_is_untouched():
    assert extract_aspect_tag("<ar:2:3>, a, <ar:16:9>") == ("a", AspectRatio(16, 9))
    assert extract_aspect_tag("a <lora:x>, b") == ("a <lora:x>, b", None)


@pytest.mark.parametrize("text", ["<ar:0:3>", "<ar:5:1>", "<ar:1:4.5>"])
def test_unusable_ratios_are_rejected(text):
    with pytest.raises(InvalidAspectRatioError) as excinfo:
        extract_aspect_tag(text)
    assert isinstance(excinfo.value, UnknownReferenceError)


def test_strip_drops_tags_even_invalid_ones():
    assert strip_aspect_tags("a, <ar:0:3>, b") == "a, b"
    assert strip_aspect_tags(None) is None


def test_size_keeps_the_pixel_count_and_snaps_to_64():
    assert size_for_aspect(1, 1, 1024, 1024) == (1024, 1024)
    assert size_for_aspect(2, 3, 1024, 1024) == (832, 1280)
    assert size_for_aspect(16, 9, 1024, 1024) == (1344, 768)
    # Another model's own resolution is kept, not SDXL's.
    assert size_for_aspect(1, 1, 768, 768) == (768, 768)


# --- typed prompts --------------------------------------------------------------


def test_resolve_profile_and_prompt_turns_the_tag_into_a_size(storage):
    *_, size = _resolve_profile_and_prompt(1, "ckpt", "1girl", storage, [_profile()])
    assert size == {}

    _, effective, _, raw_positive, _, size = _resolve_profile_and_prompt(
        1, "ckpt", "1girl, <ar:2:3>", storage, [_profile()]
    )
    assert effective == "1girl"
    assert raw_positive == "1girl, <ar:2:3>"  # Show Prompt's copy keeps it
    assert size == {"width": 832, "height": 1280}


def test_a_favorite_can_carry_the_tag(storage):
    from comfytelegram.handlers import _reference_expander

    storage.save_favorite(7, "portrait", "misc", "<ar:2:3>, upper body")
    *_, size = _resolve_profile_and_prompt(
        1,
        "ckpt",
        "1girl, $portrait",
        storage,
        [_profile()],
        references=_reference_expander(storage, 1, 7),
    )
    assert size == {"width": 832, "height": 1280}


def _message_update(text: str, storage: Storage):
    message = AsyncMock()
    message.text = text
    message.message_thread_id = None
    message.from_user.id = 7
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    update.effective_user.id = 7
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None, progress_drafts=False),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [_profile()],
    }
    return message, update, context


async def _run_generate_message(text: str, storage: Storage):
    message, update, context = _message_update(text, storage)
    with (
        patch(
            "comfytelegram.handlers._resolve_checkpoint_or_default",
            new=AsyncMock(return_value="ckpt.safetensors"),
        ),
        # Called only to build `_run_reporting_errors`' argument, which
        # then reports nothing came back — no image delivery to mock.
        patch("comfytelegram.handlers.generate", new=MagicMock()) as generate,
        patch("comfytelegram.handlers._run_reporting_errors", new=AsyncMock(return_value=None)),
        patch("comfytelegram.handlers.start_reply_status", new=AsyncMock()),
    ):
        await generate_message(update, context)
    return message, generate


@pytest.mark.asyncio
async def test_generate_message_pins_the_size_over_the_pose(storage):
    storage.set_pose(1, _skeleton(64, 64))

    message, generate = await _run_generate_message("1girl, <ar:16:9>", storage)

    kwargs = generate.call_args.kwargs
    assert kwargs["overrides"] == {"width": 1344, "height": 768}
    assert kwargs["pose_png"] is not None
    # A square pose under a 16:9 tag gets a heads-up.
    assert "aspect tag wins" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_a_matching_pose_shape_needs_no_warning(storage):
    storage.set_pose(1, _skeleton(640, 360))

    message, generate = await _run_generate_message("1girl, <ar:16:9>", storage)

    assert generate.call_args.kwargs["overrides"] == {"width": 1344, "height": 768}
    message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_message_without_a_tag_leaves_the_size_alone(storage):
    _, generate = await _run_generate_message("1girl", storage)

    assert generate.call_args.kwargs["overrides"] is None


@pytest.mark.asyncio
async def test_a_bad_tag_is_reported_without_generating(storage):
    message, generate = await _run_generate_message("1girl, <ar:9:1>", storage)

    generate.assert_not_called()
    assert "too stretched" in message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_a_regional_prompt_takes_the_tag_from_a_region_line(storage):
    _, generate = await _run_generate_message(
        "2girls\n[left] red hair <ar:3:2>\n[right] blue hair", storage
    )

    kwargs = generate.call_args.kwargs
    assert kwargs["overrides"] == {"width": 1280, "height": 832}
    assert kwargs["regions"][0].prompt == "red hair"


# --- fixed size -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_detail_prompt_drops_the_tag(storage):
    params = GenerationParams("ckpt", "1girl", "lowres")

    positive, negative = await _expand_detail_prompt_references(
        AsyncMock(), storage, 1, None, 7, params, "1girl, <ar:2:3>", None, [_profile()]
    )

    assert (positive, negative) == ("1girl", None)
