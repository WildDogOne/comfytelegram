"""`$name` references to saved characters/favorites (`prompt_refs`), and the
places that expand them: plain prompts (`_resolve_effective_prompt`,
`generate_message`) and "✏️ Detail Prompt" submissions
(`_process_one_inpaint_job`). Regional prompts are covered in
`test_regional.py`."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_detail_prompt import _callback_update, _context, _params

from comfytelegram.comfy_client import ComfyUIError
from comfytelegram.generation import GeneratedImage
from comfytelegram.handlers import (
    DETAIL_PROMPT_CALLBACK_KIND,
    _process_one_inpaint_job,
    _resolve_effective_prompt,
    _serialize_generation_params,
    generate_message,
    postprocess_callback,
)
from comfytelegram.prompt_refs import (
    ReferenceExpander,
    UnknownReferenceError,
    reference_suggestions,
)
from comfytelegram.storage import Storage

_CHARACTERS = [
    {"name": "alice", "positive_prompt": "blonde hair, cat ears", "negative_prompt": "tail"},
    {"name": "dyn", "positive_prompt": "character dyn", "negative_prompt": ""},
]
_FAVORITES = [
    {"name": "dyn", "category": "pose", "text": "dynamic pose", "note": ""},
    {"name": "wlop", "category": "artist", "text": "by wlop", "note": ""},
]


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


# --- expander -----------------------------------------------------------------


def test_expands_characters_and_favorites_with_characters_winning_a_shared_name():
    references = ReferenceExpander(_CHARACTERS, _FAVORITES)

    assert references.expand("1girl, $alice, $WLOP, $dyn") == (
        "1girl, blonde hair, cat ears, by wlop, character dyn"
    )
    assert references.negatives == "tail"


def test_text_without_references_is_untouched():
    references = ReferenceExpander(_CHARACTERS, _FAVORITES)

    assert references.expand("price US$5, $$, plain") == "price US$5, $$, plain"


def test_unknown_name_raises_listing_what_is_saved():
    with pytest.raises(UnknownReferenceError) as exc:
        ReferenceExpander(_CHARACTERS, _FAVORITES).expand("$carol")

    assert str(exc.value).splitlines() == [
        "Nothing saved under $carol.",
        "Characters: $alice, $dyn",
        "Favorites: $dyn, $wlop",
    ]


def test_non_strict_drops_an_unknown_name_and_tidies_commas():
    references = ReferenceExpander([], _FAVORITES, strict=False)

    assert references.expand("$carol, 1girl, $carol, $wlop") == "1girl, by wlop"
    assert references.unknown == ["carol"]


# --- editor autocomplete ------------------------------------------------------


def test_suggestions_list_characters_first_and_skip_shadowed_favorites():
    suggestions = reference_suggestions(_CHARACTERS, _FAVORITES)

    assert suggestions == [
        {"name": "alice", "kind": "character", "preview": "blonde hair, cat ears"},
        {"name": "dyn", "kind": "character", "preview": "character dyn"},
        {"name": "wlop", "kind": "favorite", "preview": "by wlop"},
    ]


def test_suggestion_previews_are_shortened_to_one_line():
    favorites = [{"name": "long", "text": "a,\n" + "b" * 100}]

    (suggestion,) = reference_suggestions([], favorites)

    assert "\n" not in suggestion["preview"]
    assert len(suggestion["preview"]) == 60
    assert suggestion["preview"].endswith("…")


def test_suggestions_list_every_saved_name_with_its_preview():
    # No header-size budget any more: hundreds of favorites all come through.
    favorites = [{"name": f"fav{i:03}", "text": "x" * 80} for i in range(300)]

    suggestions = reference_suggestions([], favorites)

    assert len(suggestions) == 300
    assert all(s["preview"] for s in suggestions)


@pytest.mark.asyncio
async def test_detail_prompt_button_sends_suggestions_for_the_chat_and_tapping_user(storage):
    storage.save_character(1, "alice", "blonde hair")
    storage.save_favorite(1, "wlop", "artist", "by wlop")  # `_callback_update`'s user
    storage.save_favorite(99, "other", "artist", "someone else's")
    storage.store_pending_result(
        "abc123", 1, "file123", "source.png", _serialize_generation_params(_params())
    )
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    context = _context(storage)
    context.bot_data["settings"] = MagicMock(
        allowed_user_ids=None,
        inpaint_relay_url="https://inpaint.example.com",
        inpaint_relay_shared_secret="shh",
    )
    query.message.reply_text.return_value = AsyncMock(message_id=5)
    query.message.message_thread_id = None

    with patch(
        "comfytelegram.handlers._relay_create_job", new=AsyncMock(return_value="tok1")
    ) as create_job:
        await postprocess_callback(update, context)

    references = create_job.await_args.kwargs["meta"]["references"]
    assert [r["name"] for r in references] == ["alice", "wlop"]


# --- plain prompts ------------------------------------------------------------


def test_resolve_effective_prompt_expands_both_sides_but_keeps_raw_as_typed():
    references = ReferenceExpander(_CHARACTERS, _FAVORITES)

    effective, extra_negative, raw_positive, raw_negative = _resolve_effective_prompt(
        "$alice, forest, $wlop\n---\nlowres, $wlop", None, references
    )

    assert effective == "blonde hair, cat ears, forest, by wlop"
    assert extra_negative == "lowres, by wlop, tail"
    assert (raw_positive, raw_negative) == ("$alice, forest, $wlop", "lowres, $wlop")


@pytest.mark.asyncio
async def test_generate_message_reports_an_unknown_name_without_generating(storage):
    storage.save_favorite(7, "wlop", "artist", "by wlop")
    message = AsyncMock()
    message.text = "1girl, $nobody"
    message.message_thread_id = None
    update = MagicMock()
    update.effective_message = message
    update.effective_chat.id = 1
    update.effective_user.id = 7
    context = MagicMock()
    context.chat_data = {}
    context.bot_data = {
        "settings": MagicMock(allowed_user_ids=None),
        "storage": storage,
        "comfy_client": MagicMock(),
        "profiles": [],
    }

    with (
        patch(
            "comfytelegram.handlers._resolve_checkpoint_or_default",
            new=AsyncMock(return_value="ckpt.safetensors"),
        ),
        patch("comfytelegram.handlers.generate", new=AsyncMock()) as generate,
    ):
        await generate_message(update, context)

    generate.assert_not_called()
    reply = message.reply_text.await_args.args[0]
    assert reply.startswith("⚠️ Nothing saved under $nobody.")
    assert "Favorites: $wlop" in reply


# --- "✏️ Detail Prompt" ---------------------------------------------------------


async def _run_detail_job(
    storage: Storage, positive, negative, *, user_id=7, profiles=(), params=None, disable_lora=None
):
    result_id = "abc123"
    storage.store_pending_result(
        result_id=result_id,
        chat_id=42,
        file_id="presource123",
        filename="presource.png",
        base_params=_serialize_generation_params(params or _params()),
    )
    storage.save_character(42, "alice", "blonde hair, cat ears", "tail")
    storage.save_favorite(user_id, "wlop", "artist", "by wlop")
    application = MagicMock()
    application.bot_data = {"profiles": list(profiles)}
    application.bot.send_message = AsyncMock(return_value=AsyncMock())
    application.bot.get_file = AsyncMock(
        return_value=MagicMock(download_as_bytearray=AsyncMock(return_value=bytearray(b"orig")))
    )
    application.bot.send_photo = AsyncMock(
        return_value=MagicMock(photo=[MagicMock(file_id="out-file-id")])
    )
    client = MagicMock(get_image_bytes=AsyncMock(side_effect=ComfyUIError("no file")))
    job = {
        "token": "tok1",
        "chat_id": 42,
        "message_thread_id": None,
        "result_id": result_id,
        "kind": "detail",
    }
    generated = GeneratedImage(data=b"refined", filename="out.png", full_params=_params())
    submission = {
        "mask": b"drawn-mask-bytes",
        "positive": positive,
        "negative": negative,
        "denoise": None,
        "tile_controlnet": None,
        "detailer_disable_lora": disable_lora,
        "init_data": "raw-init-data",
    }
    with (
        patch("comfytelegram.handlers._relay_poll_result", new=AsyncMock(return_value=submission)),
        patch(
            "comfytelegram.handlers.validate_webapp_init_data",
            return_value={"user": json.dumps({"id": user_id})},
        ),
        patch("comfytelegram.handlers._relay_delete_job", new=AsyncMock()),
        patch("comfytelegram.handlers.post_process", new=AsyncMock(return_value=generated)) as pp,
    ):
        await _process_one_inpaint_job(
            application, MagicMock(inpaint_relay_url="https://x"), storage, client, job
        )
    return pp.await_args.kwargs, application.bot.send_message


@pytest.mark.asyncio
async def test_detail_prompt_expands_characters_and_the_submitters_favorites(storage):
    kwargs, _ = await _run_detail_job(storage, "$alice, $wlop, detailed eyes", "lowres")

    assert kwargs["detail_prompt"] == "blonde hair, cat ears, by wlop, detailed eyes"
    assert kwargs["detail_negative_prompt"] == "lowres, tail"


@pytest.mark.asyncio
async def test_detail_prompt_blank_negative_keeps_the_images_own_plus_character_negative(storage):
    kwargs, _ = await _run_detail_job(storage, "$alice", None)

    assert kwargs["detail_negative_prompt"] == "worst quality, tail"


@pytest.mark.asyncio
async def test_detail_prompt_drops_an_unknown_name_with_a_warning(storage):
    kwargs, send_message = await _run_detail_job(storage, "$nobody, detailed eyes", "lowres")

    assert kwargs["detail_prompt"] == "detailed eyes"
    warnings = [c.args[1] for c in send_message.await_args_list if "$nobody" in c.args[1]]
    assert warnings == ["⚠️ Nothing saved under $nobody — left it out of the detail prompt."]


def _typed_params(**overrides):
    from comfytelegram.workflows import LoraSpec

    fields = {
        "positive_prompt": "masterpiece, marlow, anthro, blonde hair, a fox",
        "negative_prompt": "worst quality, tail",
        "raw_positive_prompt": "$alice, a fox",
        "raw_negative_prompt": "",
        "loras": [LoraSpec("marlow.safetensors", 0.9, 0.9)],
    }
    fields.update(overrides)
    return _params().__class__(checkpoint="ckpt.safetensors", **fields)


def _typed_profile(**overrides):
    from comfytelegram.profiles import LoraDefault, ModelProfile

    return ModelProfile(
        match=["ckpt*"],
        display_name="Typed",
        positive_prompt_prefix="masterpiece",
        negative_prompt_prefix="worst quality",
        loras=[
            LoraDefault(
                name="marlow.safetensors", default_enabled=True, trigger_words="marlow, anthro"
            )
        ],
        **overrides,
    )


def test_detail_prompt_editor_prefills_the_prompt_as_typed():
    from comfytelegram.handlers import _detail_prompt_prefill

    assert _detail_prompt_prefill(_typed_params()) == {"positive": "$alice, a fox", "negative": ""}


def test_detail_prompt_editor_prefills_the_built_prompt_without_a_typed_one():
    from comfytelegram.handlers import _detail_prompt_prefill
    from comfytelegram.workflows import RegionSpec

    regional = _typed_params(regions=[RegionSpec("left", "a", 0.0, 0.0, 0.5, 1.0)])
    for params in (_typed_params(raw_positive_prompt=""), regional):
        assert _detail_prompt_prefill(params)["positive"] == params.positive_prompt


@pytest.mark.asyncio
async def test_typed_detail_prompt_gets_prefixes_and_inherited_trigger_words_back(storage):
    kwargs, _ = await _run_detail_job(
        storage, "detailed paws", "blurry", params=_typed_params(), profiles=[_typed_profile()]
    )

    assert kwargs["detail_prompt"] == "masterpiece, marlow, anthro, detailed paws"
    assert kwargs["detail_negative_prompt"] == "worst quality, blurry"


@pytest.mark.asyncio
async def test_typed_detail_prompt_drops_trigger_words_of_loras_that_dont_apply(storage):
    profiles = [_typed_profile()]
    params = _typed_params()
    disabled, _ = await _run_detail_job(
        storage, "paws", None, params=params, profiles=profiles, disable_lora=True
    )
    switched_off, _ = await _run_detail_job(
        storage, "paws, <lora:marlow:0>", None, params=params, profiles=profiles
    )

    assert disabled["detail_prompt"] == "masterpiece, paws"
    assert switched_off["detail_prompt"] == "masterpiece, paws, <lora:marlow:0>"
    # A blank field stays blank: post_process keeps the image's own, built already.
    assert disabled["detail_negative_prompt"] is None
