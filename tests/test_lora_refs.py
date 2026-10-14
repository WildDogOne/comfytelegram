"""`<lora:…>` tags in prompts (`lora_refs`): parsing, matching against the
checkpoint profile's LoRAs, typed prompts (`_resolve_profile_and_prompt`,
`/rp`), and "✏️ Detail Prompt" (`generation._apply_detail_loras`, the
submission warning, the editor's autocomplete entries)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_prompt_refs import _run_detail_job

from comfytelegram.generation import _apply_detail_loras
from comfytelegram.handlers import _resolve_profile_and_prompt, generate_message
from comfytelegram.lora_refs import (
    LoraTag,
    UnknownLoraError,
    apply_prompt_loras,
    extract_lora_tags,
    lora_suggestions,
    lora_tag,
    match_lora,
    resolve_detail_loras,
    strip_lora_tags,
)
from comfytelegram.profiles import (
    LoraDefault,
    ModelProfile,
    resolve_generation_params,
)
from comfytelegram.prompt_refs import UnknownReferenceError
from comfytelegram.storage import Storage
from comfytelegram.workflows.builder import LoraSpec, PostProcessBaseParams

_ALICE = LoraDefault(
    name="chars/alice_v2.safetensors",
    default_enabled=False,
    trigger_words="alice_char",
    strength_model=0.9,
    strength_clip=0.9,
)
_DETAIL = LoraDefault(name="detail_tweaker.safetensors", default_enabled=True)
_STYLE = LoraDefault(name="styles\\ink.safetensors", default_enabled=False)


def _profile(*loras: LoraDefault) -> ModelProfile:
    return ModelProfile(
        match=["*"],
        display_name="p",
        positive_prompt_prefix="masterpiece",
        loras=list(loras) or [_ALICE, _DETAIL, _STYLE],
    )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


# --- parsing / matching ---------------------------------------------------------


def test_extract_takes_out_every_form_and_tidies_commas():
    cleaned, tags = extract_lora_tags(
        "1girl, <lora:chars/alice_v2>, smile, <lora:ink:0.5>, <LORA: detail : 1 : 0.25 >"
    )

    assert cleaned == "1girl, smile"
    assert tags == [
        LoraTag("chars/alice_v2"),
        LoraTag("ink", 0.5, 0.5),
        LoraTag("detail", 1.0, 0.25),
    ]


def test_extract_leaves_text_without_tags_alone():
    assert extract_lora_tags("a <b> c, (x:1.2)") == ("a <b> c, (x:1.2)", [])


@pytest.mark.parametrize(
    "name",
    [
        "chars/alice_v2.safetensors",
        "chars/alice_v2",
        "CHARS\\Alice_V2",
        "alice_v2",
        "alice_v2.safetensors",
    ],
)
def test_match_by_path_with_or_without_extension_or_by_file_name(name):
    assert match_lora(name, [_ALICE, _DETAIL]) is _ALICE


def test_match_accepts_a_windows_listed_path_typed_with_slashes():
    assert match_lora("styles/ink", [_STYLE]) is _STYLE


def test_match_returns_every_candidate_for_an_ambiguous_file_name():
    other = LoraDefault(name="old/alice_v2.safetensors")

    assert match_lora("alice_v2", [_ALICE, other]) == [_ALICE, other]
    assert match_lora("old/alice_v2", [_ALICE, other]) is other


def test_a_dot_inside_a_name_is_not_an_extension():
    versioned = LoraDefault(name="style_v1.5.safetensors")

    assert match_lora("style_v1.5", [versioned]) is versioned


# --- typed prompts ----------------------------------------------------------------


def test_apply_enables_a_lora_and_its_trigger_words_reach_the_prompt():
    profile = apply_prompt_loras(_profile(), [LoraTag("alice_v2")])

    params = resolve_generation_params("ckpt", "1girl", profile)

    assert (
        params.loras
        == [
            LoraSpec("detail_tweaker.safetensors", 1.0, 1.0),
            LoraSpec("chars/alice_v2.safetensors", 0.9, 0.9),
        ][::-1]
    )
    assert params.positive_prompt == "masterpiece, alice_char, 1girl"


def test_apply_sets_strength_and_zero_switches_a_default_off():
    profile = apply_prompt_loras(
        _profile(), [LoraTag("alice_v2", 0.6, 0.3), LoraTag("detail_tweaker", 0.0, 0.0)]
    )

    params = resolve_generation_params("ckpt", "x", profile)

    assert params.loras == [LoraSpec("chars/alice_v2.safetensors", 0.6, 0.3)]


def test_apply_rejects_an_unknown_name_listing_what_is_available():
    with pytest.raises(UnknownLoraError) as excinfo:
        apply_prompt_loras(_profile(), [LoraTag("nobody")])

    assert isinstance(excinfo.value, UnknownReferenceError)
    message = str(excinfo.value)
    assert message.startswith("No LoRA called nobody")
    assert "Available: chars/alice_v2, detail_tweaker, styles/ink" in message


def test_apply_without_a_profile_explains_there_is_nothing_to_choose_from():
    with pytest.raises(UnknownLoraError, match="doesn't list any LoRAs"):
        apply_prompt_loras(None, [LoraTag("alice_v2")])


def test_apply_reports_an_ambiguous_name_with_the_paths_to_use():
    other = LoraDefault(name="old/alice_v2.safetensors")

    with pytest.raises(UnknownLoraError, match="more than one") as excinfo:
        apply_prompt_loras(_profile(_ALICE, other), [LoraTag("alice_v2")])

    assert "<lora:chars/alice_v2>" in str(excinfo.value)
    assert "<lora:old/alice_v2>" in str(excinfo.value)


def test_resolve_profile_and_prompt_reads_tags_after_character_expansion(storage):
    from comfytelegram.handlers import _reference_expander

    storage.save_character(1, "alice", "<lora:alice_v2:0.7>, blonde hair", "")

    profile, effective, extra_negative, raw_positive, _ = _resolve_profile_and_prompt(
        1,
        "ckpt",
        "$alice, smile\n---\nlowres",
        storage,
        [_profile()],
        references=_reference_expander(storage, 1, None),
    )

    assert effective == "blonde hair, smile"
    assert extra_negative == "lowres"
    assert raw_positive == "$alice, smile"
    (alice,) = [lora for lora in profile.loras if lora.name == _ALICE.name]
    assert (alice.default_enabled, alice.strength_model) == (True, 0.7)


@pytest.mark.asyncio
async def test_generate_message_reports_an_unknown_lora_without_generating(storage):
    message = AsyncMock()
    message.text = "1girl, <lora:nobody>"
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
        "profiles": [_profile()],
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
    assert message.reply_text.await_args.args[0].startswith("⚠️ No LoRA called nobody")


# --- "✏️ Detail Prompt" -------------------------------------------------------------


def _base(**overrides) -> PostProcessBaseParams:
    fields = {
        "checkpoint": "ckpt",
        "positive_prompt": "detailed eyes",
        "negative_prompt": "lowres",
        "loras": [LoraSpec("detail_tweaker.safetensors", 1.0, 1.0)],
    }
    fields.update(overrides)
    return PostProcessBaseParams(**fields)


def test_detail_loras_add_to_the_inherited_ones_with_trigger_words_in_front():
    base = _apply_detail_loras(
        _base(positive_prompt="<lora:alice_v2>, detailed eyes"), [_profile()]
    )

    assert base.loras == [
        LoraSpec("detail_tweaker.safetensors", 1.0, 1.0),
        LoraSpec("chars/alice_v2.safetensors", 0.9, 0.9),
    ]
    assert base.positive_prompt == "alice_char, detailed eyes"


def test_detail_loras_change_or_remove_an_inherited_one():
    base = _apply_detail_loras(
        _base(positive_prompt="eyes, <lora:detail_tweaker:0>, <lora:ink:0.4>"), [_profile()]
    )

    assert base.loras == [LoraSpec("styles\\ink.safetensors", 0.4, 0.4)]


def test_detail_loras_replace_the_inherited_ones_when_lora_disabling_is_on():
    base = _apply_detail_loras(
        _base(positive_prompt="<lora:alice_v2>, alice_char, eyes", detailer_disable_lora=True),
        [_profile()],
    )

    assert base.loras == [LoraSpec("chars/alice_v2.safetensors", 0.9, 0.9)]
    # Typed LoRAs must survive `_detailer_base`'s stripping.
    assert base.detailer_disable_lora is False
    # Trigger words already in the prompt aren't added twice.
    assert base.positive_prompt == "alice_char, eyes"


def test_detail_prompt_without_tags_is_left_alone():
    base = _base(detailer_disable_lora=True)

    assert _apply_detail_loras(base, [_profile()]) == base


def test_resolve_detail_loras_reports_unknown_names_and_strip_removes_them():
    text = "<lora:nobody>, <lora:alice_v2>, eyes"

    cleaned, loras, unknown = resolve_detail_loras(text, _profile())

    assert (cleaned, [lora.name for lora in loras], unknown) == (
        "eyes",
        [_ALICE.name],
        ["nobody"],
    )
    assert strip_lora_tags(text, unknown) == "<lora:alice_v2>, eyes"


@pytest.mark.asyncio
async def test_detail_submission_drops_an_unknown_lora_with_a_warning_but_keeps_known_tags(
    storage,
):
    kwargs, send_message = await _run_detail_job(
        storage,
        "<lora:nobody>, <lora:alice_v2:0.5>, detailed eyes",
        "lowres",
        profiles=[_profile()],
    )

    # The known tag stays in the text, for post_process to apply and redo to replay.
    assert kwargs["detail_prompt"] == "<lora:alice_v2:0.5>, detailed eyes"
    warnings = [c.args[1] for c in send_message.await_args_list if "nobody" in c.args[1]]
    assert warnings == [
        "⚠️ No LoRA called <lora:nobody> for this checkpoint — left it out of the detail prompt."
    ]


def test_lora_suggestions_insert_a_tag_at_the_configured_strength():
    asymmetric = LoraDefault(name="x.safetensors", strength_model=1.0, strength_clip=0.5)

    suggestions = lora_suggestions(_profile(_ALICE, _STYLE, asymmetric))

    assert suggestions == [
        {"name": "chars/alice_v2", "strength": "0.9", "preview": "alice_char"},
        {"name": "styles/ink", "strength": "1", "preview": ""},
        {"name": "x", "strength": "1:0.5", "preview": ""},
    ]
    assert lora_tag(asymmetric) == "<lora:x:1:0.5>"
    assert lora_suggestions(None) == []


def test_lora_suggestions_list_every_lora_in_the_profile():
    # Banana Splitz XXL's 24 used to stop at 14 under the old header budget,
    # the last-added ones missing from the menu.
    many = [
        LoraDefault(
            name=f"illustrious/style_{i:03}/style_{i:03}.safetensors", trigger_words="t" * 40
        )
        for i in range(200)
    ]

    suggestions = lora_suggestions(_profile(*many))

    assert len(suggestions) == 200
    assert suggestions[-1] == {
        "name": "illustrious/style_199/style_199",
        "strength": "1",
        "preview": "t" * 29 + "…",
    }


@pytest.mark.asyncio
async def test_detail_prompt_button_sends_lora_suggestions_and_the_images_loras(storage):
    from test_detail_prompt import _callback_update, _context, _params

    from comfytelegram.handlers import (
        DETAIL_PROMPT_CALLBACK_KIND,
        _serialize_generation_params,
        postprocess_callback,
    )

    params = _params(loras=[LoraSpec("detail_tweaker.safetensors", 0.8, 0.5)])
    storage.store_pending_result(
        "abc123", 1, "file123", "source.png", _serialize_generation_params(params)
    )
    update, query = _callback_update(f"pp:{DETAIL_PROMPT_CALLBACK_KIND}:abc123")
    context = _context(storage)
    context.bot_data["profiles"] = [_profile()]
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

    meta = create_job.await_args.kwargs["meta"]
    assert [entry["name"] for entry in meta["loras"]] == [
        "chars/alice_v2",
        "detail_tweaker",
        "styles/ink",
    ]
    assert meta["active_loras"] == [
        {"name": "detail_tweaker.safetensors", "strength_model": 0.8, "strength_clip": 0.5}
    ]
