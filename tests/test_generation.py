import io
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from comfytelegram.comfy_client import JobProgress
from comfytelegram.generation import (
    _detailer_found_nothing,
    _to_post_process_base,
    generate,
    post_process,
)
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.workflows import GenerationParams, LoraSpec


def _solid_png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _solid_mask_png(width: int, height: int, value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), value).save(buf, format="PNG")
    return buf.getvalue()


class _StubComfyClient:
    """Minimal stand-in for `ComfyClient` — just enough of the queue/watch/
    history/download surface for `generate()`'s `_run_graph` call to run
    end to end without a live ComfyUI server."""

    def __init__(self) -> None:
        self.queued_graph: dict | None = None
        self.connected_client_id: str | None = None

    async def queue_prompt(self, prompt: dict, *, client_id: str) -> str:
        assert self.connected_client_id == client_id, (
            "the event socket must be connected before the prompt is queued — "
            "see ComfyClient.connect_events"
        )
        self.queued_graph = prompt
        return "prompt-1"

    @asynccontextmanager
    async def connect_events(self, *, client_id: str):
        self.connected_client_id = client_id
        yield self._Events()

    class _Events:
        async def watch(self, prompt_id: str):
            yield JobProgress(prompt_id=prompt_id, node_id=None, value=None, max=None, done=True)

    async def get_history(self, prompt_id: str) -> dict:
        save_node_id = next(
            node_id
            for node_id, node in self.queued_graph.items()
            if node["class_type"] == "SaveImage"
        )
        return {
            "status": {"status_str": "success"},
            "outputs": {
                save_node_id: {
                    "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
                }
            },
        }

    async def get_image_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        return b"fake-image-bytes"


class _StubUploadingComfyClient(_StubComfyClient):
    """`_StubComfyClient` plus `upload_image`, for `post_process()` tests.
    `output_bytes` is what the main `SaveImage` node produces. `mask_bytes`
    (grayscale PNG bytes — all-black simulates "nothing detected") is what
    the detailer's detection-check `PreviewImage` node produces (see
    `_build_detailer` — a `PreviewImage`, not `SaveImage`, so the check
    doesn't leave a stray file in ComfyUI's permanent output folder); None
    (the default) simulates that node never producing an image at all
    (e.g. an upscale graph, which has no such node)."""

    def __init__(self, output_bytes: bytes, mask_bytes: bytes | None = None) -> None:
        super().__init__()
        self._output_bytes = output_bytes
        self._mask_bytes = mask_bytes

    async def upload_image(
        self, source, *, filename: str | None = None, subfolder: str = "", overwrite: bool = False
    ) -> dict:
        return {"name": filename or "uploaded.png"}

    def _main_save_node_id(self) -> str:
        return next(
            node_id
            for node_id, node in self.queued_graph.items()
            if node["class_type"] == "SaveImage"
        )

    def _detection_node_id(self) -> str | None:
        return next(
            (
                node_id
                for node_id, node in self.queued_graph.items()
                if node["class_type"] == "PreviewImage"
            ),
            None,
        )

    async def get_history(self, prompt_id: str) -> dict:
        outputs = {
            self._main_save_node_id(): {
                "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
            }
        }
        detection_id = self._detection_node_id()
        if detection_id is not None and self._mask_bytes is not None:
            outputs[detection_id] = {
                "images": [{"filename": "mask.png", "subfolder": "", "type": "temp"}]
            }
        return {"status": {"status_str": "success"}, "outputs": outputs}

    async def get_image_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        if filename == "mask.png":
            return self._mask_bytes
        return self._output_bytes


@pytest.mark.asyncio
async def test_detailer_found_nothing_true_for_a_black_mask():
    client = AsyncMock()
    client.get_image_bytes.return_value = _solid_mask_png(4, 4, 0)
    history = {
        "outputs": {"9": {"images": [{"filename": "m.png", "subfolder": "", "type": "output"}]}}
    }

    assert await _detailer_found_nothing(client, history, "9") is True


@pytest.mark.asyncio
async def test_detailer_found_nothing_false_for_a_non_black_mask():
    client = AsyncMock()
    client.get_image_bytes.return_value = _solid_mask_png(4, 4, 200)
    history = {
        "outputs": {"9": {"images": [{"filename": "m.png", "subfolder": "", "type": "output"}]}}
    }

    assert await _detailer_found_nothing(client, history, "9") is False


@pytest.mark.asyncio
async def test_detailer_found_nothing_false_when_node_produced_no_images():
    client = AsyncMock()
    history = {"outputs": {"9": {"images": []}}}

    assert await _detailer_found_nothing(client, history, "9") is False


@pytest.mark.asyncio
async def test_detailer_found_nothing_false_when_node_id_absent():
    client = AsyncMock()
    history = {"outputs": {}}

    assert await _detailer_found_nothing(client, history, "9") is False


@pytest.mark.asyncio
async def test_detailer_found_nothing_false_when_download_fails():
    client = AsyncMock()
    client.get_image_bytes.side_effect = RuntimeError("boom")
    history = {
        "outputs": {"9": {"images": [{"filename": "m.png", "subfolder": "", "type": "output"}]}}
    }

    assert await _detailer_found_nothing(client, history, "9") is False


@pytest.mark.asyncio
async def test_post_process_face_detailer_flags_unchanged_when_mask_is_black():
    source = _solid_png(10, 10, (255, 0, 0))
    # Slightly different from the source, simulating ComfyUI's own
    # float32 round-trip noise on a true pass-through — this must NOT by
    # itself prevent "unchanged" from being reported, since the mask (not
    # the output image) is the actual detection signal now.
    output = _solid_png(10, 10, (254, 1, 1))
    mask = _solid_mask_png(10, 10, 0)
    client = _StubUploadingComfyClient(output, mask_bytes=mask)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "face", source, "source.png", params)

    assert result.unchanged is True


@pytest.mark.asyncio
async def test_post_process_hand_detailer_does_not_flag_a_real_change():
    source = _solid_png(10, 10, (255, 0, 0))
    output = _solid_png(10, 10, (0, 255, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(output, mask_bytes=mask)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "hand", source, "source.png", params)

    assert result.unchanged is False


@pytest.mark.asyncio
async def test_post_process_face_detailer_defaults_to_changed_when_mask_missing():
    """If the detection-check node's image can't be found at all,
    post_process() must default to "assume something happened" rather
    than risk wrongly suppressing a real result."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source, mask_bytes=None)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "face", source, "source.png", params)

    assert result.unchanged is False


@pytest.mark.asyncio
async def test_post_process_upscale_never_flags_unchanged():
    """An upscale graph has no detection-check node at all, so there's no
    "detector found nothing" case to flag for it."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "upscale", source, "source.png", params)

    assert result.unchanged is False


@pytest.mark.asyncio
async def test_post_process_upscale_honors_profile_upscale_denoise():
    """A profile's `defaults.upscale_denoise` (e.g. furrytoonmix_illustrious.json,
    tuned alongside its `tile_controlnet`) must actually reach the queued
    UltimateSDUpscale node instead of always using UpscaleParams' own 0.2
    default."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox",
        negative_prompt="",
        upscale_denoise=0.5,
    )

    await post_process(client, "upscale", source, "source.png", params)

    upscale_node = next(
        node for node in client.queued_graph.values() if node["class_type"] == "UltimateSDUpscale"
    )
    assert upscale_node["inputs"]["denoise"] == 0.5


@pytest.mark.asyncio
async def test_post_process_hand_manual_never_flags_unchanged():
    """A manually-marked mask has no detection-check node either (see
    `build_hand_detailer_manual`) — it's never "nothing detected"."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(
        client, "hand_manual", source, "source.png", params, point_frac=(0.5, 0.5)
    )

    assert result.unchanged is False


@pytest.mark.asyncio
async def test_post_process_hand_manual_requires_point_frac():
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    with pytest.raises(AssertionError):
        await post_process(client, "hand_manual", source, "source.png", params)


def test_to_post_process_base_carries_only_the_relevant_fields():
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        seed=42,
        steps=25,
        cfg=6.5,
        clip_skip=-2,
        loras=[LoraSpec(name="a.safetensors", strength_model=0.8, strength_clip=1.0)],
    )

    base = _to_post_process_base(params)

    assert base.checkpoint == "ckpt.safetensors"
    assert base.positive_prompt == "a fox"
    assert base.negative_prompt == "blurry"
    assert base.clip_skip == -2
    assert base.loras == params.loras
    # PostProcessBaseParams doesn't carry seed/steps/cfg/etc at all — the
    # post-processing stage re-resolves those itself via
    # UpscaleParams/FaceDetailerParams instead of inheriting the original run's.
    assert not hasattr(base, "seed")
    assert not hasattr(base, "steps")
    assert not hasattr(base, "cfg")


def test_to_post_process_base_carries_split_loader_fields():
    """Without this, a post-processing pass on a split-architecture (e.g.
    Anima) generation would silently fall back to loader="checkpoint" and
    try to load the UNET filename through CheckpointLoaderSimple."""
    params = GenerationParams(
        checkpoint="anima-aesthetic-v1.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        clip_type="stable_diffusion",
        vae_name="qwen_image_vae.safetensors",
        model_sampling_shift=3.0,
    )

    base = _to_post_process_base(params)

    assert base.loader == "split"
    assert base.clip_name == "qwen_3_06b_base.safetensors"
    assert base.clip_type == "stable_diffusion"
    assert base.vae_name == "qwen_image_vae.safetensors"
    assert base.model_sampling_shift == 3.0


def test_to_post_process_base_carries_tile_controlnet_fields():
    """Without this, a checkpoint with a tile ControlNet configured would
    lose it on every post-processing pass since build_upscale only sees
    PostProcessBaseParams, not the original GenerationParams."""
    params = GenerationParams(
        checkpoint="illustriousXL_v10.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        tile_controlnet="xinsir_tile_sdxl.safetensors",
        tile_controlnet_strength=0.55,
    )

    base = _to_post_process_base(params)

    assert base.tile_controlnet == "xinsir_tile_sdxl.safetensors"
    assert base.tile_controlnet_strength == 0.55


def test_to_post_process_base_carries_upscale_denoise():
    params = GenerationParams(
        checkpoint="illustriousXL_v10.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        upscale_denoise=0.5,
    )

    base = _to_post_process_base(params)

    assert base.upscale_denoise == 0.5


@pytest.mark.asyncio
async def test_generate_overrides_force_batch_size_regardless_of_profile_default():
    """ "/stream" passes `overrides={"batch_size": 1}` to force single-image
    generations no matter what the checkpoint's own profile defaults to —
    verify that wins over `defaults.batch_size=4` end to end."""
    profile = ModelProfile(
        match=["*ckpt*"], display_name="X", defaults=ProfileDefaults(batch_size=4)
    )
    client = _StubComfyClient()

    images = await generate(
        client, "ckpt.safetensors", "a fox", profile, overrides={"batch_size": 1}
    )

    assert len(images) == 1
    assert images[0].full_params.batch_size == 1
