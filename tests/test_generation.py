import asyncio
import io
import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from comfytelegram import generation
from comfytelegram.comfy_client import ComfyUIError, JobProgress
from comfytelegram.generation import (
    _bbox_diagonal,
    _detailer_found_nothing,
    _fix_drawn_geometry,
    _mask_bbox,
    _native_fix_mask_params,
    _to_post_process_base,
    generate,
    post_process,
)
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.workflows import DrawnMaskFixParams, GenerationParams, LoraSpec


def _solid_png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _solid_mask_png(width: int, height: int, value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), value).save(buf, format="PNG")
    return buf.getvalue()


def _rect_mask_png(width: int, height: int, box: tuple[int, int, int, int]) -> bytes:
    """A mask with a marked-white rectangle (`box`, PIL's `(x0, y0, x1, y1)`)
    on an otherwise-black background — for tests that need a specific
    bounding-box size instead of `_solid_mask_png`'s whole-image mark."""
    from PIL import ImageDraw

    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).rectangle(box, fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_fix_drawn_geometry_crops_context_around_the_mask():
    # Pass 1 must see a region proportional to the mark, not the whole
    # frame — on a 4096x4096 source, downscaling everything left the region
    # being repaired at ~200px of the pass, far too few to resolve what it
    # was meant to continue. See FixDrawnGeometry's docstring.
    mask = _rect_mask_png(4096, 4096, (1800, 1900, 2300, 2500))
    params = DrawnMaskFixParams()
    geo = _fix_drawn_geometry(mask, (4096, 4096), params)

    cx, cy, cw, ch = geo.context_crop
    assert (cw, ch) != (4096, 4096)  # not the whole frame
    # The mark, plus its grow/feather falloff, has to be inside it.
    grow, feather, _blend = _native_fix_mask_params(_bbox_diagonal(_mask_bbox(mask)))
    assert cx <= 1800 - (grow + feather)
    assert cy <= 1900 - (grow + feather)
    assert cx + cw >= 2300 + (grow + feather)
    assert cy + ch >= 2500 + (grow + feather)
    assert cw % 16 == 0 and ch % 16 == 0


def test_fix_drawn_geometry_gives_the_region_a_krita_like_share_of_pass_one():
    # The measurement that motivated context cropping: krita's first pass
    # gave the repaired region ~49% of its frame, this bot's gave it ~16%.
    mask = _rect_mask_png(4096, 4096, (1800, 1900, 2300, 2500))
    geo = _fix_drawn_geometry(mask, (4096, 4096), DrawnMaskFixParams())

    work_w, _work_h = geo.work_size
    scale = work_w / geo.context_crop[2]
    # The region here is the refine target when there is one, else the crop
    # the mask sits in; either way measure the mask's own padded extent.
    region = geo.refine_region[0] if geo.refine_region else geo.context_crop
    share = max(region[2], region[3]) * scale / work_w
    assert 0.35 < share < 0.65


def test_fix_drawn_geometry_work_size_is_bounded_and_snapped():
    mask = _rect_mask_png(4096, 4096, (1000, 1000, 3000, 3000))
    params = DrawnMaskFixParams()
    geo = _fix_drawn_geometry(mask, (4096, 4096), params)
    w, h = geo.work_size
    assert max(w, h) <= params.work_max_size
    assert w % 16 == 0 and h % 16 == 0


def test_fix_drawn_geometry_empty_mask_falls_back_to_whole_frame():
    mask = _solid_mask_png(4096, 4096, 0)
    geo = _fix_drawn_geometry(mask, (4096, 4096), DrawnMaskFixParams())
    assert geo.context_crop == (0, 0, 4096, 4096)
    assert geo.refine_region is None


def test_fix_drawn_geometry_mask_params_are_in_work_space():
    # grow/feather are computed from the mask at native resolution (krita
    # applies them before downscaling) and then scaled into the space the
    # graph actually applies them in.
    mask = _rect_mask_png(4096, 4096, (1800, 1900, 2300, 2500))
    geo = _fix_drawn_geometry(mask, (4096, 4096), DrawnMaskFixParams())
    grow_n, feather_n, _blend_n = _native_fix_mask_params(_bbox_diagonal(_mask_bbox(mask)))
    scale = geo.work_size[0] / geo.context_crop[2]
    assert geo.mask_grow == max(0, round(grow_n * scale))
    assert geo.mask_blur == max(1, round(feather_n * scale))


def test_fix_drawn_geometry_skips_refine_when_pass_one_already_resolves_it():
    # With the context crop in place pass 1 often already renders the region
    # near its native size, and a second pass has nothing to add — the same
    # condition krita's ScaledExtent.refinement_scaling reports as `none`.
    mask = _rect_mask_png(1000, 1000, (400, 400, 600, 600))
    geo = _fix_drawn_geometry(mask, (1000, 1000), DrawnMaskFixParams())
    assert geo.refine_region is None


def test_fix_drawn_geometry_refine_region_covers_the_feathered_mask():
    mask = _rect_mask_png(8192, 8192, (3600, 3800, 4600, 5000))
    geo = _fix_drawn_geometry(mask, (8192, 8192), DrawnMaskFixParams())
    if geo.refine_region is None:
        pytest.skip("no refine pass for this geometry")
    (x, y, w, h), (rw, rh) = geo.refine_region
    grow, feather, _blend = _native_fix_mask_params(_bbox_diagonal(_mask_bbox(mask)))
    assert x <= 3600 - (grow + feather)
    assert x + w >= 4600 + (grow + feather)
    assert w % 16 == 0 and h % 16 == 0
    assert rw % 16 == 0 and rh % 16 == 0
    # The refine region always sits inside what pass 1 actually saw.
    cx, cy, cw, ch = geo.context_crop
    assert x >= cx and y >= cy and x + w <= cx + cw and y + h <= cy + ch


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
            "status": {"status_str": "success", "completed": True},
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
        return {"status": {"status_str": "success", "completed": True}, "outputs": outputs}

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
async def test_post_process_homogenize_never_flags_unchanged():
    """Same reasoning as the "upscale" case above — no detection-check node
    for this kind either."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "homogenize", source, "source.png", params)

    assert result.unchanged is False


@pytest.mark.asyncio
async def test_post_process_homogenize_runs_at_upscale_by_one():
    """The whole point of "🧵 Homogenize" is a tiled img2img pass with no
    resolution change — `TiledRefineParams.upscale_by` must stay 1.0
    regardless of the profile's `upscale_denoise` (that override is
    documented as `"upscale"`-only, see `GenerationParams.upscale_denoise`,
    and shouldn't leak into this separate, lower-denoise pass)."""
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox",
        negative_prompt="",
        upscale_denoise=0.9,
    )

    await post_process(client, "homogenize", source, "source.png", params)

    refine_node = next(
        node for node in client.queued_graph.values() if node["class_type"] == "UltimateSDUpscale"
    )
    assert refine_node["inputs"]["upscale_by"] == 1.0
    assert refine_node["inputs"]["denoise"] != 0.9


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


@pytest.mark.asyncio
async def test_run_graph_falls_back_to_history_when_websocket_event_is_lost(monkeypatch):
    """Reproduces a real, observed failure: ComfyUI can finish and drop a
    fast job's terminal event before the websocket watcher ever sees it
    (see `_run_graph`'s `_WS_EVENT_FALLBACK_SECONDS` docstring) — the job
    still shows up in `/history` as completed, and `_run_graph` must not
    hang forever waiting on an event that's already gone."""
    monkeypatch.setattr(generation, "_WS_EVENT_FALLBACK_SECONDS", 0.01)

    class _HangingEvents:
        async def watch(self, prompt_id: str):
            await asyncio.Event().wait()
            yield  # pragma: no cover - never reached; makes this an async generator

    class _StubHangingComfyClient(_StubUploadingComfyClient):
        @asynccontextmanager
        async def connect_events(self, *, client_id: str):
            self.connected_client_id = client_id
            yield _HangingEvents()

    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubHangingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(
        client, "hand_manual", source, "source.png", params, point_frac=(0.5, 0.5)
    )

    assert result.data == source


@pytest.mark.asyncio
async def test_run_graph_keeps_polling_past_a_premature_terminal_event(monkeypatch):
    """Reproduces a real, observed failure: ComfyUI fired the terminal
    `executing{node: null}` websocket event within milliseconds of a job
    being queued — right after its leading, already-`execution_cached`
    nodes — while the job's actual (non-cached, GPU-bound) work went on to
    take roughly 90 more seconds. Trusting that event outright reported a
    job that later succeeded as an instant failure. `_run_graph` must
    corroborate a `done` event against `/history` and, if it disagrees,
    keep polling on the same cadence a lost event would use instead of
    giving up."""
    monkeypatch.setattr(generation, "_WS_EVENT_FALLBACK_SECONDS", 0)

    class _PrematureDoneEvents:
        async def watch(self, prompt_id: str):
            yield JobProgress(prompt_id=prompt_id, node_id=None, value=None, max=None, done=True)

    class _StubPrematureDoneClient(_StubUploadingComfyClient):
        def __init__(self, output_bytes: bytes) -> None:
            super().__init__(output_bytes)
            self.history_calls = 0

        @asynccontextmanager
        async def connect_events(self, *, client_id: str):
            self.connected_client_id = client_id
            yield _PrematureDoneEvents()

        async def get_history(self, prompt_id: str) -> dict | None:
            self.history_calls += 1
            if self.history_calls < 3:
                return {"status": {"status_str": "running", "completed": False}}
            return await super().get_history(prompt_id)

    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubPrematureDoneClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "upscale", source, "source.png", params)

    assert result.data == source
    assert client.history_calls >= 3


@pytest.mark.asyncio
async def test_run_graph_still_raises_when_the_job_itself_errors():
    """A job that ComfyUI genuinely finishes with an error must still be
    reported as one — the premature-terminal-event fix above only defers
    judgment until `/history` actually says the job is complete, it
    doesn't suppress a real failure once it is."""

    class _StubErroringClient(_StubUploadingComfyClient):
        async def get_history(self, prompt_id: str) -> dict | None:
            return {"status": {"status_str": "error", "completed": True}, "outputs": {}}

    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubErroringClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    with pytest.raises(ComfyUIError, match="failed"):
        await post_process(client, "upscale", source, "source.png", params)


@pytest.mark.asyncio
async def test_post_process_hand_drawn_never_flags_unchanged():
    """A freehand-drawn mask has no detection-check node either (see
    `build_hand_detailer_drawn_mask`) — it's never "nothing detected"."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "hand_drawn", source, "source.png", params, mask_bytes=mask)

    assert result.unchanged is False
    mask_load = next(n for n in client.queued_graph.values() if n["class_type"] == "LoadImageMask")
    assert mask_load["inputs"]["image"] == "mask_source.png"


@pytest.mark.asyncio
async def test_post_process_hand_drawn_requires_mask_bytes():
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    with pytest.raises(AssertionError):
        await post_process(client, "hand_drawn", source, "source.png", params)


@pytest.mark.asyncio
async def test_post_process_fix_drawn_never_flags_unchanged():
    """Same reasoning as `hand_drawn` — a freehand-drawn mask has no
    detection-check node (see `build_fix_drawn_mask`), so it's never
    reported as "nothing detected"."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    result = await post_process(client, "fix_drawn", source, "source.png", params, mask_bytes=mask)

    assert result.unchanged is False
    mask_load = next(n for n in client.queued_graph.values() if n["class_type"] == "LoadImageMask")
    assert mask_load["inputs"]["image"] == "mask_source.png"
    detailer = next(n for n in client.queued_graph.values() if n["class_type"] == "DetailerForEach")
    assert detailer["inputs"]["denoise"] == DrawnMaskFixParams().denoise


@pytest.mark.asyncio
async def test_post_process_fix_drawn_logs_whether_anima_lllite_patch_engaged(caplog):
    """The graph itself doesn't make it obvious from the outside whether
    `anima_lllite_inpaint_patch` actually fired for a given run (e.g. a
    stale `full_params` from before a profile added the field, vs. a
    checkpoint that never had it configured) — this log line is what lets
    that be confirmed from the bot's own logs instead of guessing."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    caplog.set_level(logging.INFO, logger="comfytelegram.generation")

    caplog.clear()
    unset_params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )
    await post_process(client, "fix_drawn", source, "source.png", unset_params, mask_bytes=mask)
    assert "skipped (loader='checkpoint', not 'split')" in caplog.text

    caplog.clear()
    engaged_params = GenerationParams(
        checkpoint="anima_unet.safetensors",
        positive_prompt="a fox",
        negative_prompt="",
        loader="split",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
        anima_lllite_inpaint_patch_strength=0.8,
    )
    await post_process(client, "fix_drawn", source, "source.png", engaged_params, mask_bytes=mask)
    assert "patch='anima-lllite-inpainting-v2.safetensors'" in caplog.text
    assert "strength=0.8" in caplog.text


@pytest.mark.asyncio
async def test_post_process_fix_drawn_sizes_mask_grow_from_drawn_mask_not_fixed_default():
    """The Anima path used to run every "🩹 Fix Artifact" job through the
    same fixed `DrawnMaskFixParams.mask_grow`/`mask_blur`/`blend` regardless
    of how big the actual drawn mark was — krita-ai-diffusion computes these
    per job from the mark's own bounding-box size instead
    (`_dynamic_fix_mask_params`). A small mark and a huge one on the same
    image size should come out with different `INPAINT_ExpandMask.grow`."""
    source = _solid_png(2000, 2000, (255, 0, 0))
    small_mask = _rect_mask_png(2000, 2000, (975, 975, 1025, 1025))  # 50x50
    huge_mask = _rect_mask_png(2000, 2000, (100, 100, 1900, 1900))  # 1800x1800
    params = GenerationParams(
        checkpoint="anima_unet.safetensors",
        positive_prompt="a fox",
        negative_prompt="",
        loader="split",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
        anima_lllite_inpaint_patch_strength=1.0,
    )

    small_client = _StubUploadingComfyClient(source)
    await post_process(
        small_client, "fix_drawn", source, "source.png", params, mask_bytes=small_mask
    )
    small_grow = next(
        n["inputs"]["grow"]
        for n in small_client.queued_graph.values()
        if n["class_type"] == "INPAINT_ExpandMask" and n["inputs"]["blur"] > 0
    )

    huge_client = _StubUploadingComfyClient(source)
    await post_process(huge_client, "fix_drawn", source, "source.png", params, mask_bytes=huge_mask)
    huge_grow = next(
        n["inputs"]["grow"]
        for n in huge_client.queued_graph.values()
        if n["class_type"] == "INPAINT_ExpandMask" and n["inputs"]["blur"] > 0
    )

    assert huge_grow > small_grow
    # Neither should just be DrawnMaskFixParams()'s static fallback default —
    # both jobs' marks differ enough in size from the one krita reference
    # job that default was captured from.
    assert small_grow != DrawnMaskFixParams().mask_grow
    assert huge_grow != DrawnMaskFixParams().mask_grow


def _fix_artifact_override_profile() -> ModelProfile:
    return ModelProfile(
        match=["anima*aesthetic*"],
        display_name="Anima Aesthetic",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        clip_type="stable_diffusion",
        vae_name="qwen_image_vae.safetensors",
        model_sampling_shift=3.0,
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
        anima_lllite_inpaint_patch_strength=1.0,
        fix_artifact_checkpoint="anima-base-v1.0.safetensors",
        negative_prompt_prefix="worst quality, low quality",
    )


@pytest.mark.asyncio
async def test_post_process_fix_drawn_uses_fix_artifact_checkpoint_override():
    """A furrytoonmix-generated image's own checkpoint has no inpainting-aware
    path wired — "🩹 Fix Artifact" should switch to whichever profile sets
    `fix_artifact_checkpoint` (Anima Aesthetic) instead, regardless of what
    generated the image."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
    )

    await post_process(
        client,
        "fix_drawn",
        source,
        "source.png",
        params,
        mask_bytes=mask,
        profiles=[_fix_artifact_override_profile()],
    )

    unet = next(n for n in client.queued_graph.values() if n["class_type"] == "UNETLoader")
    assert unet["inputs"]["unet_name"] == "anima-base-v1.0.safetensors"
    class_types = [n["class_type"] for n in client.queued_graph.values()]
    assert "CheckpointLoaderSimple" not in class_types
    # positive_prompt is "<profile prefix>, background scenery" (no prefix
    # set on this profile, so just "background scenery") — not blank, and
    # not the original furrytoonmix image's "a fox"; negative comes from
    # the override profile too, not the original image's "blurry".
    texts = {
        n["inputs"]["text"]
        for n in client.queued_graph.values()
        if n["class_type"] == "CLIPTextEncode"
    }
    assert "background scenery" in texts
    assert "worst quality, low quality" in texts
    assert "blurry" not in texts
    assert "a fox" not in texts


@pytest.mark.asyncio
async def test_post_process_fix_drawn_without_override_profile_keeps_own_checkpoint():
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
    )

    await post_process(
        client, "fix_drawn", source, "source.png", params, mask_bytes=mask, profiles=[]
    )

    ckpt = next(
        n for n in client.queued_graph.values() if n["class_type"] == "CheckpointLoaderSimple"
    )
    assert ckpt["inputs"]["ckpt_name"] == "furrytoonmix_xlIllustriousV2.safetensors"


@pytest.mark.asyncio
async def test_post_process_fix_drawn_anima_own_checkpoint_without_override_still_gets_prompt():
    """No profile sets fix_artifact_checkpoint (so no override_base matches),
    but this image's own checkpoint already has loader='split' and
    anima_lllite_inpaint_patch set — build_fix_drawn_mask's routing check
    only looks at those two fields, so it still routes to
    _build_anima_fix_drawn_mask regardless of the missing override. That
    pipeline documents it expects a non-blank positive prompt; the old
    unconditional `positive_prompt=""` fallback would have silently
    violated that."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="anima_unet.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        vae_name="qwen_image_vae.safetensors",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
    )

    await post_process(
        client, "fix_drawn", source, "source.png", params, mask_bytes=mask, profiles=[]
    )

    texts = {
        n["inputs"]["text"]
        for n in client.queued_graph.values()
        if n["class_type"] == "CLIPTextEncode"
    }
    assert "" not in texts
    assert "a fox" not in texts


@pytest.mark.asyncio
async def test_post_process_fix_drawn_clears_positive_prompt():
    """The original positive prompt describes the whole scene, including
    whatever the user just drew a mask over to remove — conditioning the
    inpaint on it steers the model toward a nicer version of the exact
    content being removed instead of erasing it, so `fix_drawn` must not
    forward it into the graph's `CLIPTextEncode`."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt="blurry"
    )

    await post_process(client, "fix_drawn", source, "source.png", params, mask_bytes=mask)

    encodes = [n for n in client.queued_graph.values() if n["class_type"] == "CLIPTextEncode"]
    texts = {n["inputs"]["text"] for n in encodes}
    assert "a fox" not in texts
    assert "" in texts
    assert "blurry" in texts


@pytest.mark.asyncio
async def test_post_process_fix_drawn_requires_mask_bytes():
    source = _solid_png(10, 10, (255, 0, 0))
    client = _StubUploadingComfyClient(source)
    params = GenerationParams(
        checkpoint="ckpt.safetensors", positive_prompt="a fox", negative_prompt=""
    )

    with pytest.raises(AssertionError):
        await post_process(client, "fix_drawn", source, "source.png", params)


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


def test_to_post_process_base_carries_anima_lllite_inpaint_patch_fields():
    """Without this, "🩹 Fix Artifact" on an Anima checkpoint with the patch
    configured would lose it, since build_fix_drawn_mask only sees
    PostProcessBaseParams, not the original GenerationParams."""
    params = GenerationParams(
        checkpoint="anima-aesthetic-v1.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        loader="split",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
        anima_lllite_inpaint_patch_strength=0.8,
    )

    base = _to_post_process_base(params)

    assert base.anima_lllite_inpaint_patch == "anima-lllite-inpainting-v2.safetensors"
    assert base.anima_lllite_inpaint_patch_strength == 0.8


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


@pytest.mark.asyncio
async def test_post_process_fix_drawn_uses_fix_artifact_prompt_overrides():
    """When a profile sets the fix-artifact-only prompts, the removal pass
    uses those rather than the profile's generation prefixes — and still
    appends "background scenery"."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    profile = _fix_artifact_override_profile()
    profile.positive_prompt_prefix = "generation only"
    profile.negative_prompt_prefix = "generation negative"
    profile.fix_artifact_positive_prefix = "removal quality tags"
    profile.fix_artifact_negative_prefix = "removal negative"
    params = GenerationParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
    )

    await post_process(
        client, "fix_drawn", source, "source.png", params, mask_bytes=mask, profiles=[profile]
    )

    texts = [
        n["inputs"]["text"]
        for n in client.queued_graph.values()
        if n["class_type"] == "CLIPTextEncode"
    ]
    assert "removal quality tags, background scenery" in texts
    assert "removal negative" in texts
    assert not any("generation only" in t for t in texts)


@pytest.mark.asyncio
async def test_post_process_fix_drawn_falls_back_to_generation_prefixes():
    """Without the fix-artifact-only prompts set, the removal pass keeps
    using the profile's normal prefixes — the previous behaviour."""
    source = _solid_png(10, 10, (255, 0, 0))
    mask = _solid_mask_png(10, 10, 255)
    client = _StubUploadingComfyClient(source)
    profile = _fix_artifact_override_profile()
    profile.positive_prompt_prefix = "generation only"
    profile.negative_prompt_prefix = "generation negative"
    params = GenerationParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
    )

    await post_process(
        client, "fix_drawn", source, "source.png", params, mask_bytes=mask, profiles=[profile]
    )

    texts = [
        n["inputs"]["text"]
        for n in client.queued_graph.values()
        if n["class_type"] == "CLIPTextEncode"
    ]
    assert "generation only, background scenery" in texts
    assert "generation negative" in texts
