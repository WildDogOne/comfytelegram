import pytest

from comfytelegram.comfy_client import JobProgress
from comfytelegram.generation import _to_post_process_base, generate
from comfytelegram.profiles import ModelProfile, ProfileDefaults
from comfytelegram.workflows import GenerationParams, LoraSpec


class _StubComfyClient:
    """Minimal stand-in for `ComfyClient` — just enough of the queue/watch/
    history/download surface for `generate()`'s `_run_graph` call to run
    end to end without a live ComfyUI server."""

    def __init__(self) -> None:
        self.queued_graph: dict | None = None

    async def queue_prompt(self, prompt: dict, *, client_id: str) -> str:
        self.queued_graph = prompt
        return "prompt-1"

    async def watch(self, prompt_id: str, *, client_id: str):
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
