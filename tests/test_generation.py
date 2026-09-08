from comfytelegram.generation import _to_post_process_base
from comfytelegram.workflows import GenerationParams, LoraSpec


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
