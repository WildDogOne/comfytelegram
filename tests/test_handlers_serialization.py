from comfytelegram.handlers import _deserialize_generation_params, _serialize_generation_params
from comfytelegram.workflows import GenerationParams, LoraSpec


def _sample_params() -> GenerationParams:
    return GenerationParams(
        checkpoint="ckpt.safetensors",
        positive_prompt="a fox",
        negative_prompt="blurry",
        steps=25,
        cfg=6.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        width=896,
        height=1152,
        batch_size=2,
        clip_skip=-2,
        loras=[LoraSpec(name="a.safetensors", strength_model=0.8, strength_clip=1.0)],
    )


def test_generation_params_roundtrip_through_serialization():
    params = _sample_params()
    data = _serialize_generation_params(params)
    restored = _deserialize_generation_params(data)

    assert restored.checkpoint == params.checkpoint
    assert restored.positive_prompt == params.positive_prompt
    assert restored.negative_prompt == params.negative_prompt
    assert restored.steps == params.steps
    assert restored.cfg == params.cfg
    assert restored.sampler_name == params.sampler_name
    assert restored.scheduler == params.scheduler
    assert restored.width == params.width
    assert restored.height == params.height
    assert restored.batch_size == params.batch_size
    assert restored.clip_skip == params.clip_skip
    assert restored.loras == params.loras
    assert restored.loader == params.loader
    assert restored.clip_name == params.clip_name
    assert restored.clip_type == params.clip_type
    assert restored.vae_name == params.vae_name
    assert restored.model_sampling_shift == params.model_sampling_shift


def test_split_loader_params_roundtrip_through_serialization():
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
    restored = _deserialize_generation_params(_serialize_generation_params(params))
    assert restored.loader == "split"
    assert restored.clip_name == "qwen_3_06b_base.safetensors"
    assert restored.clip_type == "stable_diffusion"
    assert restored.vae_name == "qwen_image_vae.safetensors"
    assert restored.model_sampling_shift == 3.0


def test_serialization_omits_seed_so_regenerate_gets_a_fresh_roll():
    data = _serialize_generation_params(_sample_params())
    assert "seed" not in data


def test_deserialize_falls_back_to_defaults_for_pre_regenerate_rows():
    # rows written before the "Regenerate" button existed only stored the
    # post-processing subset of fields (checkpoint/prompts/clip_skip/loras)
    legacy_row = {
        "checkpoint": "ckpt.safetensors",
        "positive_prompt": "a fox",
        "negative_prompt": "blurry",
        "clip_skip": -2,
        "loras": [],
    }
    restored = _deserialize_generation_params(legacy_row)
    assert restored.checkpoint == "ckpt.safetensors"
    assert restored.clip_skip == -2
    assert restored.steps == 30  # GenerationParams' own generic default
    assert restored.cfg == 7.0
    assert restored.loader == "checkpoint"  # pre-split-loader rows default to the old behavior
