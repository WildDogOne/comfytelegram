from comfytelegram.workflows.builder import (
    FaceDetailerParams,
    GenerationParams,
    HandDetailerParams,
    LoraSpec,
    PostProcessBaseParams,
    UpscaleParams,
    _resolve_seed,
    build_face_detailer,
    build_hand_detailer,
    build_txt2img,
    build_upscale,
)


def _class_types(prompt: dict) -> list[str]:
    return [node["class_type"] for node in prompt.values()]


def test_build_txt2img_basic_structure():
    params = GenerationParams(
        checkpoint="sd_xl_base_1.0.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        seed=42,
    )
    prompt, save_id = build_txt2img(params)

    assert prompt[save_id]["class_type"] == "SaveImage"
    types = _class_types(prompt)
    assert types.count("CheckpointLoaderSimple") == 1
    assert types.count("LoraLoader") == 0
    assert "CLIPSetLastLayer" not in types  # clip_skip default (-1) means "don't add the node"
    assert types.count("KSampler") == 1
    assert types.count("VAEDecode") == 1

    ksampler = next(n for n in prompt.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["seed"] == 42
    assert ksampler["inputs"]["cfg"] == params.cfg
    assert ksampler["inputs"]["steps"] == params.steps


def test_build_txt2img_with_loras_and_clip_skip_chains_correctly():
    params = GenerationParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a wolf",
        negative_prompt="",
        clip_skip=-2,
        loras=[
            LoraSpec(name="a.safetensors", strength_model=0.8, strength_clip=1.0),
            LoraSpec(name="b.safetensors", strength_model=0.5, strength_clip=0.5),
        ],
    )
    prompt, _save_id = build_txt2img(params)

    lora_nodes = [n for n in prompt.values() if n["class_type"] == "LoraLoader"]
    assert len(lora_nodes) == 2

    ckpt_id = next(nid for nid, n in prompt.items() if n["class_type"] == "CheckpointLoaderSimple")
    first_lora = next(n for n in lora_nodes if n["inputs"]["model"] == [ckpt_id, 0])
    assert first_lora["inputs"]["lora_name"] == "a.safetensors"

    clip_skip_node = next(n for n in prompt.values() if n["class_type"] == "CLIPSetLastLayer")
    assert clip_skip_node["inputs"]["stop_at_clip_layer"] == -2

    # positive/negative CLIPTextEncode must read from the clip-skip output, not raw checkpoint clip
    clip_skip_id = next(
        nid for nid, n in prompt.items() if n["class_type"] == "CLIPSetLastLayer"
    )
    text_encodes = [n for n in prompt.values() if n["class_type"] == "CLIPTextEncode"]
    assert len(text_encodes) == 2
    for node in text_encodes:
        assert node["inputs"]["clip"] == [clip_skip_id, 0]


def test_resolved_seed_is_random_when_unset():
    params = GenerationParams(checkpoint="x.safetensors", positive_prompt="p", negative_prompt="n")
    seed_a = params.resolved_seed()
    seed_b = params.resolved_seed()
    # not asserting inequality (a 1/2^32 flake is possible but absurd) — just that it's an int in range
    assert isinstance(seed_a, int) and 0 <= seed_a < 2**32
    assert isinstance(seed_b, int) and 0 <= seed_b < 2**32


def test_resolve_seed_returns_given_seed_unchanged():
    assert _resolve_seed(12345) == 12345
    assert _resolve_seed(0) == 0


def test_resolve_seed_generates_random_int_when_none():
    seed = _resolve_seed(None)
    assert isinstance(seed, int) and 0 <= seed < 2**32


def test_build_upscale_honors_explicit_seed():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_upscale("uploaded.png", base, UpscaleParams(seed=777))
    upscale_node = next(n for n in prompt.values() if n["class_type"] == "UltimateSDUpscale")
    assert upscale_node["inputs"]["seed"] == 777


def test_build_face_detailer_honors_explicit_seed():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_face_detailer("uploaded.png", base, FaceDetailerParams(seed=888))
    detailer = next(n for n in prompt.values() if n["class_type"] == "FaceDetailer")
    assert detailer["inputs"]["seed"] == 888


def test_build_upscale_wires_source_image_and_saves():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_upscale("uploaded.png", base, UpscaleParams())

    load_node = next(n for n in prompt.values() if n["class_type"] == "LoadImage")
    assert load_node["inputs"]["image"] == "uploaded.png"

    upscale_node = next(n for n in prompt.values() if n["class_type"] == "UltimateSDUpscale")
    load_id = next(nid for nid, n in prompt.items() if n["class_type"] == "LoadImage")
    assert upscale_node["inputs"]["image"] == [load_id, 0]
    assert prompt[save_id]["inputs"]["images"] == [
        next(nid for nid, n in prompt.items() if n["class_type"] == "UltimateSDUpscale"),
        0,
    ]


def test_build_face_detailer_wires_detector_and_sam():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_face_detailer("uploaded.png", base, FaceDetailerParams())

    detailer = next(n for n in prompt.values() if n["class_type"] == "FaceDetailer")
    assert "bbox_detector" in detailer["inputs"]
    assert "sam_model_opt" in detailer["inputs"]
    assert prompt[save_id]["class_type"] == "SaveImage"


def test_build_hand_detailer_honors_explicit_seed():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_hand_detailer("uploaded.png", base, HandDetailerParams(seed=888))
    detailer = next(n for n in prompt.values() if n["class_type"] == "FaceDetailer")
    assert detailer["inputs"]["seed"] == 888


def test_build_hand_detailer_wires_detector_and_sam_with_hand_bbox_model():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_hand_detailer("uploaded.png", base, HandDetailerParams())

    detector = next(n for n in prompt.values() if n["class_type"] == "UltralyticsDetectorProvider")
    assert detector["inputs"]["model_name"] == "bbox/hand_yolov8s.pt"

    detailer = next(n for n in prompt.values() if n["class_type"] == "FaceDetailer")
    assert "bbox_detector" in detailer["inputs"]
    assert "sam_model_opt" in detailer["inputs"]
    assert prompt[save_id]["class_type"] == "SaveImage"
