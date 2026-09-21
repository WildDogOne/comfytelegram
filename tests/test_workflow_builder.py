from comfytelegram.workflows.builder import (
    DrawnMaskHandDetailerParams,
    FaceDetailerParams,
    GenerationParams,
    HandDetailerParams,
    LoraSpec,
    ManualHandDetailerParams,
    PostProcessBaseParams,
    TiledRefineParams,
    UpscaleParams,
    _resolve_seed,
    build_face_detailer,
    build_hand_detailer,
    build_hand_detailer_drawn_mask,
    build_hand_detailer_manual,
    build_tiled_refine,
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
    clip_skip_id = next(nid for nid, n in prompt.items() if n["class_type"] == "CLIPSetLastLayer")
    text_encodes = [n for n in prompt.values() if n["class_type"] == "CLIPTextEncode"]
    assert len(text_encodes) == 2
    for node in text_encodes:
        assert node["inputs"]["clip"] == [clip_skip_id, 0]


def test_build_txt2img_split_loader_wires_unet_clip_vae_and_sampling_shift():
    params = GenerationParams(
        checkpoint="anima-aesthetic-v1.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        clip_type="stable_diffusion",
        vae_name="qwen_image_vae.safetensors",
        model_sampling_shift=3.0,
    )
    prompt, save_id = build_txt2img(params)

    types = _class_types(prompt)
    assert "CheckpointLoaderSimple" not in types
    assert types.count("UNETLoader") == 1
    assert types.count("CLIPLoader") == 1
    assert types.count("VAELoader") == 1
    assert types.count("ModelSamplingAuraFlow") == 1

    unet = next(n for n in prompt.values() if n["class_type"] == "UNETLoader")
    assert unet["inputs"]["unet_name"] == "anima-aesthetic-v1.safetensors"

    clip = next(n for n in prompt.values() if n["class_type"] == "CLIPLoader")
    assert clip["inputs"]["clip_name"] == "qwen_3_06b_base.safetensors"
    assert clip["inputs"]["type"] == "stable_diffusion"

    vae = next(n for n in prompt.values() if n["class_type"] == "VAELoader")
    assert vae["inputs"]["vae_name"] == "qwen_image_vae.safetensors"

    unet_id = next(nid for nid, n in prompt.items() if n["class_type"] == "UNETLoader")
    sampling = next(n for n in prompt.values() if n["class_type"] == "ModelSamplingAuraFlow")
    assert sampling["inputs"]["model"] == [unet_id, 0]
    assert sampling["inputs"]["shift"] == 3.0

    sampling_id = next(
        nid for nid, n in prompt.items() if n["class_type"] == "ModelSamplingAuraFlow"
    )
    ksampler = next(n for n in prompt.values() if n["class_type"] == "KSampler")
    assert ksampler["inputs"]["model"] == [sampling_id, 0]
    assert prompt[save_id]["class_type"] == "SaveImage"


def test_build_txt2img_split_loader_without_shift_skips_sampling_node():
    params = GenerationParams(
        checkpoint="anima-aesthetic-v1.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        vae_name="qwen_image_vae.safetensors",
    )
    prompt, _save_id = build_txt2img(params)
    assert "ModelSamplingAuraFlow" not in _class_types(prompt)


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


def test_build_tiled_refine_defaults_to_upscale_by_one():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_tiled_refine("uploaded.png", base, TiledRefineParams())

    refine_node = next(n for n in prompt.values() if n["class_type"] == "UltimateSDUpscale")
    assert refine_node["inputs"]["upscale_by"] == 1.0
    assert prompt[save_id]["inputs"]["images"] == [
        next(nid for nid, n in prompt.items() if n["class_type"] == "UltimateSDUpscale"),
        0,
    ]


def test_build_tiled_refine_honors_explicit_seed():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_tiled_refine("uploaded.png", base, TiledRefineParams(seed=777))
    refine_node = next(n for n in prompt.values() if n["class_type"] == "UltimateSDUpscale")
    assert refine_node["inputs"]["seed"] == 777


def test_build_tiled_refine_with_tile_controlnet_wires_apply_between_prompt_and_refine():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        tile_controlnet="xinsir_tile_sdxl.safetensors",
        tile_controlnet_strength=0.55,
    )
    prompt, _save_id = build_tiled_refine("uploaded.png", base, TiledRefineParams())

    types = _class_types(prompt)
    assert types.count("ControlNetLoader") == 1
    assert types.count("ControlNetApplyAdvanced") == 1


def test_build_face_detailer_honors_explicit_seed():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id, _detection_id = build_face_detailer(
        "uploaded.png", base, FaceDetailerParams(seed=888)
    )
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


def test_build_upscale_without_tile_controlnet_skips_the_branch():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_upscale("uploaded.png", base, UpscaleParams())

    types = _class_types(prompt)
    assert "ControlNetLoader" not in types
    assert "ControlNetApplyAdvanced" not in types

    text_encodes = {nid: n for nid, n in prompt.items() if n["class_type"] == "CLIPTextEncode"}
    positive_id = next(nid for nid, n in text_encodes.items() if n["inputs"]["text"] == "a fox")
    upscale_node = next(n for n in prompt.values() if n["class_type"] == "UltimateSDUpscale")
    assert upscale_node["inputs"]["positive"] == [positive_id, 0]


def test_build_upscale_with_tile_controlnet_wires_apply_between_prompt_and_upscale():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        tile_controlnet="xinsir_tile_sdxl.safetensors",
        tile_controlnet_strength=0.55,
    )
    prompt, _save_id = build_upscale("uploaded.png", base, UpscaleParams())

    types = _class_types(prompt)
    assert types.count("ControlNetLoader") == 1
    assert types.count("ControlNetApplyAdvanced") == 1

    loader = next(n for n in prompt.values() if n["class_type"] == "ControlNetLoader")
    assert loader["inputs"]["control_net_name"] == "xinsir_tile_sdxl.safetensors"

    load_id = next(nid for nid, n in prompt.items() if n["class_type"] == "LoadImage")
    apply = next(n for n in prompt.values() if n["class_type"] == "ControlNetApplyAdvanced")
    assert apply["inputs"]["image"] == [load_id, 0]
    assert apply["inputs"]["strength"] == 0.55

    apply_id = next(
        nid for nid, n in prompt.items() if n["class_type"] == "ControlNetApplyAdvanced"
    )
    upscale_node = next(n for n in prompt.values() if n["class_type"] == "UltimateSDUpscale")
    assert upscale_node["inputs"]["positive"] == [apply_id, 0]
    assert upscale_node["inputs"]["negative"] == [apply_id, 1]


def test_build_face_detailer_wires_detector_and_sam():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id, detection_id = build_face_detailer("uploaded.png", base, FaceDetailerParams())

    detailer_id = next(nid for nid, n in prompt.items() if n["class_type"] == "FaceDetailer")
    detailer = prompt[detailer_id]
    assert "bbox_detector" in detailer["inputs"]
    assert "sam_model_opt" in detailer["inputs"]
    assert prompt[save_id]["class_type"] == "SaveImage"
    assert prompt[save_id]["inputs"]["images"] == [detailer_id, 0]

    # The detection-check node is a *separate* PreviewImage (not SaveImage
    # — see _build_detailer's docstring for why) fed by the detailer's mask
    # output via MaskToImage — see post_process()'s _detailer_found_nothing.
    assert detection_id != save_id
    assert prompt[detection_id]["class_type"] == "PreviewImage"
    mask_image_id = prompt[detection_id]["inputs"]["images"][0]
    assert prompt[mask_image_id]["class_type"] == "MaskToImage"
    assert prompt[mask_image_id]["inputs"]["mask"] == [detailer_id, 3]


def test_build_hand_detailer_honors_explicit_seed():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id, _detection_id = build_hand_detailer(
        "uploaded.png", base, HandDetailerParams(seed=888)
    )
    detailer = next(n for n in prompt.values() if n["class_type"] == "FaceDetailer")
    assert detailer["inputs"]["seed"] == 888


def test_build_hand_detailer_wires_detector_and_sam_with_hand_bbox_model():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id, detection_id = build_hand_detailer("uploaded.png", base, HandDetailerParams())

    detector = next(n for n in prompt.values() if n["class_type"] == "UltralyticsDetectorProvider")
    assert detector["inputs"]["model_name"] == "bbox/hand_yolov8s.pt"

    detailer = next(n for n in prompt.values() if n["class_type"] == "FaceDetailer")
    assert "bbox_detector" in detailer["inputs"]
    assert "sam_model_opt" in detailer["inputs"]
    assert prompt[save_id]["class_type"] == "SaveImage"
    assert detection_id != save_id


def test_build_hand_detailer_manual_marks_a_box_centered_on_the_tapped_point():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_hand_detailer_manual(
        "uploaded.png",
        base,
        ManualHandDetailerParams(seed=888, box_size_frac=0.5),
        point_frac=(0.5, 0.5),
        image_size=(200, 100),
    )

    assert "UltralyticsDetectorProvider" not in _class_types(prompt)
    assert "SAMLoader" not in _class_types(prompt)

    base_mask, patch_mask = (n for n in prompt.values() if n["class_type"] == "SolidMask")
    assert base_mask["inputs"] == {"value": 0.0, "width": 200, "height": 100}
    box_size = int(min(200, 100) * 0.5)
    assert patch_mask["inputs"] == {"value": 1.0, "width": box_size, "height": box_size}

    composite = next(n for n in prompt.values() if n["class_type"] == "MaskComposite")
    assert composite["inputs"]["operation"] == "add"
    # Point at the exact center: the box should sit centered too.
    assert composite["inputs"]["x"] == 100 - box_size // 2
    assert composite["inputs"]["y"] == 50 - box_size // 2

    segs = next(n for n in prompt.values() if n["class_type"] == "MaskToSEGS")
    assert segs["inputs"]["mask"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "MaskComposite"
    )

    detailer = next(n for n in prompt.values() if n["class_type"] == "DetailerForEach")
    assert detailer["inputs"]["seed"] == 888
    assert detailer["inputs"]["segs"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "MaskToSEGS"
    )
    assert prompt[save_id]["class_type"] == "SaveImage"


def test_build_hand_detailer_manual_clamps_the_box_at_image_edges():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_hand_detailer_manual(
        "uploaded.png",
        base,
        ManualHandDetailerParams(box_size_frac=0.5),
        point_frac=(0.0, 0.0),
        image_size=(200, 100),
    )

    composite = next(n for n in prompt.values() if n["class_type"] == "MaskComposite")
    assert composite["inputs"]["x"] == 0
    assert composite["inputs"]["y"] == 0


def test_build_hand_detailer_drawn_mask_loads_mask_via_load_image_mask():
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_hand_detailer_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        base,
        DrawnMaskHandDetailerParams(seed=888),
    )

    assert "UltralyticsDetectorProvider" not in _class_types(prompt)
    assert "SAMLoader" not in _class_types(prompt)
    assert "SolidMask" not in _class_types(prompt)
    assert "MaskComposite" not in _class_types(prompt)

    mask_load = next(n for n in prompt.values() if n["class_type"] == "LoadImageMask")
    assert mask_load["inputs"] == {"image": "mask_uploaded.png", "channel": "red"}

    segs = next(n for n in prompt.values() if n["class_type"] == "MaskToSEGS")
    assert segs["inputs"]["mask"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "LoadImageMask"
    )

    detailer = next(n for n in prompt.values() if n["class_type"] == "DetailerForEach")
    assert detailer["inputs"]["seed"] == 888
    assert detailer["inputs"]["segs"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "MaskToSEGS"
    )
    assert prompt[save_id]["class_type"] == "SaveImage"
