from comfytelegram.workflows.builder import (
    DrawnMaskFixParams,
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
    build_fix_drawn_mask,
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


def test_build_fix_drawn_mask_same_shape_as_hand_drawn_mask_plus_content_aware_fill():
    # "🩹 Fix Artifact" is the same graph shape as the hand-drawn-mask
    # detailer — nothing about detecting/inpainting a user-drawn region is
    # hand-specific — plus one extra step: a content-aware (MAT) fill of the
    # masked region before the diffusion pass, since this kind means
    # "erase", not "touch up" (see `DrawnMaskFixParams.fill_model`).
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, save_id = build_fix_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        base,
        DrawnMaskFixParams(seed=888),
        image_size=(512, 512),
        work_size=(512, 512),
    )

    assert "UltralyticsDetectorProvider" not in _class_types(prompt)
    assert "SAMLoader" not in _class_types(prompt)

    mask_load = next(n for n in prompt.values() if n["class_type"] == "LoadImageMask")
    assert mask_load["inputs"] == {"image": "mask_uploaded.png", "channel": "red"}

    fill_model = next(n for n in prompt.values() if n["class_type"] == "INPAINT_LoadInpaintModel")
    assert fill_model["inputs"]["model_name"] == DrawnMaskFixParams().fill_model

    fill = next(n for n in prompt.values() if n["class_type"] == "INPAINT_InpaintWithModel")
    assert fill["inputs"]["inpaint_model"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "INPAINT_LoadInpaintModel"
    )
    assert fill["inputs"]["image"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "LoadImage"
    )
    assert fill["inputs"]["mask"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "LoadImageMask"
    )
    assert fill["inputs"]["seed"] == 888

    detailer = next(n for n in prompt.values() if n["class_type"] == "DetailerForEach")
    assert detailer["inputs"]["seed"] == 888
    # The detailer refines the content-aware-filled image, not the raw
    # source with the unwanted content still in it.
    assert detailer["inputs"]["image"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "INPAINT_InpaintWithModel"
    )
    assert prompt[save_id]["class_type"] == "SaveImage"


def test_build_hand_detailer_drawn_mask_has_no_content_aware_fill():
    # Unlike "🩹 Fix Artifact", hand touch-up should correct the masked
    # region against the subject prompt, not structurally erase it first.
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
    )
    prompt, _save_id = build_hand_detailer_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        base,
        DrawnMaskHandDetailerParams(seed=888),
    )

    assert "INPAINT_LoadInpaintModel" not in _class_types(prompt)
    assert "INPAINT_InpaintWithModel" not in _class_types(prompt)
    detailer = next(n for n in prompt.values() if n["class_type"] == "DetailerForEach")
    assert detailer["inputs"]["image"][0] == next(
        k for k, n in prompt.items() if n["class_type"] == "LoadImage"
    )


def _anima_fix_base() -> PostProcessBaseParams:
    return PostProcessBaseParams(
        checkpoint="anima_unet.safetensors",
        positive_prompt="masterpiece, best quality, background scenery",
        negative_prompt="low quality",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        vae_name="qwen_image_vae.safetensors",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
        anima_lllite_inpaint_patch_strength=0.8,
    )


def test_build_fix_drawn_mask_routes_anima_to_dedicated_pipeline():
    # loader="split" + anima_lllite_inpaint_patch set must route through
    # _build_anima_fix_drawn_mask, not the DetailerForEach-based shared
    # helper — DetailerForEach diverged from krita-ai-diffusion's own real
    # graph for this task in ways that mattered for quality (see
    # _build_anima_fix_drawn_mask's docstring).
    prompt, _save_id = build_fix_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        _anima_fix_base(),
        DrawnMaskFixParams(seed=888),
        image_size=(512, 512),
        work_size=(512, 512),
    )
    class_types = _class_types(prompt)
    assert "DetailerForEach" not in class_types
    assert "MaskToSEGS" not in class_types
    assert "SamplerCustomAdvanced" in class_types


def test_build_anima_fix_drawn_mask_matches_krita_graph_shape():
    # Every assertion here is checked against a real krita-ai-diffusion
    # "remove object" job's actual submitted graph, pulled from this
    # server's own /history for a side-by-side same-mask/same-image
    # comparison — not inferred from source reading alone.
    base = _anima_fix_base()
    prompt, save_id = build_fix_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        base,
        DrawnMaskFixParams(seed=888),
        image_size=(4096, 4096),
        work_size=(1280, 1280),
    )

    def node_of(class_type: str) -> dict:
        return next(n for n in prompt.values() if n["class_type"] == class_type)

    def id_of(class_type: str) -> str:
        return next(k for k, n in prompt.items() if n["class_type"] == class_type)

    load_id = id_of("LoadImage")
    mask_id = id_of("LoadImageMask")

    # Everything through sampling works on the *whole* source downscaled to
    # work_size, not a crop around the mask (and not the raw, un-cropped
    # full-resolution source either) — a first version of this pipeline had
    # no resize or crop at all and was clocked running a full 30-step
    # diffusion pass over an un-cropped 4096x4096 source; a second version
    # cropped to a region around the mask instead of downscaling, which was
    # confirmed (on a real removal case) to starve the model of the
    # surrounding-scene context it needs — see this function's docstring.
    image_scale_ids = [k for k, n in prompt.items() if n["class_type"] == "ImageScale"]
    # downscale source image, downscale mask-as-image, upscale result image
    # back, upscale main-mask-as-image back
    assert len(image_scale_ids) == 4
    source_downscale = node_of("ImageScale")
    assert source_downscale["inputs"]["image"][0] == load_id
    assert (source_downscale["inputs"]["width"], source_downscale["inputs"]["height"]) == (
        1280,
        1280,
    )
    assert source_downscale["inputs"]["upscale_method"] == "lanczos"
    work_image_id = id_of("ImageScale")

    mask_to_image_ids = [k for k, n in prompt.items() if n["class_type"] == "MaskToImage"]
    assert len(mask_to_image_ids) == 2  # downscale drawn mask, upscale main mask back
    mask_as_image = node_of("MaskToImage")
    assert mask_as_image["inputs"]["mask"][0] == mask_id

    # DifferentialDiffusion wraps the raw split-loader model unconditionally.
    diff = node_of("DifferentialDiffusion")
    assert diff["inputs"]["model"][0] == id_of("UNETLoader")

    # Two separate INPAINT_ExpandMask nodes share this class_type: the main
    # grow+blur one (added first, so id_of/node_of resolve to it) and a
    # second, lightly-grown-only one feeding just the content-aware fill —
    # distinguish them by which downstream node actually consumes each.
    expand_ids = [k for k, n in prompt.items() if n["class_type"] == "INPAINT_ExpandMask"]
    assert len(expand_ids) == 2
    grow = node_of("INPAINT_ExpandMask")
    assert grow["inputs"]["grow"] == DrawnMaskFixParams().mask_grow
    assert grow["inputs"]["blur"] == DrawnMaskFixParams().mask_blur
    stabilize = node_of("INPAINT_StabilizeMask")
    assert stabilize["inputs"]["mask"][0] == id_of("INPAINT_ExpandMask")
    main_mask_id = id_of("INPAINT_StabilizeMask")
    # both mask-processing branches (main + fill) trace back to the
    # downscaled drawn mask, not the raw full-resolution one directly.
    downscaled_mask_id = grow["inputs"]["mask"][0]
    assert prompt[downscaled_mask_id]["class_type"] == "ImageToMask"

    prefill = node_of("INPAINT_InpaintWithModel")
    assert prefill["inputs"]["image"][0] == work_image_id
    fill_mask_id = prefill["inputs"]["mask"][0]
    fill_mask = prompt[fill_mask_id]
    assert fill_mask["class_type"] == "INPAINT_ExpandMask"
    assert fill_mask_id in expand_ids
    assert fill_mask_id != id_of("INPAINT_ExpandMask")  # the *other* ExpandMask node
    assert fill_mask["inputs"]["mask"][0] == downscaled_mask_id
    assert fill_mask["inputs"]["grow"] == DrawnMaskFixParams().fill_mask_grow
    assert fill_mask["inputs"]["blur"] == 0

    # ETN_control_apply's image is the *raw* (downscaled) source, not the
    # content-aware filled one — matches krita's real graph; an earlier
    # local attempt fed it the filled image instead and that was a mistake
    # to revert.
    apply = node_of("ETN_control_apply")
    assert apply["inputs"]["image"][0] == work_image_id
    assert apply["inputs"]["mask"][0] == main_mask_id
    assert apply["inputs"]["strength"] == 0.8

    # Plain VAEEncode + SetLatentNoiseMask (of the content-aware-filled
    # image), not InpaintModelConditioning — matches krita's own behavior
    # for Anima (its is_inpaint_model check excludes archs with a
    # registered inpaint ControlNet, which Anima has here).
    encode = node_of("VAEEncode")
    assert encode["inputs"]["pixels"][0] == id_of("INPAINT_InpaintWithModel")
    noise_mask = node_of("SetLatentNoiseMask")
    assert noise_mask["inputs"]["mask"][0] == main_mask_id

    # Advanced sampler chain (matches krita's own, not a plain KSampler),
    # forced denoise=1.0 regardless of DrawnMaskFixParams' own field —
    # krita only engages this ControlNet-conditioned path for Anima at
    # strength=1.0, and 0.75 left the masked region almost untouched.
    guider = node_of("CFGGuider")
    assert guider["inputs"]["model"][0] == id_of("ETN_control_apply")
    scheduler = node_of("BasicScheduler")
    assert scheduler["inputs"]["denoise"] == 1.0
    sampler = node_of("SamplerCustomAdvanced")
    assert sampler["inputs"]["guider"][0] == id_of("CFGGuider")
    assert sampler["inputs"]["latent_image"][0] == id_of("SetLatentNoiseMask")
    decode = node_of("VAEDecode")
    assert decode["inputs"]["samples"] == [id_of("SamplerCustomAdvanced"), 1]

    # Color-matched against the content-aware-filled image, excluding the
    # main mask, then scaled back up to the source's own resolution and
    # composited onto the *original* full-resolution source so pixels
    # outside the mask stay pixel-identical regardless of downscale/
    # upscale round-trip drift.
    colormatch = node_of("INPAINT_ColorMatch")
    assert colormatch["inputs"]["target"][0] == id_of("VAEDecode")
    assert colormatch["inputs"]["reference"][0] == id_of("INPAINT_InpaintWithModel")
    assert colormatch["inputs"]["exclude_mask"][0] == main_mask_id

    result_upscale = next(
        n
        for n in prompt.values()
        if n["class_type"] == "ImageScale"
        and n["inputs"]["image"][0] == id_of("INPAINT_ColorMatch")
    )
    assert (result_upscale["inputs"]["width"], result_upscale["inputs"]["height"]) == (4096, 4096)
    result_upscale_id = next(
        k
        for k, n in prompt.items()
        if n["class_type"] == "ImageScale"
        and n["inputs"]["image"][0] == id_of("INPAINT_ColorMatch")
    )

    assert any(
        n["class_type"] == "MaskToImage" and n["inputs"]["mask"][0] == main_mask_id
        for n in prompt.values()
    )
    main_mask_full_id = next(
        k for k, n in prompt.items() if n["class_type"] == "ImageToMask" and k != downscaled_mask_id
    )

    composite = node_of("ImageCompositeMasked")
    assert composite["inputs"]["destination"][0] == load_id
    assert composite["inputs"]["source"][0] == result_upscale_id
    assert composite["inputs"]["mask"][0] == main_mask_full_id
    assert (composite["inputs"]["x"], composite["inputs"]["y"]) == (0, 0)
    assert prompt[save_id]["inputs"]["images"][0] == id_of("ImageCompositeMasked")


def test_build_fix_drawn_mask_skips_anima_lllite_patch_when_unset():
    base = PostProcessBaseParams(
        checkpoint="anima_unet.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        loader="split",
        clip_name="qwen_3_06b_base.safetensors",
        vae_name="qwen_image_vae.safetensors",
    )
    prompt, _save_id = build_fix_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        base,
        DrawnMaskFixParams(seed=888),
        image_size=(512, 512),
        work_size=(512, 512),
    )

    assert "ETN_control_load" not in _class_types(prompt)
    assert "ETN_control_apply" not in _class_types(prompt)


def test_build_fix_drawn_mask_skips_anima_lllite_patch_for_checkpoint_loader():
    # anima_lllite_inpaint_patch set but loader="checkpoint" (a mismatched or
    # stale profile) must not wire ETN_control_apply against a regular
    # CheckpointLoaderSimple model.
    base = PostProcessBaseParams(
        checkpoint="furrytoonmix_xlIllustriousV2.safetensors",
        positive_prompt="a fox",
        negative_prompt="low quality",
        anima_lllite_inpaint_patch="anima-lllite-inpainting-v2.safetensors",
    )
    prompt, _save_id = build_fix_drawn_mask(
        "uploaded.png",
        "mask_uploaded.png",
        base,
        DrawnMaskFixParams(seed=888),
        image_size=(512, 512),
        work_size=(512, 512),
    )

    assert "ETN_control_load" not in _class_types(prompt)
    assert "ETN_control_apply" not in _class_types(prompt)


def test_drawn_mask_fix_params_default_to_more_aggressive_than_hand():
    # Removing an arbitrary artifact needs more creative latitude and more
    # surrounding context to blend against than a hand touch-up does.
    hand_defaults = DrawnMaskHandDetailerParams()
    fix_defaults = DrawnMaskFixParams()
    assert fix_defaults.denoise > hand_defaults.denoise
    assert fix_defaults.crop_factor > hand_defaults.crop_factor
