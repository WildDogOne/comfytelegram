#!/usr/bin/env python3
"""Manual live smoke test for OpenPose ControlNet guidance — not part of the
pytest suite (needs a running ComfyUI server). Answers two questions before
any of this goes into the bot:

1. Which strength/end_percent holds a pose on a given checkpoint without
   stiffening anatomy? Sweeps `--strengths` x `--ends` at one fixed seed.
2. Does the pose ControlNet still hold when `/rp` regions patch the model
   with `AttentionCouplePPM`? Pass `--regional` (the `/rp` syntax, `\\n` for
   line breaks) instead of `--prompt`.

Each run also renders a no-ControlNet baseline at the same seed, so
"the pose held" can be told apart from "the prompt alone would have done
this", and writes a contact sheet with the skeleton overlaid on every result.

    # test 1: a skeleton you already have
    uv run python scripts/smoke_test_pose.py \\
        --checkpoint furrytoonmix_xlIllustriousV2.safetensors \\
        --controlnet illustriousXL_v10_openpose.safetensors \\
        --pose pose.png --prompt "a fox knight, full body, simple background"

    # test 2: pose extracted from a reference image, two regions
    uv run python scripts/smoke_test_pose.py \\
        --checkpoint furrytoonmix_xlIllustriousV2.safetensors \\
        --controlnet illustriousXL_v10_openpose.safetensors \\
        --pose-from two_people.png \\
        --regional "2characters, park\\n[left] a fox, red hoodie\\n[right] a wolf, blue jacket"

The ControlNet is spliced onto the graph `build_txt2img` builds, between the
resolved positive/negative and the KSampler — the shape a `pose_controlnet`
profile field would produce. `AttentionCouplePPM`'s `base_cond` keeps the
un-ControlNet'd positive; it only reads the conditioning tensors.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import itertools
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comfytelegram.comfy_client import ComfyClient
from comfytelegram.generation import _run_graph, _upload_graph_inputs
from comfytelegram.profiles import load_profiles, resolve_generation_params, resolve_profile
from comfytelegram.regional import parse_regional_prompt
from comfytelegram.workflows import build_txt2img

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PIXELS = 1024 * 1024


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _size_for_aspect(width: int, height: int) -> tuple[int, int]:
    """~1MP at the pose's own aspect, snapped to 64 — so ControlNet's resize
    of the hint to the latent never has to crop it."""
    scale = (TARGET_PIXELS / (width * height)) ** 0.5
    return max(64, round(width * scale / 64) * 64), max(64, round(height * scale / 64) * 64)


async def _extract_pose(client: ComfyClient, source: Path, xinsir_sticks: bool) -> bytes:
    """Run DWPose over a reference image on the server and return the
    rendered skeleton PNG."""
    uploaded = await client.upload_image(source, overwrite=True)
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded["name"]}},
        "2": {
            "class_type": "DWPreprocessor",
            "inputs": {
                "image": ["1", 0],
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 1024,
                "scale_stick_for_xinsr_cn": "enable" if xinsir_sticks else "disable",
            },
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "pose_smoke"},
        },
    }
    raw, _history = await _run_graph(client, graph, "3", on_progress=None)
    return raw[0][0]


def _add_pose_controlnet(
    graph: dict[str, Any], pose_filename: str, controlnet: str, strength: float, end: float
) -> None:
    """Splice LoadImage + ControlNetLoader + ControlNetApplyAdvanced in front
    of the graph's KSampler, rewiring its positive/negative through it."""
    sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")
    next_id = max(int(node_id) for node_id in graph) + 1
    load, loader, apply = (str(next_id + i) for i in range(3))
    graph[load] = {"class_type": "LoadImage", "inputs": {"image": pose_filename}}
    graph[loader] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}}
    graph[apply] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": sampler["inputs"]["positive"],
            "negative": sampler["inputs"]["negative"],
            "control_net": [loader, 0],
            "image": [load, 0],
            "strength": strength,
            "start_percent": 0.0,
            "end_percent": end,
        },
    }
    sampler["inputs"]["positive"] = [apply, 0]
    sampler["inputs"]["negative"] = [apply, 1]


def _contact_sheet(pose: bytes, results: list[tuple[str, bytes]], thumb: int = 384) -> bytes:
    """Pose first, then every result with the skeleton blended over it
    (lighten, so black background leaves the image alone) and its label."""
    pose_img = Image.open(io.BytesIO(pose)).convert("RGB")
    tiles: list[tuple[str, Image.Image]] = [("pose", pose_img)]
    for label, data in results:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        overlay = pose_img.resize(img.size)
        blended = Image.blend(img, ImageChops.lighter(img, overlay), 0.6)
        tiles.append((label, img))
        tiles.append((f"{label} +pose", blended))

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    label_h = 22
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for i, (label, img) in enumerate(tiles):
        img = img.copy()
        img.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
        sheet.paste(img, (x + (thumb - img.width) // 2, y + label_h))
        draw.text((x + 4, y + 2), label, fill="black", font=font)
    out = io.BytesIO()
    sheet.save(out, format="PNG")
    return out.getvalue()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--controlnet", required=True, help="e.g. illustriousXL_v10_openpose.safetensors"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pose", type=Path, help="a rendered OpenPose skeleton image")
    source.add_argument(
        "--pose-from", type=Path, help="a reference image to extract a pose from (DWPose)"
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="plain prompt (test 1)")
    prompt.add_argument("--regional", help="/rp syntax, \\n for line breaks (test 2)")
    parser.add_argument("--strengths", default="0.6,0.8,1.0")
    parser.add_argument("--ends", default="0.6,0.8,1.0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--width", type=int, help="default: ~1MP at the pose's aspect")
    parser.add_argument("--height", type=int)
    parser.add_argument("--no-baseline", action="store_true", help="skip the no-ControlNet run")
    parser.add_argument(
        "--xinsir-sticks",
        action="store_true",
        help="--pose-from only: thicker limbs for xinsir's ControlNet",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pose_smoke")
    args = parser.parse_args()

    profiles = load_profiles(PROJECT_ROOT / "model_profiles")
    profile = resolve_profile(args.checkpoint, profiles)
    print(f"Resolved profile: {profile.display_name if profile else '(none — generic defaults)'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d-%H%M%S")

    async with ComfyClient(
        f"http://{args.host}:{args.port}", f"ws://{args.host}:{args.port}"
    ) as client:
        controlnets = (await client.get_node_info("ControlNetLoader"))["input"]["required"][
            "control_net_name"
        ][0]
        if args.controlnet not in controlnets:
            sys.exit(f"{args.controlnet} not on the server. Available: {controlnets}")

        if args.pose_from:
            print(f"Extracting pose from {args.pose_from} …")
            pose = await _extract_pose(client, args.pose_from, args.xinsir_sticks)
        else:
            pose = args.pose.read_bytes()
        (args.out_dir / f"{run_tag}_pose.png").write_bytes(pose)

        pose_size = Image.open(io.BytesIO(pose)).size
        width, height = _size_for_aspect(*pose_size)
        width, height = args.width or width, args.height or height
        print(f"Pose {pose_size[0]}x{pose_size[1]} -> generating at {width}x{height}")

        pose_filename = f"pose_{hashlib.sha256(pose).hexdigest()[:16]}.png"
        await client.upload_image(pose, filename=pose_filename, overwrite=True)

        overrides = {"seed": args.seed, "width": width, "height": height, "batch_size": 1}
        if args.regional:
            parsed = parse_regional_prompt(args.regional.replace("\\n", "\n"))
            params = resolve_generation_params(
                args.checkpoint,
                parsed.global_prompt,
                profile,
                overrides=overrides,
                extra_negative_prompt=parsed.negative,
                regions=parsed.regions,
            )
            print(f"Regions: {[(r.label, r.prompt) for r in params.regions]}")
        else:
            params = resolve_generation_params(
                args.checkpoint, args.prompt, profile, overrides=overrides
            )
        print(
            f"cfg={params.cfg} steps={params.steps} sampler={params.sampler_name} seed={args.seed}"
        )
        print(f"positive: {params.positive_prompt}")
        await _upload_graph_inputs(client, params)

        runs: list[tuple[str, float | None, float | None]] = []
        if not args.no_baseline:
            runs.append(("no CN", None, None))
        for strength, end in itertools.product(_floats(args.strengths), _floats(args.ends)):
            runs.append((f"s={strength} end={end}", strength, end))

        results: list[tuple[str, bytes]] = []
        for label, strength, end in runs:
            graph, save_node_id = build_txt2img(params)
            if strength is not None and end is not None:
                _add_pose_controlnet(graph, pose_filename, args.controlnet, strength, end)
            t0 = time.monotonic()
            raw, _history = await _run_graph(client, graph, save_node_id, on_progress=None)
            data = raw[0][0]
            slug = label.replace(" ", "_").replace("=", "")
            (args.out_dir / f"{run_tag}_{slug}.png").write_bytes(data)
            print(f"  {label}: {time.monotonic() - t0:.1f}s")
            results.append((label, data))

        sheet = args.out_dir / f"{run_tag}_sheet.png"
        sheet.write_bytes(_contact_sheet(pose, results))
        print(f"Contact sheet: {sheet}")


if __name__ == "__main__":
    asyncio.run(main())
