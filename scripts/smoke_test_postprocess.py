#!/usr/bin/env python3
"""Manual live smoke test for the post-processing stages (upscale, face detail).

Uploads a local image to ComfyUI's input folder, then runs both post-processing
graphs against it and downloads the results.

    uv run python scripts/smoke_test_postprocess.py outputs/smoke_test_492c3f2b_0.png \
        --checkpoint furrytoonmix_xlIllustriousV2.safetensors \
        --prompt "a fox astronaut, dynamic pose"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comfytelegram.comfy_client import ComfyClient
from comfytelegram.profiles import load_profiles, resolve_profile
from comfytelegram.workflows import (
    FaceDetailerParams,
    PostProcessBaseParams,
    UpscaleParams,
    build_face_detailer,
    build_upscale,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run_stage(client: ComfyClient, name: str, prompt_graph: dict, save_node_id: str, out_dir: Path) -> None:
    t0 = time.monotonic()
    client_id = f"smoke-{name}"
    prompt_id = await client.queue_prompt(prompt_graph, client_id=client_id)
    print(f"[{name}] queued prompt_id={prompt_id}")

    async for progress in client.watch(prompt_id, client_id=client_id):
        if progress.done:
            print(f"[{name}] done in {time.monotonic() - t0:.1f}s")
            break
        if progress.value is not None:
            print(f"[{name}]   node={progress.node_id} step {progress.value}/{progress.max}")

    history = await client.get_history(prompt_id)
    if history is None:
        print(f"[{name}] ERROR: no history entry")
        return
    status = history.get("status", {})
    print(f"[{name}] status: {status.get('status_str')}")
    if status.get("status_str") != "success":
        print(f"[{name}] full status: {status}")
        return

    outputs = history.get("outputs", {})
    images = outputs.get(save_node_id, {}).get("images", [])
    if not images:
        print(f"[{name}] ERROR: no images. outputs={outputs}")
        return
    for i, img in enumerate(images):
        data = await client.get_image_bytes(img["filename"], img["subfolder"], img["type"])
        dest = out_dir / f"smoke_{name}_{prompt_id[:8]}_{i}.png"
        dest.write_bytes(data)
        print(f"[{name}] saved {dest} ({len(data)} bytes)")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_image", help="Local path to an already-generated image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--stage", choices=["upscale", "face", "both"], default="both")
    args = parser.parse_args()

    profiles = load_profiles(PROJECT_ROOT / "model_profiles")
    profile = resolve_profile(args.checkpoint, profiles)
    base = PostProcessBaseParams(
        checkpoint=args.checkpoint,
        positive_prompt=args.prompt,
        negative_prompt=args.negative or (profile.negative_prompt_prefix if profile else ""),
        clip_skip=(profile.defaults.clip_skip if profile and profile.defaults.clip_skip is not None else -1),
    )

    http_base = f"http://{args.host}:{args.port}"
    ws_base = f"ws://{args.host}:{args.port}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with ComfyClient(http_base, ws_base) as client:
        upload_result = await client.upload_image(Path(args.source_image))
        uploaded_name = upload_result["name"]
        print(f"Uploaded as: {uploaded_name} (server response: {upload_result})")

        if args.stage in ("upscale", "both"):
            prompt_graph, save_id = build_upscale(uploaded_name, base, UpscaleParams())
            await run_stage(client, "upscale", prompt_graph, save_id, out_dir)

        if args.stage in ("face", "both"):
            prompt_graph, save_id = build_face_detailer(uploaded_name, base, FaceDetailerParams())
            await run_stage(client, "face", prompt_graph, save_id, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
