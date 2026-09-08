#!/usr/bin/env python3
"""Manual live smoke test against a real ComfyUI instance.

Not part of the pytest suite (needs a running server) — run it by hand:

    uv run python scripts/smoke_test.py --checkpoint furrytoonmix_xlIllustriousV2.safetensors \
        --prompt "a fox astronaut, dynamic pose"

Builds a profile-aware txt2img request, submits it, watches progress on
stdout, and downloads the first output image to outputs/.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comfytelegram.comfy_client import ComfyClient
from comfytelegram.profiles import (
    load_profiles,
    resolve_generation_params,
    resolve_profile,
)
from comfytelegram.workflows import build_txt2img

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    profiles = load_profiles(PROJECT_ROOT / "model_profiles")
    profile = resolve_profile(args.checkpoint, profiles)
    print(f"Resolved profile: {profile.display_name if profile else '(none — using generic defaults)'}")

    params = resolve_generation_params(args.checkpoint, args.prompt, profile)
    print(f"cfg={params.cfg} steps={params.steps} sampler={params.sampler_name} "
          f"scheduler={params.scheduler} clip_skip={params.clip_skip}")
    print(f"positive: {params.positive_prompt}")
    print(f"negative: {params.negative_prompt}")

    prompt_graph, save_node_id = build_txt2img(params)
    print(f"Built graph with {len(prompt_graph)} nodes, save node id={save_node_id}")

    http_base = f"http://{args.host}:{args.port}"
    ws_base = f"ws://{args.host}:{args.port}"

    async with ComfyClient(http_base, ws_base) as client:
        checkpoints = await client.list_checkpoints()
        if args.checkpoint not in checkpoints:
            print(f"WARNING: {args.checkpoint} not in server's checkpoint list: {checkpoints}")

        t0 = time.monotonic()
        client_id = "smoke-test"
        prompt_id = await client.queue_prompt(prompt_graph, client_id=client_id)
        print(f"Queued prompt_id={prompt_id}")

        async for progress in client.watch(prompt_id, client_id=client_id):
            if progress.done:
                print(f"Done in {time.monotonic() - t0:.1f}s")
                break
            if progress.value is not None:
                print(f"  node={progress.node_id} step {progress.value}/{progress.max}")

        history = await client.get_history(prompt_id)
        if history is None:
            print("ERROR: no history entry found after completion")
            return
        status = history.get("status", {})
        print(f"status: {status.get('status_str')}")

        outputs = history.get("outputs", {})
        save_output = outputs.get(save_node_id, {})
        images = save_output.get("images", [])
        if not images:
            print(f"ERROR: no images in outputs for save node {save_node_id}. Full outputs: {outputs}")
            return

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            data = await client.get_image_bytes(img["filename"], img["subfolder"], img["type"])
            dest = out_dir / f"smoke_test_{prompt_id[:8]}_{i}.png"
            dest.write_bytes(data)
            print(f"Saved {dest} ({len(data)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
