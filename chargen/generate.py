#!/usr/bin/env python3
"""Generate a varied image set of one character, as LoRA training material.

Each image combines the fixed character prompt with a randomly drawn e621
artist (category 1 in the local tag database, weighted by post_count — a
widely posted artist is far more likely to be something the model actually
learned, so the weight is squared and the pool is capped) plus a
pose/framing/view/expression/outfit/setting from `wildcards.json`. The
character itself (prompts, class tags, excluded artists) is in
`character.json`; both are per-dataset and not in git — copy the
`.example` files to start. Talks to
ComfyUI directly, like `scripts/smoke_test.py`.

    uv run python chargen/generate.py --run project_name --target 200

--exclude TAG (repeatable) leaves matching wildcard entries and artists out
of that one run, to steer refills away from values review already keeps
too many of (see dataset_report.py).

Meant to be cycled: review chargen/output/<run>/, delete (or move into a
subfolder) every image that isn't good enough, then run the same command
again — it generates exactly enough *new* images to bring the folder back up
to --target. New images always get fresh numbers past the highest one ever
generated (per manifest.jsonl), so a removed image is never redrawn.

Per image: a PNG (with comfytelegram's own metadata chunk, so it can be sent
back to the bot as a document to upscale/detail it), a same-named .txt
caption with its tags for LoRA training, and a manifest.jsonl line.

character.json's optional `loras` attach LoRAs to every image — typically
an earlier round's character LoRA, to generate a more consistent next set
(see the README's "Next round" section). They're recorded per image in
manifest.jsonl, so a mixed folder still shows which round made what.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

CHARGEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHARGEN_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image

from comfytelegram.comfy_client import ComfyClient
from comfytelegram.generation import post_process
from comfytelegram.params_serde import serialize_generation_params
from comfytelegram.png_metadata import build_metadata, embed_metadata, extract_seed
from comfytelegram.profiles import load_profiles, resolve_generation_params, resolve_profile
from comfytelegram.profiles.schema import LoraDefault, ModelProfile
from comfytelegram.workflows import GenerationParams, build_txt2img


def load_config(name: str) -> dict | list:
    """chargen/<name>, which is per-dataset and not in git; its committed
    template is <name>.example."""
    path = CHARGEN_DIR / name
    if not path.exists():
        sys.exit(f"{path} not found — copy {name}.example to {name} and edit it")
    return json.loads(path.read_text())


# The character this dataset is for — prompts, class tags, excluded artists.
# See character.json.example for every field and why its tags were chosen.
CONFIG = load_config("character.json")
CHARACTER = CONFIG["character"]
BASE_POSITIVE = CONFIG["base_positive"]
BASE_NEGATIVE = CONFIG["base_negative"]
EXTRA_POSITIVE = CONFIG["extra_positive"]
EXTRA_NEGATIVE = CONFIG["extra_negative"]
CLASS_TAGS = CONFIG["class_tags"]
# Applied to every image, at one strength for both model and clip. No
# trigger_words: the full character prompt stays in every prompt, so the
# LoRA reinforces it rather than replacing it.
LORAS = [
    LoraDefault(name=lora["name"], strength_model=lora["strength"], strength_clip=lora["strength"])
    for lora in CONFIG.get("loras", [])
]

# SDXL-native resolutions. Which ones a job may use depends on its framing
# (`allowed_formats`); among those, `balance_formats` steers each run toward
# FORMAT_TARGET, given what the folder already holds — reviewing tends to
# keep full-body (portrait) shots, which skewed the set toward portrait.
FORMATS = {"portrait": (832, 1216), "square": (1024, 1024), "landscape": (1216, 832)}
FORMAT_TARGET = {"portrait": 0.5, "square": 0.3, "landscape": 0.2}
# Full-body poses that fit a wide canvas; every other full-body pose stays
# portrait (in landscape the model crops closer and cuts off the feet).
HORIZONTAL_POSES = {"lying on back", "lying on side", "running", "jumping", "fighting stance"}

# e621 meta tags filed as artists; character.json's artist_exclude adds the
# dataset-specific ones.
ARTIST_EXCLUDE = (
    "dnp",
    "unknown",
    "anonymous",
    "conditional",
    "third-party",
    "sound_warning",
    *CONFIG["artist_exclude"],
)


def load_artists(db_path: Path, top_n: int) -> list[tuple[str, int]]:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT name, post_count FROM tag WHERE source = 'e621' AND category = 1 "
        "ORDER BY post_count DESC LIMIT ?",
        (top_n * 2,),
    ).fetchall()
    con.close()
    # Matched with underscores as spaces on both sides, so an entry can be
    # written either way ("manmosu marimo" or "manmosu_marimo").
    excluded = [x.replace("_", " ") for x in ARTIST_EXCLUDE]
    return [(n, c) for n, c in rows if not any(x in n.replace("_", " ") for x in excluded)][:top_n]


def artist_tag(name: str, *, escape: bool = True) -> str:
    # e621 tags use underscores; prompts use spaces. Parentheses are prompt
    # weighting syntax, so a disambiguator like "_(artist)" must be escaped —
    # but not in a training caption, which is read as plain text.
    text = name.replace("_", " ")
    if escape:
        text = text.replace("(", r"\(").replace(")", r"\)")
    return f"by {text}"


def tags_of(value: str) -> set[str]:
    """A wildcard entry or artist name as comparable tags: lowercase,
    underscores as spaces, the whole entry plus each comma-separated tag."""
    value = value.lower().replace("_", " ").strip()
    return {value, *(t.strip() for t in value.split(","))}


def apply_excludes(
    terms: list[str], wildcards: dict, artists: list[tuple[str, int]]
) -> tuple[dict, list[tuple[str, int]]]:
    """Drop every wildcard entry and artist matching one of --exclude's terms
    for this run — e.g. a value dataset_report.py shows surviving review far
    more often than the rest. A term matches a whole entry or any single tag
    in it ("beach" drops "beach, ocean, sand"). Prints what went; exits on a
    term that matches nothing (likely a typo) or one that would empty a
    field."""
    wanted = {t.lower().replace("_", " ").strip() for t in terms}
    used: set[str] = set()
    kept = {}
    for field, values in wildcards.items():
        kept[field] = []
        for value in values:
            hit = wanted & tags_of(value)
            if hit:
                used |= hit
                print(f"excluded:         {field}: {value}")
            else:
                kept[field].append(value)
        if not kept[field]:
            sys.exit(f"--exclude would remove every {field} entry")
    kept_artists = []
    for name, count in artists:
        hit = wanted & tags_of(name)
        if hit:
            used |= hit
            print(f"excluded:         artist: {name}")
        else:
            kept_artists.append((name, count))
    if unmatched := wanted - used:
        sys.exit(f"--exclude matched nothing: {', '.join(sorted(unmatched))}")
    return kept, kept_artists


def build_job(index: int, seed: int, artists: list[tuple[str, int]], wildcards: dict) -> dict:
    # Seeded per index so a job never depends on how many came before it.
    rng = random.Random(f"{seed}-{index}")
    names = [n for n, _ in artists]
    # Squared so the best-known artists dominate: post_count is a proxy for
    # how well the model learned a style.
    weights = [c * c for _, c in artists]
    n_artists = 1 if rng.random() < 0.7 else 2
    picked: list[str] = []
    while len(picked) < n_artists:
        a = rng.choices(names, weights=weights)[0]
        if a not in picked:
            picked.append(a)
    parts = {key: rng.choice(values) for key, values in wildcards.items()}
    width, height = FORMATS[rng.choice(allowed_formats(parts))]
    return {
        "index": index,
        "seed": rng.randrange(2**32),
        "artists": picked,
        **parts,
        "width": width,
        "height": height,
    }


def allowed_formats(job: dict) -> list[str]:
    if job["framing"].startswith("full-length"):
        return ["portrait", "landscape"] if job["pose"] in HORIZONTAL_POSES else ["portrait"]
    if job["framing"].startswith(("headshot", "close-up")):
        return ["square"]
    return list(FORMATS)


def format_of(size: tuple[int, int]) -> str:
    width, height = size
    return "portrait" if width < height else "landscape" if width > height else "square"


def balance_formats(jobs: list[dict], existing: Counter, target_total: int, seed: int) -> None:
    """Re-pick each job's canvas so the folder moves toward FORMAT_TARGET.

    Weighted by how many images each format is still short of its target
    share, plus a constant so it's a nudge rather than a hard assignment.
    The most constrained jobs (e.g. headshots, square only) go first, so the
    flexible ones are left to fill whatever gap remains."""
    need = {f: FORMAT_TARGET[f] * target_total - existing[f] for f in FORMATS}
    for job in sorted(jobs, key=lambda j: len(allowed_formats(j))):
        allowed = allowed_formats(job)
        weights = [max(0.0, need[f]) + 0.5 for f in allowed]
        rng = random.Random(f"{seed}-{job['index']}-format")
        chosen = rng.choices(allowed, weights=weights)[0]
        need[chosen] -= 1
        job["width"], job["height"] = FORMATS[chosen]


def format_summary(counts: Counter) -> str:
    total = sum(counts.values()) or 1
    return ", ".join(f"{f} {counts[f]} ({counts[f] / total:.0%})" for f in FORMATS)


def next_index(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    indices = [json.loads(line)["index"] for line in manifest_path.read_text().splitlines() if line]
    return max(indices, default=-1) + 1


def job_prompt(
    job: dict,
    *,
    escape: bool = True,
    character: str = CHARACTER,
    extra_positive: str = EXTRA_POSITIVE,
) -> str:
    return ", ".join(
        [
            extra_positive,
            character,
            *(artist_tag(a, escape=escape) for a in job["artists"]),
            job["pose"],
            job["framing"],
            job["view"],
            job["expression"],
            job["outfit"],
            job["setting"],
        ]
    )


def lora_label(lora: LoraDefault) -> str:
    return f"{lora.name} @ {lora.strength_model:g}"


def load_profile(checkpoint: str) -> ModelProfile:
    """The checkpoint's model profile, for cfg/steps/sampler/clip_skip — with
    its prompt prefixes and LoRAs replaced by this project's own base
    prompts and character.json's loras."""
    profile = resolve_profile(checkpoint, load_profiles(PROJECT_ROOT / "model_profiles"))
    if profile is None:
        sys.exit(f"No model profile matches {checkpoint}")
    return profile.model_copy(
        update={
            "positive_prompt_prefix": BASE_POSITIVE,
            "negative_prompt_prefix": BASE_NEGATIVE,
            "loras": LORAS,
        }
    )


async def render(client: ComfyClient, params: GenerationParams, *, client_id: str) -> bytes | None:
    """Run one txt2img graph; the PNG with comfytelegram metadata, or None."""
    graph, save_node_id = build_txt2img(params)
    async with client.connect_events(client_id=client_id) as events:
        prompt_id = await client.queue_prompt(graph, client_id=client_id)
        async for progress in events.watch(prompt_id):
            if progress.done:
                break

    # /history is authoritative even if the socket missed the end event.
    history = None
    for _ in range(60):
        history = await client.get_history(prompt_id)
        if history and history.get("outputs", {}).get(save_node_id):
            break
        await asyncio.sleep(1)
    images = (history or {}).get("outputs", {}).get(save_node_id, {}).get("images", [])
    if not images:
        print(f"render failed: {(history or {}).get('status')}")
        return None

    img = images[0]
    data = await client.get_image_bytes(img["filename"], img["subfolder"], img["type"])
    return embed_metadata(
        data,
        build_metadata(
            params=serialize_generation_params(params),
            kind="txt2img",
            filename=img["filename"],
            seed=extract_seed(graph),
        ),
    )


async def detail_face(
    client: ComfyClient, data: bytes, dest: Path, params: GenerationParams
) -> tuple[bytes, str]:
    """The bot's own "Face Detail" pass (Impact Pack FaceDetailer, same
    checkpoint and prompt): re-renders just the face at higher resolution,
    which restores eye detail lost at full-image scale. The undetailed
    original is kept in raw/ for comparison. Returns the image to keep and
    what happened, for the manifest."""
    raw_dir = dest.parent / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / dest.name).write_bytes(data)
    try:
        result = await post_process(client, "face", data, dest.name, params)
    except Exception as exc:  # noqa: BLE001 — a failed refinement must not lose the image
        print(f"face detail failed for {dest.name}: {exc}")
        return data, "failed"
    if result.unchanged:
        return data, "no face found"
    return result.data, "applied"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=int, default=200, help="fill the run folder up to this many images"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run", default="aussie_v1", help="output subfolder name")
    parser.add_argument("--checkpoint", default=CONFIG["checkpoint"])
    parser.add_argument(
        "--top-artists", type=int, default=200, help="pool size, by e621 post_count"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument(
        "--no-face-detail", action="store_true", help="skip the automatic face detail pass"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TAG",
        help="leave wildcard entries/artists containing this tag out of this run (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print prompts, don't generate")
    args = parser.parse_args()

    wildcards = load_config("wildcards.json")
    artists = load_artists(PROJECT_ROOT / "data" / "tags.sqlite3", args.top_artists)
    profile = load_profile(args.checkpoint)
    if args.exclude:
        wildcards, artists = apply_excludes(args.exclude, wildcards, artists)

    out_dir = CHARGEN_DIR / "output" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if not args.dry_run:
        # Captions (and finalize_captions.py's _qc reports) of images removed
        # during review.
        for caption_path in out_dir.glob("*.txt"):
            stem = caption_path.stem.removesuffix("_qc")
            if not (out_dir / f"{stem}.png").exists():
                caption_path.unlink()

    images = list(out_dir.glob("*.png"))
    have = len(images)
    needed = max(0, args.target - have)
    start = next_index(manifest_path)
    jobs = [build_job(i, args.seed, artists, wildcards) for i in range(start, start + needed)]
    existing = Counter()
    for path in images:
        with Image.open(path) as im:
            existing[format_of(im.size)] += 1
    balance_formats(jobs, existing, args.target, args.seed)
    planned = Counter(format_of((j["width"], j["height"])) for j in jobs)
    print(f"{have} images in {out_dir.name}, generating {needed} (from #{start:04d})")
    target = ", ".join(f"{f} {share:.0%}" for f, share in FORMAT_TARGET.items())
    print(f"formats now:      {format_summary(existing)}  (target: {target})")
    print(f"new images:       {format_summary(planned)}")
    print(f"after this run:   {format_summary(existing + planned)}")
    print(f"loras:            {', '.join(lora_label(l) for l in LORAS) or 'none'}")

    if args.dry_run:
        for job in jobs:
            print(f"{job['index']:04d} {job['width']}x{job['height']}  {job_prompt(job)}")
        return

    http_base, ws_base = f"http://{args.host}:{args.port}", f"ws://{args.host}:{args.port}"
    async with ComfyClient(http_base, ws_base) as client:
        if args.checkpoint not in await client.list_checkpoints():
            sys.exit(f"{args.checkpoint} is not installed on the ComfyUI server")
        installed = await client.list_loras()
        missing = [lora.name for lora in LORAS if lora.name not in installed]
        if missing:
            sys.exit(f"LoRA not installed on the ComfyUI server: {', '.join(missing)}")

        for job in jobs:
            dest = out_dir / f"{job['index']:04d}.png"
            params = resolve_generation_params(
                args.checkpoint,
                job_prompt(job),
                profile,
                overrides={
                    "seed": job["seed"],
                    "width": job["width"],
                    "height": job["height"],
                    "batch_size": 1,
                    "filename_prefix": f"chargen_{args.run}",
                },
                extra_negative_prompt=EXTRA_NEGATIVE,
                raw_positive_prompt=job_prompt(job),
            )
            t0 = time.monotonic()
            data = await render(client, params, client_id=f"chargen-{args.run}")
            if data is None:
                print(f"{job['index']:04d} FAILED")
                continue
            face_detail = "off"
            if not args.no_face_detail:
                data, face_detail = await detail_face(client, data, dest, params)
            dest.write_bytes(data)
            # LoRA training caption, kohya-style (same stem, .txt): every tag
            # the image was prompted with, minus the generic quality tags,
            # which say nothing about this image in particular.
            caption = job_prompt(job, escape=False)
            dest.with_suffix(".txt").write_text(caption + "\n")
            with manifest_path.open("a") as f:
                record = {
                    **job,
                    "file": dest.name,
                    "caption": caption,
                    "face_detail": face_detail,
                    "loras": [{"name": l.name, "strength": l.strength_model} for l in LORAS],
                    "positive": params.positive_prompt,
                    "negative": params.negative_prompt,
                }
                f.write(json.dumps(record) + "\n")
            print(
                f"{job['index']:04d} {time.monotonic() - t0:5.1f}s  "
                f"{', '.join(artist_tag(a) for a in job['artists'])} | {job['pose']} | {job['setting']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
