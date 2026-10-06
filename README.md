# comfytelegram

**Generate, fix and refine Stable Diffusion images from Telegram, using your
own [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install.**

Send the bot a prompt and it runs the generation on your ComfyUI server,
then replies with the images. Each image has buttons for the usual follow-up
work: upscale, face/hand detailing, inpainting a region you draw on your
phone, instruction-based edits with FLUX Kontext, and turning an image back
into a prompt. Each checkpoint gets its own default settings from a JSON
profile, so switching models doesn't mean re-tuning cfg/steps/sampler by
hand.

The bot talks to ComfyUI's HTTP + WebSocket API directly. It doesn't use the
`comfy` CLI, and it needs no inbound network access: it long-polls Telegram.

- [Features](#features)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running with Docker](#running-with-docker)
- [Using the bot](#using-the-bot)
- [Optional features](#optional-features)
- [Development](#development)
- [Architecture](#architecture)

## Features

- **Prompt → image.** Any plain-text message is a prompt. You can write
  negatives inline as `-blurry -watermark`, or put a whole negative block
  below a `---` line.
- **Per-model defaults.** `/model` lists the checkpoints ComfyUI has
  installed. A matching [model profile](model_profiles/README.md) supplies
  cfg, steps, sampler, clip skip, prompt prefixes and default LoRAs.
  Split-file architectures (separate UNET / text encoder / VAE, e.g. Anima)
  are supported.
- **Post-processing buttons on every image.** 🔍 Upscale 4x, 🧵 Homogenize,
  ✨ Face Detail, 🖐️ Hand Detail (auto-detect, tap a grid cell, or draw a
  mask), 🩹 Fix Artifact, ✏️ Detail Prompt and 🪄 Kontext Edit. Each result
  gets the same buttons, so passes can be chained.
- **Freehand masks in Telegram.** A small companion web app
  ([`inpaint_relay/`](inpaint_relay/README.md)) lets you paint the region to
  inpaint with your finger, with pinch-zoom and an eraser.
- **Image → prompt.** A WD14 tagger and a Qwen-VL caption (via Ollama) run
  side by side. Each result has a 🎨 Generate button that starts a new
  generation from it.
- **PNG archives you can re-import.** Every image carries its full
  generation settings in a PNG chunk. Send the file back months later and
  all of its buttons work again.
- **Chat-side tuning.** `/settings` overrides a model's defaults and `/lora`
  toggles its LoRAs, both per chat, without editing files or restarting.
- **Saved characters, streaming, tag search.** Reusable prompt snippets,
  `/stream` for back-to-back generation, and `/tags` / `/tagcheck` against a
  local danbooru/e621 tag database.
- **State survives restarts.** Chat settings and every image's buttons are
  stored in SQLite.

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A running ComfyUI instance with these custom node packs:
  - [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
    and ComfyUI-Impact-Subpack, used by the face/hand detailers and the
    drawn-mask flows
  - [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale),
    used by 🔍 Upscale and 🧵 Homogenize
- Model files the default graphs reference by name. Stage them in ComfyUI's
  usual `models/` subfolders:

  | Used by | File (default) | ComfyUI folder |
  |---|---|---|
  | Upscale / Homogenize | `RealESRGAN_x4.pth` | `upscale_models/` |
  | Face Detail | `FacesV1.pt` | `ultralytics/bbox/` |
  | Hand Detail (auto) | `hand_yolov8s.pt` | `ultralytics/bbox/` |
  | Face / Hand Detail | `sam_vit_b_01ec64.pth` | `sams/` |
  | Upscale (optional, per profile) | `tile_controlnet`, e.g. `xinsir_tile_sdxl.safetensors` | `controlnet/` |
  | 🪄 Kontext Edit (optional) | see `FLUX_KONTEXT_*` [below](#flux-kontext-edit) | `diffusion_models/`, `text_encoders/`, `vae/` |

Optional:

- [Ollama](https://ollama.com) with a vision model, for Qwen-VL captions
- WD14 tagger files, for tag-style analysis (see [Image analysis](#image-analysis))
- A public HTTPS host for `inpaint_relay/`, needed for 🖌️ Draw Mask,
  🩹 Fix Artifact and ✏️ Detail Prompt (see [Drawn masks](#drawn-masks-inpaint_relay))

## Quick start

```bash
uv sync
cp env.example .env          # set TELEGRAM_BOT_TOKEN at minimum
uv run comfytelegram
```

Open a chat with your bot, send `/start`, pick a checkpoint with `/model`,
then send a prompt:

```
1girl, standing in a field of sunflowers, golden hour, -blurry, -watermark
```

You'll see a "Generating…" message with live progress, then the image(s)
with post-processing buttons underneath.

If the bot runs on the same machine as ComfyUI, the defaults just work.
Otherwise set `COMFYUI_HOST` to an address that reaches it.

## Configuration

All settings are environment variables, read from `.env`. The template,
with comments, is [`env.example`](env.example), and
`src/comfytelegram/settings.py` is the authoritative list.

| Variable | Default | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **required** | from @BotFather |
| `ALLOWED_USER_IDS` | *(empty = anyone)* | comma-separated Telegram user IDs allowed to use the bot |
| `COMFYUI_HOST` / `COMFYUI_PORT` | `127.0.0.1` / `8188` | where ComfyUI is reachable |
| `COMFYUI_USE_TLS` | `false` | use `https`/`wss` |
| `OLLAMA_HOST` / `OLLAMA_PORT` | `127.0.0.1` / `11434` | Ollama server for Qwen-VL captions |
| `OLLAMA_VISION_MODEL` | `qwen3.5:4b` | quick caption model. Kept small so it fits in VRAM next to the checkpoint |
| `OLLAMA_DEEP_VISION_MODEL` | `qwen3.8:latest` | larger model for 🔎 Deep Analyze and uploaded photos |
| `WD14_MODEL_REPO` | `SmilingWolf/wd-vit-tagger-v3` | where the WD14 files come from |
| `WD14_MODEL_DIR` | `models/wd14` | local dir holding `model.onnx` + `selected_tags.csv` |
| `WD14_TAG_THRESHOLD` | `0.35` | minimum tag confidence |
| `COMFYUI_LORAS_DIR` | *(unset)* | readable path to ComfyUI's `models/loras`. Enables [LoRA info and auto-discovery](#lora-info-and-auto-discovery) |
| `TAG_DB_AUTO_UPDATE` | `true` | refresh the tag database in the background at startup |
| `TAG_DB_MAX_AGE_DAYS` | `30` | how stale a tag source may get before refreshing |
| `TAG_RARE_THRESHOLD` | `100` | post count below which `/tagcheck` flags a tag as rare |
| `INPAINT_RELAY_URL` | *(unset = drawn masks disabled)* | base URL of your `inpaint_relay` deployment |
| `INPAINT_RELAY_SHARED_SECRET` | — | required if `INPAINT_RELAY_URL` is set. Must match the relay's own value |
| `INPAINT_POLL_INTERVAL_SECONDS` | `3` | how often to poll the relay for a finished mask |
| `FLUX_KONTEXT_*` | see [below](#flux-kontext-edit) | model files and sampler settings for 🪄 Kontext Edit |

> The settings module is `settings.py` (not `config.py`) and the template is
> `env.example` (not `.env.example`) because of a local tooling rule that
> blocks those file names. The names don't mean anything else.

## Running with Docker

```bash
cp docker-compose.example.yml docker-compose.yml   # edit the loras mount path
docker compose up -d --build
```

- The container uses **host networking** by default (Linux only), so
  `127.0.0.1` in `.env` reaches ComfyUI and Ollama on the host. On
  macOS/Windows, switch to the commented `extra_hosts` /
  `host.docker.internal` variant and point `COMFYUI_HOST` / `OLLAMA_HOST` at
  `host.docker.internal`.
- `data/` (SQLite state + tag database), `model_profiles/` and
  `models/wd14/` are bind-mounted from the repo. A bare `uv run` uses the
  same files, and you don't need to rebuild the image after staging new
  files there.
- The `COMFYUI_LORAS_DIR` mount has no sensible default. Point it at your
  ComfyUI install's `models/loras`, or remove it.

## Using the bot

### Commands

| Command | Effect |
|---|---|
| *(plain text)* | Generate with the current model and settings |
| *(photo)* | Analyze it with WD14 tags + a deep Qwen-VL caption, each with a 🎨 Generate button |
| *(PNG sent as a file)* | Re-import an image from its embedded settings (see [Archiving](#archiving-and-re-importing)) |
| `/model` | Pick a checkpoint (live list from ComfyUI) |
| `/settings` | Per-chat overrides for cfg, steps, sampler, scheduler, clip skip, size, batch size and prompt prefixes, plus a JPEG/PNG display toggle |
| `/lora` | Toggle and re-weight the current model's configured LoRAs |
| `/reload` | Re-read `model_profiles/*.json` and scan for new LoRAs, no restart needed |
| `/character save <name> \| <positive> [\| <negative>]` | Save a reusable prompt snippet |
| `/character delete <name>` | Delete one |
| `/characters` | List, activate, edit or rename saved characters. The active one is folded into every prompt |
| `/fav <category> <name> \| <text> [\| <note>]` | Save a tag, artist name or short phrase to your personal favorites |
| `/fav delete <name>` | Delete one |
| `/favs` | Browse your favorites by category, copy, edit or delete them |
| `/stream [prompt]` | Generate single images back-to-back (max 100) until `/stop`. Asks for the prompt if you leave it out |
| `/stop` | Stop a running stream |
| `/tags <query>` | Search the tag database. Prefix `danbooru:` / `e621:` to pick a source |
| `/tagcheck <prompt>` | Flag unknown or rarely-used tags in a prompt |
| `/start`, `/help` | Show help and install the command keyboard |

### Writing prompts

```
castle on a hill, sunset, dramatic sky, -lowres
---
blurry, jpeg artifacts, extra fingers
```

`-token` pulls a single word into the negative prompt. A mid-word hyphen
such as `well-lit` is left alone. A line of three or more dashes splits the
whole message into a positive block above and a negative block below. Both
are added on top of the profile's default negative and any active
character's prompt.

### Image buttons

Every image comes with two pages of buttons. **⋯ More** / **‹ Back** switch
between them on the same message.

**Page 1: working the image**

| Button | What it does |
|---|---|
| 🔍 Upscale 4x | Tiled 4x UltimateSDUpscale. Asks whether to use the profile's defaults or a custom denoise / tile-ControlNet strength, and asks for confirmation on images that are already large |
| 🩹 Fix Artifact | Paint over anything unwanted (a stray object, a glitch, a watermark) and it is inpainted away. *Needs the relay* |
| 🪄 Kontext Edit | Describe a change ("make it night", "remove the hat") and FLUX.1 Kontext edits the whole image. Upscale and detailer buttons on the result still use the original checkpoint |
| ✨ Face Detail | Detects faces, then re-inpaints them at higher detail |
| 🖐️ Hand Detail | Gives three options: **🤖 Auto-detect** (YOLO + SAM), **✋ Tap to mark** (a grid overlay; tap the cell with the hand, or 🔍 Finer grid for 8×8), or **🖌️ Draw Mask** (*needs the relay*) |
| ✏️ Detail Prompt | Draw a region *and* say what should be there, using editable prompt, denoise and tile-ControlNet fields pre-filled from the image's own prompt. The prompt applies to this mask only and isn't saved on the image |
| 📥 Download file | Re-sends the image as an uncompressed PNG with its settings embedded |
| 🐛 Show Prompt | Shows the exact positive and negative prompt used |

**Page 2: everything else**

| Button | What it does |
|---|---|
| 🧵 Homogenize | A low-denoise whole-image pass at 1x that blends the seams left by separate detailer patches |
| 🏷️ Analyze Image | WD14 tags + a quick Qwen-VL caption, as two messages, each with 🎨 Generate (and 📋 Copy if the prompt is ≤256 chars) |
| 🔎 Deep Analyze | Same, with the larger `OLLAMA_DEEP_VISION_MODEL` |
| 🔬 Analyze Prompt | `/tagcheck` on the prompt this image was made from |
| 🔀 Switch Model | Use a different checkpoint for this image's next passes, e.g. generate with one model and detail with another |
| 🎛 LoRAs | Toggle LoRAs for this image's current checkpoint |

Results from a drawn mask get an extra row: **🔁 Redo (same mask)** and
**🔁 x4** re-run the same mask with new seeds, and **🎚️ Redo…** lets you
pick a denoise first. The "Done" message under a fresh batch has
**🔁 Generate Again**, which reruns the whole batch with a new seed.

### Archiving and re-importing

Telegram re-encodes photos to JPEG, which strips all PNG metadata. To keep
an image you can work on later:

1. Tap **📥 Download file** and save the PNG. The bot re-fetches it from
   ComfyUI's output folder, so the original must still be there.
2. Later, send that PNG back **as a file** (not as a photo).

The bot reads its `comfytelegram` `tEXt` chunk, which holds the model,
prompts (including what you originally typed), sampler settings, LoRAs and
seed. It then re-registers the image with the full button set, even on a
fresh database or a different install. The chunk is spliced in without
re-encoding, so pixels are untouched.

A PNG from plain ComfyUI (with only ComfyUI's own `prompt` chunk) gets a
summary of the workflow and a 🎨 Generate button, but no post-processing
buttons: an executed graph doesn't contain enough to rebuild the settings
reliably. A file with no metadata is analyzed like a photo. Telegram won't
hand bots files over 20 MB, so very large upscales can't be re-imported.

### Image analysis

- **WD14 tags**: booru-style tags from a local ONNX model run with
  `onnxruntime`. **The model files are not downloaded automatically.**
  Hugging Face serves them from an LFS CDN host that restricted-egress
  networks often block. Fetch them yourself into `WD14_MODEL_DIR`:

  ```bash
  mkdir -p models/wd14 && cd models/wd14
  curl -L -o model.onnx https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx
  curl -L -o selected_tags.csv https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv
  ```

- **Qwen-VL caption**: a prose description plus a suggested negative
  prompt, from Ollama. Run `ollama pull <model>` for both
  `OLLAMA_VISION_MODEL` and `OLLAMA_DEEP_VISION_MODEL` first.

Images you generate with the bot use the quick model by default (🔎 Deep
Analyze is opt-in). Uploaded photos always use the deep model, since they
don't hold up a generation. A profile's `prompt_style` doesn't pick an
analyzer. It only decides whether 🔬 Analyze Prompt has tags to check.

## Optional features

### Drawn masks (`inpaint_relay`)

🖌️ Draw Mask, 🩹 Fix Artifact and ✏️ Detail Prompt open a
[Telegram Web App](https://core.telegram.org/bots/webapps), which needs a
public HTTPS URL. The bot has none, because it only makes outbound
connections. [`inpaint_relay/`](inpaint_relay/README.md) is a small FastAPI
service you deploy on a publicly reachable host (e.g. behind Traefik). It
serves the editor and passes the source image and drawn mask back and
forth:

```
bot ──POST image──▶ relay ◀──open editor── Telegram client
bot ◀──poll mask─── relay ◀──submit mask──
```

The relay never talks to ComfyUI and never holds the bot token. The bot
checks the WebApp `initData` signature itself after pulling a mask back.
Set `INPAINT_RELAY_URL` and `INPAINT_RELAY_SHARED_SECRET` to enable it. If
`INPAINT_RELAY_URL` is unset, 🖌️ Draw Mask is hidden and 🩹 Fix Artifact /
✏️ Detail Prompt reply that mask drawing isn't configured.

### FLUX Kontext edit

🪄 Kontext Edit runs a separate FLUX.1 Kontext [dev] model. It ignores the
image's own checkpoint and LoRAs. The source is resized to the nearest
~1 MP Kontext resolution first, because larger inputs make Kontext tile the
scene instead of editing it. Defaults (from `settings.py`, matching
[`flux_kontext_sample.json`](flux_kontext_sample.json)):

| Variable | Default |
|---|---|
| `FLUX_KONTEXT_UNET` | `flux1-dev-kontext_fp8_scaled.safetensors` |
| `FLUX_KONTEXT_CLIP_L` | `clip_l.safetensors` |
| `FLUX_KONTEXT_T5XXL` | `t5xxl_fp8_e4m3fn_scaled.safetensors` |
| `FLUX_KONTEXT_VAE` | `ae.safetensors` |
| `FLUX_KONTEXT_STEPS` | `20` |
| `FLUX_KONTEXT_GUIDANCE` | `2.5` |

### Model profiles

Each `*.json` in [`model_profiles/`](model_profiles/) tunes one checkpoint
or family of checkpoints. A profile is matched against the checkpoint
filename with a case-insensitive glob, and the first match wins. Add a file
to support a new model, no code changes needed. See
[`model_profiles/README.md`](model_profiles/README.md) for the format.

### LoRA info and auto-discovery

With `COMFYUI_LORAS_DIR` set:

- `/lora` shows an **ℹ️ Info** button per LoRA. It SHA256-hashes the file
  and looks it up on CivitAI's public
  `/api/v1/model-versions/by-hash/<hash>` endpoint for trigger words and
  base model, the same lookup ComfyUI-Custom-Scripts' Info dialog does.
  Results, including "not found", are cached until you tap 🔄 Refresh.
- At startup, and on `/reload`, LoRA files that no profile mentions are
  identified on CivitAI. Each one is added, disabled, to every profile whose
  `civitai_base_models` list matches its base model. Profiles must opt in
  to this. LoRAs that aren't on CivitAI still have to be added by hand.

ComfyUI's API only exposes LoRA filenames, so neither feature works without
read access to the files.

### Tag search

`/tags` and `/tagcheck` use a local SQLite database (`data/tags.sqlite3`)
built from [DraconicDragon/dbr-e621-lists-archive](https://github.com/DraconicDragon/dbr-e621-lists-archive).
It fills itself in the background at startup and refreshes any source
older than `TAG_DB_MAX_AGE_DAYS`. A failed refresh keeps the existing data.
Each search uses the source named by the profile's `tag_dictionary`
(`danbooru` or `e621`), or both if the profile doesn't set one.

For offline installs, set `TAG_DB_AUTO_UPDATE=false` and manage the
database by hand:

```bash
uv run python scripts/update_tag_db.py                 # both sources
uv run python scripts/update_tag_db.py --source e621   # just one
uv run python scripts/update_tag_db.py --danbooru-csv /path/to/local.csv
```

`/tagcheck` marks each comma-separated tag ✅ common, ⚠️ rare (below
`TAG_RARE_THRESHOLD` posts) or ❌ unknown. Unknown tags get a "did you
mean" suggestion when one exists.

## Development

```bash
uv run pytest                 # full suite: mocked collaborators, no network or ComfyUI needed
uv run pytest tests/test_workflow_builder.py::test_build_txt2img_basic_structure
uv run ruff check .
uv run ruff format .
```

To check against a real ComfyUI instance (these scripts are not part of
pytest):

```bash
uv run python scripts/smoke_test.py --checkpoint <ckpt> --prompt "a fox astronaut"
uv run python scripts/smoke_test_postprocess.py outputs/<generated>.png --checkpoint <ckpt> --prompt "a fox astronaut"
```

## Architecture

`main.py` wires a python-telegram-bot `Application` and stores shared
singletons (`Settings`, `ComfyClient`, profiles, `Storage`) in `bot_data`.
[`CLAUDE.md`](CLAUDE.md) has the detailed per-module design notes. In
short:

| Module | Role |
|---|---|
| `comfy_client.py` | Async wrapper for ComfyUI's `/prompt`, `/history`, `/view`, `/object_info`, `/ws`. Knows nothing about Telegram |
| `workflows/builder.py` | `PromptGraph` plus one builder per graph: txt2img, upscale/homogenize, face/hand detailers, manual and drawn-mask variants, Kontext |
| `profiles/` | `ModelProfile` schema, glob matching, layering of defaults → prompt → per-chat overrides |
| `generation.py` | Telegram-independent `generate()`, `post_process()`, `repeat()`, `kontext_edit()` |
| `analysis.py` | WD14 tagging and Ollama captioning |
| `png_metadata.py`, `params_serde.py` | Embed/read the PNG settings chunk, and the dict format shared with SQLite |
| `handlers.py` | All Telegram commands, callbacks, and the drawn-mask job poller |
| `settings_menu.py`, `lora_menu.py` | Edit-in-place `/settings` and `/lora` menus |
| `civitai.py`, `lora_discovery.py` | CivitAI hash lookup and LoRA auto-registration |
| `storage.py` | Per-chat SQLite state. Stores Telegram `file_id`s plus params, not image bytes, so buttons keep working after a restart |
| `tags/` | Tag database, CSV importer and auto-updater |
| `auth.py` | `ALLOWED_USER_IDS` check and WebApp `initData` validation |
| `topics.py`, `message_text.py` | Forum-topic scoping of pending replies, and support for Telegram "rich" multi-paragraph messages |

Post-processing graphs are submitted fresh: the chosen image is uploaded
and fed in through `LoadImage`. They aren't chained onto the original
sampler run, so any single image from a batch can be refined on its own.

### Background: the reference workflow

The node wiring and default values in `workflows/builder.py` mirror
`sample.json`, a hand-built ComfyUI graph kept at the repo root as ground
truth. It contains the base spine (checkpoint → optional LoRAs → clip skip
→ prompt encode → sampler → VAE decode), an UltimateSDUpscale 4x branch,
FaceDetailer passes, and disabled IPAdapter/openpose branches that aren't
implemented here.

One gotcha: many inputs in `sample.json` show `link: null` for
`model`/`clip`/`vae`. They aren't disconnected. They're filled at queue time
by [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere)
broadcast nodes. `builder.py` wires all of them explicitly, so the bot
depends only on node packs that add real generation capability, not
UI-convenience ones.
