# comfytelegram

A Telegram bot front-end for a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
installation. Send it a prompt, pick a checkpoint, and it drives ComfyUI over
HTTP + WebSocket to generate images — with per-model default settings,
one-tap upscale/face-detail/hand-detail post-processing, and an in-chat
settings menu.
It talks to ComfyUI's own API directly; it does not shell out to the `comfy`
CLI at runtime.

## Features

- **Text-to-image generation** — send any plain-text message as a prompt.
  Prefix any word with `-` to send it as a negative instead (e.g. `1girl,
  outdoors, -blurry, -watermark`), stacked on top of the checkpoint's own
  default negative prompt and any active character's.
- **Switchable models with smart defaults** — `/model` lists checkpoints
  ComfyUI has installed; each can have a matching *model profile* (cfg,
  steps, sampler, clip skip, prompt prefixes, default LoRAs) applied
  automatically. Most checkpoints are a single file loaded the normal way,
  but split-file architectures (separate UNET/text-encoder/VAE, e.g. Anima)
  are supported too via a profile's `loader: "split"` — see
  [`model_profiles/`](model_profiles/).
- **Post-processing, one tap away** — every generated image gets 🔍 Upscale
  (UltimateSDUpscale, 4x), ✨ Face Detail, and 🖐️ Hand Detail buttons (the
  latter two both Impact Pack's `FaceDetailer` node — it's a generic
  detect/crop/inpaint node despite the name, so hand-detailing is the same
  graph with a hand-trained bbox detector swapped in), plus 🏷️ Analyze Image (run
  *both* the WD14 tagger and a Qwen-VL caption, each sent as its own message
  with its own 🎨 Generate button, for comparing them side by side) and a
  🔁 Generate Again button on the whole batch. 🖐️ Hand Detail asks 🤖
  Auto-detect or ✋ Tap to mark first, since the YOLO bbox detector often
  can't find a hand at all — tapping a cell on a coarse grid overlay
  inpaints a small mask centered there instead, skipping detection
  entirely.
- **Image-to-prompt analysis** — 🏷️ Analyze Image runs both a WD14 tagger and a
  Qwen-VL caption via a local Ollama server, regardless of checkpoint, for
  comparing them side by side. See [Image analysis](#image-analysis) below.
- **Upload a photo for analysis** — send the bot any photo (not one of its
  own generated images — those use the 🏷️ Analyze Image button instead) and it
  runs the same side-by-side WD14/Qwen-VL analysis, each reply with its own
  🎨 Generate button.
- **`/settings`** — an in-place inline-keyboard menu to view and override a
  model's generation defaults per chat (steppers + presets for numeric
  fields, a live-populated grid for sampler/scheduler, free text for prompt
  prefixes), without editing any files.
- **Saved characters** — `/character save <name> | <prompt>` stores a
  reusable prompt snippet; `/characters` activates one so it's folded into
  every generation until you switch or clear it.
- **`/stream [prompt]`** — generate single images back-to-back from the
  same prompt (batch size forced to 1 regardless of the checkpoint's own
  default), sending each one immediately, until `/stop` or a 100-image hard
  limit ends it. Omit the prompt (or just tap the `/stream` button on the
  command keyboard, which can only ever send fixed text) and the bot asks
  for it as a follow-up message instead of erroring — with a "❌ Cancel"
  button to back out if you tapped it by mistake.
- **Context-aware command keyboard** — `/start` installs a persistent reply
  keyboard (not a button on one message, so it's still one tap away no
  matter how many images have since scrolled past) listing every top-level
  command. While a `/stream` is running it's swapped for a one-button
  `/stop` keyboard — no point offering `/model`/`/settings`/etc. mid-stream —
  then swapped back once the stream ends.
- **Survives restarts** — selected model, settings overrides, saved
  characters, and every post-processing button all persist in a local
  SQLite file, not memory. (A running `/stream` doesn't — it's a live
  background task, so it stops if the bot restarts.)
- **Tag search** — `/tags <query>` searches a local danbooru/e621 tag
  database and hands back tap-to-copy buttons for pasting matches straight
  into a prompt; `/tagcheck <prompt>` checks a whole prompt's tags against
  it, flagging unknown or rarely-used ones. See [Tag search](#tag-search)
  below.

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- A running ComfyUI instance reachable over HTTP/WebSocket, with these
  custom node packs installed (beyond ComfyUI's own core nodes):
  - [`ComfyUI-Impact-Pack`](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
    + `ComfyUI-Impact-Subpack` — face/hand detection/detailing
    (`FaceDetailer`, `UltralyticsDetectorProvider`, `SAMLoader`); hand
    detailing additionally needs a hand-trained bbox model staged under
    ComfyUI's `models/ultralytics/bbox/` (e.g. `hand_yolov8s.pt`) — not
    bundled with Impact Pack itself, same manual-staging caveat as the WD14
    tagger files (see [Image analysis](#image-analysis))
  - [`ComfyUI_UltimateSDUpscale`](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) —
    the 4x upscale pass
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

## Setup

```bash
uv sync                          # install dependencies
cp env.example .env              # then edit .env — see below
uv run comfytelegram
```

`.env` (copied from `env.example`) needs:

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | — | from @BotFather |
| `COMFYUI_HOST` | no | `127.0.0.1` | host ComfyUI is reachable at |
| `COMFYUI_PORT` | no | `8188` | ComfyUI's HTTP/WS port |
| `COMFYUI_USE_TLS` | no | `false` | use `https`/`wss` instead of `http`/`ws` |
| `OLLAMA_HOST` / `OLLAMA_PORT` | no | `127.0.0.1` / `11434` | Ollama server for 🏷️ Analyze Image's natural-language captioning |
| `OLLAMA_VISION_MODEL` | no | `qwen3.5:4b` | vision-capable Ollama model tag to caption with (`ollama pull` it first) — kept small by default since it has to coexist in VRAM with whatever checkpoint ComfyUI keeps resident |
| `WD14_MODEL_REPO` | no | `SmilingWolf/wd-vit-tagger-v3` | Hugging Face repo the WD14 tagger files come from (see [Image analysis](#image-analysis)) |
| `WD14_MODEL_DIR` | no | `models/wd14` | local directory holding `model.onnx` + `selected_tags.csv` |
| `WD14_TAG_THRESHOLD` | no | `0.35` | minimum WD14 tag confidence to include in a derived prompt |
| `ALLOWED_USER_IDS` | no | (empty = anyone) | comma-separated Telegram numeric user IDs allowed to use the bot |
| `INPAINT_RELAY_URL` | no | (unset = feature disabled) | base URL of a separately-deployed `inpaint_relay/` (see below) — backs **🖌️ Draw Mask** |
| `INPAINT_RELAY_SHARED_SECRET` | only if `INPAINT_RELAY_URL` is set | — | must match the relay's own `INPAINT_RELAY_SHARED_SECRET` |
| `INPAINT_POLL_INTERVAL_SECONDS` | no | `3` | how often to poll the relay for a finished mask drawing |

(The settings file is named `settings.py` rather than `config.py`, and the
template is `env.example` rather than `.env.example`, only because of a
local tooling rule blocking reads/writes of files literally named
`config.py`/`.env*` — no functional significance.)

If the bot runs on the same machine as ComfyUI, the defaults just work. If
it runs elsewhere (e.g. in a container without access to the host's
`localhost`), point `COMFYUI_HOST` at an address that actually reaches it —
see the Docker section below for the container case specifically.

## Running with Docker

A `Dockerfile` and `docker-compose.example.yml` are included:

```bash
docker compose up -d --build
```

The container uses host networking by default (so `COMFYUI_HOST=127.0.0.1`/
`OLLAMA_HOST=127.0.0.1` in `.env` reach ComfyUI/Ollama running directly on
the same host, matching the non-Docker setup above) — Linux-only. On
macOS/Windows, switch to the `extra_hosts`/`host.docker.internal`
alternative commented in `docker-compose.example.yml` (set both `COMFYUI_HOST` and
`OLLAMA_HOST` to `host.docker.internal` in that case). The `data/` directory
(sqlite state + tag database), `model_profiles/`, and `models/wd14/` are
bind-mounted from the repo, so they're the same files a bare `uv run
comfytelegram` would use — no Docker volume commands needed to inspect or
back them up, and no image rebuild needed to pick up a WD14 model you stage
later (see [Image analysis](#image-analysis)).

## Freehand mask drawing (🖌️ Draw Mask)

**🖐️ Hand Detail**'s **🖌️ Draw Mask** option opens a real freehand mask
editor inside Telegram, using [Telegram Web Apps](https://core.telegram.org/bots/webapps) —
a button that opens an HTTPS page in the Telegram client itself. This needs
a public HTTPS endpoint, which comfytelegram doesn't have on its own: it
runs on whatever box has ComfyUI, reachable only outbound (to Telegram's own
servers, via long-polling), never accepting inbound connections.

That's what `inpaint_relay/` (a separate top-level directory in this repo)
is for — a small, independently-deployed service meant to run on a
different, *publicly* reachable host (e.g. one already running Traefik for
other self-hosted services). It serves the mask-editor page and shuttles two
blobs between comfytelegram and the user's phone: the source image out, the
drawn mask back. It never talks to ComfyUI and never holds the Telegram bot
token — comfytelegram itself validates that a submitted mask genuinely came
from Telegram (via `Telegram.WebApp.initData`'s signature) after pulling it
back, since the relay has no way to check that itself. See
`inpaint_relay/README.md` for how to run/deploy it and
`inpaint_relay/docker-compose.example.yml` for the Traefik label pattern.

Because comfytelegram's own host typically isn't reachable from the public
internet either, the two sides never talk to each other directly in the
"public host calls back into comfytelegram" direction — comfytelegram polls
the relay outbound for a finished drawing (`INPAINT_POLL_INTERVAL_SECONDS`),
the same posture it already uses to poll Telegram itself. Leaving
`INPAINT_RELAY_URL` unset skips all of this — the **🖌️ Draw Mask** button is
simply omitted, and hand detailing falls back to **🤖 Auto-detect**/**✋ Tap
to mark** as before.

## Using the bot

| Command | Effect |
|---|---|
| *(plain text)* | Generate an image with the current model/settings — prefix any word with `-` (e.g. `-blurry`) to send it as a negative instead of a positive |
| *(photo upload)* | Analyze the photo with both WD14 tags and a Qwen-VL caption, each with its own 🎨 Generate button |
| `/model` | Pick a checkpoint (inline keyboard, populated live from ComfyUI) |
| `/settings` | View/change cfg, steps, sampler, scheduler, clip skip, width, height, batch size, and prompt prefixes for the current model, per chat |
| `/character save <name> \| <positive> [\| <negative>]` | Save a reusable prompt snippet |
| `/character delete <name>` | Delete one |
| `/characters` | List saved characters and activate one |
| `/stream [prompt]` | Generate single images from `<prompt>` back-to-back (up to 100), sending each immediately — asks for the prompt as a follow-up if omitted, and swaps the command keyboard for a one-tap `/stop` button for the duration |
| `/stop` | Stop a running `/stream` |
| `/start`, `/help` | Show the command summary |

Every generated image comes with inline buttons:

- **🔍 Upscale 4x** / **✨ Face Detail** / **🖐️ Hand Detail** — run that
  post-processing stage on this specific image and send the result (itself
  with its own buttons, so passes can be chained). **🖐️ Hand Detail** asks
  **🤖 Auto-detect** (the usual YOLO/SAM detector), **✋ Tap to mark** (sends
  the image back with a grid overlay and one button per cell, inpainting a
  small mask centered on whichever cell you tap), or — only when
  `INPAINT_RELAY_URL` is configured — **🖌️ Draw Mask**, which opens a
  Telegram WebApp for drawing a real freehand mask instead of a fixed box.
  See "Freehand mask drawing (🖌️ Draw Mask)" below for what that needs.
- **🏷️ Analyze Image** — analyze *this image* with **both** the WD14 tagger and a
  Qwen-VL caption (regardless of the checkpoint's `prompt_style`), replying
  with two separate messages — one per analyzer — each carrying its own
  **🎨 Generate** button to start a fresh generation from exactly that
  prompt, so you can compare them and pick one manually.
- **🔁 Generate Again** (on the "Done" status message) — re-run the *whole*
  last batch with a fresh seed, for quickly building up more variations
  without retyping the prompt.

## Image analysis

🏷️ Analyze Image runs both analyzers unconditionally, for comparing them side by
side. A checkpoint's model profile `prompt_style` field (see
[`model_profiles/`](model_profiles/)) doesn't pick which one runs here — it
only gates whether the "🔬 Analyze Prompt" button (a `/tagcheck`-style
tag-health check on the prompt this image was built from, separate from
"🐛 Show Prompt", which just dumps the raw prompt text) has anything to
check:

- **WD14 tags** — booru-style comma-separated tags, run locally through a
  WD14 tagger ONNX model via `onnxruntime`. **The model files are not
  downloaded automatically** — Hugging Face serves them (`model.onnx`,
  ~370MB) from an LFS/Xet-backed CDN on a different hostname than
  `huggingface.co` itself, which some restricted-egress hosts allow while
  blocking. Download both files from a machine with normal internet
  access and place them in `WD14_MODEL_DIR`:
  ```bash
  curl -L -o model.onnx https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx
  curl -L -o selected_tags.csv https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv
  ```
- **Qwen-VL caption** — a prose-style description plus a suggested negative
  prompt, via a vision-capable model on a local Ollama server
  (`OLLAMA_VISION_MODEL`, e.g. Qwen-VL). Needs `ollama pull <model>` done
  ahead of time and Ollama reachable at `OLLAMA_HOST`/`OLLAMA_PORT`.

## Model profiles

Each `*.json` file in [`model_profiles/`](model_profiles/) tunes generation
defaults for one checkpoint or a family of them (glob-matched against the
filename). Drop in a new file to add support for a model — no code changes
needed. Full format and worked examples in
[`model_profiles/README.md`](model_profiles/README.md).

## Tag search

`/tags <query>` and `/tagcheck <prompt>` are backed by a local sqlite tag
database (`Settings.tags_db_path`, default `data/tags.sqlite3`), built from CSVs
published by [DraconicDragon/dbr-e621-lists-archive](https://github.com/DraconicDragon/dbr-e621-lists-archive)
— the companion archive of the
[danbooru-e621-tag-list-processor](https://github.com/DraconicDragon/danbooru-e621-tag-list-processor)
project.

**It populates itself automatically** — at startup the bot checks each
source (danbooru, e621) and, in the background (doesn't delay startup),
downloads the newest CSV for any source that's missing or older than
`Settings.tag_db_max_age_days` (default 30, matching the archive's own
monthly refresh cadence). A fresh checkout self-populates on first run; an
existing install just stays current. A failed check (no network, GitHub
unreachable) logs a warning and leaves whatever was already imported —
`/tags`/`/tagcheck` just report "no tag data imported yet" until it
succeeds. Set `TAG_DB_AUTO_UPDATE=false` to disable this (e.g. an
offline/airgapped install) and manage the database entirely by hand
instead:

```bash
uv run python scripts/update_tag_db.py                  # both danbooru and e621
uv run python scripts/update_tag_db.py --source e621     # just one
uv run python scripts/update_tag_db.py --danbooru-csv /path/to/local.csv
```

The script (also useful with auto-update left on, e.g. to force a refresh
right now, or target one source) downloads the newest dated CSV from the
archive for each source — or imports a local file you built yourself with
the upstream processor via `--danbooru-csv`/`--e621-csv`, which also work
as a fully offline path if you'd rather not hit GitHub's API — and swaps it
in wholesale.

- **`/tags <query>`** searches tag names and aliases, ranked by post count
  (roughly, how well-represented a tag is in the model's training data), and
  replies with a tap-to-copy button per match. It's scoped to whichever
  dictionary the current chat's checkpoint prefers — a model profile's
  `tag_dictionary` field (`"e621"` for furry-trained checkpoints, `"danbooru"`
  for anime/manga-trained ones; see [`model_profiles/README.md`](model_profiles/README.md))
  — or both if that's unset. Override the scope for one query with a
  `danbooru:`/`e621:` prefix, e.g. `/tags e621:fox ears`.
- **`/tagcheck <prompt>`** splits a whole prompt on commas and checks each
  tag: ✅ known and reasonably common, ⚠️ known but rare (below
  `Settings.tag_rare_threshold` posts — a sign the model saw little of it
  during training), or ❌ not a recognized tag at all (with a "did you
  mean" suggestion when a close match exists). Same source scoping/override
  as `/tags`.

## Development

```bash
uv run pytest                # full test suite (pure logic, no network calls)
uv run pytest tests/test_workflow_builder.py::test_build_txt2img_basic_structure
uv run ruff check .           # lint
uv run ruff format .          # format
```

`scripts/smoke_test.py` and `scripts/smoke_test_postprocess.py` are manual
end-to-end checks against a *real* ComfyUI instance (not part of the pytest
suite, since they need one running):

```bash
uv run python scripts/smoke_test.py --checkpoint <ckpt> --prompt "a fox astronaut"
uv run python scripts/smoke_test_postprocess.py outputs/<generated>.png --checkpoint <ckpt>
```

## Architecture

`main.py` wires everything into a python-telegram-bot `Application`, storing
shared singletons (`Settings`, `ComfyClient`, loaded `ModelProfile`s,
`Storage`) in `application.bot_data`.

- **`comfy_client.py`** — thin async wrapper around ComfyUI's raw HTTP/WS
  API (`/prompt`, `/history`, `/view`, `/object_info`, `/ws`). No knowledge
  of Telegram or prompts/profiles.
- **`workflows/builder.py`** — `PromptGraph`, a small imperative builder for
  ComfyUI API-format node graphs. Builds `build_txt2img`, `build_upscale`
  (UltimateSDUpscale), `build_face_detailer` and `build_hand_detailer`
  (both Impact Pack's `FaceDetailer` node — it's a generic detect/crop/
  inpaint/composite node regardless of name, so the two share a
  `_build_detailer` graph builder and differ only in which bbox-detector
  model gets wired in), and `build_hand_detailer_manual` — a hand-only
  variant for when that bbox detector can't find one at all: instead of
  `UltralyticsDetectorProvider`/`SAMLoader`, it builds a small rectangular
  mask centered on a user-tapped point (`SolidMask`+`MaskComposite`), turns
  it into a `SEGS` region with Impact Pack's `MaskToSEGS`, and feeds that to
  `DetailerForEach` — the modular sibling of `FaceDetailer` that inpaints a
  ready-made `SEGS` instead of running its own detection. Post-processing
  graphs are freshly submitted (`LoadImage` from an uploaded source) rather
  than chained onto the original sampler run, so any single image from a
  batch can be picked for refinement independent of seed/batch state. The
  checkpoint/clip/vae wiring these all share (`_build_model_clip_vae`)
  branches on a profile's `loader` field: `"checkpoint"` (the default,
  single `CheckpointLoaderSimple` node) or `"split"`, for architectures
  shipped as separate UNET/text-encoder/VAE files (currently Anima) — see
  [`model_profiles/`](model_profiles/).
- **`profiles/`** — the `ModelProfile` pydantic schema plus a loader that
  glob-matches a checkpoint filename against every `*.json` in
  `model_profiles/` and layers profile defaults → user prompt → any
  per-chat override into a ready-to-build `GenerationParams`.
- **`generation.py`** — Telegram-independent glue: `generate()`,
  `post_process()`, `repeat()` turn (checkpoint, prompt, profile) into a
  submitted-and-collected result. `handlers.py` calls these directly;
  `scripts/smoke_test*.py` do the equivalent inline for manual checks.
- **`analysis.py`** — Telegram-independent image-to-prompt analysis backing
  🏷️ Analyze Image: `analyze_tags()` (WD14 tagger via `onnxruntime`) and
  `analyze_caption()` (Qwen-VL via a local Ollama server), always run
  together. See [Image analysis](#image-analysis) above.
- **`handlers.py`** — all Telegram-facing commands and callbacks, including
  the auto/manual choice behind 🖐️ Hand Detail (`_hand_mode_keyboard`) and
  the tap-a-grid-cell flow it can lead to (`_draw_hand_point_grid`/
  `_hand_point_keyboard`, handled by its own `hand_point_callback` since a
  tapped cell's callback_data needs a row/col alongside the result id).
- **`settings_menu.py`** — the `/settings` in-place inline-keyboard UI.
  Enum fields build their button grid from ComfyUI's live `/object_info`,
  so the menu can never offer a value the server would reject.
- **`storage.py`** — durable per-chat state in SQLite (stdlib `sqlite3`,
  deliberately no ORM/migrations framework): selected checkpoint, profile
  overrides, saved characters, and a post-processing result registry. That
  registry stores only a Telegram `file_id`
  (re-downloadable indefinitely via `bot.get_file()`) plus the resolved
  generation params — not raw image bytes — which is why buttons keep
  working after a bot restart.
- **`auth.py`** — the `ALLOWED_USER_IDS` allowlist check shared by
  `handlers.py` and `settings_menu.py`.
- **`tags/`** — the local danbooru/e621 tag database backing `/tags`/
  `/tagcheck` (`TagDatabase`, its own sqlite file, separate from
  `storage.py`'s per-chat state since this is bulk reference data
  wholesale-replaced by `scripts/update_tag_db.py`) plus the CSV importer
  for the DraconicDragon tag-list archive format. See
  [Tag search](#tag-search) above.

### Background: the reference workflow

Node wiring and default parameter values in `workflows/builder.py` mirror a
hand-built, frontend-format ComfyUI graph (`sample.json` at the repo root)
that was originally run manually. It's kept as ground truth for what "the
right settings" look like, and toggles several optional branches via node
`mode` (0 = enabled, 4 = bypassed):

- **Base generation spine (always on):** checkpoint → (optional, disabled
  by default) LoRA stack → clip skip → positive/negative prompt encode →
  sampler → VAE decode.
- **Post-processing branches:** an UltimateSDUpscale 4x tiled upscale
  (enabled), a FaceDetailer pass before and after the upscale (only the
  post-upscale one enabled), an IPAdapter style-transfer branch and a
  ControlNet openpose branch (both disabled, and not implemented as their
  own builder functions here — a possible future addition).

One important gotcha this codebase deliberately works around: many
downstream nodes in `sample.json` show `link: null` for inputs like
`model`/`clip`/`vae` in the raw JSON. That's *not* actually disconnected —
the graph relies on
[`cg-use-everywhere`](https://github.com/chrisgoringe/cg-use-everywhere)
("Anything Everywhere") broadcast nodes that inject those values into any
type-matching open input, resolved client-side at queue time. `builder.py`
wires all of this **explicitly** instead, so the bot only depends on
ComfyUI custom node packs that add real generation capability (Impact Pack,
UltimateSDUpscale), not ones that just add UI convenience.
