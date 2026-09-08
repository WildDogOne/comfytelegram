# comfytelegram

A Telegram bot front-end for a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
installation. Send it a prompt, pick a checkpoint, and it drives ComfyUI over
HTTP + WebSocket to generate images — with per-model default settings,
one-tap upscale/face-detail post-processing, and an in-chat settings menu.
It talks to ComfyUI's own API directly; it does not shell out to the `comfy`
CLI at runtime.

## Features

- **Text-to-image generation** — send any plain-text message as a prompt.
- **Switchable models with smart defaults** — `/model` lists checkpoints
  ComfyUI has installed; each can have a matching *model profile* (cfg,
  steps, sampler, clip skip, prompt prefixes, default LoRAs) applied
  automatically. See [`model_profiles/`](model_profiles/).
- **Post-processing, one tap away** — every generated image gets 🔍 Upscale
  (UltimateSDUpscale, 4x) and ✨ Face Detail (Impact Pack FaceDetailer)
  buttons, plus 🔁 Regenerate (same settings, fresh seed) and a 🔁 Generate
  Again button on the whole batch.
- **`/settings`** — an in-place inline-keyboard menu to view and override a
  model's generation defaults per chat (steppers + presets for numeric
  fields, a live-populated grid for sampler/scheduler, free text for prompt
  prefixes), without editing any files.
- **Saved characters** — `/character save <name> | <prompt>` stores a
  reusable prompt snippet; `/characters` activates one so it's folded into
  every generation until you switch or clear it.
- **Survives restarts** — selected model, settings overrides, saved
  characters, and every post-processing/regenerate button all persist in a
  local SQLite file, not memory.

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- A running ComfyUI instance reachable over HTTP/WebSocket, with these
  custom node packs installed (beyond ComfyUI's own core nodes):
  - [`ComfyUI-Impact-Pack`](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
    + `ComfyUI-Impact-Subpack` — face detection/detailing (`FaceDetailer`,
    `UltralyticsDetectorProvider`, `SAMLoader`)
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
| `ALLOWED_USER_IDS` | no | (empty = anyone) | comma-separated Telegram numeric user IDs allowed to use the bot |

(The settings file is named `settings.py` rather than `config.py`, and the
template is `env.example` rather than `.env.example`, only because of a
local tooling rule blocking reads/writes of files literally named
`config.py`/`.env*` — no functional significance.)

If the bot runs on the same machine as ComfyUI, the defaults just work. If
it runs elsewhere (e.g. in a container without access to the host's
`localhost`), point `COMFYUI_HOST` at an address that actually reaches it —
see the Docker section below for the container case specifically.

## Running with Docker

A `Dockerfile` and `docker-compose.yml` are included:

```bash
touch state.sqlite3   # only needed once, before the very first run
docker compose up -d --build
```

The container uses host networking by default (so `COMFYUI_HOST=127.0.0.1`
in `.env` reaches ComfyUI running directly on the same host, matching the
non-Docker setup above) — Linux-only. On macOS/Windows, switch to the
`extra_hosts`/`host.docker.internal` alternative commented in
`docker-compose.yml`. The sqlite state file and `model_profiles/` are bind
-mounted from the repo, so they're the same files a bare `uv run
comfytelegram` would use — no Docker volume commands needed to inspect or
back them up.

## Using the bot

| Command | Effect |
|---|---|
| *(plain text)* | Generate an image with the current model/settings |
| `/model` | Pick a checkpoint (inline keyboard, populated live from ComfyUI) |
| `/settings` | View/change cfg, steps, sampler, scheduler, clip skip, width, height, batch size, and prompt prefixes for the current model, per chat |
| `/character save <name> \| <positive> [\| <negative>]` | Save a reusable prompt snippet |
| `/character delete <name>` | Delete one |
| `/characters` | List saved characters and activate one |
| `/start`, `/help` | Show the command summary |

Every generated image comes with inline buttons:

- **🔍 Upscale 4x** / **✨ Face Detail** — run that post-processing stage on
  this specific image and send the result (itself with its own buttons, so
  passes can be chained).
- **🔁 Regenerate** — re-run *this image's* generation with the same
  resolved settings but a fresh random seed.
- **🔁 Generate Again** (on the "Done" status message) — re-run the *whole*
  last batch with a fresh seed, for quickly building up more variations
  without retyping the prompt.

## Model profiles

Each `*.json` file in [`model_profiles/`](model_profiles/) tunes generation
defaults for one checkpoint or a family of them (glob-matched against the
filename). Drop in a new file to add support for a model — no code changes
needed. Full format and worked examples in
[`model_profiles/README.md`](model_profiles/README.md).

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
  ComfyUI API-format node graphs. Builds three graphs: `build_txt2img`,
  `build_upscale` (UltimateSDUpscale), `build_face_detailer` (Impact Pack
  FaceDetailer). Post-processing graphs are freshly submitted (`LoadImage`
  from an uploaded source) rather than chained onto the original sampler
  run, so any single image from a batch can be picked for refinement
  independent of seed/batch state.
- **`profiles/`** — the `ModelProfile` pydantic schema plus a loader that
  glob-matches a checkpoint filename against every `*.json` in
  `model_profiles/` and layers profile defaults → user prompt → any
  per-chat override into a ready-to-build `GenerationParams`.
- **`generation.py`** — Telegram-independent glue: `generate()`,
  `post_process()`, `regenerate()`, `repeat()` turn (checkpoint, prompt,
  profile) into a submitted-and-collected result. `handlers.py` calls these
  directly; `scripts/smoke_test*.py` do the equivalent inline for manual
  checks.
- **`handlers.py`** — all Telegram-facing commands and callbacks.
- **`settings_menu.py`** — the `/settings` in-place inline-keyboard UI.
  Enum fields build their button grid from ComfyUI's live `/object_info`,
  so the menu can never offer a value the server would reject.
- **`storage.py`** — durable per-chat state in SQLite (stdlib `sqlite3`,
  deliberately no ORM/migrations framework): selected checkpoint, profile
  overrides, saved characters, and a post-processing/regenerate result
  registry. That registry stores only a Telegram `file_id`
  (re-downloadable indefinitely via `bot.get_file()`) plus the resolved
  generation params — not raw image bytes — which is why buttons keep
  working after a bot restart.
- **`auth.py`** — the `ALLOWED_USER_IDS` allowlist check shared by
  `handlers.py` and `settings_menu.py`.

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
