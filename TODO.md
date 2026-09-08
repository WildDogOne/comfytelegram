# comfytelegram — TODO

Telegram bot that drives a local ComfyUI install: one core txt2img workflow,
switchable models with per-model smart defaults, and a post-processing
step the user opts into per generated image.

**Stack decisions (locked in):** Python bot, talks to ComfyUI directly over
its HTTP + WebSocket API (no shelling out to the `comfy` CLI at runtime).

**Dev-environment note:** this coding session runs in a container; it could
not originally reach the host's ComfyUI instance (private-network address).
Fixed by relaunching the container in host-network mode, after which
`127.0.0.1:8188` was directly reachable with no ComfyUI-side change needed.
The `comfy` CLI itself still isn't installed here, so workflow construction
was built as plain Python (`workflows/builder.py`) rather than via
`comfy workflow decompose`/fragments/blueprints — that turned out to be a
reasonable fit anyway, since the bot assembles a graph per request (variable
LoRA count, optional post-processing) rather than running one fixed compiled
workflow.

**Live-verified 2026-09-08** against the real ComfyUI instance (v0.34.0,
RTX 4090): `build_txt2img()` end-to-end with the `furrytoonmix_illustrious`
profile applied, plus both `build_upscale()` and `build_face_detailer()` run
against that output. All three succeeded; node names/required-input names
matched `/object_info` on the first try (reverse-engineering from
`sample.json` held up), except `UltimateSDUpscale` needed an explicit
`batch_size` — ComfyUI's `/prompt` validation does **not** fall back to a
node's declared default for a missing `required` input, it hard-rejects with
`required_input_missing`; fixed in `builder.py`. Repro scripts kept at
`scripts/smoke_test.py` and `scripts/smoke_test_postprocess.py` for re-runs
after future changes.

---

## 0. Reference: what `sample.json` actually contains

Captured here so it doesn't need re-deriving. `sample.json` is a hand-built,
frontend-format ComfyUI graph the user runs manually, with several optional
branches toggled on/off via node `mode` (0 = enabled, 4 = bypassed).

**Base generation spine (always on):**
- `CheckpointLoaderSimple` (id 4) — currently `furrytoonmix_xlIllustriousV2.safetensors` (Illustrious/SDXL family)
- 3× `LoraLoader` chained (ids 92, 89, 336) — all currently *disabled* (mode 4); stack is model+clip in/out chained
- `CLIPSetLastLayer` (id 90) — clip skip fixed at `-2`
- Prompt assembly: several `Text Multiline` (WAS suite) fragments — quality tags, character tags, style tags, scene tags — joined via `Text Concatenate` nodes into one positive and one negative string, then `CLIPTextEncode` (ids 6/7)
- Core sampler: a **subgraph node** (id 712, custom UUID type) wrapping what is effectively KSampler + VAEDecode — exposes `steps`, `cfg`, `seed` as widgets, takes `model`/`positive`/`negative`/`vae`, outputs `images`
- Shared control values: `PrimitiveFloat` "CFG" (id 711, =5) and `PrimitiveInt` "Steps" (id 708, =40) fan out to the sampler *and* to the post-processing nodes below, so cfg/steps stay in sync across stages

**Post-processing branches (mix of enabled/disabled today):**
- `UltimateSDUpscale` (id 585, **enabled**) — 4x tiled upscale using `RealESRGAN_x4.pth`
- `FaceDetailer` (id 663, disabled) — pre-upscale face pass
- `FaceDetailer` (id 590, **enabled**) — post-upscale face pass, uses `UltralyticsDetectorProvider` (`bbox/FacesV1.pt`) + `SAMLoader` (`sam_vit_b_01ec64.pth`)
- `IPAdapterAdvanced` (id 109, disabled) — style-transfer from 4 batched reference images (`joga01-04`) via `IPAdapterModelLoader` (`ip-adapter_sdxl_vit-h`) + `CLIPVisionLoader`
- ControlNet openpose branch (ids 550/552/551/557/558/559/560, all disabled) — pose reference image → `ControlNetApplyAdvanced` → `unCLIPConditioning`
- Two `Image Save` (WAS) nodes write dated filenames (`..._4x_refined`, `..._4x_face_refiner_refined`); a `PlaySound` node fires on final save

**Important gotcha:** many downstream nodes (FaceDetailer, UltimateSDUpscale)
show `link: null` for `model`/`clip`/`vae`/etc. in this raw JSON. That's not
actually disconnected — the graph relies on **`cg-use-everywhere` ("Anything
Everywhere") broadcast nodes** (ids 77, 104, 133, 132, 652, 83, 277) that
inject VAE/UPSCALE_MODEL/CLIP_VISION/IPADAPTER/BBOX_DETECTOR/SAM_MODEL/MODEL/CLIP
into any type-matching open input, resolved client-side at queue time. When
decomposing this into fragments, resolve these broadcasts into **explicit**
wiring (e.g. queue once from the ComfyUI UI and pull the API-format prompt
history, or manually match types) — don't assume a `null` link means unused.

**Custom node packs required on the target ComfyUI install:**
`cg-use-everywhere`, `comfyui_ipadapter_plus`, `comfyui-impact-pack`,
`comfyui-impact-subpack`, `comfyui_ultimatesdupscale`, `was-node-suite-comfyui`,
`comfyui-custom-scripts`.

---

## 1. Project setup — done
- [x] Standard Python package: `src/comfytelegram/`, `pyproject.toml`, `hatchling` build backend
- [x] Dependency management via `uv` (`uv add` / `uv add --dev`); installed: `python-telegram-bot`, `aiohttp`, `pydantic`, `pydantic-settings`, `jsonschema`; dev: `pytest`, `pytest-asyncio`, `ruff`
- [x] Env-based config: `src/comfytelegram/settings.py` (a global permission rule blocks writing/reading files literally named `config.py` or `.env*` — hence `settings.py` + `env.example` as the template filename instead of `.env.example`; copy it to `.env` yourself)
- [x] `.gitignore` additions (`/outputs/`, `/state.sqlite3` — `.env`/`.venv`/`__pycache__` were already covered by the existing template)
- [x] ~~`comfy project init`~~ — n/a, see the dev-environment note above; a Python graph builder replaces the fragments/blueprints flow for this project

## 2. Workflow construction — base + two post-processing stages done, live-verified
- [x] `src/comfytelegram/workflows/builder.py` — `PromptGraph` (tiny API-format graph builder) + `build_txt2img()` (checkpoint → N chained LoRAs → optional CLIPSetLastLayer → positive/negative CLIPTextEncode → KSampler → VAEDecode → SaveImage)
- [x] `build_upscale()` — LoadImage → checkpoint/LoRA/clip-skip/prompt → UpscaleModelLoader → UltimateSDUpscale → SaveImage (defaults mirror sample.json node 585)
- [x] `build_face_detailer()` — LoadImage → checkpoint/LoRA/clip-skip/prompt → UltralyticsDetectorProvider + SAMLoader → FaceDetailer → SaveImage (defaults mirror sample.json node 590)
- [x] Anything-Everywhere broadcasts resolved: post-processing graphs wire `model`/`clip`/`vae`/positive/negative explicitly instead of relying on broadcast nodes
- [x] **Validated against the real ComfyUI instance** — see the live-verified note above; only fix needed was `UltimateSDUpscale`'s `batch_size`
- [ ] (Stretch) IPAdapter style-reference branch as its own builder function
- [ ] (Stretch) ControlNet openpose branch as its own builder function

## 3. Model profile system — done
- [x] Schema as a pydantic model: `src/comfytelegram/profiles/schema.py` (`ModelProfile`, `ProfileDefaults`, `LoraDefault`)
- [x] Loader: `src/comfytelegram/profiles/loader.py` — `load_profiles()` scans a directory, `resolve_profile()` matches checkpoint filename via case-insensitive glob, `resolve_generation_params()` layers profile defaults → user prompt → explicit overrides into a `GenerationParams`
- [x] Example profiles in `model_profiles/`: `furrytoonmix_illustrious.json` (values grounded in `sample.json` itself), `sdxl_base.json`, `ponyxl.json`, `animagine_xl.json` (community-recommended starting points, flagged as unverified in their `description`) — format documented in `model_profiles/README.md`
- [x] User-editable: any `*.json` dropped into `model_profiles/` is picked up with no code change; invalid files are logged and skipped, not fatal

## 4. ComfyUI integration layer (bot runtime) — done, live-verified
- [x] `src/comfytelegram/comfy_client.py` — `ComfyClient`: `queue_prompt`, `get_history`, `get_image_bytes`, `upload_image`, `list_checkpoints`/`list_loras`/`list_samplers`/`list_schedulers` (all via `/object_info`), `watch()` (async-generator progress stream over the websocket), `run_and_collect()` convenience wrapper
- [x] **Verified against the real server**: `queue_prompt`/`watch`/`get_history`/`get_image_bytes`/`upload_image`/`list_checkpoints` all exercised in `scripts/smoke_test*.py` runs and in the live Telegram bot (see section 0 note); websocket `progress`/`executing(node=None)` event shape confirmed correct as-implemented
- [x] Glue module: `src/comfytelegram/generation.py` — `generate()` and `post_process()` turn (checkpoint, prompt, profile) into a submitted-and-collected result; `handlers.py` calls these directly
- [x] Error handling: `ComfyUIError`/unexpected exceptions caught in `handlers.py` and reported back into the chat; a bot-wide `add_error_handler` in `main.py` catches anything that still slips through (added after a real bug — Telegram's sendPhoto 10MB limit rejected a 4x upscale and failed silently with no handler registered)
- [x] Output retrieval verified (`get_image_bytes`), including the large-file fallback (`reply_document` when a result exceeds Telegram's 10MB photo limit)

## 5. Telegram bot core — done
- [x] python-telegram-bot (v22, async); `src/comfytelegram/main.py` bootstraps `Application`, wires `Settings`/`Storage`/profiles/state into `bot_data`, registers handlers, runs polling
- [x] `/start` / `/help`
- [x] `/model` — inline-keyboard picker built from `ComfyClient.list_checkpoints()`, labeled with the matching profile's `display_name` where one exists; selection persists via `Storage`
- [x] Plain-text message → `generate_message` handler: resolves the chat's checkpoint + profile (+ any stored override) and runs `generation.generate()`
- [x] Per-chat settings persist across restarts — see section 7
- [x] Progress feedback: the "Generating…" message is edited in place with step/percent, throttled to avoid Telegram's edit rate limit

## 6. Generation + post-processing UX — done
- [x] Generated image(s) sent back per-image (not a media group, so each can carry its own buttons), with 🔍 Upscale / ✨ Face Detail inline buttons
- [x] Button callbacks (`postprocess_callback`) run the matching post-processing graph against the selected image
- [x] Result sent as a follow-up message; chaining works naturally since post-processed results also get their own Upscale/Face-Detail buttons
- [x] Multi-image batches handled correctly — each image gets its own pending-result row (see section 7) and its own keyboard
- [ ] (Stretch, not started) Style-transfer / pose-control buttons — waiting on the IPAdapter/ControlNet builder functions from section 2

## 7. State & persistence — done
- [x] `src/comfytelegram/storage.py` — sqlite (`state.sqlite3`, gitignored), two tables: per-chat selected checkpoint, per-(chat, checkpoint) profile-default overrides. Deliberately primitive: stdlib `sqlite3`, no migrations/ORM. Added after the first live test surfaced that model selection was in-memory only and lost on every restart.
- [x] In-flight job → chat/message mapping doesn't need its own storage — python-telegram-bot's `Message` object returned from `reply_text` already carries what's needed to edit it later, held as a local closure variable per request (see `generate_message`'s `status_message`)
- [x] **Fixed:** post-processing buttons ("🔍 Upscale") used to live only in an in-memory registry (`state.py`, since deleted) with the reasoning "no point persisting bytes you can't reproduce" — which missed that Telegram already stores the sent file. Now `storage.py`'s `pending_result` table keeps just the Telegram `file_id` (re-downloadable indefinitely via `bot.get_file`) plus the generation params, so a bot restart no longer breaks old "Upscale"/"Face Detail" buttons. 30-day TTL, pruned opportunistically on writes. (Buttons sent *before* this fix still can't recover — their metadata genuinely never got persisted; only new results benefit.)
- [x] `/settings` — an in-place, edited inline-keyboard menu for viewing/changing the per-chat profile overrides (`src/comfytelegram/settings_menu.py`), replacing an earlier `/override <field>=<value>` command that worked but wasn't user-friendly. Numeric fields get stepper + preset buttons; enum fields (sampler/scheduler) get a button grid sourced live from ComfyUI's `/object_info` so it can never offer an invalid value; both fall back to free-text "custom value" entry. Design rationale + sources are in the conversation that led here.

## 8. Config, secrets, deployment
- [x] Telegram bot token via `.env` (template: `env.example`, `settings.py` reads it via `pydantic-settings`), never committed
- [x] ComfyUI connection settings (host/port/tls) via `settings.py`, defaulting to local `127.0.0.1:8188`
- [ ] **Decide where the bot process actually runs.** It needs to reach ComfyUI's host/port — simplest is running the bot directly on the same machine as ComfyUI, which sidesteps the container-networking question from this dev session entirely. If it needs to run elsewhere (another container, a different box), that host/port will need to actually route to ComfyUI — figure this out before writing deployment docs
- [ ] Document how to run: start ComfyUI locally, then start the bot
- [ ] (Later) process supervision — systemd unit or simple restart wrapper

## 9. Testing / validation
- [x] Unit tests for workflow-graph construction (`tests/test_workflow_builder.py`) and model-profile resolution (`tests/test_profiles.py`) — 12 tests, all passing, all pure-logic (no ComfyUI/Telegram network calls)
- [x] Manual smoke test: generate with the `furrytoonmix_illustrious` profile applied — confirmed cfg/steps/clip_skip/prompt-prefix all landed correctly in the actual sampled image (`scripts/smoke_test.py`)
- [x] Manual smoke test: full post-processing round trip (generate → upscale, generate → face detail) — both succeeded (`scripts/smoke_test_postprocess.py`)
- [ ] Still to do once other model profiles have a matching checkpoint on the target install: repeat with SDXL base / PonyXL / Animagine to confirm those profiles too
