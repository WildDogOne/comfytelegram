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

## 4. ComfyUI integration layer (bot runtime) — HTTP/WS client done + live-verified, not wired to Telegram yet
- [x] `src/comfytelegram/comfy_client.py` — `ComfyClient`: `queue_prompt`, `get_history`, `get_image_bytes`, `upload_image`, `list_checkpoints`/`list_loras` (via `/object_info`), `watch()` (async-generator progress stream over the websocket), `run_and_collect()` convenience wrapper
- [x] **Verified against the real server**: `queue_prompt`/`watch`/`get_history`/`get_image_bytes`/`upload_image`/`list_checkpoints` all exercised in `scripts/smoke_test*.py` runs (see section 0 note); websocket `progress`/`executing(node=None)` event shape confirmed correct as-implemented
- [ ] Glue function: checkpoint + user prompt + profile → `GenerationParams` (have this, via `profiles.resolve_generation_params`) → submitted prompt dict (have this, via `workflows.build_txt2img`) → not yet composed into one call the bot handler invokes (the smoke-test scripts do this inline; the bot needs its own equivalent, likely a small `generation.py` service module)
- [ ] Error handling: translate `ComfyUIError` / connection failures into user-facing Telegram messages (not started)
- [x] Output retrieval verified (`get_image_bytes`, via the smoke tests)

## 5. Telegram bot core
- [ ] Pick library (python-telegram-bot recommended) and bootstrap bot app + config loading
- [ ] `/start` / `/help` command
- [ ] Model selection: `/model` command with inline-keyboard picker populated from checkpoints available on the ComfyUI instance (+ persists user's last-selected model)
- [ ] Prompt input flow: plain text message (or `/generate <prompt>`) triggers a run using the selected model's profile defaults
- [ ] Basic per-user/per-chat settings (current model, maybe override cfg/steps) — decide storage (in-memory vs sqlite)
- [ ] Progress feedback while a job runs (edit message with status/progress %)

## 6. Generation + post-processing UX
- [ ] On job completion, send the generated image(s) back to the user
- [ ] Attach inline keyboard per image/result set: "Upscale", "Face detail", (stretch) "Style transfer", "Pose control"
- [ ] Wire button callbacks → compose + run the corresponding post-processing fragment against the selected image
- [ ] Send post-processed result as a follow-up message; consider allowing chaining (upscale → then face-detail)
- [ ] Handle multiple images per batch cleanly (per-image button rows, track which result a callback refers to)

## 7. State & persistence
- [ ] Decide storage for: user model preference, in-flight job → chat/message mapping (for progress edits + button callbacks), generation history (optional)
- [ ] Lightweight solution first (sqlite or JSON files) — avoid over-engineering

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
