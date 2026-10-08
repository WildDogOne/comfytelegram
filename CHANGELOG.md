# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Regional prompting**: give each side of the image its own prompt so two
  characters stop swapping hair, ears and outfits. Any prompt with lines
  starting `[left]`/`[right]` (etc.) is regional, no command needed; `/rp`
  explains the format and offers a 📋 template to copy. The lines above the
  first tag are the global prompt, also added to every region, so scene,
  quality tags and the character count only need writing once. Regions are `[left]`/`[right]` or `[top]`/`[bottom]`,
  plus `[center]`/`[middle]` for three. Needs the ComfyUI-ppm node pack.
  The global/region balance can be tuned per model profile
  (`regional_base_weight`/`regional_region_weight`). Place characters with
  `$name` per region; the active character is not applied to regional
  prompts. Generate Again, Show Prompt and downloaded PNGs keep the
  regions. Post-processing (upscale, detailers) uses the global prompt only.
- **`$name` in prompts**: `$alice` inserts a saved character, `$wlop` one of
  your `/favs`, in any prompt: plain messages, `/stream`, regional prompts and
  the ✏️ Detail Prompt editor. A character's negative prompt joins the
  negatives; a character wins over a favorite with the same name. An
  unknown name gets a reply listing what's saved (in Detail Prompt it's left
  out with a warning instead, so the drawn mask isn't lost). In the Detail
  Prompt editor, typing `$` opens a menu of your characters and favorites
  (needs the updated `inpaint_relay`).
- Mask editor: Alt+click (or Alt+drag) uses the opposite tool for that one
  stroke, erasing in brush mode and painting in erase mode. Works with
  Shift-click lines too. The brush cursor changes colour while Alt is held.

### Changed

- Draw Mask, Fix Artifact and Detail Prompt: the "Draw over the area…, then
  tap Done in the editor" message (with its editor button) is now deleted
  once the drawn mask arrives and generation starts.

## 0.2.0 - 2026-10-06

### Added

- **Kontext redo buttons** under every Kontext edit result, both working
  from the original, pre-edit image: **🔁 Same prompt** retries the same
  instruction with a fresh seed, and **✏️ New prompt** asks for a different
  instruction (showing the previous one, with a copy button) instead. Redo
  results get the same buttons.
- **Favorites**: a personal library for tags, artist names and short prompt
  phrases you want to reuse.
  - `/fav <category> <name> | <text> [| <note>]` saves one, and
    `/fav delete <name>` removes it. Names and categories are short
    lowercase handles, and saving under an existing name replaces it.
  - When you save a single tag, the reply shows its tag-database entry
    (source, category, post count) as a typo check.
  - `/favs` browses favorites by category. Each favorite can be copied,
    edited or deleted. Edit changes text, note, category or name one at a
    time; category can be picked from your existing categories.
  - Favorites are stored per Telegram user, not per chat, so they follow you
    across chats and topics and nobody else sees them.
- `/favs` on the main command keyboard.

## 0.1.0 - 2026-10-06

Everything the bot could do when this changelog was started. It was
reconstructed from the git history (commit messages, diffs and the tests each
commit added), one section per day, newest first.

### 2026-10-06

- Added "🪄 Kontext Edit": describe a change in plain English and FLUX.1
  Kontext edits the image. It uses its own Kontext model, not the image's
  checkpoint, and first resizes the source to about 1 megapixel because
  Kontext repeats the scene at larger sizes.
- Reorganized the README.

### 2026-10-01

- Added "🎚️ Redo…" on drawn-mask results: redo with the same mask at a
  different denoise, picked from presets or typed. Later redos keep the
  chosen value.
- Mask editor: the options under the canvas fold away to give the canvas
  more room. They start folded on phones, the editor remembers your choice,
  and a one-line summary shows what "Done" will submit.

### 2026-09-30

- Added "🔀 Switch Model" and "🎛 LoRAs" on an image. They change which
  checkpoint and LoRAs that image's later buttons (detailers, upscale,
  "🔁 Generate Again") use, keeping the prompt.
- Model profiles can skip LoRAs during detailer passes
  (`detailer_disable_lora`), and the mask editor has a toggle for it.
- Model profiles can load a separate VAE with a checkpoint.

### 2026-09-29

- Added the Banana Splitz XXL model profile.
- LoRA discovery adds LoRAs that were already identified to profiles that
  opt in later.
- Detailers use the profile's own denoise, CFG and steps, read from the
  current profile rather than the values saved with the image.
- "✏️ Detail Prompt" can condition the detailer on the tile ControlNet. A
  profile sets the default, and the mask editor has a checkbox for it.
- Fixed: renaming a forum topic and other service messages no longer get an
  "I didn't understand that" reply.

### 2026-09-28

- Mask editor: undo/redo, and Shift-click to draw a straight line from the
  last point.
- Fixed: the "🩹 Fix Artifact" crop could pull away from the image edge.

### 2026-09-27

- LoRA trigger words: set `trigger_words` on a LoRA in a model profile and
  it is added to the prompt whenever that LoRA is active. Missing keys are
  filled in automatically so they're easy to find in the JSON.

### 2026-09-26

- Added LoRA support.
  - `/lora` turns a model's LoRAs on or off and sets their strengths, per
    chat.
  - "ℹ️ Info" looks up a LoRA's trigger words and base model on CivitAI.
    Results are cached.
  - New LoRA files are found at startup, identified on CivitAI, and added to
    matching profiles.
  - `/reload` and "🔁 Reload profiles" pick up edited profiles without a
    restart.
- `docker-compose.yml` became `docker-compose.example.yml`.
- Fixed: `/tagcheck` now splits on sentence-ending periods but keeps tags
  that contain a period intact.
- Fixed: "⚙️ Customize" upscale values were lost when the "already large"
  confirmation came up. Starting an upscale on another image while one was
  waiting for custom values now tells you the first one was dropped.

### 2026-09-25

- "🔍 Upscale 4x" asks whether to use the profile defaults or "⚙️ Customize"
  (denoise and tile ControlNet strength).
- Images are sent as compressed JPEGs by default. A "🖼️ Display" toggle in
  `/settings` switches a chat back to lossless PNG.
- "✏️ Detail Prompt" moved into the mask editor: you mark the region and
  write its prompt and denoise in one place. The prompt is used once and is
  only kept for "🔁 Redo (same mask)".

### 2026-09-23

- Added "✏️ Detail Prompt", a prompt override for detailer passes.
- The post-processing buttons are split over two pages ("⋯ More" /
  "‹ Back").
- Added "📥 Download file", which sends the image as a PNG file with the
  generation settings embedded. Sending that file back to the bot restores
  all its buttons, even without a database entry. Other ComfyUI PNGs show
  their workflow; images without metadata are analyzed.
- Uploads to the mask editor are much smaller (a JPEG at full resolution)
  and have their own, longer timeout.
- Longer Telegram upload timeouts, so large upscales no longer fail at the
  final send.

### 2026-09-22

- "🩹 Fix Artifact" works more like Krita's inpainting: it crops context
  around the mask and runs a second refining pass. Profiles can set their
  own Fix Artifact prompts.
- Mask editor: "🪣 Fill" fills areas you have fully enclosed with the brush.
- Fixed: the CLIP type in the Anima profiles.

### 2026-09-21

- Added "🩹 Fix Artifact": paint over anything unwanted in the mask editor
  to remove or repaint it. Anima models use a dedicated pipeline, and a
  profile can name a different checkpoint for this.
- Added "🔁 x4" to redo a drawn-mask result four times in one tap.
- Added "🧵 Homogenize", a tiled refine pass at the image's own resolution.
- Mask editor: zoom and pan on desktop, and a sharper image when zoomed in.
- Fixed: when ComfyUI signalled "finished" too early (for example on a
  mostly cached job), the bot stopped waiting before the image existed. It
  now confirms completion against ComfyUI's history.

### 2026-09-20

- Added "🖌️ Draw Mask" for hand detailing: a mask editor that opens inside
  Telegram, backed by the separate `inpaint_relay` service. It has a brush,
  an eraser, a brush-size cursor, and two-finger zoom and pan on phones.
- Added "🔁 Redo (same mask)".
- Upscaling can use a tile ControlNet, set per profile.
- Fixed: a fast job could finish before its progress events arrived and wait
  forever; the bot now falls back to asking ComfyUI's history.

### 2026-09-19

- "🖐️ Hand Detail" offers "🤖 Auto-detect" or "✋ Tap to mark" (pick a cell on
  a grid), with "🔍 Finer grid" for small hands.
- A face or hand detailer that finds nothing now says so instead of sending
  back an unchanged image.
- "🐛 Show Prompt" shows exactly what you typed, with a copy button. The tag
  check moved to its own "🔬 Analyze Prompt" button.
- `/model` labels tell apart several versions of one model.
- Added the AutismMix SDXL model profile.
- Fixed: the hand grid preview failed for very large images.

### 2026-09-18

- Added a local tag database with `/tags` search and `/tagcheck`. It updates
  itself monthly from the danbooru/e621 archive. Search ranks by
  popularity and falls back to fuzzy matching; `/tagcheck` sees through
  weight syntax like `(tag:1.2)`.
- Fixed: in forum groups, a reply in one topic could be taken as the answer
  to a question asked in another.
- Progress messages are sent silently.

### 2026-09-17

- Characters can be edited and renamed from `/characters`.
- Line breaks in a prompt are treated like commas.
- Commands show up in Telegram's "/" menu.
- "🔁 Generate Again" now appears below the new images instead of above them.
- Added a confirmation before upscaling an image that is already large.

### 2026-09-13

- Fixed: multi-paragraph messages from newer Telegram clients were silently
  ignored.
- Unknown commands and unsupported messages (stickers, voice notes) now get
  a reply instead of silence.
- The image caption also suggests a negative prompt.

### 2026-09-12

- Added a "---" line to separate a block of negative prompt from the
  positive prompt.
- Added a "📋 Copy" button on analyzed prompts.
- Added the Anima Aesthetic and Anima Turbo model profiles.
- Fixed: when several post-processing buttons were tapped quickly, only the
  first one completed.

### 2026-09-11

- Support for models shipped as separate UNET, text encoder and VAE files
  (Anima).
- Added "🔎 Deep Analyze" with a larger vision model. Photos you upload use
  it automatically.

### 2026-09-10

- Added `/stream` and `/stop` for generating images back to back, with a
  command keyboard at the bottom of the chat.
- Negative prompts with `-tag`.
- Upload a photo to get prompts from it, each with a "🎨 Generate" button.
- Added "🖐️ Hand Detail".

### 2026-09-09

- Added "🏷️ Analyze Image": WD14 tags and a Qwen-VL caption.

### 2026-09-08

- First version: generate images from a prompt through ComfyUI, with live
  progress, `/model`, `/settings`, model profiles, "🔍 Upscale 4x" and
  "✨ Face Detail".
- Saved characters (`/character`, `/characters`).
- "🔁 Generate Again" with a new seed.
- Buttons keep working after a restart.
- Allowlist of Telegram users.
- Docker setup.
