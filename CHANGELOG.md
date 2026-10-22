# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **`<ar:2:3>` in a prompt** picks that image's aspect ratio (also
  `<ar:16x9>`, `<ar:1.5>`, `<aspectratio:…>`). It keeps the model's usual
  pixel count, so `<ar:2:3>` on a 1024×1024 model gives 832×1280. Works in
  plain prompts, `/stream`, `/rp` (global part or any region line) and
  inside a saved favorite; **🔁 Generate Again** keeps the size. It wins
  over a `/pose` skeleton's shape, with a note when the two don't match.
  Ratios past 4:1 are rejected.

- **🦴 Pose strength…** can compare several strengths at once: tick any
  number of presets (✅), then **▶ Send** to render the image once per
  strength, lowest first. **✏️ Custom…** takes several values too
  (`0.45 0.65`), added to the ticked ones. The chat-wide strength under
  `/pose` → **🎚 Strength** still applies one value on tap.

- **Coloured buttons**: green for the button that starts or confirms the
  work (🔁 Generate Again, 🎨 Generate, ✅ Apply/Use Defaults/Upscale
  anyway, the editor buttons), red for delete/clear/reset (and the
  `/stop` key while streaming), blue for the current choice in a picker.
  Older Telegram apps show them as before.

- **🦴 `/pose`**: send an OpenPose skeleton (coloured limbs on black) after
  `/pose`, or as a photo captioned `/pose`, and every image in the chat
  follows it until `/pose off`. Images take the pose's aspect ratio, the
  progress message says "with pose 🦴" while one is active, and
  **🔁 Generate Again** keeps the pose the image was made with. Works with
  regional prompts: write `[left]`/`[right]` for the people in the pose
  and each character lands on their own skeleton. Needs `pose_controlnet`
  in the model profile — set for FurryToonMix and Banana Splitz
  (`illustriousXL_v10_openpose`); other checkpoints say so and generate
  without the pose.
- **🦴 Pose detection and editor**: send `/pose` a photo of people posing
  instead of a skeleton and it finds everyone's pose (DWPose, run locally —
  stage its two model files, see `env.example`). **🦴 Copy pose** (under
  "⋯ More") does the same for a generated image: same pose, new prompt.
  **✏️ Edit pose** opens a touch skeleton editor in the mask-editor relay,
  over the photo the pose came from: drag joints, drag a body to move that
  person, add, remove, mirror, resize and rotate people, hide joints that
  aren't visible, two fingers to zoom. Without a photo it starts from a
  standing figure on a blank canvas. The canvas shape (and so the
  generated image's) can be changed either way — over a photo, the photo
  is fitted inside with black around it, never stretched. If nobody
  is detected, the editor opens over the photo so you can place the
  skeleton yourself. Every person is drawn in the same OpenPose colours on
  purpose — the colour says which limb, the ControlNet tells people apart
  by position.
- **🦴 Pose strength…** under every image generated with a pose: pick a
  strength (0.3–1.0, or type one up to 2) and that image is rendered again
  with the same seed, so only the pose's hold changes. A "Done" message
  follows it with **🔁 Generate Again** for a full new batch at that
  strength. The pick stays
  with that image and what's made from it; new images use the chat's
  pose strength, which `/pose` shows with a **🎚 Strength** button to set
  it (until `/pose off`).
- **🎨 Paint Regions** (under "⋯ More"): paint the two regions of a
  regional prompt instead of splitting the picture into straight halves.
  Generate something like "2boys, cuddling", tap the button, and the mask
  editor opens over that image with the left half already painted. Fix
  the line where the characters meet, write a prompt for the red side and
  one for the blue rest (plus the whole-image prompt), then Done. You get
  one image with the same checkpoint, settings and noise as the image you
  painted over — even if it was the third image of a batch, which takes
  as long as generating that batch again — so it tends to keep that
  layout; **🔁 Generate Again** then tries your usual batch
  with new seeds, same painting. This also works on a "📥 Download file"
  PNG sent back to the bot later: the file now records which image of its
  batch it was, next to the seed. Paint a character *plus roughly their half* —
  the blue side covers everything you didn't paint, background included.
  `$names` and `<lora:…>` work as in `/rp`; a region's LoRA is applied
  afterwards like any region LoRA, and now follows the painted line
  instead of a straight box edge.
- **LoRAs in the prompt**: write `<lora:name>` (or `<lora:name:0.8>`, or
  `<lora:name:0.8:0.5>` for model/clip strength) anywhere in a prompt to
  switch on one of the checkpoint's LoRAs for that image, trigger words
  included. Folders work (`<lora:chars/alice>`), `:0` switches off one
  that's on by default, and a saved character or favorite can carry its
  own tag. Only LoRAs listed under `/lora` for the checkpoint can be used.
- **A LoRA per character in regional prompts**: put a `<lora:…>` in a
  region (`[left] <lora:alice>, red hoodie`) and that character gets its
  own LoRA instead of it bleeding onto everyone. It's applied after the
  image is generated: tap **🎭 Region LoRAs** under the image, pick the
  region, and the mask editor opens with that character already masked
  and the region's prompt and LoRA filled in — fix the mask where it's off
  (mostly where the characters touch), then Done. Do the next region on
  the result. **🤖 All, automatically** skips the checking, and
  `/settings` → "🎭 Region LoRAs: Automatic" does that for every new image
  right away.
- **LoRAs in ✏️ Detail Prompt**: type `<` in the editor's prompt fields to
  pick a LoRA — any part of its name finds it (`<marlow`, even `<mlw3`), no
  need to type its folder. It's added to the ones the image was made with for that one
  detail pass (the editor now lists those); with "Disable LoRAs" checked,
  the typed ones replace them instead. 🔁 Redo keeps the same LoRAs.
- **Bring back an edited image**: edited one of the bot's images in another
  app? Send the edit as a reply to the original (or to its 📥 Download file
  copy) and it gets the original's settings and all its buttons back, even
  if the editor wiped the embedded metadata. Works as a photo or as a file,
  and skips the automatic image analysis. The downloaded copy now has the
  buttons too.
- **✨ Smart select in the mask editor**: tap an object (a hand, a shirt
  print, the sky) and it's masked in one go, edges and all. Tap it again to
  cycle through sizes (finger → hand → arm); with 🧹 Erase on, tap inside
  the selection to cut a part out, or tap elsewhere to remove a whole object
  from the mask. Mixes freely with the brush and undo/redo. Runs MobileSAM:
  the bot encodes the image (needs `mobile_sam_image_encoder.onnx` staged,
  see `env.example`), the decoder runs on your phone.
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

- **Quiet messages**: the bot no longer buzzes your phone for menus,
  status lines, prompts and other replies. Only finished images
  (generations, post-processing results, `/stream` images) notify.
- **"🔁 x4" keeps ComfyUI busy**: two redos are now queued at a time, so
  the next one is already running while the previous result is still
  being sent. Results arrive in the order they finish. One status line
  covers the whole tap ("Detailing 2/4… 40%") instead of one per redo,
  and drawn-mask jobs (Draw Mask, Fix Artifact, Detail Prompt and their
  redos) now show a percentage.
- **Detailers decode in tiles**: Face/Hand Detail, Draw Mask, Detail
  Prompt, Fix Artifact and the region LoRA pass now use tiled VAE
  encode/decode straight away. On a large (upscaled) image ComfyUI used to
  run out of memory on the full decode first and then retry tiled, which
  cost time on every run.
- **Progress as a live draft**: in a private chat, "Generating… 40%" (and
  every other job's progress) is now Telegram's animated typing-style
  preview instead of a status message that gets edited and deleted. It
  never stays in the chat, shows below the newest image during `/stream`,
  and lists every job at once when several run in parallel. Errors and
  "No faces found"-style outcomes still arrive as normal messages. Groups
  keep the old status message; `PROGRESS_DRAFTS=false` brings it back
  everywhere.
- **✏️ Detail Prompt starts from the prompt as you typed it**: the editor
  now pre-fills the image's prompt without the profile's quality tags, the
  active character's prompt or LoRA trigger words, and the bot adds the
  prefixes and trigger words back when you tap Done (only for LoRAs that
  still run in that pass). They used to be in the text already and got
  added a second time. Regional images still pre-fill the full prompt.
- **Mask editor info is folded away**: the detailer settings and LoRA note
  above the prompt fields now sit in a collapsed "Info" dropdown, giving
  the prompts and canvas more room. Needs a relay redeploy.
- **✏️ Detail Prompt's `$` menu lists all your characters and favorites**:
  with a lot saved, the ones at the end of the list used to be left out
  (and some lost their preview). Its search is looser too: letters typed in
  order find a name even with gaps between them. The bot and the mask editor relay need
  updating together for this.
- Mask editor: the brush size is now measured on screen, so it's the same
  in Draw Mask, Fix Artifact and Detail Prompt and on every image. It used
  to depend on the image's resolution and on how much room the editor had
  at load, so Fix Artifact's default brush came out much bigger than Detail
  Prompt's. The slider's default and range changed to match.

- Draw Mask, Fix Artifact and Detail Prompt: the "Draw over the area…, then
  tap Done in the editor" message (with its editor button) is now deleted
  once the drawn mask arrives and generation starts.

### Security

- Mask/pose editor submissions are only accepted from the user the editor
  was opened for, if they're still on `ALLOWED_USER_IDS`, and only within
  30 minutes of opening it. Before, any validly signed Telegram WebApp
  data was accepted, and it never expired.
- inpaint_relay: rejects malformed submissions (bad or out-of-range
  denoise, oversized masks) with a 400/413 instead of a 500, accepts one
  submission per job, no longer serves FastAPI's `/docs`, and sends
  `Referrer-Policy: no-referrer`, `nosniff` and a Content-Security-Policy
  on the editor pages.
- inpaint_relay: the editor checks onnxruntime-web's files (script,
  `.mjs`, `.wasm`) against pinned SHA-384 hashes before running them.
- inpaint_relay: `docker-compose.example.yml` now actually attaches its
  security-headers middleware to the router; it was defined but unused.
  Add the same `traefik.http.routers.inpaint-relay.middlewares` label to
  an existing deployment.

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
