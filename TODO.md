# TODO

Ideas and follow-ups that were discussed but not built yet. When something
here ships, move it to `CHANGELOG.md` under "Unreleased".

## Favorites

Step 1 (`/fav`, `/favs`, edit, delete) and `$name` expansion in prompts are done. Still open:

### Use favorites in prompts (step 2)

- [ ] **`$category?` wildcards**: picks a random favorite from that category
  for each image. Combined with `/stream`, this gives an exploration mode
  ("50 images, each with a random artist from my list").
  - Plain `$name` expansion shipped (`prompt_refs.py`): an unknown name is an
    error. An empty category should probably be one too.
  - "🔁 Generate Again" reruns the *resolved* params, so a wildcard would
    not re-roll. Re-rolling means re-expanding `raw_positive_prompt`.

### Saving more easily

- [ ] **Guided `/fav` flow for mobile**: a bare `/fav` asks for the text,
  then the name, then a category from buttons, then an optional note.
- [ ] **⭐ Save button on `/tags` results**, next to the 📋 copy button.
- [ ] **Save from an image's prompt**: under "🐛 Show Prompt" or
  "🔬 Analyze Prompt", list the prompt's tags as toggle buttons and save the
  ones you pick.
- [ ] **Save from analysis output**, such as WD14 tags from "🏷️ Analyze Image".

### Browsing and organizing

- [ ] **Paging in `/favs`**: a category's favorites aren't paged yet, which
  will matter once a category holds dozens of entries.
- [ ] **Example image per favorite**: keep the Telegram `file_id` of an image
  that shows what the tag does, and show it in `/favs`.
- [ ] **Show matching favorites first**: list favorites that fit the current
  model's `tag_dictionary` (danbooru vs. e621) before the others.
- [ ] **Basket**: toggle several favorites in `/favs`, then "🎨 Generate with
  these".

## Regional prompts (/rp)

Step 1 (`/rp` text syntax, Attention Couple, per-profile weights) is done.
Still open:

- [ ] **Painted regions: follow-ups.** Shipped (see CHANGELOG). Open:
  - Images from before `pending_result.noise_index` existed have none, so
    painting over a later image of such a batch still gets image 1's
    noise. Self-healing: every new image records it.
  - Show Prompt's copy of a painted image gives its nearest box layout
    (`[left]`/`[right]`), not the painting — there's no text form for it.
- [ ] **Region-aware detailers**: Face/Hand Detail on a regional image only
  sees the global prompt, so character-specific tags are missing there.
  Pick the region whose box contains the detected face and add its prompt.
- [ ] **Regional `/stream`**: a `/stream` prompt with `[left]`/`[right]`
  lines currently generates as one plain prompt, tags included.
- [ ] **Weights in `/settings`**: `regional_base_weight`/
  `regional_region_weight` can only be set in profile JSON for now.
- [ ] **Region LoRAs: follow-ups.** Shipped (see CHANGELOG): a region's
  `<lora:…>` is applied after generating by a detailer pass over that
  region's character, mask checked in the editor first or run
  automatically. Tested and working as is, so these are parked — kept for
  reference in case problems show up, not planned:
  - **Masks where characters touch.** The suggested mask is "people inside
    the region's box", so between touching characters its edge is the
    box's straight line. Neither the COCO person segmenter (one merged
    person in 8 of 8 hug/kiss test images) nor MobileSAM with
    positive/negative points (heads merged in ~3 of 8) could separate them.
    Worth trying if a model turns up that's trained on anime/furry
    characters and separates touching ones (instance segmentation, or a
    person detector + SAM per box) — test it on the same kind of images
    first.
  - **Optional finishing Homogenize** after the automatic passes, if seams
    show up in practice.
  - **Alternative without masks:** core hook nodes (`CreateHookLora`,
    `PairConditioningSetProperties`) with mask-based regions
    (`ConditioningSetMask`) instead of Attention Couple — the LoRA applies
    only while its region's conditioning is sampled, so no seams, but
    roughly one model evaluation per region (slower) and coarser coherence
    between regions.

## Poses (/pose)

Step 1 (`/pose` upload, `pose_controlnet` profile fields, aspect from the
pose, works with `/rp`) is done. Still open, in order:

- [ ] **Test more checkpoints** with `scripts/smoke_test_pose.py`:
  `xinsir_openpose` on autismmix/pony/sdxl_base (it may want
  `--xinsir-sticks` thicker limbs), harder poses (overlapping limbs,
  people touching), and whether `anima-lllite-any-test-like-v2` does
  anything with a pose on Anima.
- [x] ~~**🦴 Use pose button**~~ — shipped as "🦴 Copy pose" plus detection
  on `/pose` uploads (DWPose run in-process, `pose_detect.py`), with a
  body-only skeleton editor in the relay (`static/pose.html`). Keypoints
  are stored per chat (`chat_pose.pose_json`), so the items below can use
  them.
- [ ] **OpenPose JSON upload**: accept a `.json` file from other editors
  (`parse_pose_json` already reads that format).
- [ ] **Hands and face in the editor**: body only for now. Hand keypoints
  need the hand-trained ControlNets and are fiddly to place on a phone.
- [ ] **Regions from the pose**: per-person boxes (left to right) instead of
  fixed halves, so `[left]`/`[right]` mean "person 1/2" even when the
  people don't split the canvas evenly.
- [ ] **Region LoRA masks from the pose**: a per-person skeleton grown into
  a mask separates touching characters, where `region_lora_mask` currently
  cuts merged person detections with the box's straight line.
- [ ] **Paint Regions over the pose**: open the editor with the skeleton as
  its layout guide and person 1 pre-painted.
- [ ] **Pose in the Region LoRA pass**: the high-denoise detailer redrawing
  a character drifts from the pose; condition it on the cropped skeleton
  (`ImpactControlNetApplySEGS`).

## Mask editor

- [ ] **Superpixel brush, alongside ✨ Smart select**: SAM picks whole
  objects, but sometimes the region to mask isn't an object (part of a
  background, a smudge, a gradient). Split the image into a few thousand
  small same-colour regions that follow its edges (SLIC, ~100 lines of JS,
  computed once in the page on a ~1024px copy), then let a brush stroke
  select every region it touches and the eraser remove them. Edge-snapping
  like quick selection, no tolerance to tune, no model needed — so it also
  works when no SAM encoder is staged. Undo can store touched region ids
  instead of bitmaps, since the regions never change. A coarse/fine choice
  for region size could go in the options panel.
- [ ] **`disableVerticalSwipes()` in `index.html`**: `pose.html` already
  calls it, the mask editor doesn't, so a downward paint stroke can
  minimise or close the Mini App.
- [ ] **Full-screen editors** (`requestFullscreen()`, Bot API 8.0): both
  pages only `expand()`; full screen gives the canvas more room on a phone.
  Mind the safe-area insets.
- [ ] **`DeviceStorage`** (Bot API 9.0) instead of `localStorage` for the
  options panel's folded state, since Telegram's webview can clear
  `localStorage`.
- [ ] **Check the Mini App domain lock** (Bot API 10.2, on since
  2026-07-20): Mini App methods now only work from the Mini App's own
  origin. The editors are served from the relay and only load
  `telegram-web-app.js` from telegram.org, so they should be fine — try
  both once to confirm.

## Telegram Bot API features

Ideas from the Bot API changelog (8.0–10.3, reviewed 2026-10-19).

### Available in python-telegram-bot

PTB 22.8 supports Bot API 10.0.

- [ ] **Custom emoji icons on buttons** (`icon_custom_emoji_id`, Bot API
  9.4): needs the bot owner to have Premium (or bought Fragment
  usernames). Button colours (`style`) shipped.
- [ ] **Topics in the private chat** (Bot API 9.3/9.4): the bot can enable
  topics in its DM and `create_forum_topic` there, e.g. one thread per
  character or project. `message_thread_id` is already passed everywhere,
  so this should mostly just work.

### Waiting for python-telegram-bot

These need Bot API 10.1–10.3. Until PTB supports them, they're only
reachable as raw `bot.do_api_request(...)` calls, without PTB models for
the new update types. Check PTB's changelog for a release that supports
Bot API 10.3.

- [ ] **Stop button on drafts** (`can_stop`/`keep_on_stop` on
  `sendMessageDraft`, `stopped_message_generation` update, Bot API 10.3):
  a native "stop" on the progress draft (`progress.py`), which would
  interrupt the ComfyUI job (`/interrupt`) or end a `/stream` — instead of
  `_STREAMING_KEYBOARD`'s "/stop" button. Needs PTB to dispatch the new
  update type.
- [ ] **`force_reply` on inline keyboards** (Bot API 10.3): opens the reply
  box as if the user had tapped "Reply". Fits every follow-up question
  (`/stream` prompt, Kontext instruction, `/settings` custom value, `/fav`
  edit, "🎚️ Redo…" custom denoise, `/rp`). The answer then arrives as a
  reply to the question, so it could be matched by `reply_to_message`
  instead of the `awaiting_*` flags in `chat_data`. Can't be changed by
  editing the keyboard later.
- [ ] **Disabled buttons** (`InlineKeyboardButton.disabled`/
  `DisabledButton`, Bot API 10.3): grey out a tapped "🖌️ Draw Mask"/
  upscale button while it runs, instead of `uploads_in_progress` plus a
  "still uploading" reply. Also a greyed-out "📋 Copy" for prompts over
  256 characters, rather than leaving the button out.
- [ ] **Ephemeral messages** (Bot API 10.2/10.3): group messages only one
  user sees. Fits `/settings`, `/lora`, `/favs` and error notes in a group;
  `replace_callback_query_message` can show a menu in place of the message
  that opened it. Only worth it if the bot is used in groups.
- [ ] **Sending rich messages** (`sendRichMessage`, Bot API 10.1–10.3):
  compact tables, collapsible quotes and button blocks — for
  "🐛 Show Prompt", `/tagcheck`, LoRA info and the `/settings` summary.
  Receiving them is already handled (`message_text.py`); once PTB models
  `Message.rich_message`, that module's `api_kwargs` workaround can go too.

## Housekeeping

- [ ] `scripts/smoke_test.py` and `scripts/smoke_test_postprocess.py` aren't
  `ruff format`-clean, so `uv run ruff format --check .` fails on them.
