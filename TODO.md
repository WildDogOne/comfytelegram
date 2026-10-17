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

## Housekeeping

- [ ] `scripts/smoke_test.py` and `scripts/smoke_test_postprocess.py` aren't
  `ruff format`-clean, so `uv run ruff format --check .` fails on them.
