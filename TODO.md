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

- [ ] **Painted regions**: a "regions" mode in the `inpaint_relay` editor.
  You paint one colour per region, with a prompt box each, either on a
  blank canvas or over an earlier result as a layout guide. Core
  `ImageColorToMask` can split one colour-coded PNG into masks. This would
  be the first relay job that starts a generation rather than
  post-processing.
- [ ] **Region-aware detailers**: Face/Hand Detail on a regional image only
  sees the global prompt, so character-specific tags are missing there.
  Pick the region whose box contains the detected face and add its prompt.
- [ ] **Regional `/stream`**: a `/stream` prompt with `[left]`/`[right]`
  lines currently generates as one plain prompt, tags included.
- [ ] **Weights in `/settings`**: `regional_base_weight`/
  `regional_region_weight` can only be set in profile JSON for now.
- [ ] **Per-region LoRAs** with core hook nodes (`CreateHookLora`,
  `PairConditioningSetProperties`). Probably doesn't combine with Attention
  Couple; needs testing.

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
