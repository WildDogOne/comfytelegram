# TODO

Ideas and follow-ups that were discussed but not built yet. When something
here ships, move it to `CHANGELOG.md` under "Unreleased".

## Favorites

Step 1 (`/fav`, `/favs`, edit, delete) is done. Still open:

### Use favorites in prompts (step 2)

- [ ] **`$name` expansion**: typing `1girl, $pose, $wlop, forest` swaps in the
  saved text. This belongs in `handlers._resolve_effective_prompt`, so it
  also works in `/stream`, "🔁 Generate Again" and everywhere else prompts
  are resolved.
- [ ] **`$category?` wildcards**: picks a random favorite from that category
  for each image. Combined with `/stream`, this gives an exploration mode
  ("50 images, each with a random artist from my list").
  - Decide what to do with an unknown `$name` or an empty category: leave the
    text as is, or reply with an error.
  - "🔁 Generate Again" should probably re-roll the wildcard. Check what
    `raw_positive_prompt` stores before deciding.

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

## Housekeeping

- [ ] `scripts/smoke_test.py` and `scripts/smoke_test_postprocess.py` aren't
  `ruff format`-clean, so `uv run ruff format --check .` fails on them.
