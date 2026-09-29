# Model profiles

Each `*.json` file here is one `ModelProfile` (schema:
`src/comfytelegram/profiles/schema.py`) — generation defaults tuned for a
checkpoint or a family of checkpoints. The bot loads every file in this
directory at startup (`MODEL_PROFILES_DIR`, see `env.example`) and, when a
user picks a checkpoint, applies the first profile whose `match` glob
patterns hit that checkpoint's filename.

Add your own by dropping in a new `*.json` file — no code changes needed.
Files starting with `_` are ignored (reserved for docs/schema files).

## Fields

```jsonc
{
  "match": ["furrytoonmix_*"],       // fnmatch globs, case-insensitive, checked in file order
  "display_name": "FurryToonMix XL", // shown in the /model picker
  "description": "free text",
  "defaults": {                      // all optional; unset falls back to generic defaults
    "cfg": 5.0,
    "steps": 40,
    "sampler_name": "euler_ancestral",
    "scheduler": "normal",
    "clip_skip": -2,
    "width": 1024,
    "height": 1024,
    "upscale_denoise": 0.2              // UltimateSDUpscale's denoise for "🔍 Upscale 4x" (omit to use its own default) — see below
  },
  "positive_prompt_prefix": "quality tags prepended before the user's prompt",
  "negative_prompt_prefix": "used as the negative prompt whenever the user doesn't supply one",
  "tag_dictionary": "e621",            // "danbooru", "e621", or omit to search/check both — see below

  "loras": [
    {
      "name": "family/style.safetensors",
      "strength_model": 0.8,
      "strength_clip": 1.0,
      "default_enabled": false,       // true = always applied for this model; false = documented but off
      "trigger_words": ""             // auto-added to the prompt when this LoRA is active — fill in by hand only, see below
    }
  ],
  "civitai_base_models": [],           // opts this profile into boot-time LoRA auto-discovery — see below
  "loader": "checkpoint",              // "checkpoint" (default) or "split" — see below
  "clip_name": "",                     // "split" only: CLIPLoader's text-encoder filename
  "clip_type": "stable_diffusion",     // "split" only: CLIPLoader's `type` input
  "vae_name": "",                      // "split" only: VAELoader's filename
  "model_sampling_shift": null,        // "split" only: shift for a ModelSamplingAuraFlow node (omit/null to skip it)
  "tile_controlnet": null,             // ControlNet Tile model filename for the upscale pass (omit/null to skip it) — see below
  "tile_controlnet_strength": 0.4,     // ControlNetApplyAdvanced's strength for tile_controlnet
  "anima_lllite_inpaint_patch": null,  // "split" (Anima) only: ETN_control_load weights filename for the "🩹 Fix Artifact" pass (omit/null to skip it) — see below
  "anima_lllite_inpaint_patch_strength": 1.0, // ETN_control_apply's strength for anima_lllite_inpaint_patch
  "fix_artifact_checkpoint": null      // exact filename "🩹 Fix Artifact" always uses instead of the image's own checkpoint (omit/null to keep using each image's own) — see below
}
```

`furrytoonmix_illustrious.json` is grounded in the actual working values
from the hand-built `sample.json` workflow at the project root (see the
main [README](../README.md#background-the-reference-workflow)). The other
three are common community-recommended starting points for those model
families, not values verified against a specific checkpoint on this
install — tune them once you've run a few generations.

### `tag_dictionary` — scoping `/tags`/`/tagcheck`

Independent of `prompt_style` (which just picks WD14 vs. Qwen-VL for image
analysis). `tag_dictionary` picks which local tag database `/tags`
(search) and `/tagcheck` (prompt scanner) default to for this checkpoint —
`"e621"` for furry-trained models, `"danbooru"` for anime/manga-trained
ones. Leave it unset for a checkpoint that isn't clearly one or the other
(both commands then search/check against both dictionaries); either
command's caller can still override it per-call with a `danbooru:`/`e621:`
prefix. See the main README's "Tag search" section for how the databases
themselves get populated.

### `civitai_base_models` — auto-registering downloaded LoRAs into `loras`

Hand-typing a `loras` entry for every LoRA you download is exactly the
tedium `civitai_base_models` exists to cut down on. If `Settings.
comfyui_loras_dir` is set (a filesystem path to ComfyUI's own
`models/loras`, readable by the bot — see the main README's
[LoRA info lookup](../README.md#lora-info-lookup) section), the bot scans
that directory at startup for files no profile's `loras` list mentions
yet, hashes each one, and looks it up on CivitAI. A LoRA whose
CivitAI-reported `baseModel` (e.g. `"SDXL 1.0"`, `"Pony"`,
`"Illustrious"` — exact strings, case-insensitive) matches an entry in
this list gets appended to `loras` automatically, `default_enabled: false`
so it doesn't silently start applying to generations — toggle it on via
`/lora` once you've confirmed it's the one you want.

Left empty (the default), a profile is untouched by this entirely — set
it explicitly to opt in, e.g. `["SDXL 1.0"]` for a plain SDXL profile, or
`["Pony", "SDXL 1.0"]` if a profile is happy to accept LoRAs trained
against either. This only ever *adds* entries, never removes or edits an
existing one, and only for a LoRA CivitAI actually has a hash record for —
anything privately/custom-trained (like `furrytoonmix_illustrious.json`'s
three character LoRAs above) still needs to be added here by hand, the
same as before this existed.

The `/lora` menu's toggleable set is exactly *this profile's own* `loras`
list, not "every LoRA this architecture could use" — a LoRA another
profile already lists never shows up here on its own, even if both
profiles opt into the same `civitai_base_models` entry, because the
file-scan above only ever considers a filename *no* profile mentions yet.
The first profile to claim a name (by this scan or by hand) permanently
removes it from that scan's consideration. A second, cache-only pass
(`_propagate_known_loras`) covers exactly this case: for every LoRA name
some profile already lists with a cached CivitAI lookup — from a past
discovery run *or* a `/lora` "ℹ️ Info" tap, both land in the same
`lora_civitai_cache` table — it adds that LoRA to every other opted-in
profile that accepts its base model and doesn't have it yet, on the next
boot or `/reload`. It never hashes or queries CivitAI itself, so a LoRA
nobody has ever actually looked up (no cache entry yet) isn't propagated
until something populates one — tap "ℹ️ Info" on it once under whichever
profile already lists it, or just add it to the new profile's `loras` by
hand, either one seeds the cache (or list) the same way.

### `loras[].trigger_words` — auto-injected, but manual-only to fill in

Free text for whatever word(s)/phrase a LoRA actually needs in the prompt
to activate. Whenever this LoRA ends up active for a generation
(`default_enabled`, after any `/lora` override), `resolve_generation_params`
folds this straight into the positive prompt automatically — right after
`positive_prompt_prefix` and before whatever the user typed — so you don't
have to retype a LoRA's trigger word by hand on every single message just
to get it to actually do anything. A LoRA that's toggled off (`/lora`, or
`default_enabled: false` in the JSON) contributes nothing, same as its
`strength_model`/`strength_clip` not applying either. This only ever
touches the built prompt — not `LoraDefault.to_spec()`, which still only
ever carries `name`/`strength_model`/`strength_clip` into the actual
ComfyUI graph, and not what the user actually typed (`raw_positive_prompt`,
what "🐛 Show Prompt"'s "As typed" reply shows) — "🐛 Show Prompt"'s main
reply, which shows the fully-resolved prompt, is where an injected trigger
word is visible.

Filling the field in, though, is deliberately manual-only: nothing in this
codebase ever writes an actual *value* into it automatically, even though
CivitAI's API does return a `trainedWords` list right alongside everything
auto-discovery already pulls from it. The key itself does get added
automatically — a newly auto-registered LoRA gets `"trigger_words": ""`
from the start (`lora_discovery.py`'s auto-registration), and every boot
plus every `/reload` backfills the same empty key onto any older entry
that predates this field, or that a human wrote by hand without knowing
about it (`lora_discovery.backfill_trigger_words` — pure JSON-file
hygiene, so opening the file shows the field as a visible placeholder
either way) — but the *value* stays empty until you type one in yourself.
That's on purpose: a LoRA's hash can get re-uploaded under a different/
renamed listing, CivitAI's own trigger-word list is sometimes incomplete
or wrong, and a privately/
custom-trained LoRA has no CivitAI entry to pull one from at all — silently
saving a wrong trigger word as if it were verified, and then silently
injecting it into every prompt, is worse than leaving the field empty and
obviously still TODO. Fill it in by hand once you've actually confirmed
the trigger word works (checking `/lora`'s "ℹ️ Info" button's own
CivitAI-sourced trigger words is a fine starting point, just verify before
copying it in here). Left empty (the default, folds in as nothing) on
every profile right now.

### `loader: "split"` — Anima-style architectures

Most checkpoints are one `.safetensors` file loaded through
`CheckpointLoaderSimple`, matched by `match` against that filename — the
default (`loader: "checkpoint"`). Some newer architectures instead ship as
separate UNET/text-encoder/VAE files loaded through
`UNETLoader`+`CLIPLoader`+`VAELoader` — currently just
[Anima](https://civitai.com/ecosystems/anima) (`anima_aesthetic.json`,
`anima_turbo.json`). For those, set `"loader": "split"` and:

- `match`/the `/model` picker target the **UNET filename** (e.g.
  `anima-aesthetic-v1.safetensors`, staged under ComfyUI's
  `models/diffusion_models/`) — `ComfyClient.list_checkpoints()` merges
  `CheckpointLoaderSimple`'s and `UNETLoader`'s filename lists into one
  `/model` list, so both kinds of model show up together.
- `clip_name`/`clip_type` point at the separate text-encoder file (Anima
  uses a Qwen-3 0.6B encoder, staged under `models/text_encoders/`, loaded
  with `clip_type: "omnigen2"` despite not being an OmniGen2 model — that's
  just the `CLIPLoader` bucket krita-ai-diffusion's own `workflow.py` uses
  for this exact text encoder (`case Arch.anima: clip = w.load_clip(...,
  type="omnigen2")`); `"stable_diffusion"` loads without erroring but is
  the wrong wrapper and was silently producing worse text conditioning.
- `vae_name` points at the separate VAE file (`models/vae/`).
- `model_sampling_shift`, if set, inserts a `ModelSamplingAuraFlow` node
  right after the UNET load — Anima's AuraFlow-style sampling shift.
  Leave it `null`/omitted for split architectures that don't need that node.
  "🩹 Fix Artifact"'s dedicated Anima pipeline (`_build_anima_fix_drawn_mask`)
  always ignores this field regardless of what a profile sets it to — a
  real krita "remove object" job's own graph has no such node at all, and
  krita's own `workflow.py` never applies one for Anima in any workflow.

`cfg`/`steps`/`sampler_name`/`scheduler`/`width`/`height` all still work the
normal way through `defaults` — Anima Aesthetic and Anima Turbo just want
very different values for them (the two shipped profiles already reflect
that). `clip_skip` isn't meaningful for a non-CLIP text encoder like
Anima's, so leave it unset (`-1`, i.e. "don't add that node").

### `tile_controlnet` — detail across the whole image, not just faces/hands

The face/hand detailers only touch the regions Impact Pack's detector
crops out, so a scene's background/clothing/etc. can end up looking flat
next to a sharp face after a 4x upscale. Setting `tile_controlnet` to a
ControlNet Tile model's filename (staged under `models/controlnet/`) adds
a `ControlNetLoader`+`ControlNetApplyAdvanced` branch to the upscale pass,
conditioned on the pre-upscale source image — `UltimateSDUpscale` crops
that hint per tile itself, so it stays structurally faithful to the source
while still re-diffusing (and therefore adding texture/detail to) every
tile, not just the detailer regions. Must be a ControlNet trained for this
checkpoint's base architecture (an SDXL tile ControlNet won't work on an
Anima/AuraFlow checkpoint, for example) — leave it `null`/omitted if you
don't have a matching one staged. `tile_controlnet_strength` tunes how
strongly it holds the upscale to the source; too high fights the extra
denoise you actually want, too low lets tiles drift/seam.

`tile_controlnet` alone doesn't add detail, though — it just makes it
*safe* to raise `defaults.upscale_denoise` (how much `UltimateSDUpscale`
actually re-diffuses per tile) without the image drifting off-structure.
The two are meant to be tuned together: `furrytoonmix_illustrious.json`
sets `upscale_denoise: 0.5` (up from the conservative `0.2` the base
UpscaleParams default mirrors from `sample.json`) precisely because it
also has a `tile_controlnet` staged to anchor that extra denoise.

### `anima_lllite_inpaint_patch` — making "🩹 Fix Artifact" actually inpainting-aware

`"loader": "split"` only. Without this, "🩹 Fix Artifact"/"🖌️ Draw Mask"
runs plain noise-masked img2img on a checkpoint that was never trained on
"here's a masked hole, infer what belongs there" — it just has to hope the
denoise pass reconstructs something plausible, which is why results can be
hit-or-miss (sometimes it cleanly erases the marked region, sometimes it
reconstructs a nicer-looking version of the very thing you masked out).
Setting `anima_lllite_inpaint_patch` to an Anima LLLite inpainting weights
filename (staged under ComfyUI's `models/controlnet/` — `ETN_control_load`
scans the same folder a plain `ControlNetLoader` does) patches the model
via `ETN_control_load`+`ETN_control_apply` (`comfyui-tooling-nodes`, Acly's
own node pack — the *same one `krita-ai-diffusion` itself uses* for this
exact weight format, via its own `apply_controlnet_lllite`) before that
diffusion pass, conditioned directly on the source image and the drawn
mask, instead of relying on generic img2img. **Do not** point this at
ComfyUI core's `ModelPatchLoader`/`AnimaLLLiteApply` pair instead — despite
the similar name and despite also listing this same file in its own
dropdown, it's a different, unrelated mechanism that can't actually
interpret this weight format; a first pass wired against those core nodes
produced very poor results (confirmed by cross-checking krita-ai-diffusion's
own source, which uses `ETN_control_load`/`ETN_control_apply` specifically).
`anima_lllite_inpaint_patch_strength` tunes how strongly the patch holds.
No SDXL/SD1.5 equivalent exists yet — see `fix_artifact_checkpoint` below
for how checkpoints without one are handled in the meantime.

### `fix_artifact_checkpoint` — routing "🩹 Fix Artifact" to a checkpoint that can actually inpaint

Most checkpoints have no inpainting-aware path wired at all (see
`anima_lllite_inpaint_patch` above) — running "🩹 Fix Artifact" on an image
from one of them is a coin flip regardless of how the pass itself is
tuned. Setting `fix_artifact_checkpoint` on a profile makes "🩹 Fix
Artifact" **always** load that exact checkpoint filename (plus that
profile's `loader`/`clip_name`/`clip_type`/`vae_name`/`loras`/
`negative_prompt_prefix`/`anima_lllite_inpaint_patch` — *not*
`model_sampling_shift`, which the dedicated Anima fix-drawn-mask pipeline
always ignores; see below) instead of the image's own original one —
regardless of which checkpoint the image was actually generated with.
`anima_aesthetic.json` sets this to `anima-base-v1.0.safetensors` — the
plain base checkpoint, not the aesthetic finetune this profile otherwise
loads for normal generation, matched against a real krita-ai-diffusion
"remove object" job pulled from this server's own `/history` — so "🩹 Fix
Artifact" on *any* image (furrytoonmix, SDXL, whatever) routes through
Anima's base checkpoint + its lllite inpainting patch.

Unlike `match` (a glob matched against whatever checkpoint the user
picked), this is a literal filename — it names the exact file to switch
*to*, not which profile applies to a file you already have. Only set it
on one profile; if several do, the first one in file-load order wins.
The positive prompt is still forced empty for this pass either way — see
`generation.post_process`'s `fix_drawn` branch — so the removed content
isn't reconstructed from a text description regardless of which
checkpoint ends up handling it.

Trade-off worth knowing: the source image's own checkpoint and the
`fix_artifact_checkpoint` one can be completely different architectures
(Anima's AuraFlow-style 2B model vs. an SDXL/Illustrious checkpoint like
furrytoonmix). A clean removal is the goal, but the patched region is
rendered by a *different* model than the rest of the image, so it can
carry a slightly different color grade/lineweight/rendering style than
its surroundings — a different failure mode than the "reconstructs the
removed thing" problem this is meant to fix, not a strictly better result
in every case. The bot logs which checkpoint a given "🩹 Fix Artifact" run
actually used (`Fix Artifact: checkpoint=... (fix_artifact_checkpoint
override / image's own checkpoint) ...`), so this is easy to confirm
against real results rather than guessing.
