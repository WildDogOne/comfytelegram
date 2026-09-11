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
    "height": 1024
  },
  "positive_prompt_prefix": "quality tags prepended before the user's prompt",
  "negative_prompt_prefix": "used as the negative prompt whenever the user doesn't supply one",
  "loras": [
    {
      "name": "family/style.safetensors",
      "strength_model": 0.8,
      "strength_clip": 1.0,
      "default_enabled": false        // true = always applied for this model; false = documented but off
    }
  ],
  "loader": "checkpoint",              // "checkpoint" (default) or "split" — see below
  "clip_name": "",                     // "split" only: CLIPLoader's text-encoder filename
  "clip_type": "stable_diffusion",     // "split" only: CLIPLoader's `type` input
  "vae_name": "",                      // "split" only: VAELoader's filename
  "model_sampling_shift": null         // "split" only: shift for a ModelSamplingAuraFlow node (omit/null to skip it)
}
```

`furrytoonmix_illustrious.json` is grounded in the actual working values
from the hand-built `sample.json` workflow at the project root (see the
main [README](../README.md#background-the-reference-workflow)). The other
three are common community-recommended starting points for those model
families, not values verified against a specific checkpoint on this
install — tune them once you've run a few generations.

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
  with `clip_type: "stable_diffusion"` despite not being a CLIP model —
  that's just the `CLIPLoader` bucket the Anima authors' own workflow uses).
- `vae_name` points at the separate VAE file (`models/vae/`).
- `model_sampling_shift`, if set, inserts a `ModelSamplingAuraFlow` node
  right after the UNET load — Anima's AuraFlow-style sampling shift.
  Leave it `null`/omitted for split architectures that don't need that node.

`cfg`/`steps`/`sampler_name`/`scheduler`/`width`/`height` all still work the
normal way through `defaults` — Anima Aesthetic and Anima Turbo just want
very different values for them (the two shipped profiles already reflect
that). `clip_skip` isn't meaningful for a non-CLIP text encoder like
Anima's, so leave it unset (`-1`, i.e. "don't add that node").
