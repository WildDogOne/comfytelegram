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
  ]
}
```

`furrytoonmix_illustrious.json` is grounded in the actual working values
from the hand-built `sample.json` workflow at the project root (see the
main [README](../README.md#background-the-reference-workflow)). The other
three are common community-recommended starting points for those model
families, not values verified against a specific checkpoint on this
install — tune them once you've run a few generations.
