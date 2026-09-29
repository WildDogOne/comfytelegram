"""Boot-time LoRA auto-discovery.

Scans `Settings.comfyui_loras_dir` for files no model profile's `loras`
list already mentions, identifies each via `civitai.py`'s hash lookup, and
appends a disabled `LoraDefault` entry for it to every profile whose
`ModelProfile.civitai_base_models` accepts its CivitAI-reported base
model — replacing the "download a LoRA, hunt down its trigger words, hand-
type a model_profiles/*.json entry" tedium for the common CivitAI-hosted
case.

That file-scan only ever considers a name *no* profile currently mentions
— the first profile to claim a given filename (whether through this scan
or a hand-written entry) permanently removes it from consideration, even
for a same-architecture profile added later that would also have accepted
it. `_propagate_known_loras` is the other half of this: for every LoRA
name already sitting in *some* profile's `loras` list with a cached
CivitAI lookup (`lora_civitai_cache` — populated either by a past
discovery run or by `/lora`'s own "ℹ️ Info" button, both write through the
same table), it adds that LoRA to every other opted-in profile that
accepts its base model and doesn't have it yet. This is what makes adding
a new profile for an architecture this install already has LoRAs staged
for (e.g. a second Illustrious-based checkpoint alongside
`furrytoonmix_illustrious.json`) pick those up automatically on the next
boot or `/reload`, instead of requiring the exact same hand-copy this
module exists to avoid in the first place. It never hashes or queries
CivitAI itself, though — a LoRA nobody has ever actually looked up (no
cache entry yet, most commonly a hand-typed entry that predates any Info
tap or discovery run) isn't propagated until something populates its
cache entry first.

This is NOT full automation, by design: a LoRA CivitAI has no hash record
for at all — most commonly a privately/custom-trained one, exactly the
case `furrytoonmix_illustrious.json`'s three character LoRAs describe —
can't be identified this way and still needs a human to add it to a
profile's `loras` list by hand, same as before this existed. So does one
whose base model doesn't match any profile's `civitai_base_models` — that
list is empty by default, so a profile has to opt in before this job ever
touches its file.

Writes go through a full `json.dump(..., indent=2)` rewrite of each
touched profile file rather than a hand-rolled bracket-aware text splice.
That's a real trade-off: any field a human wrote compactly on one line
(e.g. a short `"match": [...]` list) gets exploded to one-value-per-line
the first time its file is touched this way, since that's what
`json.dump(indent=2)` always does for a list. Accepted as a one-time
cosmetic reformat rather than writing much more code to preserve
hand-formatting exactly, on a config file only this bot and its
maintainer ever edit.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from comfytelegram.civitai import (
    CivitaiLookupError,
    fetch_civitai_info,
    hash_lora_file,
    resolve_lora_path,
)
from comfytelegram.profiles import ModelProfile, load_profiles
from comfytelegram.profiles.loader import load_profile_files
from comfytelegram.settings import Settings
from comfytelegram.storage import Storage

logger = logging.getLogger(__name__)

#: Extensions ComfyUI's own `folder_paths.py` recognizes for any model file
#: (`supported_pt_extensions` there, confirmed against its source) —
#: scanning for exactly these avoids hashing/querying CivitAI for files
#: that could never show up in `LoraLoader`'s dropdown anyway (preview
#: images, `.civitai.info`/`.txt` sidecars ComfyUI-Custom-Scripts itself
#: leaves behind, etc).
LORA_EXTENSIONS = {".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"}

#: Politeness delay between CivitAI requests when a run finds several new
#: LoRAs at once — this is an unattended boot-time background job, so
#: there's no reason to hit a public, no-API-key endpoint as fast as the
#: network allows.
_REQUEST_DELAY_SECONDS = 1.0

#: `LoraDefault.strength_model`/`strength_clip` for an auto-registered
#: entry — the schema's own defaults, spelled out here since this module
#: writes plain dicts, not `LoraDefault` instances.
_DEFAULT_STRENGTH = 1.0


def _scan_lora_files(loras_dir: Path) -> list[str]:
    """Every recognized-extension file under `loras_dir`, recursively, as
    forward-slash-joined paths relative to it — the same relative-path
    form a model profile's `loras[].name` (and `LoraLoader`'s own
    dropdown) use."""
    return sorted(
        p.relative_to(loras_dir).as_posix()
        for p in loras_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in LORA_EXTENSIONS
    )


def _propagate_known_loras(
    entries: list[tuple[Path, dict, ModelProfile]],
    opt_in: list[tuple[Path, dict, ModelProfile]],
    storage: Storage,
) -> set[Path]:
    """For every LoRA name some profile's `loras` list already mentions
    with a cached CivitAI lookup, add it to every *other* opted-in profile
    that accepts its base model and doesn't have it yet. See this module's
    own docstring for why this needs to exist as a step separate from the
    file-scan below (that scan permanently excludes any name already
    claimed by a profile, even for a different, later-added profile that
    would also accept it) and why it's deliberately cache-only (never
    hashes or queries CivitAI, so a name nobody has looked up yet just
    isn't propagated until something populates its cache entry)."""
    touched: set[Path] = set()
    known_names = sorted({lora.name for _, _, profile in entries for lora in profile.loras})
    for name in known_names:
        cached = storage.get_lora_civitai_cache(name)
        if cached is None or not cached["found"]:
            continue
        base_model = cached["base_model"] or ""
        for path, data, profile in opt_in:
            if any(lora["name"] == name for lora in data.get("loras", [])):
                continue  # this profile already has it — nothing to propagate
            if not any(b.lower() == base_model.lower() for b in profile.civitai_base_models):
                continue
            data.setdefault("loras", []).append(
                {
                    "name": name,
                    "strength_model": _DEFAULT_STRENGTH,
                    "strength_clip": _DEFAULT_STRENGTH,
                    "default_enabled": False,
                    "trigger_words": "",
                }
            )
            touched.add(path)
            logger.info(
                "LoRA auto-discovery: propagated already-known %s (%s, %s) into %s, "
                "disabled by default",
                name,
                cached["model_name"],
                base_model,
                path.name,
            )
    return touched


async def discover_new_loras(loras_dir: Path, profiles_dir: Path, storage: Storage) -> bool:
    """Run one discovery pass. Returns True if any profile file under
    `profiles_dir` was actually modified, so the caller (`main.py`'s
    `_start_lora_discovery`) knows to reload `bot_data['profiles']` for the
    change to take effect without a restart."""
    entries = load_profile_files(profiles_dir)
    opt_in = [
        (path, data, profile) for path, data, profile in entries if profile.civitai_base_models
    ]
    if not opt_in:
        logger.info("LoRA auto-discovery: no profile sets civitai_base_models, nothing to do")
        return False

    touched: set[Path] = _propagate_known_loras(entries, opt_in, storage)

    known_names = {lora.name for _, _, profile in entries for lora in profile.loras}
    candidates = [name for name in _scan_lora_files(loras_dir) if name not in known_names]
    if not candidates:
        logger.info("LoRA auto-discovery: no new files under %s", loras_dir)
    else:
        logger.info(
            "LoRA auto-discovery: checking %d new file(s) under %s", len(candidates), loras_dir
        )

    for name in candidates:
        cached = storage.get_lora_civitai_cache(name)
        if cached is None:
            path = resolve_lora_path(loras_dir, name)
            if path is None:  # just scanned it, but stay defensive against a race
                continue
            sha256 = await hash_lora_file(path)
            try:
                info = await fetch_civitai_info(sha256)
            except CivitaiLookupError as exc:
                logger.warning(
                    "LoRA auto-discovery: CivitAI lookup failed (%s) — stopping this run, "
                    "remaining files will be retried on the next boot",
                    exc,
                )
                break
            storage.set_lora_civitai_cache(name, sha256, info)
            cached = storage.get_lora_civitai_cache(name)
            await asyncio.sleep(_REQUEST_DELAY_SECONDS)

        if not cached["found"]:
            logger.info(
                "LoRA auto-discovery: %s has no CivitAI match — add it to a profile "
                "manually if it's custom-trained",
                name,
            )
            continue

        base_model = cached["base_model"] or ""
        matches = [
            (path, data)
            for path, data, profile in opt_in
            if any(b.lower() == base_model.lower() for b in profile.civitai_base_models)
        ]
        if not matches:
            logger.info(
                "LoRA auto-discovery: %s is CivitAI's %r (base model %r), but no profile's "
                "civitai_base_models accepts that — add it manually or configure a profile",
                name,
                cached["model_name"],
                base_model,
            )
            continue

        for path, data in matches:
            data.setdefault("loras", []).append(
                {
                    "name": name,
                    "strength_model": _DEFAULT_STRENGTH,
                    "strength_clip": _DEFAULT_STRENGTH,
                    "default_enabled": False,
                    "trigger_words": "",
                }
            )
            touched.add(path)
        logger.info(
            "LoRA auto-discovery: registered %s (%s, %s) into %d profile(s), disabled by default",
            name,
            cached["model_name"],
            base_model,
            len(matches),
        )

    data_by_path = {path: data for path, data, _ in entries}
    written: set[Path] = set()
    for path in touched:
        try:
            path.write_text(json.dumps(data_by_path[path], indent=2, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Most commonly a read-only bind mount (docker-compose.example.yml's
            # `model_profiles` mount was `:ro` before this feature needed write
            # access — see its updated comment) — surfaced loudly here rather
            # than left to `asyncio.create_task`'s silent "exception never
            # retrieved" fate, since `_start_lora_discovery` never awaits this
            # task. Without this, every "registered ... disabled by default"
            # log line above is a lie: the match was computed, but nothing was
            # ever actually written.
            logger.error(
                "LoRA auto-discovery: failed to write %s (%s) — check that "
                "model_profiles/ is writable (not mounted `:ro` in Docker)",
                path,
                exc,
            )
            continue
        written.add(path)

    return bool(written)


def backfill_trigger_words(profiles_dir: Path) -> bool:
    """Add an explicit `"trigger_words": ""` to every `loras[]` entry
    across `model_profiles/*.json` that doesn't already have the key —
    pure JSON-file hygiene, not a discovery step: pydantic already
    defaults a missing key to `""` in memory (see
    `LoraDefault.trigger_words`), so this changes nothing about how a
    profile actually resolves. It exists purely so opening the file to
    fill one in by hand shows the field as a visible placeholder instead
    of requiring you to already know the schema has it (freshly
    auto-registered entries already get the key from `discover_new_loras`
    itself; this is for everything that predates it, or that a human wrote
    without knowing about it). Deliberately synchronous and independent of
    `Settings.comfyui_loras_dir` — unlike `discover_new_loras`, this needs
    no filesystem access to the actual LoRA files, only to
    `model_profiles/*.json` itself, so it runs unconditionally at every
    boot (`main.py`'s `build_application`) and every on-demand reload
    (`reload_profiles_and_discover` below). Returns True if any file was
    actually modified."""
    entries = load_profile_files(profiles_dir)
    dirty: set[Path] = set()
    for path, data, _ in entries:
        for lora in data.get("loras", []):
            if "trigger_words" not in lora:
                lora["trigger_words"] = ""
                dirty.add(path)

    data_by_path = {path: data for path, data, _ in entries}
    written: set[Path] = set()
    for path in dirty:
        try:
            path.write_text(json.dumps(data_by_path[path], indent=2, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error(
                "trigger_words backfill: failed to write %s (%s) — check that "
                "model_profiles/ is writable",
                path,
                exc,
            )
            continue
        written.add(path)

    return bool(written)


async def reload_profiles_and_discover(
    settings: Settings, storage: Storage
) -> tuple[list[ModelProfile], bool]:
    """The on-demand counterpart to `main.py`'s `_start_lora_discovery`,
    which only ever runs once, at boot. Backs both `/reload` and `/lora`'s
    "🔁 Reload profiles" button (see `lora_menu.py`): run one LoRA
    auto-discovery pass first if `Settings.comfyui_loras_dir` is
    configured (so a LoRA file dropped in since the bot last started gets
    registered too, not just picked up on the *next* restart), then
    backfill any missing `trigger_words` key (see `backfill_trigger_words`
    — unconditional, unlike discovery), then unconditionally reload every
    `model_profiles/*.json` from disk — unlike `_start_lora_discovery`'s
    own reload, which only happens when *discovery itself* wrote
    something, this always re-reads every file, since the whole point of
    an on-demand reload is picking up a profile someone hand-edited in
    between (a manually added LoRA entry, a new `civitai_base_models`
    list, tweaked defaults, a brand-new profile file) — something
    discovery's own "did I write anything" tracking can't see at all.
    Returns the freshly loaded profiles and whether discovery found
    anything new, so callers can tailor their confirmation message."""
    discovered = False
    if settings.comfyui_loras_dir:
        discovered = await discover_new_loras(
            settings.comfyui_loras_dir, settings.model_profiles_dir, storage
        )
    backfill_trigger_words(settings.model_profiles_dir)
    profiles = load_profiles(settings.model_profiles_dir)
    return profiles, discovered
