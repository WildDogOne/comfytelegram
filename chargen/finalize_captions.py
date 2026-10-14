#!/usr/bin/env python3
"""Turn a reviewed run's prompt captions into final LoRA training captions.

    uv run python chargen/finalize_captions.py --run project_name

The captions generate.py writes record what each image was *prompted*
with, and the model often draws something else (a "headshot portrait" that
came out full body, "pants" that came out shorts). This re-tags every image
in the run folder with the WD14 tagger (the bot's own, see analysis.py),
which describes what is actually in the image, and merges that with the
prompt:

- Fixed prefix (character.json's class_tags): the trigger word, then the
  class tags. kohya only falls back to the dataset folder's class token for
  images without a caption file, so the trigger has to be in the caption;
  train with keep_tokens = 1 so shuffling keeps the trigger in front.
  CLASS_TAG_DROPOUT of the images get the trigger word alone, without the
  class tags: if "marlow" only ever appears next to "anthro, male,
  australian shepherd", the identity is learned by all four together and
  the trigger doesn't work on its own (seen on the first training run).
- Identity tags (fur/marking/eye colours, ears, tail, freckles…) are dropped
  from both sources: whatever is always present gets absorbed into the
  trigger word, and anything left in a caption is learned as optional.
- Artist tags come from the prompt (WD14 can't see artists), so each style
  is attributed to its artist instead of being baked into the LoRA.
- Everything else (pose, clothing, expression, background, framing, view)
  comes from WD14, translated from Danbooru to e621 tag names via the local
  e621 tag database (exact name, else e621's own alias table); tags with no
  e621 equivalent are dropped. A prompt tag WD14 also sees, just below the
  usual threshold, is kept as a cross-check.

Per image this writes:
- <name>.txt — the final caption (kohya reads the .txt with the image's name)
- <name>_qc.txt — review report: final vs. prompt caption, QC flags
  (watermark/text, female, multiple views… as detected by WD14) and every
  dropped tag with the reason
- captions_prompt/<name>.txt — the original prompt caption, backed up once,
  so re-running this script always starts from the prompt caption
plus caption_report.txt, listing only the images that have QC flags. A flag
that turns out wrong on inspection is dismissed in qc_dismissed.json
({"0490.png": ["signature"]}) and stays out of the report on every re-run.
Captions, reports and backups of deleted images are removed.

WD14 scores are cached in wd14_scores.json, keyed by file name and checked
against the image's SHA-256, so a re-run only tags images that are new or
were edited since (e.g. a signature cropped out). Captions are still rebuilt
from the cache for every image, so a change to the merge rules here reaches
the whole set without re-tagging anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

CHARGEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHARGEN_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate import CLASS_TAGS

from comfytelegram.analysis import _load_wd14_model, _prepare_wd14_image

THRESHOLD = 0.35
# A prompt tag WD14 also detects at this confidence or above is kept.
CROSS_CHECK_THRESHOLD = 0.2
# e621 tags rarer than this are Danbooru-isms that happen to exist on e621.
MIN_POSTS = 50
# Share of images captioned with the trigger word but not the rest of
# class_tags. Picked per image by a hash of its file name, so a re-run
# gives every image the same caption prefix as before.
CLASS_TAG_DROPOUT = 0.4

# Character identity, in e621 names: dropped from every caption. A colour
# on a body part ("blue_fur", "orange_eyes", "two_tone_hair") is identity;
# the same part with a non-colour word ("v-shaped_eyebrows",
# "raised_eyebrows") is an expression and stays.
IDENTITY_COLORS = (
    "white|black|blue|dark_blue|light_blue|orange|brown|tan|grey|gray|red|yellow|green|pink"
    "|purple|cream|beige|amber|teal|cyan|aqua|multicolored|two_tone|tricolored"
)
IDENTITY_PARTS = (
    "fur|hair|eyes|body|ears|inner_ear|tail|tail_tip|nose|skin|markings|face|head|cheeks"
    "|snout|muzzle|freckles|eyebrows|hands|feet|toes|fingers|paws|pawpads|spots|belly|chest"
    "|sclera|pupils|claws"
)
IDENTITY_RE = re.compile(rf"(?:{IDENTITY_COLORS})_(?:{IDENTITY_PARTS})")
IDENTITY_TAGS = {
    "anthro",
    "male",
    "australian_shepherd",
    "canine",
    "canid",
    "canis",
    "mammal",
    "domestic_dog",
    "herding_dog",
    "pastoral_dog",
    "dog",
    "freckles",
    "tail",
    "fur",
    "hair",
    "animal_ears",
    "floppy_ears",
    "dog_ears",
    "dog_tail",
    "animal_nose",
    "body_fur",
    "short_hair",
    "long_hair",
    "medium_hair",
    "messy_hair",
    "hair_between_eyes",
    "fluffy",
    "fluffy_tail",
    "markings",
    "facial_markings",
    "eye_markings",
    "cheek_markings",
    "spots",
    "spotted_body",
    "spotted_fur",
    "multicolored_body",
    "multicolored_fur",
    "two_tone_body",
    "two_tone_fur",
    "multicolored_tail",
    "two_tone_tail",
    "5_fingers",
    "fingers",
    "clothed",
    "clothing",
    "solo",
}
# Wrong or useless once mapped to e621 (WD14 name or mapped name).
BLOCKED = {"facing_viewer", "male_focus", "furry", "furry_male", "1boy"}
# The character is male ("male" is in CLASS_TAGS), so WD14's female tags —
# mostly a reaction to the cute art style — are dropped without a QC flag.
FEMALE_TAGS = {"female", "furry_female", "1girl", "2girls", "multiple_girls", "girl", "dog_girl"}
BLOCKED |= FEMALE_TAGS
# Danbooru names whose e621 counterpart isn't reachable through e621's
# alias table (or would land on a near-empty Danbooru-ism).
TRANSLATE = {
    "upper_body": "half-length_portrait",
    "portrait": "headshot_portrait",
    "cowboy_shot": "three-quarter_portrait",
    "holding": "holding_object",
}
# Detected → never captioned, but reported for review.
FLAG_TAGS = {
    "twitter_username": "watermark",
    "patreon_username": "watermark",
    "patreon_logo": "watermark",
    "watermark": "watermark",
    "web_address": "watermark",
    "url": "watermark",
    "signature": "signature",
    "artist_name": "signature",
    "dated": "signature",
    "text": "text",
    "english_text": "text",
    "speech_bubble": "text",
    # A drawing defect rather than a gender call, so still worth a look.
    "breasts": "breasts",
    "multiple_views": "multiple views",
    "multiple_angles": "multiple views",
    "model_sheet": "multiple views",
    "reference_sheet": "multiple views",
    "2boys": "multiple characters",
    "multiple_boys": "multiple characters",
    "duo": "multiple characters",
    "group": "multiple characters",
    "nude": "nudity",
    "completely_nude": "nudity",
    "penis": "nudity",
    "nipples": "nudity",
    "bottomless": "nudity",
}


class E621:
    def __init__(self, db_path: Path):
        self.con = sqlite3.connect(db_path)

    def resolve(self, name: str) -> tuple[str | None, int]:
        """A WD14 (Danbooru) tag name → (e621 tag name, post count), or
        (None, 0) if e621 has no such tag or alias."""
        row = self.con.execute(
            "SELECT post_count FROM tag WHERE source = 'e621' AND name = ?", (name,)
        ).fetchone()
        if row:
            return name, row[0]
        alias = self.con.execute(
            "SELECT tag_name FROM tag_alias WHERE source = 'e621' AND alias = ?", (name,)
        ).fetchone()
        if alias:
            row = self.con.execute(
                "SELECT post_count FROM tag WHERE source = 'e621' AND name = ?", (alias[0],)
            ).fetchone()
            return alias[0], row[0] if row else 0
        return None, 0


def is_identity(name: str) -> bool:
    return name in IDENTITY_TAGS or IDENTITY_RE.fullmatch(name) is not None


def caption_prefix(image_name: str) -> list[str]:
    """The trigger word, plus the class tags unless this image is one of
    the CLASS_TAG_DROPOUT share that goes without them."""
    bucket = int.from_bytes(hashlib.sha256(image_name.encode()).digest()[:4]) / 2**32
    return CLASS_TAGS[:1] if bucket < CLASS_TAG_DROPOUT else CLASS_TAGS


def spaced(name: str) -> str:
    return name.replace("_", " ")


def prompt_tags(record: dict | None) -> set[str]:
    """The prompt's variable tags (pose, outfit…), as underscore names."""
    if record is None:
        return set()
    fields = ("pose", "framing", "view", "expression", "outfit", "setting")
    return {
        t.strip().replace(" ", "_")
        for f in fields
        for t in record.get(f, "").split(",")
        if t.strip()
    }


def wd14_scores(data: bytes, model) -> list[tuple[str, float]]:
    """Every general/character tag at or above CROSS_CHECK_THRESHOLD, most
    confident first."""
    probs = model.session.run(
        None, {model.input_name: _prepare_wd14_image(data, model.target_size)}
    )[0][0]
    return sorted(
        (
            (name, round(float(p), 4))
            for (name, category), p in zip(model.tags, probs)
            if category in (0, 4) and p >= CROSS_CHECK_THRESHOLD
        ),
        key=lambda t: -t[1],
    )


class ScoreCache:
    """wd14_scores.json: file name -> {sha256, min_score, scores}. An entry
    is reused only if the image bytes are unchanged and it was stored with
    a threshold at least as low as the current CROSS_CHECK_THRESHOLD."""

    def __init__(self, path: Path):
        self.path = path
        self.entries = json.loads(path.read_text()) if path.exists() else {}
        self._model = None
        self.tagged = 0

    def scores(self, image: Path) -> list[tuple[str, float]]:
        data = image.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        entry = self.entries.get(image.name)
        if entry and entry["sha256"] == digest and entry["min_score"] <= CROSS_CHECK_THRESHOLD:
            return [(n, s) for n, s in entry["scores"] if s >= CROSS_CHECK_THRESHOLD]
        if self._model is None:  # loaded only if something actually needs tagging
            self._model = _load_wd14_model(
                str(PROJECT_ROOT / "models" / "wd14" / "model.onnx"),
                str(PROJECT_ROOT / "models" / "wd14" / "selected_tags.csv"),
            )
        scored = wd14_scores(data, self._model)
        self.entries[image.name] = {
            "sha256": digest,
            "min_score": CROSS_CHECK_THRESHOLD,
            "scores": scored,
        }
        self.tagged += 1
        return scored

    def save(self, keep: set[str]) -> None:
        """Write the cache, dropping entries for images no longer in the folder."""
        self.entries = {k: v for k, v in self.entries.items() if k in keep}
        self.path.write_text(json.dumps(self.entries, indent=1))


def finalize(
    scored: list[tuple[str, float]],
    record: dict | None,
    prompt_caption: str,
    e621: E621,
    dismissed: list[str],
    prefix: list[str],
) -> tuple[str, str, list[str]]:
    expected = prompt_tags(record)

    kept: list[str] = []
    flags: dict[str, str] = {}
    dropped: list[str] = []
    cross_checked: list[str] = []
    for name, score in scored:
        mapped, posts = e621.resolve(TRANSLATE.get(name, name))
        label = f"{name} {score:.2f}" + (f" -> {mapped}" if mapped and mapped != name else "")
        flag = FLAG_TAGS.get(name) or (FLAG_TAGS.get(mapped) if mapped else None)
        if flag:
            if score >= THRESHOLD:
                flags.setdefault(flag, label)
            continue
        if score < THRESHOLD and not (mapped and mapped in expected):
            continue
        if name in BLOCKED or (mapped and mapped in BLOCKED):
            dropped.append(f"{label} (blocked)")
        elif mapped is None:
            dropped.append(f"{label} (no e621 tag)")
        elif posts < MIN_POSTS:
            dropped.append(f"{label} (rare on e621: {posts} posts)")
        elif is_identity(mapped):
            dropped.append(f"{label} (identity)")
        elif mapped not in kept:
            kept.append(mapped)
            if score < THRESHOLD:
                cross_checked.append(label)

    artists = [f"by {spaced(a)}" for a in (record or {}).get("artists", [])]
    caption = ", ".join(prefix + artists + [spaced(t) for t in kept])

    missing = sorted(spaced(t) for t in expected if t not in kept and not is_identity(t))
    qc = [
        f"final:  {caption}",
        f"prompt: {prompt_caption}",
        "",
        "flags: " + ("; ".join(f"{k} ({v})" for k, v in flags.items()) or "none"),
        "dismissed flags (qc_dismissed.json): " + (", ".join(dismissed) or "none"),
        "prompted but not detected: " + (", ".join(missing) or "none"),
        "kept via prompt cross-check: " + (", ".join(cross_checked) or "none"),
        "dropped:",
        *(f"  {d}" for d in dropped),
    ]
    return caption, "\n".join(qc) + "\n", [f for f in flags if f not in dismissed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="aussie_v1", help="output subfolder name")
    args = parser.parse_args()

    out_dir = CHARGEN_DIR / "output" / args.run
    backup_dir = out_dir / "captions_prompt"
    backup_dir.mkdir(exist_ok=True)
    records = {}
    for line in (out_dir / "manifest.jsonl").read_text().splitlines():
        if line:
            record = json.loads(line)
            records[record["file"]] = record  # last one wins (resumed runs)

    cache = ScoreCache(out_dir / "wd14_scores.json")
    e621 = E621(PROJECT_ROOT / "data" / "tags.sqlite3")

    images = sorted(p for p in out_dir.glob("*.png") if re.fullmatch(r"\d+", p.stem))
    # Leftovers of images deleted during review: captions (still prompt
    # captions if the image went before it was ever finalized), _qc reports
    # and prompt-caption backups.
    names = {image.stem for image in images}
    for path in [*out_dir.glob("*.txt"), *backup_dir.glob("*.txt")]:
        stem = path.stem.removesuffix("_qc")
        if re.fullmatch(r"\d+", stem) and stem not in names:
            path.unlink()
    # False QC flags, confirmed by eye: {"0490.png": ["signature"], ...}.
    dismissed_path = out_dir / "qc_dismissed.json"
    dismissed = json.loads(dismissed_path.read_text()) if dismissed_path.exists() else {}
    report = []
    for image in images:
        caption_path = image.with_suffix(".txt")
        backup = backup_dir / caption_path.name
        if not backup.exists() and caption_path.exists():
            backup.write_text(caption_path.read_text())
        prompt_caption = backup.read_text().strip() if backup.exists() else ""

        caption, qc, flags = finalize(
            cache.scores(image),
            records.get(image.name),
            prompt_caption,
            e621,
            dismissed.get(image.name, []),
            caption_prefix(image.name),
        )
        caption_path.write_text(caption + "\n")
        (out_dir / f"{image.stem}_qc.txt").write_text(qc)
        if flags:
            report.append(f"{image.name}: {', '.join(flags)}")
        print(f"{image.name}  {'FLAGS: ' + ', '.join(flags) if flags else 'ok'}")

    cache.save({image.name for image in images})
    (out_dir / "caption_report.txt").write_text(
        f"{len(report)} of {len(images)} images flagged\n\n" + "\n".join(report) + "\n"
    )
    print(
        f"\nWD14 tagged {cache.tagged} new/changed images, {len(images) - cache.tagged} from cache"
    )
    print(f"{len(report)} of {len(images)} images flagged — see caption_report.txt")


if __name__ == "__main__":
    main()
