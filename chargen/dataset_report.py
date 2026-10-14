#!/usr/bin/env python3
"""Report which prompt choices survive review more (or less) often than others.

    uv run python chargen/dataset_report.py --run project_name

generate.py draws every wildcard value (and artist) evenly, but review
doesn't keep them evenly: some values make the character come out right
more often (e.g. "serious" tends to show the eyebrows and marks clearly),
others mostly get deleted. Refilling doesn't fix that — each round draws
evenly again — so the kept set ends up at roughly "drawn share × survival
rate" per value, however many cycles are run.

This compares manifest.jsonl (everything ever generated, including deleted
images) with the images still directly in the run folder (what generate.py
also counts as kept). Per field and value:

- kept/gen and survival: how many were kept out of how many generated
- lift: survival relative to the run's overall survival rate (2.0 = kept
  twice as often as average, so twice its fair share of the kept set)
- share: its share of the kept set
- flag: ▲/▼ when the lift is >= 2 or <= 0.5 *and* a binomial test says it's
  unlikely to be chance (p < 0.05), and at least MIN_GENERATED were drawn —
  with a few dozen images per value, a 0/6 or 6/20 is often just noise

Values in wildcards.json that were never drawn, and drawn values no longer
in it, are noted too. Artists are many and thinly spread, so only flagged
ones are listed unless --all.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

CHARGEN_DIR = Path(__file__).resolve().parent

WILDCARD_FIELDS = ["pose", "framing", "view", "expression", "outfit", "setting"]
LIFT_HIGH = 2.0
LIFT_LOW = 0.5
P_THRESHOLD = 0.05
# Below this many generated, a value is never flagged: with ~100 artists each
# tested separately, a 2/2 clears p < 0.05 by luck alone now and then.
MIN_GENERATED = 8


def binom_tail(k: int, n: int, p: float, upper: bool) -> float:
    """P(X >= k) if upper else P(X <= k), for X ~ Binomial(n, p)."""
    ks = range(k, n + 1) if upper else range(k + 1)
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in ks))


def format_of(record: dict) -> str:
    w, h = record["width"], record["height"]
    return "portrait" if w < h else "landscape" if w > h else "square"


def values_of(record: dict, field: str) -> list[str]:
    if field == "format":
        return [format_of(record)]
    if field == "artists":
        return record["artists"]
    if field == "loras":
        return [lora["name"] for lora in record.get("loras") or []] or ["(none)"]
    return [record[field]]


def flag_of(kept: int, gen: int, rate: float) -> str:
    if gen < MIN_GENERATED:
        return ""
    lift = kept / gen / rate
    if lift >= LIFT_HIGH and binom_tail(kept, gen, rate, upper=True) < P_THRESHOLD:
        return "▲"
    if lift <= LIFT_LOW and binom_tail(kept, gen, rate, upper=False) < P_THRESHOLD:
        return "▼"
    return ""


def report_field(
    field: str,
    records: dict,
    kept: set[str],
    rate: float,
    current: list[str] | None,
    show_all: bool,
) -> tuple[list[str], list[str]]:
    gen, keep = Counter(), Counter()
    for name, record in records.items():
        for value in values_of(record, field):
            gen[value] += 1
            if name in kept:
                keep[value] += 1
    total_kept = sum(keep.values()) or 1
    rows = []
    flagged = []
    for value in sorted(gen, key=lambda v: (-keep[v] / gen[v], -gen[v])):
        flag = flag_of(keep[value], gen[value], rate)
        if flag:
            flagged.append(f"{flag} {field}: {value} ({keep[value]}/{gen[value]})")
        if not (show_all or flag):
            continue
        note = "  (not in wildcards.json)" if current is not None and value not in current else ""
        rows.append(
            f"  {flag or ' '} {value[:40]:40} {keep[value]:4}/{gen[value]:<4} "
            f"{keep[value] / gen[value]:5.0%}  {keep[value] / gen[value] / rate:5.2f}  "
            f"{keep[value] / total_kept:6.1%}{note}"
        )
    if current is not None:
        for value in sorted(set(current) - set(gen)):
            rows.append(f"    {value[:40]:40}  never generated")
    header = f"[{field}]" + ("" if show_all else "  (flagged only, --all for every value)")
    lines = [header, f"    {'value':40} kept/gen  surv   lift   share", *rows]
    if not rows:
        lines.append("    nothing flagged")
    return lines + [""], flagged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="aussie_v1", help="output subfolder name")
    parser.add_argument("--all", action="store_true", help="list every artist, not just flagged")
    args = parser.parse_args()

    out_dir = CHARGEN_DIR / "output" / args.run
    records = {}
    for line in (out_dir / "manifest.jsonl").read_text().splitlines():
        if line:
            record = json.loads(line)
            records[record["file"]] = record  # last one wins (resumed runs)
    images = {p.name for p in out_dir.glob("*.png") if re.fullmatch(r"\d+", p.stem)}
    kept = images & records.keys()
    if not kept:
        raise SystemExit(f"no kept images from manifest.jsonl in {out_dir}")
    rate = len(kept) / len(records)

    wildcards_path = CHARGEN_DIR / "wildcards.json"
    wildcards = json.loads(wildcards_path.read_text()) if wildcards_path.exists() else {}

    print(f"{args.run}: {len(records)} generated, {len(kept)} kept ({rate:.0%} survival)")
    if unknown := images - records.keys():
        print(f"{len(unknown)} kept images not in manifest.jsonl, ignored: {sorted(unknown)[:5]}")
    print()

    fields = [*WILDCARD_FIELDS, "format", "artists"]
    if any(r.get("loras") for r in records.values()):
        fields.append("loras")
    all_flagged = []
    for field in fields:
        lines, flagged = report_field(
            field,
            records,
            kept,
            rate,
            wildcards.get(field) if field in WILDCARD_FIELDS else None,
            show_all=args.all or field != "artists",
        )
        print("\n".join(lines))
        all_flagged += flagged

    print(
        f"summary: {len(all_flagged)} flagged (lift >= {LIFT_HIGH} or <= {LIFT_LOW}, p < {P_THRESHOLD})"
    )
    for line in all_flagged:
        print(f"  {line}")


if __name__ == "__main__":
    main()
