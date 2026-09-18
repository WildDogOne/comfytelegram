#!/usr/bin/env python3
"""Manual/offline CLI for the local danbooru/e621 tag database
(`Settings.tags_db_path`, backing `/tags` and `/tagcheck`). The bot itself
already does this automatically at startup (see `tags/update.py`'s
`refresh_if_stale`, wired up in `main.py`) for any source that's missing or
older than `Settings.tag_db_max_age_days` — reach for this script when you
want to force a refresh right now, target one source, or import a CSV file
by hand instead of letting the bot fetch from GitHub:

    uv run python scripts/update_tag_db.py                  # both sources
    uv run python scripts/update_tag_db.py --source e621
    uv run python scripts/update_tag_db.py --danbooru-csv /path/to/local.csv

--danbooru-csv/--e621-csv accept a local file instead of downloading, e.g.
output from the upstream danbooru-e621-tag-list-processor project run by
hand (https://github.com/DraconicDragon/danbooru-e621-tag-list-processor) —
same CSV format either way. Not part of the pytest suite (needs network,
unless using the local-file flags).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comfytelegram.settings import load_settings
from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.importer import import_csv
from comfytelegram.tags.schema import TagSource
from comfytelegram.tags.update import refresh_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["danbooru", "e621", "both"], default="both")
    parser.add_argument("--danbooru-csv", type=Path, help="Use a local CSV instead of downloading")
    parser.add_argument("--e621-csv", type=Path, help="Use a local CSV instead of downloading")
    args = parser.parse_args()

    settings = load_settings()
    db = TagDatabase(settings.tags_db_path)
    print(f"Tag database: {settings.tags_db_path}")

    wanted = (
        [TagSource.DANBOORU, TagSource.E621] if args.source == "both" else [TagSource(args.source)]
    )
    local_override = {TagSource.DANBOORU: args.danbooru_csv, TagSource.E621: args.e621_csv}

    for source in wanted:
        override = local_override[source]
        if override is not None:
            print(f"{source.value}: importing local file {override}")
            tag_count, alias_count = import_csv(db, source, override)
        else:
            print(f"{source.value}: downloading latest CSV...")
            tag_count, alias_count = refresh_source(db, source)
        print(f"{source.value}: imported {tag_count} tags, {alias_count} aliases")

    db.close()


if __name__ == "__main__":
    main()
