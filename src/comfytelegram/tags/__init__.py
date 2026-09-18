"""Local danbooru/e621 tag database backing `/tags` and `/tagcheck`.

See `db.py` for the storage layer, `importer.py` for turning the
DraconicDragon tag-list archive's CSV format into rows, and `update.py` for
fetching that archive's CSVs (used both by `scripts/update_tag_db.py`, the
manual CLI, and `main.py`'s automatic startup refresh).
"""

from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.schema import CATEGORY_LABELS, TagResult, TagRow, TagSource, category_label
from comfytelegram.tags.update import refresh_if_stale

__all__ = [
    "CATEGORY_LABELS",
    "TagDatabase",
    "TagResult",
    "TagRow",
    "TagSource",
    "category_label",
    "refresh_if_stale",
]
