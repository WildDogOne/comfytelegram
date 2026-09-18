import time
from pathlib import Path

import pytest

from comfytelegram.tags.db import TagDatabase
from comfytelegram.tags.importer import import_csv, parse_csv
from comfytelegram.tags.schema import TagRow, TagSource


@pytest.fixture
def db(tmp_path: Path) -> TagDatabase:
    database = TagDatabase(tmp_path / "tags.sqlite3")
    yield database
    database.close()


def _csv(tmp_path: Path, name: str, rows: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_parse_csv_splits_aliases(tmp_path: Path):
    path = _csv(
        tmp_path,
        "sample.csv",
        [
            'anthro,0,4156082,"anthromorph,anthropomorph"',
            "hi_res,7,3908371,",
        ],
    )
    rows = list(parse_csv(path))
    assert rows[0] == TagRow("anthro", 0, 4156082, ("anthromorph", "anthropomorph"))
    assert rows[1] == TagRow("hi_res", 7, 3908371, ())


def test_import_csv_populates_stats(db: TagDatabase, tmp_path: Path):
    path = _csv(tmp_path, "e621.csv", ['fox,5,50000,"foxes"', "male,0,3000000,"])
    tag_count, alias_count = import_csv(db, TagSource.E621, path)
    assert (tag_count, alias_count) == (2, 1)
    assert db.stats() == {TagSource.DANBOORU: 0, TagSource.E621: 2}


def test_last_imported_is_none_until_a_source_is_imported(db: TagDatabase):
    assert db.last_imported() == {TagSource.DANBOORU: None, TagSource.E621: None}


def test_last_imported_tracks_replace_source(db: TagDatabase, tmp_path: Path):
    before = time.time()
    import_csv(db, TagSource.E621, _csv(tmp_path, "e.csv", ["fox,5,1000,"]))
    after = time.time()

    imported = db.last_imported()
    assert imported[TagSource.DANBOORU] is None
    assert before <= imported[TagSource.E621] <= after


def test_replace_source_swaps_without_touching_other_source(db: TagDatabase, tmp_path: Path):
    danbooru_csv = _csv(tmp_path, "danbooru.csv", ["1girl,0,100000,"])
    import_csv(db, TagSource.DANBOORU, danbooru_csv)

    e621_csv_v1 = _csv(tmp_path, "e621_v1.csv", ["fox,5,50000,"])
    import_csv(db, TagSource.E621, e621_csv_v1)
    assert db.stats() == {TagSource.DANBOORU: 1, TagSource.E621: 1}

    e621_csv_v2 = _csv(tmp_path, "e621_v2.csv", ["wolf,5,40000,"])
    import_csv(db, TagSource.E621, e621_csv_v2)
    assert db.stats() == {TagSource.DANBOORU: 1, TagSource.E621: 1}
    assert db.lookup_exact("fox", [TagSource.E621]) is None
    assert db.lookup_exact("wolf", [TagSource.E621]) is not None
    assert db.lookup_exact("1girl", [TagSource.DANBOORU]) is not None


def test_search_ranks_prefix_matches_above_substring_matches(db: TagDatabase, tmp_path: Path):
    path = _csv(
        tmp_path,
        "e621.csv",
        [
            "anthro,0,4000000,",
            "big_anthro_dude,0,100,",
        ],
    )
    import_csv(db, TagSource.E621, path)

    results = db.search("anthro", [TagSource.E621], limit=10)
    assert [r.name for r in results] == ["anthro", "big_anthro_dude"]


def test_search_ranks_by_post_count_within_same_rank(db: TagDatabase, tmp_path: Path):
    path = _csv(
        tmp_path,
        "e621.csv",
        [
            "fox,5,1000,",
            "fox_ears,5,50000,",
        ],
    )
    import_csv(db, TagSource.E621, path)

    results = db.search("fox", [TagSource.E621], limit=10)
    assert [r.name for r in results] == ["fox_ears", "fox"]


def test_search_by_frequency_ranks_a_common_substring_match_above_a_rare_prefix_match(
    db: TagDatabase, tmp_path: Path
):
    path = _csv(
        tmp_path,
        "e621.csv",
        [
            "anthro,0,100,",
            "big_anthro_dude,0,4000000,",
        ],
    )
    import_csv(db, TagSource.E621, path)

    results = db.search("anthro", [TagSource.E621], limit=10, by_frequency=True)
    assert [r.name for r in results] == ["big_anthro_dude", "anthro"]


def test_search_by_frequency_still_caps_at_limit(db: TagDatabase, tmp_path: Path):
    path = _csv(
        tmp_path,
        "e621.csv",
        [
            "fox,5,1000,",
            "fox_ears,5,50000,",
            "fox_tail,5,20000,",
        ],
    )
    import_csv(db, TagSource.E621, path)

    results = db.search("fox", [TagSource.E621], limit=2, by_frequency=True)
    assert [r.name for r in results] == ["fox_ears", "fox_tail"]


def test_search_falls_back_to_fuzzy_matching_when_no_substring_hits(
    db: TagDatabase, tmp_path: Path
):
    path = _csv(tmp_path, "danbooru.csv", ["1girl,0,5000000,", "solo,0,4000000,"])
    import_csv(db, TagSource.DANBOORU, path)

    # "1gril" shares no substring with "1girl" (the "ri"/"ir" are
    # transposed), so plain substring matching alone finds nothing.
    results = db.search("1gril", [TagSource.DANBOORU], limit=10)
    assert [r.name for r in results] == ["1girl"]


def test_search_never_pads_real_matches_out_with_fuzzy_guesses(db: TagDatabase, tmp_path: Path):
    # Regression test for a reported bug: "/tags dimple" had 5 genuine
    # substring hits, well under limit=15, and the old "top up to limit"
    # fuzzy fallback padded the rest of the page with unrelated tags —
    # "temple", "nipples", "male" — that just happened to score above the
    # ratio threshold against a handful of e621's most common tags.
    path = _csv(
        tmp_path,
        "e621.csv",
        [
            "dimple,0,147,",
            "back_dimples,0,659,",
            "tail_dimple,0,277,",
            "dimple_piercing,0,128,",
            "butt_dimples,0,78,",
            "temple,0,1650,",
            "nipples,0,1559290,",
            "male,0,3093404,",
        ],
    )
    import_csv(db, TagSource.E621, path)

    results = db.search("dimple", [TagSource.E621], limit=15, by_frequency=True)
    assert {r.name for r in results} == {
        "dimple",
        "back_dimples",
        "tail_dimple",
        "dimple_piercing",
        "butt_dimples",
    }


def test_search_fuzzy_fallback_never_runs_with_any_real_match(db: TagDatabase, tmp_path: Path):
    path = _csv(
        tmp_path,
        "danbooru.csv",
        [
            "big_anthro_dude,0,100,",
            "anthro,0,4000000,",
        ],
    )
    import_csv(db, TagSource.DANBOORU, path)

    # Even a single real substring match should stand on its own, not get
    # topped up with fuzzy guesses to fill out `limit`.
    results = db.search("anthro", [TagSource.DANBOORU], limit=15)
    assert [r.name for r in results] == ["anthro", "big_anthro_dude"]


def test_search_fuzzy_fallback_ranks_by_closeness_then_frequency(db: TagDatabase, tmp_path: Path):
    path = _csv(
        tmp_path,
        "danbooru.csv",
        [
            "grey_eyes,0,1000,",  # closer match, far fewer posts
            "gray_ears,0,5000000,",  # less close, but way more popular
        ],
    )
    import_csv(db, TagSource.DANBOORU, path)

    # Closeness of match outranks raw popularity in the fuzzy fallback —
    # unlike a real substring/prefix match, a fuzzy guess is only useful if
    # it's actually the tag the user meant.
    results = db.search("gray_eyes", [TagSource.DANBOORU], limit=10)
    assert [r.name for r in results] == ["grey_eyes", "gray_ears"]


def test_search_fuzzy_fallback_ignores_unrelated_tags(db: TagDatabase, tmp_path: Path):
    path = _csv(tmp_path, "danbooru.csv", ["humanoid,0,1000,"])
    import_csv(db, TagSource.DANBOORU, path)

    assert db.search("anthro", [TagSource.DANBOORU], limit=10) == []


def test_search_matches_aliases_and_resolves_to_canonical_tag(db: TagDatabase, tmp_path: Path):
    path = _csv(tmp_path, "e621.csv", ['anthro,0,4000000,"anthropomorphic,antro"'])
    import_csv(db, TagSource.E621, path)

    results = db.search("antro", [TagSource.E621], limit=10)
    assert len(results) == 1
    assert results[0].name == "anthro"
    assert results[0].matched_alias == "antro"


def test_search_name_match_wins_over_alias_match_for_same_tag(db: TagDatabase, tmp_path: Path):
    # Searching "fox" matches both this tag's own name and its "foxes"
    # alias (which also contains "fox") — the direct name match must win,
    # not get shadowed by its own alias hit.
    path = _csv(tmp_path, "e621.csv", ['fox,5,50000,"foxes"'])
    import_csv(db, TagSource.E621, path)

    results = db.search("fox", [TagSource.E621], limit=10)
    assert results[0].matched_alias is None


def test_search_is_scoped_to_requested_sources(db: TagDatabase, tmp_path: Path):
    import_csv(db, TagSource.DANBOORU, _csv(tmp_path, "d.csv", ["1girl,0,100000,"]))
    import_csv(db, TagSource.E621, _csv(tmp_path, "e.csv", ["anthro,0,4000000,"]))

    assert [r.name for r in db.search("1girl", [TagSource.E621])] == []
    assert [r.name for r in db.search("1girl", [TagSource.DANBOORU])] == ["1girl"]
    assert {r.source for r in db.search("a", [TagSource.DANBOORU, TagSource.E621])} <= {
        TagSource.DANBOORU,
        TagSource.E621,
    }


def test_search_accepts_spaces_in_place_of_underscores(db: TagDatabase, tmp_path: Path):
    import_csv(db, TagSource.E621, _csv(tmp_path, "e.csv", ["hi_res,7,3000000,"]))
    assert [r.name for r in db.search("hi res", [TagSource.E621])] == ["hi_res"]


def test_lookup_exact_does_not_match_substrings(db: TagDatabase, tmp_path: Path):
    import_csv(db, TagSource.E621, _csv(tmp_path, "e.csv", ["anthro,0,4000000,"]))
    assert db.lookup_exact("anthr", [TagSource.E621]) is None
    assert db.lookup_exact("anthro", [TagSource.E621]) is not None


def test_lookup_exact_resolves_aliases(db: TagDatabase, tmp_path: Path):
    import_csv(db, TagSource.E621, _csv(tmp_path, "e.csv", ['anthro,0,4000000,"antro"']))
    result = db.lookup_exact("antro", [TagSource.E621])
    assert result is not None
    assert result.name == "anthro"
    assert result.matched_alias == "antro"
