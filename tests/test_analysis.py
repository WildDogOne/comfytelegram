from comfytelegram.analysis import select_tags


def test_select_tags_filters_by_threshold():
    tags = [("1girl", 0, 0.95), ("solo", 0, 0.10)]
    assert select_tags(tags, threshold=0.35) == "1girl"


def test_select_tags_orders_character_before_general():
    tags = [("outdoors", 0, 0.9), ("hatsune_miku", 4, 0.8)]
    assert select_tags(tags, threshold=0.35) == "hatsune miku, outdoors"


def test_select_tags_sorts_by_descending_confidence_within_category():
    tags = [("blue_hair", 0, 0.5), ("long_hair", 0, 0.9), ("smile", 0, 0.7)]
    assert select_tags(tags, threshold=0.35) == "long hair, smile, blue hair"


def test_select_tags_excludes_rating_category():
    tags = [("1girl", 0, 0.9), ("explicit", 9, 0.99)]
    assert select_tags(tags, threshold=0.35) == "1girl"


def test_select_tags_underscore_to_space_except_kaomoji():
    tags = [("long_hair", 0, 0.9), ("^_^", 0, 0.8)]
    assert select_tags(tags, threshold=0.35) == "long hair, ^_^"


def test_select_tags_empty_when_nothing_meets_threshold():
    assert select_tags([("1girl", 0, 0.1)], threshold=0.35) == ""


def test_select_tags_excludes_noise_tags():
    tags = [("1girl", 0, 0.9), ("watermark", 0, 0.85), ("artist_name", 0, 0.8), ("signature", 0, 0.7)]
    assert select_tags(tags, threshold=0.35) == "1girl"
