from pathlib import Path

import pytest

from comfytelegram.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def test_checkpoint_roundtrip(storage: Storage):
    assert storage.get_checkpoint(1) is None
    storage.set_checkpoint(1, "furrytoonmix_xlIllustriousV2.safetensors")
    assert storage.get_checkpoint(1) == "furrytoonmix_xlIllustriousV2.safetensors"


def test_checkpoint_is_per_chat(storage: Storage):
    storage.set_checkpoint(1, "a.safetensors")
    storage.set_checkpoint(2, "b.safetensors")
    assert storage.get_checkpoint(1) == "a.safetensors"
    assert storage.get_checkpoint(2) == "b.safetensors"


def test_checkpoint_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.set_checkpoint(42, "furrytoonmix_xlIllustriousV2.safetensors")
    s1.close()

    s2 = Storage(db_path)
    assert s2.get_checkpoint(42) == "furrytoonmix_xlIllustriousV2.safetensors"
    s2.close()


def test_override_roundtrip_and_merge(storage: Storage):
    assert storage.get_override(1, "ckpt.safetensors") == {}
    storage.set_override_fields(1, "ckpt.safetensors", {"cfg": 6.0})
    assert storage.get_override(1, "ckpt.safetensors") == {"cfg": 6.0}

    # a second call merges into the existing override, doesn't replace it
    storage.set_override_fields(1, "ckpt.safetensors", {"steps": 25})
    assert storage.get_override(1, "ckpt.safetensors") == {"cfg": 6.0, "steps": 25}

    # overwriting an existing key updates it in place
    storage.set_override_fields(1, "ckpt.safetensors", {"cfg": 4.5})
    assert storage.get_override(1, "ckpt.safetensors") == {"cfg": 4.5, "steps": 25}


def test_override_is_scoped_per_checkpoint(storage: Storage):
    storage.set_override_fields(1, "a.safetensors", {"cfg": 6.0})
    assert storage.get_override(1, "b.safetensors") == {}


def test_clear_override(storage: Storage):
    storage.set_override_fields(1, "ckpt.safetensors", {"cfg": 6.0})
    storage.clear_override(1, "ckpt.safetensors")
    assert storage.get_override(1, "ckpt.safetensors") == {}


def test_character_save_and_get(storage: Storage):
    assert storage.get_character(1, "fox") is None
    storage.save_character(1, "fox", "a fox girl, blue eyes", "extra limbs")
    char = storage.get_character(1, "fox")
    assert char == {"name": "fox", "positive_prompt": "a fox girl, blue eyes", "negative_prompt": "extra limbs"}


def test_character_save_overwrites_by_name(storage: Storage):
    storage.save_character(1, "fox", "first version")
    storage.save_character(1, "fox", "second version", "bad hands")
    char = storage.get_character(1, "fox")
    assert char["positive_prompt"] == "second version"
    assert char["negative_prompt"] == "bad hands"


def test_character_is_scoped_per_chat(storage: Storage):
    storage.save_character(1, "fox", "chat one's fox")
    assert storage.get_character(2, "fox") is None


def test_list_characters_sorted_by_name(storage: Storage):
    storage.save_character(1, "zebra", "z")
    storage.save_character(1, "ant", "a")
    names = [c["name"] for c in storage.list_characters(1)]
    assert names == ["ant", "zebra"]


def test_character_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.save_character(1, "fox", "a fox girl")
    s1.close()

    s2 = Storage(db_path)
    assert s2.get_character(1, "fox")["positive_prompt"] == "a fox girl"
    s2.close()


def test_active_character_roundtrip(storage: Storage):
    assert storage.get_active_character_name(1) is None
    storage.save_character(1, "fox", "a fox girl")
    storage.set_active_character(1, "fox")
    assert storage.get_active_character_name(1) == "fox"

    storage.clear_active_character(1)
    assert storage.get_active_character_name(1) is None


def test_deleting_active_character_clears_activation(storage: Storage):
    storage.save_character(1, "fox", "a fox girl")
    storage.set_active_character(1, "fox")
    storage.delete_character(1, "fox")
    assert storage.get_character(1, "fox") is None
    assert storage.get_active_character_name(1) is None


def test_deleting_inactive_character_keeps_other_active(storage: Storage):
    storage.save_character(1, "fox", "a fox girl")
    storage.save_character(1, "wolf", "a wolf boy")
    storage.set_active_character(1, "fox")
    storage.delete_character(1, "wolf")
    assert storage.get_active_character_name(1) == "fox"
