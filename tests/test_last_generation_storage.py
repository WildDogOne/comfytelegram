from pathlib import Path

import pytest

from comfytelegram.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "state.sqlite3")
    yield s
    s.close()


def test_last_generation_roundtrip(storage: Storage):
    assert storage.get_last_generation(42) is None

    params = {
        "checkpoint": "ckpt.safetensors",
        "positive_prompt": "a fox",
        "negative_prompt": "low quality",
        "steps": 25,
        "cfg": 6.5,
    }
    storage.set_last_generation(42, params)

    assert storage.get_last_generation(42) == params


def test_last_generation_overwrites_for_same_chat(storage: Storage):
    storage.set_last_generation(42, {"checkpoint": "a.safetensors"})
    storage.set_last_generation(42, {"checkpoint": "b.safetensors"})

    assert storage.get_last_generation(42) == {"checkpoint": "b.safetensors"}


def test_last_generation_is_scoped_per_chat(storage: Storage):
    storage.set_last_generation(1, {"checkpoint": "a.safetensors"})
    storage.set_last_generation(2, {"checkpoint": "b.safetensors"})

    assert storage.get_last_generation(1) == {"checkpoint": "a.safetensors"}
    assert storage.get_last_generation(2) == {"checkpoint": "b.safetensors"}


def test_last_generation_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.set_last_generation(42, {"checkpoint": "ckpt.safetensors"})
    s1.close()

    s2 = Storage(db_path)
    assert s2.get_last_generation(42) == {"checkpoint": "ckpt.safetensors"}
    s2.close()
