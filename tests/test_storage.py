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
