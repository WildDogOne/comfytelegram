from pathlib import Path

import pytest

from comfytelegram.storage import IMAGE_FORMAT_JPEG, IMAGE_FORMAT_PNG, Storage


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


def test_image_format_defaults_to_jpeg(storage: Storage):
    assert storage.get_image_format(1) == IMAGE_FORMAT_JPEG


def test_image_format_roundtrip_and_is_per_chat(storage: Storage):
    storage.set_image_format(1, IMAGE_FORMAT_PNG)
    assert storage.get_image_format(1) == IMAGE_FORMAT_PNG
    assert storage.get_image_format(2) == IMAGE_FORMAT_JPEG


def test_image_format_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.set_image_format(42, IMAGE_FORMAT_PNG)
    s1.close()

    s2 = Storage(db_path)
    assert s2.get_image_format(42) == IMAGE_FORMAT_PNG
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
    assert char == {
        "name": "fox",
        "positive_prompt": "a fox girl, blue eyes",
        "negative_prompt": "extra limbs",
    }


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


def test_rename_character_keeps_its_prompt(storage: Storage):
    storage.save_character(1, "fox", "a fox girl", "extra limbs")
    storage.rename_character(1, "fox", "vixen")
    assert storage.get_character(1, "fox") is None
    char = storage.get_character(1, "vixen")
    assert char["positive_prompt"] == "a fox girl"
    assert char["negative_prompt"] == "extra limbs"


def test_renaming_active_character_updates_the_activation_pointer(storage: Storage):
    storage.save_character(1, "fox", "a fox girl")
    storage.set_active_character(1, "fox")
    storage.rename_character(1, "fox", "vixen")
    assert storage.get_active_character_name(1) == "vixen"


def test_renaming_inactive_character_leaves_activation_alone(storage: Storage):
    storage.save_character(1, "fox", "a fox girl")
    storage.save_character(1, "wolf", "a wolf boy")
    storage.set_active_character(1, "fox")
    storage.rename_character(1, "wolf", "coyote")
    assert storage.get_active_character_name(1) == "fox"


def test_inpaint_job_roundtrip_carries_message_thread_id(storage: Storage):
    storage.store_inpaint_job("tok1", "result1", 42, message_thread_id=827915)
    (job,) = storage.list_inpaint_jobs()
    assert job == {
        "token": "tok1",
        "result_id": "result1",
        "chat_id": 42,
        "message_thread_id": 827915,
        "kind": "hand",
    }


def test_inpaint_job_message_thread_id_defaults_to_none(storage: Storage):
    # A chat with no forum topics has nothing to record here — Telegram
    # itself gives such messages no message_thread_id at all.
    storage.store_inpaint_job("tok1", "result1", 42)
    (job,) = storage.list_inpaint_jobs()
    assert job["message_thread_id"] is None


def test_inpaint_job_roundtrip_carries_kind(storage: Storage):
    # "🩹 Fix Artifact" shares the same relay/poller machinery as "🖌️ Draw
    # Mask" — `kind` is what `_process_one_inpaint_job` uses to tell which
    # post_process kind/label/redo button applies once the mask comes back.
    storage.store_inpaint_job("tok1", "result1", 42, kind="fix")
    (job,) = storage.list_inpaint_jobs()
    assert job["kind"] == "fix"


def test_delete_inpaint_job_removes_it(storage: Storage):
    storage.store_inpaint_job("tok1", "result1", 42)
    storage.delete_inpaint_job("tok1")
    assert storage.list_inpaint_jobs() == []


def test_inpaint_job_message_thread_id_column_is_added_to_a_pre_existing_table(tmp_path: Path):
    # Reproduces a real deployed database: `inpaint_job` already existed
    # (created before `message_thread_id`/`kind` were added), so `CREATE
    # TABLE IF NOT EXISTS` is a no-op against it and each new column needs
    # its own guarded ALTER TABLE — see Storage.__init__'s
    # _add_column_if_missing calls and CLAUDE.md's storage.py note on this
    # exact pattern.
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    with s1._conn:
        s1._conn.execute("DROP TABLE inpaint_job")
        s1._conn.execute(
            "CREATE TABLE inpaint_job ("
            "token TEXT PRIMARY KEY, result_id TEXT NOT NULL, "
            "chat_id INTEGER NOT NULL, created_at REAL NOT NULL)"
        )
    s1.close()

    s2 = Storage(db_path)
    s2.store_inpaint_job("tok1", "result1", 42, message_thread_id=5)
    assert s2.list_inpaint_jobs() == [
        {
            "token": "tok1",
            "result_id": "result1",
            "chat_id": 42,
            "message_thread_id": 5,
            "kind": "hand",
        }
    ]
    s2.close()


def test_inpaint_redo_roundtrip(storage: Storage):
    assert storage.get_inpaint_redo("result1") is None
    storage.store_inpaint_redo("result1", "file123", "source.png", b"\x89PNGmaskbytes")
    assert storage.get_inpaint_redo("result1") == {
        "source_file_id": "file123",
        "source_filename": "source.png",
        "mask_png": b"\x89PNGmaskbytes",
    }


def test_inpaint_redo_is_scoped_per_result_id(storage: Storage):
    storage.store_inpaint_redo("result1", "file123", "a.png", b"mask-a")
    assert storage.get_inpaint_redo("result2") is None


def test_inpaint_redo_survives_reopen(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    s1 = Storage(db_path)
    s1.store_inpaint_redo("result1", "file123", "source.png", b"mask-bytes")
    s1.close()

    s2 = Storage(db_path)
    assert s2.get_inpaint_redo("result1") == {
        "source_file_id": "file123",
        "source_filename": "source.png",
        "mask_png": b"mask-bytes",
    }
    s2.close()
