from pathlib import Path

from server.backend_state import BackendStateStore


def test_default_state_is_embedding_and_not_dirty(tmp_path: Path) -> None:
    store = BackendStateStore(tmp_path / "backend_state.json")

    state = store.load()

    assert state.active_backend == "embedding"
    assert state.model_dirty is False
    assert state.mindspore_ready is False
    assert state.training is False
    assert state.last_error == ""


def test_mark_dirty_and_activate_mindspore_are_persisted(tmp_path: Path) -> None:
    store = BackendStateStore(tmp_path / "backend_state.json")

    dirty = store.mark_dirty(reason="register")
    assert dirty.model_dirty is True
    assert dirty.last_error == ""

    activated = store.activate_mindspore(
        model_version="20260412-193000",
        class_count=3,
        sample_count=18,
        label_map_path="metadata/mindspore_label_map.json",
        model_path="models/mindspore_active.ckpt",
        trained_at="2026-04-12T19:30:00",
    )

    reloaded = store.load()
    assert activated.active_backend == "mindspore_cnn"
    assert reloaded.model_dirty is False
    assert reloaded.mindspore_ready is True
    assert reloaded.active_model_version == "20260412-193000"
    assert reloaded.class_count == 3
