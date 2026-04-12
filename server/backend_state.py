from dataclasses import asdict, dataclass, replace
from json import JSONDecodeError, dump, loads
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class BackendState:
    active_backend: str = "embedding"
    model_dirty: bool = False
    mindspore_ready: bool = False
    training: bool = False
    last_error: str = ""
    active_model_version: str = ""
    class_count: int = 0
    sample_count: int = 0
    label_map_path: str = ""
    model_path: str = ""
    trained_at: str = ""
    dataset_signature: str = ""


class BackendStateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> BackendState:
        if not self._path.exists():
            return self._save(BackendState())

        try:
            raw = loads(self._path.read_text())
        except (JSONDecodeError, OSError):
            return self._save(BackendState())

        state = self._deserialize(raw)
        return state

    def mark_dirty(self, reason: str = "") -> BackendState:
        state = self.load()
        updated = replace(state, model_dirty=True, training=False, last_error="")
        return self._save(updated)

    def set_training(self, training: bool) -> BackendState:
        state = self.load()
        updated = replace(state, training=training)
        return self._save(updated)

    def set_error(self, message: str) -> BackendState:
        state = self.load()
        updated = replace(state, last_error=message, model_dirty=True)
        return self._save(updated)

    def activate_mindspore(
        self,
        model_version: str,
        class_count: int,
        sample_count: int,
        label_map_path: str,
        model_path: str,
        trained_at: str,
    ) -> BackendState:
        state = self.load()
        updated = replace(
            state,
            active_backend="mindspore_cnn",
            model_dirty=False,
            mindspore_ready=True,
            training=False,
            active_model_version=model_version,
            class_count=class_count,
            sample_count=sample_count,
            label_map_path=label_map_path,
            model_path=model_path,
            trained_at=trained_at,
        )
        return self._save(updated)

    def _save(self, state: BackendState) -> BackendState:
        with self._path.open("w", encoding="utf-8") as out:
            dump(asdict(state), out, ensure_ascii=False, indent=2)
        return state

    @staticmethod
    def _deserialize(raw: Any) -> BackendState:
        if not isinstance(raw, dict):
            return BackendState()

        filtered: Dict[str, Any] = {}
        for field in BackendState.__dataclass_fields__.keys():
            if field in raw:
                filtered[field] = raw[field]
        return BackendState(**filtered)
